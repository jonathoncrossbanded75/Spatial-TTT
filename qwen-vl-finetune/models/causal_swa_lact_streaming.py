# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .causal_swa_lact import (
    FLASH_ATTN_AVAILABLE,
    LaCTCache,
    LaCTLayerState,
    Qwen3VLLaCTSWIGLULayer,
    apply_partial_rotary_pos_emb,
    apply_rotary_pos_emb,
    flash_attn_varlen_func,
    l2_norm,
    silu_backprop,
    zeropower_via_newtonschulz5,
)


class Qwen3VLLaCTSWIGLULayerStreaming(Qwen3VLLaCTSWIGLULayer):
    """
    Streaming-prefill variant to reduce peak memory for long sequences.
    """

    def _build_fast_qkv_chunk(
        self,
        q_normed: torch.Tensor,
        k_normed: torch.Tensor,
        v_raw: torch.Tensor,
        start: int,
        end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = q_normed.shape[0]
        chunk_len = end - start

        q_chunk = q_normed[:, start:end, :, :].reshape(batch_size, chunk_len, -1)
        q_chunk = q_chunk * self.q_scale.view(1, 1, -1) + self.q_offset.view(1, 1, -1)

        n_rep = self.num_attn_heads // self.num_kv_heads
        if n_rep == 1:
            k_chunk = k_normed[:, start:end, :, :].reshape(batch_size, chunk_len, -1)
            k_chunk = (
                k_chunk * self.k_scale.view(1, 1, -1) + self.k_offset.view(1, 1, -1)
            )
            v_chunk = v_raw[:, start:end, :]
            return q_chunk, k_chunk, v_chunk

        k_chunk = k_normed[:, start:end, :, :]
        v_chunk = v_raw[:, start:end, :]
        device = q_normed.device
        dtype = q_normed.dtype

        k_out = torch.empty(
            batch_size, chunk_len, self.hidden_size, device=device, dtype=dtype
        )
        v_out = torch.empty_like(k_out)

        k_view = k_out.view(batch_size, chunk_len, self.num_kv_heads, n_rep, self.head_dim)
        v_view = v_out.view(batch_size, chunk_len, self.num_kv_heads, n_rep, self.head_dim)

        k_scale = self.k_scale.view(self.num_kv_heads, n_rep, self.head_dim)
        k_offset = self.k_offset.view(self.num_kv_heads, n_rep, self.head_dim)

        k_in = k_chunk.unsqueeze(3)
        torch.mul(k_in, k_scale, out=k_view)
        k_view.add_(k_offset)

        v_in = v_chunk.view(batch_size, chunk_len, self.num_kv_heads, self.head_dim)
        v_view.copy_(v_in.unsqueeze(3).expand(-1, -1, -1, n_rep, -1))

        return q_chunk, k_out, v_out

    def _prepare_fast_qkv(
        self,
        q_chunk: torch.Tensor,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fast_q = rearrange(q_chunk, "b s (h d) -> (b h) s d", h=self.num_fw_heads)
        fast_k = rearrange(k_chunk, "b s (h d) -> (b h) s d", h=self.num_fw_heads)
        fast_v = rearrange(v_chunk, "b s (h d) -> (b h) s d", h=self.num_fw_heads)

        if self.qkv_silu:
            fast_q = F.silu(fast_q)
            fast_k = F.silu(fast_k)
            if not self.no_v_silu:
                fast_v = F.silu(fast_v)

        fast_q = l2_norm(fast_q)
        fast_k = l2_norm(fast_k)

        if not self.ttt_nope:
            fast_q, fast_k = apply_partial_rotary_pos_emb(
                fast_q, fast_k, cos, sin, rope_dim=self.head_dim
            )

        return fast_q, fast_k, fast_v

    def _init_cache_from_prefill_streaming(
        self,
        hidden_states: torch.Tensor,
        q_normed: torch.Tensor,
        k_normed: torch.Tensor,
        v_raw: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        lact_cache: LaCTCache,
        q_flat: Optional[torch.Tensor] = None,
        k_flat: Optional[torch.Tensor] = None,
        v_expanded: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()
        assert batch_size == 1, "Decode cache only supports batch_size=1"

        cos_full, sin_full = position_embeddings

        if self.w0_w2_low_rank > 0:
            fw_w0 = self.w0()
            fw_w2 = self.w2()
        else:
            fw_w0 = self.w0.clone()
            fw_w2 = self.w2.clone()
        fw_w1 = self.w1.clone()

        if self.fp32_states:
            fw_w0 = fw_w0.to(torch.float32)
            fw_w1 = fw_w1.to(torch.float32)
            fw_w2 = fw_w2.to(torch.float32)

        w0_norm = fw_w0.norm(dim=2, keepdim=True)
        w1_norm = fw_w1.norm(dim=2, keepdim=True)
        w2_norm = fw_w2.norm(dim=2, keepdim=True)

        if self.use_momentum:
            dw0_momentum = torch.zeros_like(fw_w0)
            dw1_momentum = torch.zeros_like(fw_w1)
            dw2_momentum = torch.zeros_like(fw_w2)
        else:
            dw0_momentum = None
            dw1_momentum = None
            dw2_momentum = None

        ttt_output = torch.empty_like(hidden_states)
        chunk_size = self.lact_chunk_size
        main_end = max(seq_len - chunk_size, 0)

        def build_lr_momentum(start: int, end: int):
            lr = self.lr_proj(hidden_states[:, start:end, :])
            if self.lr_parameterization == "mamba":
                lr = F.softplus(lr.float() + self.base_lr_inv)
            fw_lr = rearrange(lr, "b s (h d) -> (b h) s d", h=self.num_fw_heads)
            fw_lr1, fw_lr2, fw_lr3 = fw_lr.chunk(3, dim=-1)
            if self.use_momentum:
                momentum = self.momentum_proj(hidden_states[:, start:end, :]).float()
                momentum = rearrange(
                    momentum, "b s (h d) -> (b h) s d", h=self.num_fw_heads
                )
            else:
                momentum = None
            return fw_lr1, fw_lr2, fw_lr3, momentum

        def write_output(
            start: int, end: int, fw_x: torch.Tensor, hs_chunk: torch.Tensor
        ) -> None:
            ttt_x_normed = self.ttt_norm(fw_x)
            if self.learnable_ttt_scale:
                ttt_scale = F.silu(self.ttt_scale_proj(hs_chunk), inplace=False)
                ttt_scale = rearrange(
                    ttt_scale, "b s (n d) -> (b n) s d", n=self.num_fw_heads
                )
                ttt_x_normed = ttt_x_normed * ttt_scale

            ttt_chunk = rearrange(
                ttt_x_normed,
                "(b n) s d -> b s (n d)",
                n=self.num_fw_heads,
                b=batch_size,
            )
            ttt_output[:, start:end, :] = ttt_chunk.type_as(hidden_states)

        for start in range(0, main_end, chunk_size):
            end = start + chunk_size
            if q_flat is None:
                q_chunk, k_chunk, v_chunk = self._build_fast_qkv_chunk(
                    q_normed, k_normed, v_raw, start, end
                )
            else:
                q_chunk = q_flat[:, start:end, :]
                k_chunk = k_flat[:, start:end, :]
                v_chunk = v_expanded[:, start:end, :]
            cos = cos_full[:, start:end, :]
            sin = sin_full[:, start:end, :]
            fast_q, fast_k, fast_v = self._prepare_fast_qkv(
                q_chunk, k_chunk, v_chunk, cos, sin
            )

            fw_lr1, fw_lr2, fw_lr3, momentum = build_lr_momentum(start, end)

            q_t = fast_q.transpose(1, 2)
            v_t = fast_v.transpose(1, 2)
            if self.fp32_states:
                q_t = q_t.float()
                v_t = v_t.float()
                fast_k = fast_k.float()

            h = torch.bmm(fw_w2, q_t)
            gate = F.silu(torch.bmm(fw_w0, q_t), inplace=True)
            output = torch.bmm(fw_w1, gate * h)
            fw_x = output.transpose(1, 2)
            write_output(start, end, fw_x, hidden_states[:, start:end, :])

            gate_before_act = torch.bmm(fw_w0, fast_k.transpose(1, 2))
            hidden_before_mul = torch.bmm(fw_w2, fast_k.transpose(1, 2))
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
            dhidden = torch.bmm(fw_w1.transpose(1, 2), v_t)
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            dw1 = torch.bmm(v_t, (hidden.transpose(1, 2) * fw_lr2).type_as(v_t))
            dw0 = torch.bmm(
                dgate_before_act, (fast_k * fw_lr1).type_as(dgate_before_act)
            )
            dw2 = torch.bmm(
                dhidden_before_mul, (fast_k * fw_lr3).type_as(dhidden_before_mul)
            )

            if momentum is not None:
                m_i = momentum.mean(dim=1, keepdim=True)
                dw0 = dw0 + dw0_momentum * m_i
                dw1 = dw1 + dw1_momentum * m_i
                dw2 = dw2 + dw2_momentum * m_i
                dw0_momentum = dw0
                dw1_momentum = dw1
                dw2_momentum = dw2

            if self.use_muon:
                dw0 = zeropower_via_newtonschulz5(dw0)
                dw1 = zeropower_via_newtonschulz5(dw1)
                dw2 = zeropower_via_newtonschulz5(dw2)

            fw_w0 = fw_w0 + dw0
            fw_w1 = fw_w1 + dw1
            fw_w2 = fw_w2 + dw2

            fw_w0 = fw_w0 / (fw_w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
            fw_w1 = fw_w1 / (fw_w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
            fw_w2 = fw_w2 / (fw_w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

        start = main_end
        end = seq_len
        if q_flat is None:
            q_chunk, k_chunk, v_chunk = self._build_fast_qkv_chunk(
                q_normed, k_normed, v_raw, start, end
            )
        else:
            q_chunk = q_flat[:, start:end, :]
            k_chunk = k_flat[:, start:end, :]
            v_chunk = v_expanded[:, start:end, :]
        cos = cos_full[:, start:end, :]
        sin = sin_full[:, start:end, :]
        fast_q, fast_k, fast_v = self._prepare_fast_qkv(q_chunk, k_chunk, v_chunk, cos, sin)
        fw_lr1, fw_lr2, fw_lr3, momentum = build_lr_momentum(start, end)

        q_t = fast_q.transpose(1, 2)
        v_t = fast_v.transpose(1, 2)
        if self.fp32_states:
            q_t = q_t.float()
            v_t = v_t.float()
            fast_k = fast_k.float()

        h = torch.bmm(fw_w2, q_t)
        gate = F.silu(torch.bmm(fw_w0, q_t), inplace=True)
        output = torch.bmm(fw_w1, gate * h)
        fw_x = output.transpose(1, 2)
        write_output(start, end, fw_x, hidden_states[:, start:end, :])

        state = LaCTLayerState(
            w0=fw_w0,
            w1=fw_w1,
            w2=fw_w2,
            w0_norm=w0_norm,
            w1_norm=w1_norm,
            w2_norm=w2_norm,
            dw0_momentum=dw0_momentum,
            dw1_momentum=dw1_momentum,
            dw2_momentum=dw2_momentum,
            pending_k=fast_k,
            pending_v=fast_v,
            pending_lr0=fw_lr1,
            pending_lr1=fw_lr2,
            pending_lr2=fw_lr3,
            pending_momentum=momentum,
        )
        lact_cache.set_layer_state(self.layer_idx, state)

        return ttt_output

    def _forward_prefill_streaming(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values,
        cache_position: Optional[torch.LongTensor],
        lact_cache: LaCTCache,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.size()

        q = self.attn_layer.q_proj(hidden_states)
        k = self.attn_layer.k_proj(hidden_states)
        v = self.attn_layer.v_proj(hidden_states)

        q_normed = self.attn_layer.q_norm(
            rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        )
        k_normed = self.attn_layer.k_norm(
            rearrange(k, "... (h d) -> ... h d", d=self.head_dim)
        )

        q_flat = None
        k_flat = None
        v_expanded = None
        if self.use_conv_layer:
            video_mask = kwargs.get("video_mask", None)
            video_grid_thw = kwargs.get("video_grid_thw", None)
            if video_mask is not None:
                t, h, w = video_grid_thw[0]
                h, w = h // 2, w // 2

                def short_conv(x, conv_layer, inference: bool):
                    if inference:
                        x_clone = x
                    else:
                        x_clone = x.clone()
                    video_tokens = x[video_mask]
                    num_videos = video_tokens.size(0) // (t * h * w)
                    video_reshaped = rearrange(
                        video_tokens,
                        "(n t h w) c -> n c t h w",
                        n=num_videos,
                        t=t,
                        h=h,
                        w=w,
                    )
                    video_conv = conv_layer(video_reshaped)
                    x_clone[video_mask] = rearrange(video_conv, "n c t h w -> (n t h w) c")
                    return x_clone

                q_flat = rearrange(q_normed, "... h d -> ... (h d)")
                k_flat = rearrange(k_normed, "... h d -> ... (h d)")
                n_rep = self.num_attn_heads // self.num_kv_heads
                if n_rep > 1:
                    k_flat = k_flat.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
                    k_flat = k_flat.unsqueeze(3).expand(-1, -1, -1, n_rep, -1)
                    k_flat = k_flat.reshape(batch_size, seq_len, -1)
                    v_expanded = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
                    v_expanded = v_expanded.unsqueeze(3).expand(-1, -1, -1, n_rep, -1)
                    v_expanded = v_expanded.reshape(batch_size, seq_len, -1)
                else:
                    v_expanded = v

                q_flat, k_flat = self._rescale_qk(q_flat, k_flat)
                q_flat = short_conv(q_flat, self.conv_q, lact_cache is not None)
                k_flat = short_conv(k_flat, self.conv_k, lact_cache is not None)
                v_expanded = short_conv(v_expanded, self.conv_v, lact_cache is not None)

        ttt_output = self._init_cache_from_prefill_streaming(
            hidden_states,
            q_normed,
            k_normed,
            v,
            position_embeddings,
            lact_cache,
            q_flat=q_flat,
            k_flat=k_flat,
            v_expanded=v_expanded,
        )

        query_states = q_normed.transpose(1, 2)
        key_states = k_normed.transpose(1, 2)
        value_states = rearrange(v, "b s (h d) -> b h s d", d=self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=1
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        use_varlen_attn = (
            FLASH_ATTN_AVAILABLE
            and self.config._attn_implementation == "flash_attention_2"
        )

        if attention_mask is None:
            cu_seqlens_q = torch.tensor(
                [0, query_states.shape[2]],
                device=hidden_states.device,
                dtype=torch.int32,
            )
            cu_seqlens_k = torch.tensor(
                [0, key_states.shape[2]],
                device=hidden_states.device,
                dtype=torch.int32,
            )
        else:
            cu_seqlens_q = attention_mask
            cu_seqlens_k = attention_mask

        if use_varlen_attn:
            query_fa = query_states.transpose(1, 2).squeeze(0)
            key_fa = key_states.transpose(1, 2).squeeze(0)
            value_fa = value_states.transpose(1, 2).squeeze(0)

            with torch.no_grad():
                max_seqlen_q = max(
                    [
                        cu_seqlens_q[idx + 1] - cu_seqlens_q[idx]
                        for idx in range(cu_seqlens_q.size(0) - 1)
                    ]
                ).item()
                max_seqlen_k = max(
                    [
                        cu_seqlens_k[idx + 1] - cu_seqlens_k[idx]
                        for idx in range(cu_seqlens_k.size(0) - 1)
                    ]
                ).item()

            attn_output = flash_attn_varlen_func(
                query_fa,
                key_fa,
                value_fa,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                window_size=(self.window_size - 1, 0),
                causal=True,
            )
            attn_output = attn_output.unsqueeze(0)
            attn_weights = None
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]
            attn_kwargs = dict(kwargs)
            attn_kwargs["sliding_window"] = self.window_size
            attn_output, attn_weights = attention_interface(
                self.attn_layer,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attn_layer.attention_dropout,
                scaling=self.scaling,
                **attn_kwargs,
            )

        if past_key_values is not None:
            cache_len = key_states.shape[2]
            if cache_len > self.window_size:
                key_states = key_states[:, :, -self.window_size :, :]
                value_states = value_states[:, :, -self.window_size :, :]
                past_key_values.layers[self.layer_idx].keys = key_states
                past_key_values.layers[self.layer_idx].values = value_states

        attn_output = attn_output.reshape(batch_size, seq_len, -1).contiguous()
        output = attn_output + ttt_output
        output = self.attn_layer.o_proj(output)

        return output, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        lact_cache: Optional[LaCTCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        seq_len = hidden_states.shape[1]

        if (
            lact_cache is not None
            and not lact_cache.has_layer(self.layer_idx)
            and seq_len > 1
        ):
            return self._forward_prefill_streaming(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
                lact_cache,
                **kwargs,
            )

        return super().forward(
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            lact_cache=lact_cache,
            **kwargs,
        )
