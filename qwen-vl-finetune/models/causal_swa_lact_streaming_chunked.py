# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .causal_swa_lact import (
    FLASH_ATTN_AVAILABLE,
    LaCTCache,
    LaCTLayerState,
    apply_rotary_pos_emb,
    flash_attn_varlen_func,
    silu_backprop,
    zeropower_via_newtonschulz5,
)
from .causal_swa_lact_streaming import Qwen3VLLaCTSWIGLULayerStreaming


@dataclass
class _VideoIndex:
    token_pos_to_frame: torch.Tensor
    frame_token_positions: torch.Tensor
    tokens_per_frame: int
    frames_per_video: int
    num_videos: int
    h: int
    w: int


class Qwen3VLLaCTSWIGLULayerStreamingChunked(Qwen3VLLaCTSWIGLULayerStreaming):
    """
    Streaming-prefill variant with chunked attention and conv-aware frame caching.
    """

    def __init__(self, *args, attn_chunk_size: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_chunk_size = (
            attn_chunk_size if attn_chunk_size is not None else self.lact_chunk_size
        )

    def _compute_qkv(self, hidden_states: torch.Tensor):
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
        return q_normed, k_normed, v, q_flat, k_flat, v_expanded

    def _build_chunk_ranges(
        self,
        seq_len: int,
        chunk_size: int,
        video_mask: Optional[torch.Tensor],
    ) -> list[tuple[int, int]]:
        if chunk_size <= 0 or chunk_size >= seq_len:
            return [(0, seq_len)]

        if video_mask is None:
            return [
                (i, min(i + chunk_size, seq_len)) for i in range(0, seq_len, chunk_size)
            ]

        mask = video_mask
        if mask.dim() > 1:
            mask = mask[0]

        chunks = []
        start = 0
        while start < seq_len:
            end = min(start + chunk_size, seq_len)
            if end < seq_len and mask[end - 1]:
                while end < seq_len and mask[end]:
                    end += 1
            chunks.append((start, end))
            start = end
        return chunks

    def _build_video_index(
        self,
        video_mask: torch.Tensor,
        video_grid_thw: torch.Tensor,
        seq_len: int,
    ) -> Optional[_VideoIndex]:
        mask = video_mask
        if mask.dim() > 1:
            mask = mask[0]
        video_token_positions = torch.nonzero(mask, as_tuple=False).flatten()
        if video_token_positions.numel() == 0:
            return None

        t, h, w = video_grid_thw[0]
        h, w = h // 2, w // 2
        tokens_per_frame = int(h * w)
        if tokens_per_frame <= 0:
            return None

        num_video_tokens = video_token_positions.numel()
        if num_video_tokens % tokens_per_frame != 0:
            raise RuntimeError(
                f"video tokens ({num_video_tokens}) not divisible by frame size ({tokens_per_frame})."
            )
        total_frames = num_video_tokens // tokens_per_frame
        if total_frames % int(t) != 0:
            raise RuntimeError(
                f"total frames ({total_frames}) not divisible by t ({int(t)})."
            )
        num_videos = total_frames // int(t)

        frame_token_positions = video_token_positions.view(
            total_frames, tokens_per_frame
        )
        token_pos_to_frame = torch.full(
            (seq_len,),
            -1,
            device=video_token_positions.device,
            dtype=torch.long,
        )
        frame_ids = torch.arange(total_frames, device=video_token_positions.device)
        token_pos_to_frame[video_token_positions] = frame_ids.repeat_interleave(
            tokens_per_frame
        )
        return _VideoIndex(
            token_pos_to_frame=token_pos_to_frame,
            frame_token_positions=frame_token_positions,
            tokens_per_frame=tokens_per_frame,
            frames_per_video=int(t),
            num_videos=num_videos,
            h=int(h),
            w=int(w),
        )

    def _compute_frame_qkv(
        self,
        hidden_states: torch.Tensor,
        frame_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hs_frame = hidden_states[:, frame_positions, :]
        _, _, _, q_flat, k_flat, v_expanded = self._compute_qkv(hs_frame)
        return q_flat, k_flat, v_expanded

    def _apply_conv_frames(
        self,
        frames: list[torch.Tensor],
        conv_layer: torch.nn.Module,
        h: int,
        w: int,
    ) -> torch.Tensor:
        video_tokens = torch.stack(frames, dim=0)
        video_tokens = rearrange(video_tokens, "t (h w) c -> 1 c t h w", h=h, w=w)
        video_conv = conv_layer(video_tokens)
        return rearrange(video_conv, "1 c t h w -> t (h w) c")

    def _apply_streaming_conv(
        self,
        q_flat: torch.Tensor,
        k_flat: torch.Tensor,
        v_expanded: torch.Tensor,
        hidden_states: torch.Tensor,
        chunk_start: int,
        chunk_end: int,
        video_index: _VideoIndex,
        prev_frame_cache: Dict[
            int, Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frame_ids = video_index.token_pos_to_frame[chunk_start:chunk_end]
        frame_ids = frame_ids[frame_ids >= 0]
        if frame_ids.numel() == 0:
            return q_flat, k_flat, v_expanded

        frame_ids = torch.unique(frame_ids)

        for video_idx in range(video_index.num_videos):
            video_frame_start = video_idx * video_index.frames_per_video
            video_frame_end = video_frame_start + video_index.frames_per_video - 1
            frames_in_video = frame_ids[
                (frame_ids >= video_frame_start) & (frame_ids <= video_frame_end)
            ]
            if frames_in_video.numel() == 0:
                continue

            frame_start = int(frames_in_video.min().item())
            frame_end = int(frames_in_video.max().item())

            last_frame_positions = video_index.frame_token_positions[frame_end]
            last_pos_in_chunk = last_frame_positions - chunk_start
            raw_last_q = q_flat[:, last_pos_in_chunk, :].clone()
            raw_last_k = k_flat[:, last_pos_in_chunk, :].clone()
            raw_last_v = v_expanded[:, last_pos_in_chunk, :].clone()

            local_start = frame_start - video_frame_start
            local_end = frame_end - video_frame_start
            prev_frame_idx = frame_start - 1 if local_start > 0 else None
            next_frame_idx = (
                frame_end + 1 if local_end < video_index.frames_per_video - 1 else None
            )

            frame_order = []
            if prev_frame_idx is not None:
                frame_order.append(prev_frame_idx)
            frame_order.extend(range(frame_start, frame_end + 1))
            if next_frame_idx is not None:
                frame_order.append(next_frame_idx)

            q_frames = []
            k_frames = []
            v_frames = []
            for frame_idx in frame_order:
                if frame_start <= frame_idx <= frame_end:
                    frame_positions = video_index.frame_token_positions[frame_idx]
                    pos_in_chunk = frame_positions - chunk_start
                    q_frames.append(q_flat[:, pos_in_chunk, :].squeeze(0))
                    k_frames.append(k_flat[:, pos_in_chunk, :].squeeze(0))
                    v_frames.append(v_expanded[:, pos_in_chunk, :].squeeze(0))
                else:
                    cached = prev_frame_cache.get(video_idx)
                    if cached is not None and cached[0] == frame_idx:
                        q_frame, k_frame, v_frame = cached[1], cached[2], cached[3]
                    else:
                        frame_positions = video_index.frame_token_positions[frame_idx]
                        q_frame, k_frame, v_frame = self._compute_frame_qkv(
                            hidden_states, frame_positions
                        )
                        q_frame = q_frame.squeeze(0)
                        k_frame = k_frame.squeeze(0)
                        v_frame = v_frame.squeeze(0)
                    q_frames.append(q_frame)
                    k_frames.append(k_frame)
                    v_frames.append(v_frame)

            q_conv = self._apply_conv_frames(
                q_frames, self.conv_q, video_index.h, video_index.w
            )
            k_conv = self._apply_conv_frames(
                k_frames, self.conv_k, video_index.h, video_index.w
            )
            v_conv = self._apply_conv_frames(
                v_frames, self.conv_v, video_index.h, video_index.w
            )

            offset = 1 if prev_frame_idx is not None else 0
            for i, frame_idx in enumerate(range(frame_start, frame_end + 1)):
                frame_positions = video_index.frame_token_positions[frame_idx]
                pos_in_chunk = frame_positions - chunk_start
                q_flat[:, pos_in_chunk, :] = q_conv[offset + i].unsqueeze(0)
                k_flat[:, pos_in_chunk, :] = k_conv[offset + i].unsqueeze(0)
                v_expanded[:, pos_in_chunk, :] = v_conv[offset + i].unsqueeze(0)

            prev_frame_cache[video_idx] = (
                frame_end,
                raw_last_q.squeeze(0),
                raw_last_k.squeeze(0),
                raw_last_v.squeeze(0),
            )

        return q_flat, k_flat, v_expanded

    def _update_fast_weights(
        self,
        fw_w0: torch.Tensor,
        fw_w1: torch.Tensor,
        fw_w2: torch.Tensor,
        w0_norm: torch.Tensor,
        w1_norm: torch.Tensor,
        w2_norm: torch.Tensor,
        dw0_momentum: Optional[torch.Tensor],
        dw1_momentum: Optional[torch.Tensor],
        dw2_momentum: Optional[torch.Tensor],
        fast_k: torch.Tensor,
        fast_v: torch.Tensor,
        lr0: torch.Tensor,
        lr1: torch.Tensor,
        lr2: torch.Tensor,
        momentum: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        ki = fast_k
        vi = fast_v.transpose(1, 2)

        if self.fp32_states:
            ki = ki.float()
            vi = vi.float()

        gate_before_act = torch.bmm(fw_w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(fw_w2, ki.transpose(1, 2))
        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
        dhidden = torch.bmm(fw_w1.transpose(1, 2), vi)
        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1).type_as(vi))
        dw0 = torch.bmm(dgate_before_act, (ki * lr0).type_as(dgate_before_act))
        dw2 = torch.bmm(dhidden_before_mul, (ki * lr2).type_as(dhidden_before_mul))

        if momentum is not None and dw0_momentum is not None:
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

        return fw_w0, fw_w1, fw_w2, dw0_momentum, dw1_momentum, dw2_momentum

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

        pending_k = None
        pending_v = None
        pending_lr0 = None
        pending_lr1 = None
        pending_lr2 = None
        pending_momentum = None
        pending_start = None
        tail_k = None
        tail_v = None
        tail_lr0 = None
        tail_lr1 = None
        tail_lr2 = None
        tail_momentum = None
        update_cutoff = max(seq_len - self.lact_chunk_size, 0)

        output = torch.empty_like(hidden_states)
        ttt_output = torch.empty_like(hidden_states)

        video_mask = kwargs.get("video_mask", None)
        if video_mask is not None and video_mask.dim() > 2:
            video_mask = video_mask[..., 0]

        video_index = None
        prev_frame_cache_ttt: Dict[
            int, Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        if self.use_conv_layer and video_mask is not None:
            video_grid_thw = kwargs.get("video_grid_thw", None)
            if video_grid_thw is None:
                raise RuntimeError("video_grid_thw required when use_conv_layer=True.")
            video_index = self._build_video_index(video_mask, video_grid_thw, seq_len)

        ttt_chunks = self._build_chunk_ranges(seq_len, self.lact_chunk_size, video_mask)
        attn_chunks = self._build_chunk_ranges(
            seq_len, self.attn_chunk_size, video_mask
        )

        use_varlen_attn = (
            FLASH_ATTN_AVAILABLE
            and self.config._attn_implementation == "flash_attention_2"
        )
        attention_interface = None
        if not use_varlen_attn:
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        for chunk_start, chunk_end in ttt_chunks:
            hs_chunk = hidden_states[:, chunk_start:chunk_end, :]
            cos = cos_full[:, chunk_start:chunk_end, :]
            sin = sin_full[:, chunk_start:chunk_end, :]

            q_normed, k_normed, v, q_flat, k_flat, v_expanded = self._compute_qkv(
                hs_chunk
            )

            if self.use_conv_layer and video_index is not None:
                q_flat, k_flat, v_expanded = self._apply_streaming_conv(
                    q_flat,
                    k_flat,
                    v_expanded,
                    hidden_states,
                    chunk_start,
                    chunk_end,
                    video_index,
                    prev_frame_cache_ttt,
                )

            fast_q, fast_k, fast_v = self._prepare_fast_qkv(
                q_flat, k_flat, v_expanded, cos, sin
            )

            lr = self.lr_proj(hs_chunk)
            if self.lr_parameterization == "mamba":
                lr = F.softplus(lr.float() + self.base_lr_inv)
            fw_lr = rearrange(lr, "b s (h d) -> (b h) s d", h=self.num_fw_heads)
            fw_lr1, fw_lr2, fw_lr3 = fw_lr.chunk(3, dim=-1)

            if self.use_momentum:
                momentum = self.momentum_proj(hs_chunk).float()
                momentum = rearrange(
                    momentum, "b s (h d) -> (b h) s d", h=self.num_fw_heads
                )
            else:
                momentum = None

            ttt_chunk = torch.empty_like(hs_chunk)
            cursor = 0
            while cursor < fast_q.shape[1]:
                pending_len = 0 if pending_k is None else pending_k.shape[1]
                step = min(
                    self.lact_chunk_size - pending_len,
                    fast_q.shape[1] - cursor,
                )
                if step <= 0:
                    raise RuntimeError(
                        "LaCT streaming chunked hit non-positive step; "
                        f"pending_len={pending_len}, cursor={cursor}, "
                        f"seq_len={seq_len}, update_cutoff={update_cutoff}."
                    )
                seg_global_start = chunk_start + cursor

                fast_q_seg = fast_q[:, cursor : cursor + step, :]
                fast_k_seg = fast_k[:, cursor : cursor + step, :]
                fast_v_seg = fast_v[:, cursor : cursor + step, :]
                lr0_seg = fw_lr1[:, cursor : cursor + step, :]
                lr1_seg = fw_lr2[:, cursor : cursor + step, :]
                lr2_seg = fw_lr3[:, cursor : cursor + step, :]
                momentum_seg = (
                    momentum[:, cursor : cursor + step, :]
                    if momentum is not None
                    else None
                )

                q_t = fast_q_seg.transpose(1, 2)
                v_t = fast_v_seg.transpose(1, 2)
                if self.fp32_states:
                    q_t = q_t.float()
                    v_t = v_t.float()

                h = torch.bmm(fw_w2, q_t)
                gate = F.silu(torch.bmm(fw_w0, q_t), inplace=True)
                fw_x = torch.bmm(fw_w1, gate * h).transpose(1, 2)

                ttt_x_normed = self.ttt_norm(fw_x)
                if self.learnable_ttt_scale:
                    ttt_scale = F.silu(
                        self.ttt_scale_proj(hs_chunk[:, cursor : cursor + step, :]),
                        inplace=False,
                    )
                    ttt_scale = rearrange(
                        ttt_scale, "b s (n d) -> (b n) s d", n=self.num_fw_heads
                    )
                    ttt_x_normed = ttt_x_normed * ttt_scale

                ttt_seg = rearrange(
                    ttt_x_normed,
                    "(b n) s d -> b s (n d)",
                    n=self.num_fw_heads,
                    b=batch_size,
                )
                ttt_chunk[:, cursor : cursor + step, :] = ttt_seg.type_as(hidden_states)

                fast_k_seg = fast_k_seg.float() if self.fp32_states else fast_k_seg

                if tail_k is None:
                    tail_k = fast_k_seg
                    tail_v = fast_v_seg
                else:
                    tail_k = torch.cat([tail_k, fast_k_seg], dim=1)
                    tail_v = torch.cat([tail_v, fast_v_seg], dim=1)

                if tail_k.shape[1] > self.lact_chunk_size:
                    tail_k = tail_k[:, -self.lact_chunk_size :, :]
                    tail_v = tail_v[:, -self.lact_chunk_size :, :]

                if pending_k is None:
                    pending_k = fast_k_seg
                    pending_v = fast_v_seg
                    pending_lr0 = lr0_seg
                    pending_lr1 = lr1_seg
                    pending_lr2 = lr2_seg
                    pending_momentum = momentum_seg
                    pending_start = seg_global_start
                else:
                    pending_k = torch.cat([pending_k, fast_k_seg], dim=1)
                    pending_v = torch.cat([pending_v, fast_v_seg], dim=1)
                    pending_lr0 = torch.cat([pending_lr0, lr0_seg], dim=1)
                    pending_lr1 = torch.cat([pending_lr1, lr1_seg], dim=1)
                    pending_lr2 = torch.cat([pending_lr2, lr2_seg], dim=1)
                    if pending_momentum is not None and momentum_seg is not None:
                        pending_momentum = torch.cat(
                            [pending_momentum, momentum_seg], dim=1
                        )

                while (
                    pending_k is not None and pending_k.shape[1] >= self.lact_chunk_size
                ):
                    if pending_start is None:
                        raise RuntimeError("pending_start missing for LaCT update.")
                    if pending_start >= update_cutoff:
                        break
                    ki = pending_k[:, : self.lact_chunk_size, :]
                    vi = pending_v[:, : self.lact_chunk_size, :]
                    lr0i = pending_lr0[:, : self.lact_chunk_size, :]
                    lr1i = pending_lr1[:, : self.lact_chunk_size, :]
                    lr2i = pending_lr2[:, : self.lact_chunk_size, :]
                    mi = (
                        pending_momentum[:, : self.lact_chunk_size, :]
                        if pending_momentum is not None
                        else None
                    )

                    (
                        fw_w0,
                        fw_w1,
                        fw_w2,
                        dw0_momentum,
                        dw1_momentum,
                        dw2_momentum,
                    ) = self._update_fast_weights(
                        fw_w0,
                        fw_w1,
                        fw_w2,
                        w0_norm,
                        w1_norm,
                        w2_norm,
                        dw0_momentum,
                        dw1_momentum,
                        dw2_momentum,
                        ki,
                        vi,
                        lr0i,
                        lr1i,
                        lr2i,
                        mi,
                    )

                    if pending_k.shape[1] > self.lact_chunk_size:
                        pending_k = pending_k[:, self.lact_chunk_size :, :]
                        pending_v = pending_v[:, self.lact_chunk_size :, :]
                        pending_lr0 = pending_lr0[:, self.lact_chunk_size :, :]
                        pending_lr1 = pending_lr1[:, self.lact_chunk_size :, :]
                        pending_lr2 = pending_lr2[:, self.lact_chunk_size :, :]
                        if pending_momentum is not None:
                            pending_momentum = pending_momentum[
                                :, self.lact_chunk_size :, :
                            ]
                        pending_start += self.lact_chunk_size
                    else:
                        pending_k = None
                        pending_v = None
                        pending_lr0 = None
                        pending_lr1 = None
                        pending_lr2 = None
                        pending_momentum = None
                        pending_start = None

                cursor += step

            ttt_output[:, chunk_start:chunk_end, :] = ttt_chunk

        for chunk_start, chunk_end in attn_chunks:
            hs_chunk = hidden_states[:, chunk_start:chunk_end, :]
            cos = cos_full[:, chunk_start:chunk_end, :]
            sin = sin_full[:, chunk_start:chunk_end, :]

            q_normed, k_normed, v, _, _, _ = self._compute_qkv(hs_chunk)

            query_states = q_normed.transpose(1, 2)
            key_states = k_normed.transpose(1, 2)
            value_states = rearrange(v, "b s (h d) -> b h s d", d=self.head_dim)

            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=1
            )

            if past_key_values is not None:
                cache_position_chunk = torch.arange(
                    chunk_start,
                    chunk_end,
                    device=hidden_states.device,
                )
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position_chunk,
                }
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
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

                attn_output = flash_attn_varlen_func(
                    query_fa,
                    key_fa,
                    value_fa,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=query_fa.shape[0],
                    max_seqlen_k=key_fa.shape[0],
                    window_size=(self.window_size - 1, 0),
                    causal=True,
                )
                attn_output = attn_output.unsqueeze(0)
            else:
                attn_kwargs = dict(kwargs)
                attn_kwargs["sliding_window"] = self.window_size
                attn_mask = attention_mask
                if attn_mask is None:
                    past_len = key_states.shape[2] - query_states.shape[2]
                    mask_window = None
                    if self.config._attn_implementation not in ("sdpa", "sdpa_paged"):
                        mask_window = self.window_size
                    attn_mask = _prepare_4d_causal_attention_mask(
                        attention_mask=None,
                        input_shape=(batch_size, query_states.shape[2]),
                        inputs_embeds=hs_chunk,
                        past_key_values_length=past_len,
                        sliding_window=mask_window,
                    )
                attn_output, _ = attention_interface(
                    self.attn_layer,
                    query_states,
                    key_states,
                    value_states,
                    attn_mask,
                    dropout=0.0
                    if not self.training
                    else self.attn_layer.attention_dropout,
                    scaling=self.scaling,
                    **attn_kwargs,
                )

            if past_key_values is not None and use_varlen_attn:
                cache_len = key_states.shape[2]
                if cache_len > self.window_size:
                    key_states = key_states[:, :, -self.window_size :, :]
                    value_states = value_states[:, :, -self.window_size :, :]
                    past_key_values.layers[self.layer_idx].keys = key_states
                    past_key_values.layers[self.layer_idx].values = value_states

            attn_chunk = attn_output.reshape(batch_size, chunk_end - chunk_start, -1)
            combined = attn_chunk + ttt_output[:, chunk_start:chunk_end, :]
            output[:, chunk_start:chunk_end, :] = self.attn_layer.o_proj(combined)

        tail_start = max(seq_len - self.lact_chunk_size, 0)
        tail_hidden = hidden_states[:, tail_start:seq_len, :]
        tail_lr = self.lr_proj(tail_hidden)
        if self.lr_parameterization == "mamba":
            tail_lr = F.softplus(tail_lr.float() + self.base_lr_inv)
        tail_fw_lr = rearrange(tail_lr, "b s (h d) -> (b h) s d", h=self.num_fw_heads)
        tail_lr0, tail_lr1, tail_lr2 = tail_fw_lr.chunk(3, dim=-1)
        if self.use_momentum:
            tail_momentum = self.momentum_proj(tail_hidden).float()
            tail_momentum = rearrange(
                tail_momentum, "b s (h d) -> (b h) s d", h=self.num_fw_heads
            )
        else:
            tail_momentum = None

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
            pending_k=tail_k,
            pending_v=tail_v,
            pending_lr0=tail_lr0,
            pending_lr1=tail_lr1,
            pending_lr2=tail_lr2,
            pending_momentum=tail_momentum,
        )
        lact_cache.set_layer_state(self.layer_idx, state)

        if past_key_values is not None and not use_varlen_attn:
            cache = past_key_values.layers[self.layer_idx]
            cache_len = cache.keys.shape[2]
            if cache_len > self.window_size:
                cache.keys = cache.keys[:, :, -self.window_size :, :]
                cache.values = cache.values[:, :, -self.window_size :, :]

        return output, None
