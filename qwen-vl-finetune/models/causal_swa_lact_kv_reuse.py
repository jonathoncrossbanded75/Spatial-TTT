# Adapted from causal_swa_lact.py with KV-cache reuse for LaCT fast KV.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

from .causal_swa_lact import (
    LaCTLayerState as _BaseLayerState,
    Qwen3VLLaCTSWIGLULayer as _BaseLaCTLayer,
    apply_partial_rotary_pos_emb,
    l2_norm,
    rotate_half,
    silu_backprop,
    zeropower_via_newtonschulz5,
)


def apply_rotary_pos_emb_inverse(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) - (rotate_half(x) * sin)


@dataclass
class LaCTLayerStateKVReuse:
    w0: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor

    w0_norm: torch.Tensor
    w1_norm: torch.Tensor
    w2_norm: torch.Tensor

    dw0_momentum: Optional[torch.Tensor] = None
    dw1_momentum: Optional[torch.Tensor] = None
    dw2_momentum: Optional[torch.Tensor] = None

    pending_len: int = 0
    pending_lr0: Optional[torch.Tensor] = None
    pending_lr1: Optional[torch.Tensor] = None
    pending_lr2: Optional[torch.Tensor] = None
    pending_momentum_sum: Optional[torch.Tensor] = None


class LaCTCacheKVReuse:
    def __init__(self):
        self._layer_states: Dict[int, LaCTLayerStateKVReuse] = {}
        self.position_ids: Optional[torch.Tensor] = None

    def has_layer(self, layer_idx: int) -> bool:
        return layer_idx in self._layer_states

    def get_layer_state(self, layer_idx: int) -> LaCTLayerStateKVReuse:
        return self._layer_states[layer_idx]

    def set_layer_state(self, layer_idx: int, state):
        if isinstance(state, _BaseLayerState):
            pending_len = 0
            if state.pending_lr0 is not None:
                pending_len = state.pending_lr0.shape[1]
            pending_momentum_sum = None
            if state.pending_momentum is not None:
                pending_momentum_sum = state.pending_momentum.sum(dim=1, keepdim=True)
            converted = LaCTLayerStateKVReuse(
                w0=state.w0,
                w1=state.w1,
                w2=state.w2,
                w0_norm=state.w0_norm,
                w1_norm=state.w1_norm,
                w2_norm=state.w2_norm,
                dw0_momentum=state.dw0_momentum,
                dw1_momentum=state.dw1_momentum,
                dw2_momentum=state.dw2_momentum,
                pending_len=pending_len,
                pending_lr0=state.pending_lr0,
                pending_lr1=state.pending_lr1,
                pending_lr2=state.pending_lr2,
                pending_momentum_sum=pending_momentum_sum,
            )
            self._layer_states[layer_idx] = converted
        else:
            self._layer_states[layer_idx] = state

    def get_pending_length(self, layer_idx: int) -> int:
        if layer_idx not in self._layer_states:
            return 0
        return self._layer_states[layer_idx].pending_len

    def set_position_ids(self, position_ids: torch.Tensor) -> None:
        self.position_ids = position_ids

    def append_position_ids(self, position_ids_step: torch.Tensor) -> None:
        if self.position_ids is None:
            self.position_ids = position_ids_step
        else:
            self.position_ids = torch.cat([self.position_ids, position_ids_step], dim=-1)

    def reset(self) -> None:
        self._layer_states.clear()
        self.position_ids = None


class Qwen3VLLaCTSWIGLULayerKVReuse(_BaseLaCTLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rotary_emb = None
        self._kv_cache = None

    def _collect_fast_kv_from_cache(
        self,
        past_key_values,
        lact_cache: LaCTCacheKVReuse,
        hidden_states: torch.Tensor,
        pending_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rotary_emb is None:
            raise RuntimeError("KV-reuse LaCT requires rotary_emb to be set.")
        if lact_cache.position_ids is None:
            raise RuntimeError("KV-reuse LaCT requires position_ids in cache.")

        layer_cache = past_key_values.layers[self.layer_idx]
        key_states = getattr(layer_cache, "keys", None)
        value_states = getattr(layer_cache, "values", None)
        if key_states is None or value_states is None:
            raise RuntimeError("Missing KV cache for LaCT layer.")

        cache_len = key_states.shape[2]
        if pending_len > cache_len:
            raise RuntimeError(
                f"pending_len={pending_len} exceeds KV cache length={cache_len}"
            )

        key_states = key_states[:, :, -pending_len:, :]
        value_states = value_states[:, :, -pending_len:, :]

        n_rep = self.num_attn_heads // self.num_kv_heads
        if n_rep > 1:
            key_states = key_states.unsqueeze(2).expand(
                -1, -1, n_rep, -1, -1
            )
            key_states = key_states.reshape(
                key_states.shape[0], self.num_attn_heads, pending_len, self.head_dim
            )
            value_states = value_states.unsqueeze(2).expand(
                -1, -1, n_rep, -1, -1
            )
            value_states = value_states.reshape(
                value_states.shape[0], self.num_attn_heads, pending_len, self.head_dim
            )

        pos_ids = lact_cache.position_ids[..., -pending_len:]
        if pos_ids.device != hidden_states.device:
            pos_ids = pos_ids.to(hidden_states.device)
        cos, sin = self.rotary_emb(hidden_states, pos_ids)

        key_states = apply_rotary_pos_emb_inverse(key_states, cos, sin, unsqueeze_dim=1)

        key_flat = key_states.transpose(1, 2).reshape(
            key_states.shape[0], pending_len, -1
        )
        key_flat = key_flat * self.k_scale.view(1, 1, -1) + self.k_offset.view(1, 1, -1)
        key_flat = rearrange(key_flat, "b s (h d) -> (b h) s d", h=self.num_fw_heads)

        if self.qkv_silu:
            key_flat = F.silu(key_flat)
        key_flat = l2_norm(key_flat)

        if not self.ttt_nope:
            dummy = key_flat
            _, key_flat = apply_partial_rotary_pos_emb(
                dummy, key_flat, cos, sin, rope_dim=self.head_dim
            )

        value_flat = value_states.transpose(1, 2).reshape(
            value_states.shape[0], pending_len, -1
        )
        value_flat = rearrange(
            value_flat, "b s (h d) -> (b h) s d", h=self.num_fw_heads
        )
        if self.qkv_silu and not self.no_v_silu:
            value_flat = F.silu(value_flat)

        if self.fp32_states:
            key_flat = key_flat.float()
            value_flat = value_flat.float()

        return key_flat, value_flat

    def _compute_ttt_output_decode(
        self,
        hidden_states: torch.Tensor,
        fast_q: torch.Tensor,
        fast_k: torch.Tensor,
        fast_v: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        lact_cache: LaCTCacheKVReuse,
    ) -> torch.Tensor:
        if self._kv_cache is None:
            raise RuntimeError("KV cache not set for KV-reuse decode.")

        batch_size, seq_len, _ = hidden_states.size()
        assert batch_size == 1, "Decode cache only supports batch_size=1"
        assert seq_len == 1, "Decode should process one token at a time"

        state = lact_cache.get_layer_state(self.layer_idx)

        fast_q = rearrange(fast_q, "b s (h d) -> (b h) s d", h=self.num_fw_heads)
        fast_k = rearrange(fast_k, "b s (h d) -> (b h) s d", h=self.num_fw_heads)
        fast_v = rearrange(fast_v, "b s (h d) -> (b h) s d", h=self.num_fw_heads)

        if self.qkv_silu:
            fast_q = F.silu(fast_q)
            fast_k = F.silu(fast_k)
            if not self.no_v_silu:
                fast_v = F.silu(fast_v)

        fast_q = l2_norm(fast_q)
        fast_k = l2_norm(fast_k)

        if not self.ttt_nope:
            cos, sin = position_embeddings
            fast_q, fast_k = apply_partial_rotary_pos_emb(
                fast_q, fast_k, cos, sin, rope_dim=self.head_dim
            )

        lr = self.lr_proj(hidden_states)
        if self.lr_parameterization == "mamba":
            lr = F.softplus(lr.float() + self.base_lr_inv)
        fw_lr = rearrange(lr, "b s (h d) -> (b h) s d", h=self.num_fw_heads)
        fw_lr1, fw_lr2, fw_lr3 = fw_lr.chunk(3, dim=-1)

        if self.use_momentum:
            momentum = self.momentum_proj(hidden_states).float()
            momentum = rearrange(
                momentum, "b s (h d) -> (b h) s d", h=self.num_fw_heads
            )
        else:
            momentum = None

        fw_w0 = state.w0
        fw_w1 = state.w1
        fw_w2 = state.w2

        q_t = fast_q.transpose(1, 2)
        if self.fp32_states:
            q_t = q_t.float()
        h = torch.bmm(fw_w2, q_t)
        gate = F.silu(torch.bmm(fw_w0, q_t), inplace=True)
        output = torch.bmm(fw_w1, gate * h)
        fw_x = output.transpose(1, 2)

        if state.pending_len == 0:
            state.pending_lr0 = fw_lr1
            state.pending_lr1 = fw_lr2
            state.pending_lr2 = fw_lr3
            state.pending_len = 1
            if momentum is not None:
                state.pending_momentum_sum = momentum.sum(dim=1, keepdim=True)
        else:
            assert state.pending_lr0 is not None
            assert state.pending_lr1 is not None
            assert state.pending_lr2 is not None
            state.pending_lr0 = torch.cat([state.pending_lr0, fw_lr1], dim=1)
            state.pending_lr1 = torch.cat([state.pending_lr1, fw_lr2], dim=1)
            state.pending_lr2 = torch.cat([state.pending_lr2, fw_lr3], dim=1)
            state.pending_len += 1
            if momentum is not None:
                if state.pending_momentum_sum is None:
                    state.pending_momentum_sum = momentum.sum(dim=1, keepdim=True)
                else:
                    state.pending_momentum_sum = (
                        state.pending_momentum_sum + momentum.sum(dim=1, keepdim=True)
                    )

        pending_len = state.pending_len
        if pending_len >= self.lact_chunk_size:
            if pending_len != self.lact_chunk_size:
                raise RuntimeError(
                    "pending_len exceeds lact_chunk_size; decode must be step-wise."
                )
            if (
                state.pending_lr0 is None
                or state.pending_lr1 is None
                or state.pending_lr2 is None
            ):
                raise RuntimeError("Missing pending LR state for LaCT update.")

            fast_k_cache, fast_v_cache = self._collect_fast_kv_from_cache(
                self._kv_cache, lact_cache, hidden_states, self.lact_chunk_size
            )
            ki = fast_k_cache
            vi = fast_v_cache.transpose(1, 2)
            lr0i = state.pending_lr0[:, : self.lact_chunk_size, :]
            lr1i = state.pending_lr1[:, : self.lact_chunk_size, :]
            lr2i = state.pending_lr2[:, : self.lact_chunk_size, :]

            gate_before_act = torch.bmm(fw_w0, ki.transpose(1, 2))
            hidden_before_mul = torch.bmm(fw_w2, ki.transpose(1, 2))
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
            dhidden = torch.bmm(fw_w1.transpose(1, 2), vi)
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1i).type_as(vi))
            dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
            dw2 = torch.bmm(
                dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul)
            )

            if state.pending_momentum_sum is not None and state.dw0_momentum is not None:
                assert state.dw1_momentum is not None
                assert state.dw2_momentum is not None
                m_i = state.pending_momentum_sum / float(self.lact_chunk_size)
                dw0 = dw0 + state.dw0_momentum * m_i
                dw1 = dw1 + state.dw1_momentum * m_i
                dw2 = dw2 + state.dw2_momentum * m_i
                state.dw0_momentum = dw0
                state.dw1_momentum = dw1
                state.dw2_momentum = dw2

            if self.use_muon:
                dw0 = zeropower_via_newtonschulz5(dw0)
                dw1 = zeropower_via_newtonschulz5(dw1)
                dw2 = zeropower_via_newtonschulz5(dw2)

            state.w0 = fw_w0 + dw0
            state.w1 = fw_w1 + dw1
            state.w2 = fw_w2 + dw2

            state.w0 = (
                state.w0 / (state.w0.norm(dim=2, keepdim=True) + 1e-5)
            ) * state.w0_norm
            state.w1 = (
                state.w1 / (state.w1.norm(dim=2, keepdim=True) + 1e-5)
            ) * state.w1_norm
            state.w2 = (
                state.w2 / (state.w2.norm(dim=2, keepdim=True) + 1e-5)
            ) * state.w2_norm

            state.pending_len = 0
            state.pending_lr0 = None
            state.pending_lr1 = None
            state.pending_lr2 = None
            state.pending_momentum_sum = None

        ttt_x_normed = self.ttt_norm(fw_x)
        if self.learnable_ttt_scale:
            ttt_scale = F.silu(self.ttt_scale_proj(hidden_states), inplace=False)
            ttt_scale = rearrange(
                ttt_scale, "b s (h d) -> (b h) s d", h=self.num_fw_heads
            )
            ttt_x_normed = ttt_x_normed * ttt_scale

        ttt_output = rearrange(
            ttt_x_normed, "(b h) s d -> b s (h d)", h=self.num_fw_heads
        )
        return ttt_output.type_as(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        lact_cache: Optional[LaCTCacheKVReuse] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._kv_cache = past_key_values
        output = super().forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            lact_cache=lact_cache,
            **kwargs,
        )
        self._kv_cache = None
        return output
