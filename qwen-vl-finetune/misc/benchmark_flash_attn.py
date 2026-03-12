import time
from functools import partial

import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_varlen_func


def benchmark_runtime(fn, *args, iters=10, warmup=2, **kwargs):
    print("Benchmarking ...")
    # warm up
    for _ in range(warmup):
        fn(*args, **kwargs)
    # benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.time()
    print(f"Avg runtime: {(end - start) / iters * 1000:.2f} ms")


seqlen = 16484
batch_size = 2
head_num = 16
head_dim = 128
window_size = 2048

x = torch.randn(
    seqlen * batch_size, head_num, head_dim, device="cuda", dtype=torch.bfloat16
)
cu_seqlens = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * seqlen

flash_attn_with_window = partial(
    flash_attn_varlen_func,
    x,
    x,
    x,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=seqlen,
    max_seqlen_k=seqlen,
    window_size=(window_size - 1, 0),
    causal=True,
)

flash_attn_without_window = partial(
    flash_attn_varlen_func,
    x,
    x,
    x,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=seqlen,
    max_seqlen_k=seqlen,
    causal=True,
)

benchmark_runtime(flash_attn_with_window, iters=20)
benchmark_runtime(flash_attn_without_window, iters=20)


# a simple ffn
class FFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


ffn = FFN(head_num * head_dim, head_num * head_dim * 4).to("cuda").bfloat16()


def ffn_forward(x, ffn):
    batch_size, seqlen, dim = x.shape
    x = x.view(batch_size * seqlen, dim)
    out = ffn(x)
    out = out.view(batch_size, seqlen, dim)
    return out


class SelfAttn(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        qkv = self.qkv_proj(x)  # (batch_size, seqlen, dim * 3)
        qkv = qkv.view(batch_size * seq_len, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seqlen,
            max_seqlen_k=seqlen,
            causal=True,
        )
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)
        return attn_output


selfattn = SelfAttn(head_num * head_dim, head_num).to("cuda").bfloat16()


def vanilla_attn(q, k, v):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1))
    attn_probs = torch.softmax(scores, dim=-1)
    attn_output = torch.matmul(attn_probs, v)
    return attn_output


benchmark_runtime(
    ffn_forward, x.view(batch_size, seqlen, head_num * head_dim), ffn, iters=20
)
benchmark_runtime(selfattn, x.view(batch_size, seqlen, head_num * head_dim), iters=20)

benchmark_runtime(
    vanilla_attn,
    x.view(batch_size, seqlen, head_num, head_dim),
    x.view(batch_size, seqlen, head_num, head_dim),
    x.view(batch_size, seqlen, head_num, head_dim),
    iters=20,
)
