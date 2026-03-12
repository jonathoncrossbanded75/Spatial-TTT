import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def get_vram_usage():
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def reset_vram_stats():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# Optimized Tiled Conv: No torch.cat to avoid memory doubling
def tiled_conv3d_optimized(layer, x, t_chunk=1, use_cp=True):
    b, c, t, h, w = x.shape
    # Pre-allocate output to avoid torch.cat peak memory
    out = torch.empty_like(x)
    x_padded = F.pad(x, (0, 0, 0, 0, 1, 1), mode="replicate")

    def run_conv(inp):
        return F.conv3d(
            inp,
            layer.weight,
            layer.bias,
            layer.stride,
            (0, 1, 1),
            layer.dilation,
            layer.groups,
        )

    for i in range(1, t + 1, t_chunk):
        t_end = min(i + t_chunk, t + 1)
        chunk = x_padded[:, :, i - 1 : t_end + 1, :, :]

        if use_cp and x.requires_grad:
            out_chunk = checkpoint(run_conv, chunk, use_reentrant=False)
        else:
            out_chunk = run_conv(chunk)

        # In-place assignment saves memory compared to append + cat
        out[:, :, i - 1 : t_end] = out_chunk

    return out


# A Deep Model with 4 layers to see Checkpoint effect
class DeepModel(nn.Module):
    def __init__(self, hidden_size, mode="native"):
        super().__init__()
        self.mode = mode
        self.layers = nn.ModuleList(
            [
                nn.Conv3d(
                    hidden_size,
                    hidden_size,
                    3,
                    padding=1,
                    padding_mode="replicate",
                    bias=False,
                )
                for _ in range(20)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            if self.mode == "native":
                x = layer(x)
            else:
                x = tiled_conv3d_optimized(layer, x, t_chunk=1, use_cp=True)
        return x


# Config
hidden_size = 1024  # Reduced to fit multiple layers in most GPUs
T, H, W = 128, 8, 10
device = "cuda"
dtype = torch.bfloat16

# Shared Input
x_input = torch.randn(
    1, hidden_size, T, H, W, device=device, dtype=dtype, requires_grad=True
)

# 1. Native Deep Test
reset_vram_stats()
model_native = DeepModel(hidden_size, mode="native").to(device).to(dtype)
print("--- 4-Layer Native Conv3d ---")
try:
    out = model_native(x_input)
    print(f"Forward VRAM: {get_vram_usage():.2f} MiB")
    out.sum().backward()
    print(f"Total VRAM (Fwd+Bwd): {get_vram_usage():.2f} MiB")
except RuntimeError as e:
    print(e)

# 2. Tiled Deep Test
del model_native, out
reset_vram_stats()
model_tiled = DeepModel(hidden_size, mode="tiled").to(device).to(dtype)
print("\n--- 4-Layer Tiled + Checkpoint ---")
# Use a fresh input clone
x_input2 = x_input.detach().clone().requires_grad_(True)
try:
    out = model_tiled(x_input2)
    print(f"Forward VRAM: {get_vram_usage():.2f} MiB")
    out.sum().backward()
    print(f"Total VRAM (Fwd+Bwd): {get_vram_usage():.2f} MiB")
except RuntimeError as e:
    print(e)
