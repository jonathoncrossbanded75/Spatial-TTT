import math

import numpy as np
import torch
import torch.nn.functional as F
from transformers.video_utils import load_video


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 32,
    min_pixels: int = 128 * 128,
    max_pixels: int = 16 * 16 * 2 * 2 * 2 * 6144,
):
    if num_frames < temporal_factor:
        raise ValueError(
            f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}"
        )
    if height < factor or width < factor:
        raise ValueError(
            f"height:{height} or width:{width} must be larger than factor:{factor}"
        )
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def serch_max_pixels(orig_shape, target_shape):
    for value in range(math.prod(orig_shape), 2 * math.prod(orig_shape), 1000):
        h, w = smart_resize(
            orig_shape[0],
            orig_shape[1],
            orig_shape[2],
            max_pixels=value,
        )
        if h == target_shape[1] and w == target_shape[2]:
            return value


def fetch_videos(self, video_url_or_urls, sample_indices_fn=None):
    backend = "torchcodec"
    if isinstance(video_url_or_urls, list):
        return list(
            zip(
                *[
                    self.fetch_videos(x, sample_indices_fn=sample_indices_fn)
                    for x in video_url_or_urls
                ]
            )
        )

    if video_url_or_urls.endswith(".pt"):
        torchfile = torch.load(video_url_or_urls, weights_only=False)
        return torchfile["video"], torchfile["metadata"]

    return load_video(
        video_url_or_urls, backend=backend, sample_indices_fn=sample_indices_fn
    )


def preprocess_video(
    video: torch.Tensor,
    max_pixels: int = 1024 * 28 * 28,
    min_pixels: int = 4096,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    image_mean: tuple = (0.5, 0.5, 0.5),
    image_std: tuple = (0.5, 0.5, 0.5),
) -> tuple[torch.Tensor, torch.Tensor]:
    if video.shape[-1] == 3:  # (T, H, W, C)
        video = video.permute(0, 3, 1, 2)

    T, C, H, W = video.shape

    factor = patch_size * merge_size
    resized_height, resized_width = smart_resize(
        num_frames=T,
        height=H,
        width=W,
        temporal_factor=temporal_patch_size,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    t_bar = round(T / temporal_patch_size) * temporal_patch_size
    if t_bar != T:
        indices = np.linspace(0, T - 1, t_bar).round().astype(int)
        video = video[indices]
        T = t_bar

    video = video.float()
    video = F.interpolate(
        video,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )

    if video.max() > 1.0:
        video = video / 255.0

    mean = torch.tensor(image_mean, device=video.device).view(1, 3, 1, 1)
    std = torch.tensor(image_std, device=video.device).view(1, 3, 1, 1)
    video = (video - mean) / std

    grid_t = T // temporal_patch_size
    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size

    video = video.unsqueeze(0)  # (1, T, C, H, W)
    B = 1

    patches = video.view(
        B,
        grid_t,
        temporal_patch_size,
        C,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    flatten_patches = patches.reshape(
        B,
        grid_t * grid_h * grid_w,
        C * temporal_patch_size * patch_size * patch_size,
    )

    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]])

    return flatten_patches, grid_thw

