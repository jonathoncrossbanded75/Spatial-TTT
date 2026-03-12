import math


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


# for value in range(19_660_800, 25_000_000, 1000):
#     h, w = smart_resize(64, 968, 1296, max_pixels=value)
#     if h == 480 and w == 640:
#         print(f"Found value: {value}, h: {h}, w: {w}")
#         break

for value in range(9_830_400, 25_000_000, 1000):
    h, w = smart_resize(32, 968, 1296, max_pixels=value)
    if h == 480 and w == 640:
        print(f"Found value: {value}, h: {h}, w: {w}")
        break
