import sys

sys.path.insert(0, "/home/lff/bigdata1/cjw/projs/qwen3vl/dependency")
import os
from glob import glob
from typing import Callable, Optional, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers.video_utils import VIDEO_DECODERS


def fetch_videos(
    video_url_or_urls: Union[str, list[str], list[list[str]]],
    sample_indices_fn=None,
):
    backend = "torchcodec"
    if isinstance(video_url_or_urls, list):
        return list(
            zip(
                *[
                    fetch_videos(x, sample_indices_fn=sample_indices_fn)
                    for x in video_url_or_urls
                ]
            )
        )
    else:
        return load_video(
            video_url_or_urls,
            num_frames=256,  # hard coded
            backend=backend,
            sample_indices_fn=sample_indices_fn,
        )


def default_sample_indices_fn(metadata, num_frames=None, fps=None, **kwargs):
    total_num_frames = metadata.total_num_frames
    video_fps = metadata.fps

    # If num_frames is not given but fps is, calculate num_frames from fps
    if num_frames is None and fps is not None:
        num_frames = int(total_num_frames / video_fps * fps)
        if num_frames > total_num_frames:
            raise ValueError(
                f"When loading the video with fps={fps}, we computed num_frames={num_frames} "
                f"which exceeds total_num_frames={total_num_frames}. Check fps or video metadata."
            )

    if num_frames is not None:
        indices = np.arange(
            0, total_num_frames, total_num_frames / num_frames, dtype=int
        )
    else:
        indices = np.arange(0, total_num_frames, dtype=int)
    return indices


def load_video(
    video,
    num_frames: Optional[int] = None,
    fps: Optional[Union[int, float]] = None,
    backend: str = "pyav",
    sample_indices_fn: Optional[Callable] = None,
    **kwargs,
):
    # If `sample_indices_fn` is given, we can accept any args as those might be needed by custom `sample_indices_fn`
    if fps is not None and num_frames is not None and sample_indices_fn is None:
        raise ValueError(
            "`num_frames`, `fps`, and `sample_indices_fn` are mutually exclusive arguments, please use only one!"
        )

    # If user didn't pass a sampling function, create one on the fly with default logic
    if sample_indices_fn is None:

        def sample_indices_fn_func(metadata, **fn_kwargs):
            return default_sample_indices_fn(
                metadata, num_frames=num_frames, fps=fps, **fn_kwargs
            )

        sample_indices_fn = sample_indices_fn_func

    # Early exit if provided an array or `PIL` frames
    if not isinstance(video, str):
        metadata = [None] * len(video)
        return video, metadata

    if os.path.isfile(video):
        file_obj = video
    else:
        raise TypeError(
            "Incorrect format used for video. Should be an url linking to an video or a local path."
        )

    video_decoder = VIDEO_DECODERS[backend]
    video, metadata = video_decoder(file_obj, sample_indices_fn, **kwargs)
    return video, metadata


if __name__ == "__main__":
    raw_video_rootdir = "/home/lff/bigdata1/cjw/projs/qwen3vl/qwen-vl-finetune/data"
    processed_video_rootdir = (
        "/home/lff/bigdata1/cjw/projs/qwen3vl/qwen-vl-finetune/data_processed"
    )
    raw_video_urls = glob(f"{raw_video_rootdir}/*.mp4")
    os.makedirs(processed_video_rootdir, exist_ok=True)

    # benchmark decoding time
    #
    # sample_video_url = raw_video_urls[0]
    # import time
    #
    # # warm up
    # for _ in range(5):
    #     fetch_videos(sample_video_url, sample_indices_fn=None)
    # start_time = time.time()
    # for _ in range(10):
    #     fetch_videos(sample_video_url, sample_indices_fn=None)
    # end_time = time.time()
    # print(f"Average decoding time: {(end_time - start_time) / 10:.4f} seconds")

    for video_url in tqdm(raw_video_urls):
        video_array, metadata = fetch_videos(
            video_url, sample_indices_fn=None
        )  # Load full video
        if isinstance(video_array, np.ndarray):
            video_tensor = torch.from_numpy(video_array)
        else:
            video_tensor = video_array

        save_path = (
            f"{processed_video_rootdir}/{os.path.basename(video_url).split('.')[0]}.pt"
        )
        torch.save({"video": video_tensor, "metadata": metadata}, save_path)
