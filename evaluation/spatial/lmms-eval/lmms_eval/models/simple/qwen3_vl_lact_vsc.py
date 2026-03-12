import copy
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[6]
print(f"Repo root: {repo_root}")
sys.path.append(str(repo_root / "qwen-vl-finetune"))
print(f"sys.path: {sys.path}")

import re
from typing import List, Optional, Tuple, Union

import decord
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning(
        "Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`"
    )

from models.lact_inference import load_lact_model


@register_model("qwen3_vl_lact_vsc")
class Qwen3_VL_LaCT_VSC(lmms):
    """
    Qwen3_VL Model
    "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-4B-Instruct",
        lact_checkpoint_path: Optional[str] = None,
        num_lact_heads: int = 4,
        w0_w2_low_rank: int = 32,
        use_fused_kernel: bool = False,
        use_conv_layer: bool = False,
        lact_chunk_size: int = 2560,
        window_size: int = 2560,
        lact_layers: Optional[str] = None,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        resize_height: int = 480,
        resize_width: int = 640,
        max_num_frames: int = 32,
        video_chunk_size: int = 600,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[
            float
        ] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[
            int
        ] = None,  # Only applicable if use_custom_video_loader is True
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(
                f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}"
            )

        # Lact parameters
        self.num_lact_heads = num_lact_heads
        self.w0_w2_low_rank = w0_w2_low_rank
        self.use_fused_kernel = use_fused_kernel
        self.use_conv_layer = use_conv_layer
        self.lact_chunk_size = lact_chunk_size
        self.window_size = window_size
        self.lact_layers = lact_layers

        self.resize_height = resize_height
        self.resize_width = resize_width

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError(
                "max_image_size is only applicable if use_custom_video_loader is True"
            )

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "dtype": "bfloat16",
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # load the model with LaCT
        self._model = load_lact_model(
            model_path=pretrained,
            num_lact_heads=num_lact_heads,
            w0_w2_low_rank=w0_w2_low_rank,
            use_fused_kernel=self.use_fused_kernel,
            use_conv_layer=self.use_conv_layer,
            lact_chunk_size=lact_chunk_size,
            window_size=window_size,
            lact_layers=lact_layers,
            checkpoint_path=lact_checkpoint_path,
            device=self.device_map,
        ).eval()

        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self.video_chunk_size = video_chunk_size

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.processor = AutoProcessor.from_pretrained(
            pretrained, max_pixels=max_pixels, min_pixels=min_pixels
        )

        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(
                    f"Using {accelerator.num_processes} devices with data parallelism"
                )
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _num_video_frames(self, video) -> int:
        if hasattr(video, "shape"):
            return int(video.shape[0])
        return len(video)

    def _split_video_with_metadata(self, video, metadata, chunk_size: int):
        total_frames = self._num_video_frames(video)
        if total_frames <= chunk_size:
            return [(video, metadata)]

        chunks = []
        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)
            chunk_video = video[start:end]
            chunk_metadata = None
            if metadata is not None:
                if isinstance(metadata, dict):
                    chunk_metadata = metadata.copy()
                    frames_indices = chunk_metadata.get("frames_indices")
                else:
                    chunk_metadata = copy.copy(metadata)
                    frames_indices = getattr(chunk_metadata, "frames_indices", None)

                if frames_indices is not None:
                    if isinstance(frames_indices, torch.Tensor):
                        frames_indices = frames_indices.tolist()
                    elif not isinstance(frames_indices, list):
                        frames_indices = list(frames_indices)
                    frames_indices = frames_indices[start:end]
                    if isinstance(chunk_metadata, dict):
                        chunk_metadata["frames_indices"] = frames_indices
                    else:
                        setattr(chunk_metadata, "frames_indices", frames_indices)

            chunks.append((chunk_video, chunk_metadata))
        return chunks

    def _extract_numeric_value(self, text: str) -> Optional[float]:
        if text is None:
            return None
        cleaned = text.replace(",", "")
        matches = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", cleaned)
        if not matches:
            return None
        try:
            return float(matches[-1])
        except ValueError:
            return None

    def _format_numeric_sum(self, value: float) -> str:
        if abs(value - round(value)) < 1e-6:
            return str(int(round(value)))
        return str(value)

    def _run_generation(
        self,
        texts: List[str],
        image_inputs,
        video_inputs,
        video_metadatas,
        video_kwargs,
        current_gen_kwargs,
        until: List[str],
    ):
        if self.batch_size > 1 and len(texts) > 1:
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadatas=video_metadatas,
                do_resize=False,
                padding=True,
                padding_side="left",
                return_tensors="pt",
                **video_kwargs,
            )
        else:
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadatas=video_metadatas,
                do_resize=False,
                return_tensors="pt",
                **video_kwargs,
            )

        if self.device_map == "auto":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to(self.device)

        if "pixel_values_videos" in inputs:
            eval_logger.debug(
                f"inputs['pixel_values_videos'].shape: {inputs['pixel_values_videos'].shape}"
            )

        cont = self.model.generate_with_lact(
            inputs["input_ids"],
            pixel_values_videos=inputs.get("pixel_values_videos", None),
            video_grid_thw=inputs.get("video_grid_thw", None),
            pixel_values=inputs.get("pixel_values", None),
            image_grid_thw=inputs.get("image_grid_thw", None),
            max_new_tokens=current_gen_kwargs["max_new_tokens"],
            do_sample=False,
            eos_token_id=self.eot_token_id,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, cont)
        ]
        answers = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for i, ans in enumerate(answers):
            for term in until:
                if len(term) > 0:
                    ans = ans.split(term)[0]
            answers[i] = ans

        return answers

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests), disable=(self.rank != 0), desc="Model Responding"
        )
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [
                doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id
            ]
            gen_kwargs = all_gen_kwargs[0]

            # Set default until or update values from gen_kwargs if present
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])

            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(
                    f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}"
                )

            # Avoid using '\n\n' as a stopper for Qwen2.5VL to prevent truncation, which can lead to incorrect results
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": self.system_prompt}]
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                if visual_list[i] is not None:
                    if isinstance(visual_list[i][0], Image.Image):
                        image_list = []

                        for visual in visual_list[i]:
                            image_list += [visual, visual]
                        processed_visuals.append(
                            {
                                "type": "video",
                                "video": image_list,
                                "resized_height": self.resize_height,
                                "resized_width": self.resize_width,
                                "sample_fps": 1,
                            }
                        )

                    else:
                        for visual in visual_list[i]:
                            if isinstance(visual, str) and visual.endswith(
                                (".mp4", ".avi", ".mov")
                            ):  # Video file
                                vr = decord.VideoReader(visual)
                                first_frame = vr[0].asnumpy()
                                height, width = first_frame.shape[:2]
                                # max_pixels = height * width
                                processed_visuals.append(
                                    {
                                        "type": "video",
                                        "video": visual,
                                        "resized_height": self.resize_height,
                                        "resized_width": self.resize_width,
                                        "nframes": self.max_num_frames,
                                    }
                                )

                # temporal fix; super hacky
                context = (
                    context
                    + " There maybe many rooms in the video, sum the counts from all rooms/scenes to produce the final total."
                )

                if self.interleave_visuals is False:
                    message.append(
                        {
                            "role": "user",
                            "content": processed_visuals
                            + [{"type": "text", "text": context}],
                        }
                    )
                else:  # currently support find <image x> in the context
                    image_placeholders = re.findall(r"<image \d+>", context)
                    content_parts = []
                    text_parts = re.split(r"<image \d+>", context)
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})

                    for i, placeholder in enumerate(image_placeholders):
                        img_idx = (
                            int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        )
                        image_idx = (
                            min(img_idx, len(processed_visuals) - 1)
                            if processed_visuals
                            else 0
                        )
                        if processed_visuals and image_idx < len(processed_visuals):
                            content_parts.append(processed_visuals[image_idx])
                        if i + 1 < len(text_parts) and text_parts[i + 1]:
                            content_parts.append(
                                {"type": "text", "text": text_parts[i + 1]}
                            )

                    message.append(
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    )

                batched_messages.append(message)

            texts = self.processor.apply_chat_template(
                batched_messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                batched_messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs, video_metadatas = (
                    list(video_inputs),
                    list(video_metadatas),
                )
            else:
                video_metadatas = None

            eval_logger.debug(
                f"video_inputs.shape: {None if video_inputs is None else [v.shape for v in video_inputs]}"
            )

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            needs_chunking = False
            if video_inputs is not None:
                needs_chunking = any(
                    self._num_video_frames(video) > self.video_chunk_size
                    for video in video_inputs
                )

            if not needs_chunking:
                answers = self._run_generation(
                    texts,
                    image_inputs,
                    video_inputs,
                    video_metadatas,
                    video_kwargs,
                    current_gen_kwargs,
                    until,
                )
                for ans, context in zip(answers, contexts):
                    clean_ans = parse_reasoning_model_answer(ans)
                    res.append(clean_ans)
                    self.cache_hook.add_partial(
                        "generate_until", (context, gen_kwargs), clean_ans
                    )
                    pbar.update(1)

                    eval_logger.debug(f"Question: {context}")
                    eval_logger.debug(f"Model Raw Response: {ans}")
                    eval_logger.debug(f"Model Clean Response: {clean_ans}")
            else:
                def _count_media(conversation):
                    image_count = 0
                    video_count = 0
                    for message in conversation:
                        content = message.get("content")
                        if isinstance(content, list):
                            for ele in content:
                                if not isinstance(ele, dict):
                                    continue
                                if ele.get("type") == "video" or "video" in ele:
                                    video_count += 1
                                if (
                                    ele.get("type") in ("image", "image_url")
                                    or "image" in ele
                                    or "image_url" in ele
                                ):
                                    image_count += 1
                    return image_count, video_count

                image_offset = 0
                video_offset = 0
                per_sample_images = []
                per_sample_videos = []
                per_sample_video_metadatas = []
                for conversation in batched_messages:
                    image_count, video_count = _count_media(conversation)
                    if image_inputs is None or image_count == 0:
                        per_sample_images.append(None)
                    else:
                        per_sample_images.append(
                            image_inputs[image_offset : image_offset + image_count]
                        )
                        image_offset += image_count

                    if video_inputs is None or video_count == 0:
                        per_sample_videos.append(None)
                        per_sample_video_metadatas.append(None)
                    else:
                        per_sample_videos.append(
                            video_inputs[video_offset : video_offset + video_count]
                        )
                        if video_metadatas is None:
                            per_sample_video_metadatas.append(None)
                        else:
                            per_sample_video_metadatas.append(
                                video_metadatas[
                                    video_offset : video_offset + video_count
                                ]
                            )
                        video_offset += video_count

                for idx, (context, text) in enumerate(zip(contexts, texts)):
                    sample_images = per_sample_images[idx]
                    sample_videos = per_sample_videos[idx]
                    sample_video_metadatas = per_sample_video_metadatas[idx]

                    if not sample_videos:
                        answers = self._run_generation(
                            [text],
                            sample_images,
                            None,
                            None,
                            video_kwargs,
                            current_gen_kwargs,
                            until,
                        )
                        ans = answers[0]
                        clean_ans = parse_reasoning_model_answer(ans)
                        res.append(clean_ans)
                        self.cache_hook.add_partial(
                            "generate_until", (context, gen_kwargs), clean_ans
                        )
                        pbar.update(1)

                        eval_logger.debug(f"Question: {context}")
                        eval_logger.debug(f"Model Raw Response: {ans}")
                        eval_logger.debug(f"Model Clean Response: {clean_ans}")
                        continue

                    if len(sample_videos) != 1:
                        raise ValueError(
                            "Video chunking expects exactly one video per sample."
                        )

                    sample_video = sample_videos[0]
                    sample_metadata = (
                        sample_video_metadatas[0]
                        if sample_video_metadatas is not None
                        else None
                    )
                    if self._num_video_frames(sample_video) > self.video_chunk_size:
                        chunks = self._split_video_with_metadata(
                            sample_video,
                            sample_metadata,
                            self.video_chunk_size,
                        )
                        numeric_sum = 0.0
                        had_non_numeric = False
                        for chunk_idx, (chunk_video, chunk_metadata) in enumerate(chunks):
                            chunk_metadatas = (
                                [chunk_metadata] if chunk_metadata is not None else None
                            )
                            answers = self._run_generation(
                                [text],
                                sample_images,
                                [chunk_video],
                                chunk_metadatas,
                                video_kwargs,
                                current_gen_kwargs,
                                until,
                            )
                            chunk_ans = answers[0]
                            chunk_clean = parse_reasoning_model_answer(chunk_ans)
                            value = self._extract_numeric_value(chunk_clean)
                            if value is None:
                                had_non_numeric = True
                                value = 0.0
                                eval_logger.warning(
                                    "Non-numeric chunk answer encountered; treating as 0 for summation."
                                )
                            numeric_sum += value

                            eval_logger.debug(
                                f"Chunk {chunk_idx + 1}/{len(chunks)} Raw Response: {chunk_ans}"
                            )
                            eval_logger.debug(
                                f"Chunk {chunk_idx + 1}/{len(chunks)} Clean Response: {chunk_clean}"
                            )

                        final_ans = self._format_numeric_sum(numeric_sum)
                        if had_non_numeric:
                            eval_logger.warning(
                                "One or more chunk answers were non-numeric; summed value may be underestimated."
                            )
                        res.append(final_ans)
                        self.cache_hook.add_partial(
                            "generate_until", (context, gen_kwargs), final_ans
                        )
                        pbar.update(1)

                        eval_logger.debug(f"Question: {context}")
                        eval_logger.debug(
                            f"Model Summed Response ({len(chunks)} chunks): {final_ans}"
                        )
                    else:
                        sample_metadatas = (
                            [sample_metadata] if sample_metadata is not None else None
                        )
                        answers = self._run_generation(
                            [text],
                            sample_images,
                            [sample_video],
                            sample_metadatas,
                            video_kwargs,
                            current_gen_kwargs,
                            until,
                        )
                        ans = answers[0]
                        clean_ans = parse_reasoning_model_answer(ans)
                        res.append(clean_ans)
                        self.cache_hook.add_partial(
                            "generate_until", (context, gen_kwargs), clean_ans
                        )
                        pbar.update(1)

                        eval_logger.debug(f"Question: {context}")
                        eval_logger.debug(f"Model Raw Response: {ans}")
                        eval_logger.debug(f"Model Clean Response: {clean_ans}")
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
