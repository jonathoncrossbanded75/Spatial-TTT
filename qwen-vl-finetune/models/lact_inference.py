# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import tyro
from safetensors import safe_open
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration
from transformers.cache_utils import DynamicCache
from utils.general import fetch_videos

from .causal_swa_lact import LaCTCache, Qwen3VLLaCTSWIGLULayer
from .causal_swa_lact_streaming import Qwen3VLLaCTSWIGLULayerStreaming
from .causal_swa_lact_streaming_chunked import (
    Qwen3VLLaCTSWIGLULayerStreamingChunked,
)


def wrap_model_with_lact(
    model: Qwen3VLForConditionalGeneration,
    num_lact_heads: int = 4,
    w0_w2_low_rank: int = 32,
    use_fused_kernel: bool = True,
    lact_chunk_size: int = 2560,
    window_size: int = 2560,
    lact_layers: Optional[str] = None,
    **kwargs,
) -> Qwen3VLForConditionalGeneration:
    if lact_layers is not None:
        lact_layer_indices = set(int(x.strip()) for x in lact_layers.split("/"))
    else:
        lact_layer_indices = None  # apply to all layers

    for idx, layer in enumerate(model.model.language_model.layers):
        if lact_layer_indices is not None and idx not in lact_layer_indices:
            continue

        old_attn = layer.self_attn
        model_dtype = next(old_attn.parameters()).dtype
        model_device = next(old_attn.parameters()).device

        lact_layer = Qwen3VLLaCTSWIGLULayerStreamingChunked(
            attn_layer=old_attn,
            num_lact_heads=num_lact_heads,
            w0_w2_low_rank=w0_w2_low_rank,
            use_fused_kernel=use_fused_kernel,
            lact_chunk_size=lact_chunk_size,
            window_size=window_size,
            **kwargs,
        )
        lact_layer = lact_layer.to(dtype=model_dtype, device=model_device)
        layer.self_attn = lact_layer

    return model


class LaCTGenerationMixin:
    @torch.no_grad()
    def generate_with_lact(
        self,
        input_ids: torch.Tensor,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = False,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        assert batch_size == 1, "LaCT generation currently only supports batch_size=1"

        device = input_ids.device
        qwen_model = self._model
        inputs_embeds = qwen_model.model.language_model.embed_tokens(input_ids)
        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = qwen_model.model.get_image_features(
                pixel_values, image_grid_thw
            )
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = qwen_model.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = qwen_model.model.get_video_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = qwen_model.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(
                deepstack_image_embeds, deepstack_video_embeds
            ):
                embed_joint = img_embed.new_zeros(
                    visual_pos_masks.sum(), img_embed.shape[-1]
                ).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        model = self.model.language_model
        rotary_emb = model.rotary_emb
        norm = model.norm
        lm_head = self.lm_head

        # prefill phase
        hidden_states = inputs_embeds
        position_ids, rope_deltas = self.model.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw
        )
        self.rope_deltas = rope_deltas
        position_embeddings = rotary_emb(hidden_states, position_ids)

        lact_cache = LaCTCache()
        past_key_values = DynamicCache()

        cache_position = torch.tensor([0], device=device)
        seq_len = hidden_states.shape[1]

        # run through all layers (prefill)
        for layer_idx, layer in enumerate(model.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            if isinstance(layer.self_attn, Qwen3VLLaCTSWIGLULayer) or isinstance(
                layer.self_attn,
                (Qwen3VLLaCTSWIGLULayerStreaming, Qwen3VLLaCTSWIGLULayerStreamingChunked),
            ):
                hidden_states, _ = layer.self_attn(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    lact_cache=lact_cache,
                    video_mask=video_mask,
                    video_grid_thw=video_grid_thw,
                )
            else:
                hidden_states, _ = layer.self_attn(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    attention_mask=None,
                )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(
                len(deepstack_visual_embeds)
            ):
                hidden_states = model._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = norm(hidden_states)
        logits = lm_head(hidden_states[:, -1:, :])

        next_token = self._sample_token(logits, temperature, top_k, top_p, do_sample)

        generated_tokens = [next_token]

        embed_tokens = model.embed_tokens

        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

            hidden_states = embed_tokens(next_token)
            position_ids_step = (
                torch.tensor([seq_len], device=device) + rope_deltas
            ).to(hidden_states.device)

            position_ids_step = position_ids_step.unsqueeze(0).expand(3, -1, -1)
            position_embeddings = rotary_emb(hidden_states, position_ids_step)
            cache_position = torch.tensor([seq_len], device=device)
            # run through all layers (decode)
            for layer_idx, layer in enumerate(model.layers):
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)

                if isinstance(layer.self_attn, Qwen3VLLaCTSWIGLULayer) or isinstance(
                    layer.self_attn,
                    (
                        Qwen3VLLaCTSWIGLULayerStreaming,
                        Qwen3VLLaCTSWIGLULayerStreamingChunked,
                    ),
                ):
                    hidden_states, _ = layer.self_attn(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        lact_cache=lact_cache,
                        video_mask=None,
                        video_grid_thw=video_grid_thw,
                    )
                else:
                    hidden_states, _ = layer.self_attn(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        attention_mask=None,
                    )
                hidden_states = residual + hidden_states

                residual = hidden_states
                hidden_states = layer.post_attention_layernorm(hidden_states)
                hidden_states = layer.mlp(hidden_states)
                hidden_states = residual + hidden_states

            hidden_states = norm(hidden_states)
            logits = lm_head(hidden_states)
            next_token = self._sample_token(
                logits, temperature, top_k, top_p, do_sample
            )
            generated_tokens.append(next_token)
            seq_len += 1

        all_tokens = torch.cat([input_ids] + generated_tokens, dim=1)
        return all_tokens

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        do_sample: bool,
    ) -> torch.Tensor:
        logits = logits[:, -1, :]

        # greedy
        if not do_sample:
            return logits.argmax(dim=-1, keepdim=True)

        # apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # apply top-k
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # apply top-p (nucleus sampling)
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        # sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token


class Qwen3VLLaCTForConditionalGeneration(LaCTGenerationMixin):
    def __init__(
        self,
        model: Qwen3VLForConditionalGeneration,
    ):
        self.model = model.model
        self.lm_head = model.lm_head
        self._model = model

    def __getattr__(self, name):
        # delegate to underlying model
        if name in ("model", "lm_head", "_model"):
            return object.__getattribute__(self, name)
        return getattr(self._model, name)

    def to(self, *args, **kwargs):
        self._model = self._model.to(*args, **kwargs)
        self.model = self._model.model
        self.lm_head = self._model.lm_head
        return self

    def eval(self):
        self._model.eval()
        return self

    def train(self, mode=True):
        self._model.train(mode)
        return self


def load_lact_model(
    model_path: str,
    num_lact_heads: int = 4,
    w0_w2_low_rank: int = 32,
    use_fused_kernel: bool = True,
    use_conv_layer: bool = False,
    lact_chunk_size: int = 2560,
    window_size: int = 2560,
    lact_layers: Optional[str] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    **kwargs,
) -> Qwen3VLLaCTForConditionalGeneration:
    # load base model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        **kwargs,
    )

    model = wrap_model_with_lact(
        model,
        num_lact_heads=num_lact_heads,
        w0_w2_low_rank=w0_w2_low_rank,
        use_fused_kernel=use_fused_kernel,
        lact_chunk_size=lact_chunk_size,
        window_size=window_size,
        use_conv_layer=use_conv_layer,
        lact_layers=lact_layers,
    )

    # Load trained checkpoint if provided
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_dir():
            # Look for model.safetensors in the directory
            safetensors_path = checkpoint_path / "model.safetensors"
            if not safetensors_path.exists():
                raise FileNotFoundError(
                    f"model.safetensors not found in {checkpoint_path}"
                )
        else:
            safetensors_path = checkpoint_path

        print(f"Loading checkpoint from {safetensors_path}...")
        state_dict = {}
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

        # Load state dict
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
        print(f"Loaded {len(state_dict)} parameters from checkpoint")

    if device is not None:
        # model = model.to(device)
        model = model.to("cuda")

    # convert to inference wrapper
    lact_model = Qwen3VLLaCTForConditionalGeneration(model)

    return lact_model


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def load_dataset_samples(
    annotation_path: str,
    num_samples: int = 3,
    shuffle: bool = True,
    seed: int = 42,
):
    data = read_jsonl(annotation_path)

    if shuffle:
        import random

        random.seed(seed)
        random.shuffle(data)

    return data[:num_samples]


def build_messages_from_sample(sample: Dict, data_path: str = "") -> tuple:
    base_path = Path(data_path) if data_path else Path("")

    # Extract media
    images = sample.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = sample.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Build media pools with absolute paths
    def make_abs_path(p):
        p = Path(p)
        if p.is_absolute():
            return str(p)
        return str((base_path / p).resolve())

    image_pool = [{"type": "image", "image": make_abs_path(img)} for img in images]
    video_pool = [{"type": "video", "video": make_abs_path(vid)} for vid in videos]

    messages = []
    ground_truth = None

    for turn in sample["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if image_pool:
                        content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if video_pool:
                        content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # assistant message is the ground truth
            ground_truth = text

    return messages, ground_truth


def run_inference_on_dataset(
    model,
    processor: AutoProcessor,
    annotation_path: str,
    num_samples: int = 3,
    max_new_tokens: int = 128,
    data_path: str = "",
    device: str = "cuda",
    use_native_generate: bool = False,
    video_max_pixels: int = 1024 * 28 * 28,
):
    print(f"\n{'=' * 60}")
    print(f"Running inference on {num_samples} samples from:")
    print(f"  {annotation_path}")
    print(f"video_max_pixels: {video_max_pixels}")
    print(f"{'=' * 60}\n")

    samples = load_dataset_samples(annotation_path, num_samples)

    results = []

    for idx, sample in enumerate(samples):
        print(f"\n[Sample {idx + 1}/{num_samples}]")
        print("-" * 40)

        messages, ground_truth = build_messages_from_sample(sample, data_path)

        # Print question (last user message)
        user_content = messages[-1]["content"]
        question_text = " ".join(
            c["text"] for c in user_content if c.get("type") == "text"
        )
        print(f"Question: {question_text[:200]}...")
        print(f"GT: {ground_truth}")
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids")

        inputs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }

        if use_native_generate:
            raw_model = model._model if hasattr(model, "_model") else model
            with torch.no_grad():
                output_ids = raw_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
        else:
            output_ids = model.generate_with_lact(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_len:]
        prediction = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(f"Pred: {prediction}")

        results.append(
            {
                "question": question_text,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "match": prediction.strip() == ground_truth.strip(),
            }
        )

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    num_correct = sum(1 for r in results if r["match"])
    print(
        f"Exact match: {num_correct}/{len(results)} ({100 * num_correct / len(results):.1f}%)"
    )

    return results


@dataclass
class InferenceConfig:
    model_path: str = (
        "/mnt/public/wdk/proj/ECCV2026/qwenvl-lact/checkpoints/Spatial-2B-base"
    )
    checkpoint_path: Optional[str] = None
    annotation_path: str = "/mnt/public/wdk/proj/ECCV2026/Spatial-Data/datasets/annotations/processed/sp-mllm-vlm3r-reweighted-70k.jsonl"
    num_samples: int = 1
    max_new_tokens: int = 1024
    num_lact_heads: int = 4
    w0_w2_low_rank: int = 128
    lact_chunk_size: int = 2468
    lact_layers: Optional[str] = None
    window_size: int = 2468
    # video_max_pixels: int = 1024 * 28 * 28
    video_max_pixels: int = 9_871_400
    use_conv_layer: bool = False
    simple_test: bool = False


def main(config: InferenceConfig):
    print("=" * 60)
    print("LaCT Inference Testing")
    print("=" * 60)
    print(f"Model path: {config.model_path}")
    print(f"Checkpoint: {config.checkpoint_path or 'None (using base model)'}")
    print(
        f"LaCT config: heads={config.num_lact_heads}, low_rank={config.w0_w2_low_rank}"
    )
    print(
        f"             chunk_size={config.lact_chunk_size}, window_size={config.window_size}"
    )
    print("=" * 60)

    print("\nLoading model...")
    model = load_lact_model(
        config.model_path,
        num_lact_heads=config.num_lact_heads,
        w0_w2_low_rank=config.w0_w2_low_rank,
        use_fused_kernel=False,
        use_conv_layer=config.use_conv_layer,
        lact_chunk_size=config.lact_chunk_size,
        window_size=config.window_size,
        lact_layers=config.lact_layers,
        checkpoint_path=config.checkpoint_path,
        device="cuda",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(config.model_path)
    type(processor.video_processor).fetch_videos = fetch_videos

    if config.simple_test:
        # simple text generation test
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        prompt = "Introduce yourself!"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

        print(f"\nIn: {prompt}")
        print("Generating...")

        output_ids = model.generate_with_lact(
            input_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Output: {output_text}")
    else:
        # dataset inference test
        run_inference_on_dataset(
            model=model,
            processor=processor,
            annotation_path=config.annotation_path,
            num_samples=config.num_samples,
            max_new_tokens=config.max_new_tokens,
            device="cuda",
            use_native_generate=False,
            video_max_pixels=config.video_max_pixels,
        )


if __name__ == "__main__":
    config = tyro.cli(InferenceConfig)
    main(config)
