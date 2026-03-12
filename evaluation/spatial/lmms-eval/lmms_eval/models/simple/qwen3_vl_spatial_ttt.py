"""
Spatial-TTT eval model alias: same implementation as qwen3_vl_lact, user-facing name.
"""
from lmms_eval.api.registry import register_model

from .qwen3_vl_lact import Qwen3_VL_LaCT


@register_model("qwen3_vl_spatial_ttt")
class Qwen3_VL_SpatialTTT(Qwen3_VL_LaCT):
    """Spatial-TTT (Qwen3-VL with Test-Time Training). Alias of Qwen3_VL_LaCT."""

    pass
