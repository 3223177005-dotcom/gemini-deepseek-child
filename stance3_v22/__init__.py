# -*- coding: utf-8 -*-
"""stance3 v22 日本 2ch 立场分类器

三类标签：
    yes   支持改宪与国防强化议题
    no    反对上述议题
    other 无立场
"""
from .predictor import LABELS, StancePredictorV22
from .context import build_context

__version__ = "22.0"
__all__ = ["StancePredictorV22", "LABELS", "build_context"]
