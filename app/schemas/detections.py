from __future__ import annotations
from pydantic import BaseModel


class DetectionOut(BaseModel):
    bbox             : tuple[float, float, float, float]
    class_name       : str
    score            : float
    inside_cartouche : bool
    cartouche_id     : int
