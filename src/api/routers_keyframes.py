from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
import os
from src.main.keyframe_extractor.keyframe_pipeline import HybridPipeline, load_hybrid_settings
from src.main.utils.io_utils import flush_dir

router = APIRouter(prefix="/keyframes", tags=["keyframes"])

class KeyframeRequest(BaseModel):
    video_path: str = Field(..., description="Local video file path to process")

class KeyframeResponse(BaseModel):
    mode: str
    keyframes: List[float]
    count: int
    capture_path: str | None = None

@router.post('/extract', response_model=KeyframeResponse, summary="Extract keyframes from local file")
def extract_keyframes(req: KeyframeRequest):
    if not os.path.isfile(req.video_path):
        raise HTTPException(status_code=404, detail='Video not found')
    config_dir = os.path.join('src', 'main', 'keyframe_extractor', 'config')
    cfg = load_hybrid_settings(config_dir)
    cfg['video_path'] = req.video_path
    if cfg.get('output_dir'):
        flush_dir(cfg['output_dir'])
    if cfg.get('save_debug_masks'):
        flush_dir(cfg.get('mask_output_dir', 'hybrid_keyframes/masks'))
    pipeline = HybridPipeline(cfg)
    timestamps = pipeline.run()
    return KeyframeResponse(mode='file', keyframes=timestamps, count=len(timestamps), capture_path=req.video_path)
