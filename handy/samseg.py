"""SAM segmentation for the wrist camera: one-shot text prompt + frame-to-frame mask tracking.

SAM 3 turns a text prompt ("screwdriver") into an initial mask on one frame; a SAM 2 video
session in streaming mode then tracks that mask frame to frame. Downstream code only needs
the mask centroid.

Ported from ~/so-101-rl/real/tracking/sam_seg.py (CUDA/bf16 there); here the device is picked
at load time: Apple GPU (mps) if available, else CPU. Weights come from the HF hub cache.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # CPU fallback for ops mps lacks

import cv2
import numpy as np
import torch

SAM3_ID = "facebook/sam3"
SAM2_MODELS = {"tiny": "facebook/sam2.1-hiera-tiny", "base+": "facebook/sam2.1-hiera-base-plus"}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# A streaming session keeps every frame's memory features for as long as it lives; SAM 2 only
# reads the most recent few plus the conditioning frame. Roll the session at this many frames,
# re-seeding it from the mask just produced so tracking carries on unbroken.
SESSION_MAX_FRAMES = 200


def load_sam3():
    """SAM 3 model + processor for one-shot text prompting (fp32; it runs once per pick)."""
    from transformers import Sam3Model, Sam3Processor
    model = Sam3Model.from_pretrained(SAM3_ID).to(DEVICE).eval()
    return model, Sam3Processor.from_pretrained(SAM3_ID)


def find_text_mask(sam3, frame_bgr: np.ndarray, prompt: str, threshold: float = 0.5):
    """Highest-scoring mask for `prompt`: (HxW bool, score), or None when nothing matches."""
    model, processor = sam3
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, text=prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=threshold, mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist())[0]
    scores = results["scores"]
    if len(scores) == 0:
        return None
    top = int(scores.argmax())
    return results["masks"][top].cpu().numpy().astype(bool), float(scores[top])


class MaskTracker:
    """Streaming SAM 2 mask tracker bound to one camera stream."""

    def __init__(self, model: str = "tiny", dtype: torch.dtype = torch.float32,
                 session_max_frames: int = SESSION_MAX_FRAMES):
        from transformers import Sam2VideoModel, Sam2VideoProcessor
        repo = SAM2_MODELS[model]
        self._dtype = dtype
        self._model = Sam2VideoModel.from_pretrained(repo, dtype=dtype).to(DEVICE).eval()
        self._processor = Sam2VideoProcessor.from_pretrained(repo)
        self._session = None
        self._session_max_frames = session_max_frames
        self._frames = 0

    def prime(self, frame_bgr: np.ndarray, mask: np.ndarray) -> None:
        """Seed a fresh streaming session with an initial mask on this frame."""
        self._session = self._processor.init_video_session(inference_device=DEVICE, dtype=self._dtype)
        self._frames = 0
        inputs = self._preprocess(frame_bgr)
        self._processor.add_inputs_to_inference_session(
            inference_session=self._session, frame_idx=0, obj_ids=1,
            input_masks=mask, original_size=inputs.original_sizes[0])
        self._forward(inputs)

    def track(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Next frame's HxW bool mask; all False means the object was lost this frame."""
        mask = self._forward(self._preprocess(frame_bgr))
        self._frames += 1
        if self._frames >= self._session_max_frames and mask.any():
            self.prime(frame_bgr, mask)
        return mask

    def _preprocess(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._processor(images=rgb, device=DEVICE, return_tensors="pt")

    def _forward(self, inputs) -> np.ndarray:
        with torch.inference_mode():
            out = self._model(inference_session=self._session,
                              frame=inputs.pixel_values[0].to(self._dtype))
        masks = self._processor.post_process_masks([out.pred_masks], original_sizes=inputs.original_sizes)[0]
        return masks[0, 0].cpu().numpy().astype(bool)


def mask_centroid(mask: np.ndarray):
    """(u, v) pixel centroid of a bool mask, or None for an empty mask."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())
