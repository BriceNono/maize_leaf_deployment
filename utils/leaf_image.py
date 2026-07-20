"""
utils/leaf_image.py
===================
LeafImage — single-image preprocessing for Flask inference.

EXACTLY mirrors Colab Cell 12 (LeafImage._apply_preprocessing):
  1. PIL.open().convert('RGB')
  2. .resize(img_size, PILImage.LANCZOS)
  3. np.array(..., dtype=np.float32) / 255.0   ← matches rescale=1/255
  4. np.expand_dims(arr, axis=0)               ← shape (1,224,224,3)

DO NOT change to preprocess_input — training used rescale=1.0/255.
"""

import os
import io
import numpy as np
from PIL import Image as PILImage


class LeafImage:
    """
    Represents a single maize leaf image prepared for VGG16 inference.
    Mirrors Colab Cell 12 exactly.

    Attributes:
        source    : str path or '<bytes>'
        img_array : np.ndarray shape (1, 224, 224, 3), float32, [0,1]
        _pil_img  : PIL.Image for display
    """

    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    def __init__(self):
        self.source    = None
        self.img_array = None
        self._pil_img  = None

    @classmethod
    def from_path(cls, path: str, img_size: tuple = (224, 224)) -> 'LeafImage':
        """Load and preprocess from a file path."""
        ext = os.path.splitext(path)[1]
        if ext not in cls.ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{ext}'. Allowed: {cls.ALLOWED_EXTENSIONS}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: '{path}'")
        obj = cls()
        obj.source   = path
        obj._pil_img = PILImage.open(path).convert('RGB')
        obj._apply_preprocessing(img_size)
        return obj

    @classmethod
    def from_bytes(cls, raw_bytes: bytes, img_size: tuple = (224, 224)) -> 'LeafImage':
        """Load and preprocess from raw bytes (Flask upload)."""
        obj = cls()
        obj.source   = '<bytes>'
        obj._pil_img = PILImage.open(io.BytesIO(raw_bytes)).convert('RGB')
        obj._apply_preprocessing(img_size)
        return obj

    def _apply_preprocessing(self, img_size: tuple) -> None:
        """
        Preprocessing pipeline — identical to Colab Cell 12:
          Step 1: resize to img_size with LANCZOS resampling
          Step 2: convert to float32 numpy array
          Step 3: divide by 255.0  (matches ImageDataGenerator rescale=1/255)
          Step 4: expand dims  →  shape (1, H, W, 3)
        """
        resized        = self._pil_img.resize(img_size, PILImage.LANCZOS)
        arr            = np.array(resized, dtype=np.float32)
        arr            = arr / 255.0                   # rescale=1/255
        self.img_array = np.expand_dims(arr, axis=0)  # (1,224,224,3)
