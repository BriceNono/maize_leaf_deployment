"""
utils/cnn_model.py
==================
CNNModel — loads vgg16_maize_best.h5 and runs inference.
Mirrors Colab Cell 6 CNNModel.load() and the predict() call in Cell 15.
Model is loaded ONCE at Flask startup; reused for every request.
"""

import logging
import numpy as np

log = logging.getLogger('CNNModel')


class CNNModel:
    """
    Wraps the saved Keras VGG16 model for single-image inference.

    Attributes:
        num_classes (int)         : 4
        img_size    (tuple)       : (224, 224)
        model       (keras.Model) : loaded model, None until load()
    """

    def __init__(self, num_classes: int = 4, img_size: tuple = (224, 224)):
        self.num_classes = num_classes
        self.img_size    = img_size
        self.model       = None

    def load(self, model_path: str) -> None:
        """
        Load .h5 model from disk. Called once at Flask startup.
        Raises RuntimeError if TensorFlow cannot load the file.
        """
        import tensorflow as tf
        try:
            self.model = tf.keras.models.load_model(model_path)
            log.info(
                "Model loaded from '%s' | params: %s",
                model_path,
                f"{self.model.count_params():,}"
            )
        except Exception as exc:
            log.error("Model load failed: %s", exc)
            raise RuntimeError(f"Model load failed: {exc}") from exc

    def predict(self, img_array: np.ndarray) -> np.ndarray:
        """
        Run inference. Mirrors Cell 15: model.predict(leaf.img_array, verbose=0)[0]

        Args:
            img_array: shape (1,224,224,3), float32, values in [0,1]

        Returns:
            np.ndarray shape (4,) — softmax probs
            Index: 0=Blight  1=Common_Rust  2=Gray_Leaf_Spot  3=Healthy
        """
        if self.model is None:
            raise RuntimeError("Call CNNModel.load(path) before predict().")
        probs = self.model.predict(img_array, verbose=0)
        return probs[0]   # shape (4,)
