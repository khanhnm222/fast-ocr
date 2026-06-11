import json
import os

import cv2
import numpy as np
import tensorflow as tf

# Ordered by preference: EMNIST (full alphanumeric) before MNIST (digits only)
_CANDIDATES = [
    ('saved_models/emnist_cnn.keras', 'saved_models/emnist_labels.json'),
    ('saved_models/mnist_cnn.keras', 'saved_models/mnist_labels.json'),
]

_FALLBACK_LABELS = {
    10: [str(i) for i in range(10)],
    62: (
        [str(i) for i in range(10)]
        + [chr(ord('A') + i) for i in range(26)]
        + [chr(ord('a') + i) for i in range(26)]
    ),
}


def _resolve_paths(model_path: str | None) -> tuple[str, str | None]:
    """Return (model_path, labels_path) — auto-detect if model_path is None."""
    if model_path is not None:
        base = os.path.splitext(model_path)[0]
        labels_path = base + '_labels.json'
        return model_path, labels_path if os.path.exists(labels_path) else None

    for mp, lp in _CANDIDATES:
        if os.path.exists(mp):
            return mp, lp if os.path.exists(lp) else None

    raise FileNotFoundError(
        "No trained model found in saved_models/. "
        "Run `python train.py` (MNIST) or `python train.py --dataset emnist`."
    )


class OCRPredictor:
    def __init__(self, model_path: str | None = None):
        resolved_model, resolved_labels = _resolve_paths(model_path)

        self.model: tf.keras.Model = tf.keras.models.load_model(resolved_model)
        print(f"Loaded model: {resolved_model}")

        if resolved_labels:
            with open(resolved_labels) as f:
                self.labels: list[str] = json.load(f)
            print(f"Loaded labels: {resolved_labels}  ({len(self.labels)} classes)")
        else:
            num_classes = self.model.output_shape[-1]
            self.labels = _FALLBACK_LABELS.get(num_classes, [str(i) for i in range(num_classes)])
            print(f"No labels file found — using fallback for {num_classes} classes.")

    def predict_character(self, char_image: np.ndarray) -> tuple[str, float]:
        """Predict a single character from a grayscale binary image."""
        resized = cv2.resize(char_image, (28, 28), interpolation=cv2.INTER_AREA)
        tensor = resized.astype('float32') / 255.0
        tensor = tensor.reshape(1, 28, 28, 1)
        probs = self.model.predict(tensor, verbose=0)[0]
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])

    def predict_characters(self, char_images: list[np.ndarray]) -> list[tuple[str, float]]:
        """Batch predict a list of character images."""
        if not char_images:
            return []
        batch = np.stack([
            cv2.resize(img, (28, 28)).astype('float32') / 255.0
            for img in char_images
        ]).reshape(-1, 28, 28, 1)
        probs = self.model.predict(batch, verbose=0)
        return [
            (self.labels[int(np.argmax(p))], float(np.max(p)))
            for p in probs
        ]