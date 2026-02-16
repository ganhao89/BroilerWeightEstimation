# =========================
# Segmentation Strategy
# =========================

from abc import ABC, abstractmethod
import cv2
import numpy as np


class Segmenter(ABC):
    @abstractmethod
    def segment(self, frame: np.ndarray) -> np.ndarray:
        pass


class DeepLearningSegmenter(Segmenter):

    def __init__(self, model_path: str):
        self.model_path = model_path
        print(f"[DeepLearningSegmenter] Initialized with model: {model_path}")

    def segment(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError("DeepLearningSegmenter is not yet implemented.")


class ClassicalSegmenter(Segmenter):

    def __init__(self):
        print("[ClassicalSegmenter] Initialized")

    def segment(self, frame: np.ndarray) -> np.ndarray:

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        blur = cv2.GaussianBlur(v, (5, 5), 0)
        _, mask = cv2.threshold(
            blur, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Remove orange color (feeder) from mask
        feeder_mask = ((h > 8) & (h < 20) & (s > 150) & (v > 100)).astype(np.uint8) * 255
        mask[feeder_mask == 255] = 0

        kernel = np.ones((5, 5), np.uint8)
        mask_morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, kernel, iterations=2)

        mask_morph = cv2.erode(mask_morph, kernel, iterations = 1)
        mask_morph = cv2.dilate(mask_morph, kernel, iterations = 1)

        return mask_morph
