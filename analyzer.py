"""
analyzer.py - OpenCV-based road damage image analysis module.
Extracts damage features from road images using computer vision techniques.
"""

import cv2
import numpy as np
import base64
from PIL import Image
import io
import os


class RoadDamageAnalyzer:
    """Analyzes road images to extract damage features using OpenCV."""

    def __init__(self):
        self.min_contour_area = 100  # Minimum area to count as damaged region

    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from path and return as numpy array."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from: {image_path}")
        return img

    def extract_features(self, image_path: str) -> dict:
        """
        Extract damage features from road image.

        Returns:
            dict with keys:
                - damage_percentage: float (0-100)
                - num_damaged_regions: int
                - crack_density: float (0-1)
                - texture_roughness: float (0-1)
                - dark_surface_percentage: float (0-100)
                - processed_image_b64: base64-encoded processed image
                - original_image_b64: base64-encoded original image
        """
        img = self.load_image(image_path)
        original = img.copy()

        # --- Damage Mask via Adaptive Thresholding ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        adaptive = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 3
        )

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

        total_pixels = img.shape[0] * img.shape[1]
        damaged_pixels = int(np.sum(cleaned > 0))
        damage_percentage = round((damaged_pixels / total_pixels) * 100, 2)

        # --- Contour Detection (Damaged Regions) ---
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_contours = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]
        num_damaged_regions = len(significant_contours)

        # --- Crack Density via Canny Edge Detection ---
        edges = cv2.Canny(blurred, 50, 150)
        crack_density = round(float(np.sum(edges > 0)) / total_pixels, 4)

        # --- Texture Roughness via Laplacian Variance ---
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        raw_roughness = laplacian.var()
        texture_roughness = round(min(raw_roughness / 5000.0, 1.0), 4)

        # --- Dark Surface Percentage (damaged asphalt appears darker) ---
        dark_mask = gray < 80
        dark_surface_percentage = round((np.sum(dark_mask) / total_pixels) * 100, 2)

        # --- Build Processed Image with Overlays ---
        processed = original.copy()

        # Draw significant contours in red with fill
        overlay = processed.copy()
        cv2.drawContours(overlay, significant_contours, -1, (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.4, processed, 0.6, 0, processed)

        # Draw contour borders in bright red
        cv2.drawContours(processed, significant_contours, -1, (0, 0, 220), 2)

        # Overlay edges in yellow
        edge_overlay = np.zeros_like(processed)
        edge_overlay[edges > 0] = [0, 255, 255]
        cv2.addWeighted(edge_overlay, 0.3, processed, 1.0, 0, processed)

        # --- Encode images to base64 ---
        original_b64 = self._encode_image(original)
        processed_b64 = self._encode_image(processed)

        return {
            "damage_percentage": damage_percentage,
            "num_damaged_regions": num_damaged_regions,
            "crack_density": crack_density,
            "texture_roughness": texture_roughness,
            "dark_surface_percentage": dark_surface_percentage,
            "processed_image_b64": processed_b64,
            "original_image_b64": original_b64,
        }

    def _encode_image(self, img: np.ndarray) -> str:
        """Encode an OpenCV image as a base64 JPEG string."""
        success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise RuntimeError("Failed to encode image")
        return base64.b64encode(buffer).decode("utf-8")
