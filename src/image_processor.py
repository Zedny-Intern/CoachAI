"""
Basic image processing for the Multimodal Learning Coach
"""
import numpy as np
import streamlit as st
from PIL import Image


class ImageProcessor:
    """Basic image validation and resizing for Qwen3-VL processing"""

    @staticmethod
    def validate_image(image):
        """Validate image format and basic quality checks"""
        try:
            img_array = np.array(image)
            height, width = img_array.shape[:2]

            # Check minimum size for meaningful analysis
            min_pixels = 224 * 224
            if height * width < min_pixels:
                st.warning(f"⚠️ Image too small ({width}x{height}). Minimum: {int(min_pixels**0.5)}x{int(min_pixels**0.5)} pixels")
                return False

            # Check maximum size (Qwen3-VL has limits)
            max_pixels = 1280 * 1280
            if height * width > max_pixels:
                st.info(f"📏 Large image detected ({width}x{height}). Will be resized for optimal processing.")

            return True
        except Exception as e:
            st.error(f"❌ Invalid image format: {e}")
            return False

    @staticmethod
    def resize_image(image, max_pixels=1280*1280):
        """Resize image if it's too large for efficient processing"""
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        if height * width <= max_pixels:
            return image

        # Calculate new dimensions maintaining aspect ratio
        aspect_ratio = width / height
        new_height = int((max_pixels / aspect_ratio) ** 0.5)
        new_width = int(new_height * aspect_ratio)

        # Ensure minimum size for quality
        new_height = max(new_height, 224)
        new_width = max(new_width, 224)

        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized