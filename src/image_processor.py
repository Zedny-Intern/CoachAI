"""
Image processing and OCR functionality for the Multimodal Learning Coach
"""
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import pytesseract


class ImageProcessor:
    """Advanced image preprocessing and OCR"""

    @staticmethod
    def validate_image(image):
        """Validate image format and quality"""
        try:
            img_array = np.array(image)
            height, width = img_array.shape[:2]

            # Check minimum size
            min_pixels = 224 * 224
            if height * width < min_pixels:
                st.warning(f"⚠️ Image too small ({width}x{height}). Minimum: {int(min_pixels**0.5)}x{int(min_pixels**0.5)} pixels")
                return False

            # Check maximum size
            max_pixels = 1280 * 1280
            if height * width > max_pixels:
                st.info(f"📏 Large image detected ({width}x{height}). Will be resized for processing.")

            return True
        except Exception as e:
            st.error(f"❌ Invalid image format: {e}")
            return False

    @staticmethod
    def resize_image(image, max_pixels=1280*1280):
        """Resize image if it's too large for processing"""
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        if height * width <= max_pixels:
            return image

        # Calculate new dimensions maintaining aspect ratio
        aspect_ratio = width / height
        new_height = int((max_pixels / aspect_ratio) ** 0.5)
        new_width = int(new_height * aspect_ratio)

        # Ensure minimum size
        new_height = max(new_height, 224)
        new_width = max(new_width, 224)

        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized

    @staticmethod
    def enhance_image_for_math(image):
        """Enhanced preprocessing specifically for math equations and handwritten text"""
        img_array = np.array(image)

        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Resize for better OCR (Tesseract works better with certain sizes)
        height, width = gray.shape
        if height < 300 or width < 300:
            scale_factor = max(300 / height, 300 / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Enhance contrast for handwritten text
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # Apply morphological operations to clean up handwriting
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        cleaned = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)

        # Try different thresholding approaches
        # Method 1: Adaptive thresholding
        binary1 = cv2.adaptiveThreshold(
            cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Method 2: Otsu's thresholding
        _, binary2 = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Method 3: Simple binary threshold
        _, binary3 = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)

        # Choose the best result (most text-like)
        binaries = [binary1, binary2, binary3]
        best_binary = max(binaries, key=lambda x: ImageProcessor._count_text_pixels(x))

        # Final denoising
        denoised = cv2.fastNlMeansDenoising(best_binary, h=10)

        return Image.fromarray(denoised)

    @staticmethod
    def _count_text_pixels(binary_image):
        """Count pixels that are likely to be text (heuristic for choosing best preprocessing)"""
        # Simple heuristic: prefer images with moderate black pixel percentage
        total_pixels = binary_image.size
        black_pixels = np.count_nonzero(binary_image == 0)
        percentage = black_pixels / total_pixels
        # Ideal text images have 10-40% black pixels
        if 0.1 <= percentage <= 0.4:
            return percentage
        else:
            return 0

    @staticmethod
    def preprocess_image(image, mode="general"):
        """Enhanced preprocessing with mode selection"""
        if mode == "math":
            return ImageProcessor.enhance_image_for_math(image)
        else:
            # Original preprocessing for general text
            img_array = np.array(image)

            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            denoised = cv2.fastNlMeansDenoising(binary)

            return Image.fromarray(denoised)

    @staticmethod
    def extract_text(image, mode="general"):
        """Extract text using OCR with processing mode and multiple attempts"""
        try:
            processed = ImageProcessor.preprocess_image(image, mode)

            # Try multiple OCR configurations for better results
            configs = []
            if mode == "math":
                # Multiple configurations for math/handwritten content
                configs = [
                    '--psm 6 --oem 3',  # Uniform block of text
                    '--psm 8 --oem 3',  # Single word
                    '--psm 3 --oem 3',  # Fully automatic
                ]
            else:
                configs = ['--psm 3 --oem 3']  # Standard configuration

            best_text = ""
            for config in configs:
                try:
                    text = pytesseract.image_to_string(processed, config=config)
                    text = text.strip()
                    if len(text) > len(best_text):  # Prefer longer results
                        best_text = text
                except:
                    continue

            # Post-process the text to improve physics/math term recognition
            if mode == "math" and best_text:
                best_text = ImageProcessor._post_process_physics_text(best_text)

            return best_text
        except Exception as e:
            st.error(f"OCR Error: {str(e)}")
            return ""

    @staticmethod
    def _post_process_physics_text(text):
        """Advanced post-processing for physics text with fuzzy matching"""
        if not text or len(text.strip()) < 3:
            return text

        result = text.lower()

        # Fuzzy pattern matching for physics terms (handles OCR corruption)
        physics_patterns = {
            # Newton's patterns (very fuzzy to catch corrupted OCR)
            r'n[aeiou]*w*t[aeiou]*[nm]': 'Newton',
            r'n[aeiou]*t[aeiou]*[nm]': 'Newton',
            r'l[aeiou]*[vw]': 'law',
            r's[aeiou]*c[aeiou]*n*d': 'second',
            r'f[aeiou]*r*s*t': 'first',
            r'th[aeiou]*r*d': 'third',

            # Force and motion
            r'f[aeiou]*r*c': 'force',
            r'm[aeiou]*s': 'mass',
            r'[ae]*c[aeiou]*l[aeiou]*r*[aeiou]*t*[aeiou]*[aeiou]*n': 'acceleration',
            r'v[aeiou]*l*[aeiou]*c*[aeiou]*t*[aeiou]*': 'velocity',
            r'g[aeiou]*[r]*[aeiou]*v*[aeiou]*t*': 'gravity',
            r'm[aeiou]*m[aeiou]*n*t[aeiou]*m': 'momentum',
            r'[aeiou]*n[aeiou]*r*g': 'energy',
            r'w[aeiou]*r*k': 'work',
            r'p[aeiou]*w[aeiou]*r': 'power',
            r'[aeiou]*n[aeiou]*r*t*[aeiou]*[ae]': 'inertia',

            # Common equations (very fuzzy)
            r'f=ma': 'F=ma',
            r'f=m*a': 'F=m*a',
            r'w=f*d': 'W=F*d',
            r'p=m*v': 'p=m*v',
            r'ke=1/2*m*v': 'KE=1/2*m*v²',
            r'pe=m*g*h': 'PE=m*g*h',
        }

        # Apply fuzzy corrections
        import re
        corrected_result = result
        for pattern, replacement in physics_patterns.items():
            corrected_result = re.sub(pattern, replacement, corrected_result, flags=re.IGNORECASE)

        # Capitalize physics terms
        corrections = {
            'newton': 'Newton',
            'force': 'force',
            'mass': 'mass',
            'acceleration': 'acceleration',
            'velocity': 'velocity',
            'gravity': 'gravity',
            'momentum': 'momentum',
            'energy': 'energy',
            'work': 'work',
            'power': 'power',
            'inertia': 'inertia',
            'second': 'second',
            'first': 'first',
            'third': 'third',
        }

        for wrong, correct in corrections.items():
            corrected_result = corrected_result.replace(wrong, correct)

        # Advanced physics concept detection with fuzzy matching
        physics_concepts = []
        concept_patterns = {
            'Newton': [r'n[aeiou]*[w]*t[aeiou]*[nm]', r'n[aeiou]*t[aeiou]*[nm]'],
            'force': [r'f[aeiou]*r*c', r'f[aeiou]*r*s*t'],
            'mass': [r'm[aeiou]*s', r'm[aeiou]*s*s'],
            'acceleration': [r'[ae]*c[aeiou]*l[aeiou]*r*[aeiou]*t*[aeiou]*[aeiou]*n'],
            'velocity': [r'v[aeiou]*l*[aeiou]*c*[aeiou]*t*[aeiou]*'],
            'gravity': [r'g[aeiou]*[r]*[aeiou]*v*[aeiou]*t*'],
            'momentum': [r'm[aeiou]*m[aeiou]*n*t[aeiou]*m'],
            'energy': [r'[aeiou]*n[aeiou]*r*g', r'[aeiou]*n[aeiou]*r*j*[aeiou]'],
            'second law': [r's[aeiou]*c[aeiou]*n*d.*l[aeiou]*[vw]'],
        }

        for concept, patterns in concept_patterns.items():
            for pattern in patterns:
                if re.search(pattern, corrected_result, re.IGNORECASE):
                    physics_concepts.append(concept)
                    break

        # Remove duplicates
        physics_concepts = list(set(physics_concepts))

        if physics_concepts:
            corrected_result += f"\n\n🔍 Detected physics concepts: {', '.join(physics_concepts)}"

        return corrected_result
