"""
Model loading and inference handling for the Multimodal Learning Coach
"""
import torch
import streamlit as st
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class ModelHandler:
    """Handles model loading and inference"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.processor = None
        self.device = None

    def load_model(self):
        """Load Qwen3-VL model"""
        try:
            st.info(f"Loading model from: {self.config.MODEL_PATH}")

            self.processor = AutoProcessor.from_pretrained(
                self.config.MODEL_PATH,
                trust_remote_code=True
            )

            # Force CPU usage only (GPU is incompatible with CUDA requirements)
            self.device = "cpu"

            # Use lower precision for CPU to reduce memory usage
            dtype = torch.float16  # Use half precision for memory efficiency

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.config.MODEL_PATH,
                dtype=dtype,
                device_map={"": "cpu"},  # Force all on CPU
                trust_remote_code=True,
                low_cpu_mem_usage=True  # Enable memory optimization
            )

            st.success(f"✓ Model loaded on {self.device}")
            return True

        except Exception as e:
            st.error(f"Model loading error: {str(e)}")
            st.info(f"Please ensure model is at: {self.config.MODEL_PATH}")
            return False

    def generate(self, messages, max_new_tokens=None, temperature=None):
        """Generate response"""
        if self.model is None:
            return "Error: Model not loaded"

        try:
            max_new_tokens = max_new_tokens or self.config.MAX_TOKENS
            temperature = temperature or self.config.TEMPERATURE

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=self.config.DO_SAMPLE if temperature > 0 else False,
                    top_p=self.config.TOP_P,
                    repetition_penalty=self.config.REPETITION_PENALTY,
                    length_penalty=self.config.LENGTH_PENALTY,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, outputs)
            ]

            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            return response

        except Exception as e:
            return f"Generation error: {str(e)}"
