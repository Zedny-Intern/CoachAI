"""
Model loading and inference handling for the Multimodal Learning Coach
"""
import torch
import streamlit as st
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info


class ModelHandler:
    """Handles model loading and inference"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.processor = None
        self.device = None

    def _check_memory_requirements(self):
        """Check if system has enough memory for the model"""
        if not torch.cuda.is_available():
            return True  # CPU has virtually unlimited memory

        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

        # Qwen3-VL-2B rough memory requirements (varies with quantization)
        if hasattr(self.config, 'quantization') and self.config.quantization == '4bit':
            required_memory = 6  # ~6GB for 4-bit quantized
        elif hasattr(self.config, 'quantization') and self.config.quantization == '8bit':
            required_memory = 10  # ~10GB for 8-bit quantized
        else:
            required_memory = 16  # ~16GB for full precision

        if gpu_memory_gb < required_memory:
            st.warning(f"⚠️ GPU has {gpu_memory_gb:.1f}GB VRAM, model needs ~{required_memory}GB")
            st.info("💡 Consider using quantization or CPU offloading")
            return False
        return True

    def _get_quantization_config(self):
        """Get quantization configuration based on settings"""
        if not hasattr(self.config, 'quantization') or self.config.quantization == 'none':
            return None

        quantization = self.config.quantization.lower()

        if quantization == '4bit':
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == '8bit':
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
        else:
            st.warning(f"⚠️ Unknown quantization '{quantization}', using full precision")
            return None

    def _get_device(self):
        """Determine which device to use based on config and availability"""
        config_device = self.config.device.lower()

        if config_device == 'cpu':
            return 'cpu'
        elif config_device == 'cuda':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                st.warning("⚠️ CUDA requested but not available, falling back to CPU")
                return 'cpu'
        elif config_device == 'auto':
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                st.info(f"✓ CUDA available: {gpu_name} ({gpu_memory:.1f}GB)")
                return 'cuda'
            else:
                st.info("ℹ️ CUDA not available, using CPU")
                return 'cpu'
        else:
            st.warning(f"⚠️ Unknown device '{config_device}', defaulting to auto detection")
            return self._get_device() if hasattr(self.config, 'device') and self.config.device.lower() == 'auto' else 'cpu'

    def load_model(self):
        """Load Qwen3-VL model with memory optimizations"""
        try:
            st.info(f"Loading model from: {self.config.MODEL_PATH}")

            # Check memory requirements first
            if not self._check_memory_requirements():
                st.warning("💡 Memory requirements not met, attempting with optimizations...")

            self.processor = AutoProcessor.from_pretrained(
                self.config.MODEL_PATH,
                trust_remote_code=True
            )

            # Determine device based on config and availability
            self.device = self._get_device()

            # Get quantization config
            quantization_config = self._get_quantization_config()
            if quantization_config:
                st.info(f"🔧 Using {self.config.quantization} quantization")

            # Set up device mapping with CPU offloading as fallback
            if self.device == "cuda" and quantization_config is None:
                # Check if we need CPU offloading for non-quantized models
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory_gb < 16:  # RTX 4060 Ti has ~16GB
                    st.info("🔄 Using CPU offloading for layers that don't fit in GPU memory")
                    device_map = "auto"  # This enables automatic CPU offloading
                else:
                    device_map = "auto"
            else:
                device_map = "auto" if self.device == "cuda" else {"": "cpu"}

            # Load the model with optimizations
            try:
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.config.MODEL_PATH,
                    quantization_config=quantization_config,
                    device_map=device_map,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    st.warning("🚨 GPU out of memory! Falling back to CPU...")
                    self.device = "cpu"
                    device_map = {"": "cpu"}
                    quantization_config = None  # Disable quantization for CPU

                    self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                        self.config.MODEL_PATH,
                        torch_dtype=torch.float32,
                        device_map=device_map,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                else:
                    raise e

            device_display = f"{self.device.upper()}"
            if quantization_config:
                device_display += f" ({self.config.quantization})"

            st.success(f"✓ Model loaded on {device_display}")
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
