#!/usr/bin/env python3
"""
Test script to verify the model loads correctly on CPU
Run with: python test_model.py
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

def test_model_loading():
    model_path = "./src/Qwen3-VL-2B-Instruct"

    print("Testing model loading...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    print()

    try:
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        print("Loading model on CPU...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )

        print("✓ Model loaded successfully!")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")

        # Test a simple generation
        print("\nTesting generation...")
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello, what is 2+2?"}]}]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )

        response = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        print(f"Test response: {response}")
        print("✓ All tests passed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True

if __name__ == "__main__":
    test_model_loading()
