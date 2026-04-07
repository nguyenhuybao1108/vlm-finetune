import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"


def load_model_and_processor():
    """Load base model and processor with optimizations for 7B."""
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,   # bf16 better than fp16 for 7B
        device_map="auto",
        load_in_4bit=True              # QLoRA — saves VRAM
    )

    return model, processor


def apply_lora(model):
    """Apply LoRA adapters to model."""
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # → trainable params: ~40M / 7B ≈ 0.5%
    return model
