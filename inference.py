import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

BASE_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
ADAPTER_DIR = "./adapters/pathvqa-qwen2vl-lora"


def load_finetuned():
    """Load fine-tuned model with LoRA adapters."""
    processor = AutoProcessor.from_pretrained(ADAPTER_DIR)
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    return model, processor


def ask(model, processor, image: Image.Image, question: str) -> str:
    """Ask a question about an image."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True
        )

    response = processor.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response


# Test
if __name__ == "__main__":
    from datasets import load_dataset

    model, processor = load_finetuned()
    ds = load_dataset("flaviagiammarino/path-vqa", split="test[:5]")

    for sample in ds:
        image = sample["image"]
        question = sample["question"]
        gt = sample["answer"]
        pred = ask(model, processor, image, question)

        print(f"Q: {question}")
        print(f"GT:   {gt}")
        print(f"Pred: {pred}")
        print("─" * 40)
