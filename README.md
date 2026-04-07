# Qwen2-VL-7B Fine-tuning on PathVQA

Fine-tune Qwen2-VL-7B on the PathVQA dataset using QLoRA for efficient adaptation.

## Setup

### Install dependencies
```bash
uv add torch torchvision transformers peft datasets accelerate bitsandbytes trl pillow qwen-vl-utils
```

Or with pip:
```bash
pip install -r requirements.txt
```

### Project structure
```
pathvqa-qwen2vl-finetune/
├── dataset.py          # Dataset loading and formatting
├── model.py            # Model loading and LoRA setup
├── collate.py          # Batch collation function
├── train.py            # Training script
├── inference.py        # Inference and evaluation
├── checkpoints/        # Saved checkpoints during training
├── adapters/           # Final LoRA adapters
└── pyproject.toml      # Project metadata
```

## Quick Start

### 1. Prepare dataset
The PathVQA dataset is loaded automatically from Hugging Face.

### 2. Train
```bash
python train.py
```

Training configuration:
- **Model**: Qwen2-VL-7B-Instruct
- **Method**: QLoRA (4-bit quantization + LoRA)
- **Trainable parameters**: ~40M (0.5% of model)
- **Batch size**: 1 (with gradient accumulation × 8 = effective batch 8)
- **Learning rate**: 2e-4
- **Epochs**: 1
- **Hardware**: Optimized for 24GB+ VRAM GPUs

### 3. Inference
After training, run inference on test samples:
```bash
python inference.py
```

Or use programmatically:
```python
from inference import load_finetuned, ask
from PIL import Image

model, processor = load_finetuned()
image = Image.open("path_image.jpg")
answer = ask(model, processor, image, "What is visible in this image?")
print(answer)
```

## Training Details

### Configuration
- **Quantization**: 4-bit (QLoRA)
- **Dtype**: bfloat16 (better stability than fp16 for 7B models)
- **Gradient checkpointing**: Enabled
- **LoRA rank**: 16
- **LoRA alpha**: 32
- **Dropout**: 0.05

### Optimization
- **Learning rate scheduler**: Cosine with 5% warmup
- **Optimizer**: AdamW (default in Trainer)
- **Evaluation**: Every 200 steps
- **Checkpointing**: Save top 3 models

## Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size` (already at 1)
- Reduce `gradient_accumulation_steps`
- Enable `gradient_checkpointing` in model.py
- Use a GPU with more VRAM (40GB+ recommended)

### Slow data loading
- Increase `dataloader_num_workers` in train.py
- Pre-cache the dataset with `.map(..., cache_file_name="...")`

### Dataset not loading
Ensure internet connection for downloading from Hugging Face.

## References
- [Qwen2-VL Documentation](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [PathVQA Dataset](https://huggingface.co/datasets/flaviagiammarino/path-vqa)
- [PEFT LoRA](https://huggingface.co/docs/peft)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
