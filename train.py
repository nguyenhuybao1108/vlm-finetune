import torch
from transformers import TrainingArguments, Trainer
from dataset import load_pathvqa
from model import load_model_and_processor, apply_lora
from collate import make_collate_fn


def main():
    # ── 1. Load data ───────────────────────────────────────
    print("Loading dataset...")
    ds = load_pathvqa()
    print(f"Train: {len(ds['train'])} | Val: {len(ds['validation'])}")

    # ── 2. Load model ──────────────────────────────────────
    print("Loading model...")
    model, processor = load_model_and_processor()
    model = apply_lora(model)

    # ── 3. Training args ───────────────────────────────────
    training_args = TrainingArguments(
        output_dir="./checkpoints/pathvqa-qwen2vl",
        num_train_epochs=1,
        per_device_train_batch_size=2,      # Increase batch size for RTX 5090 (32GB)
        gradient_accumulation_steps=4,      # effective batch = 8 (same as before)
        learning_rate=2e-4,
        warmup_steps=500,
        lr_scheduler_type="cosine",
        fp16=True,                          # Use fp16
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        eval_strategy="steps",
        save_total_limit=3,
        remove_unused_columns=False,        # important for VLM
        dataloader_num_workers=8,           # More workers for faster loading
        pin_memory=False,                   # Disable pin_memory warning
        report_to="none"                    # change to "wandb" for tracking
    )

    # ── 4. Trainer ─────────────────────────────────────────
    collate_fn = make_collate_fn(processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collate_fn,
    )

    # ── 5. Train ───────────────────────────────────────────
    print("Starting training...")
    trainer.train()

    # ── 6. Save ────────────────────────────────────────────
    model.save_pretrained("./adapters/pathvqa-qwen2vl-lora")
    processor.save_pretrained("./adapters/pathvqa-qwen2vl-lora")
    print("Done! Saved to ./adapters/pathvqa-qwen2vl-lora")


if __name__ == "__main__":
    main()
