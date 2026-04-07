from qwen_vl_utils import process_vision_info
import torch


def make_collate_fn(processor):
    """Create a collate function for batching with on-the-fly formatting."""
    def collate_fn(batch):
        # Format samples on-the-fly (raw PathVQA → Qwen2-VL format)
        messages_list = []
        for item in batch:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image"]},
                        {"type": "text", "text": item["question"]}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": item["answer"]}]
                }
            ]
            messages_list.append(messages)

        # Apply chat template
        texts = [
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            for messages in messages_list
        ]

        # Get image inputs from messages
        image_inputs_list = []
        for messages in messages_list:
            image_inputs, _ = process_vision_info(messages)
            image_inputs_list.append(image_inputs)

        # Tokenize
        inputs = processor(
            text=texts,
            images=image_inputs_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )

        # Labels: mask padding and user portion, only compute loss on assistant portion
        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        inputs["labels"] = labels
        return inputs

    return collate_fn
