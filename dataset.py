from datasets import load_dataset


def load_pathvqa():
    """Load PathVQA dataset from Hugging Face (raw, no formatting)."""
    ds = load_dataset("flaviagiammarino/path-vqa")
    return ds
