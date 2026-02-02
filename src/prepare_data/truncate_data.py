from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import concatenate_datasets, load_dataset, load_from_disk
dataset=load_from_disk("./datasets/OpenHermes-2.5/preprocessed-Qwen3-14B_nothink/")
print(dataset)
max_len = 1024
def truncate_batch(batch):
    batch["input_ids"] = [x[:max_len] for x in batch["input_ids"]]
    batch["attention_mask"] = [x[:max_len] for x in batch["attention_mask"]]
    if "labels" in batch:
        batch["labels"] = [x[:max_len] for x in batch["labels"]]
    if "length" in batch:
        batch["length"] = [min(l, max_len) for l in batch["length"]]
    return batch

dataset = dataset.map(
    truncate_batch,
    batched=True,
    batch_size=1000,
    desc=f"Truncating to {max_len}"
)
dataset = dataset.remove_columns(["labels"])
print(dataset)
dataset.save_to_disk("datasets/OpenHermes-2.5/preprocessed-Qwen3-14B_nothink_truncated_to_1024")