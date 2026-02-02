from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import argparse
import yaml

parser = argparse.ArgumentParser(description="Prepare data for training.")
parser.add_argument(
        "--config_file", type=str, help="Path to the config file"
    )
args = parser.parse_args()
with open(args.config_file, "r") as file:
    config = yaml.safe_load(file)

def modify_conversation_keys(row):
    """Modify conversation entry keys to standardize role names.

    Converts conversation role names from custom format (e.g., 'human', 'gpt')
    to standard format ('user', 'assistant').

    Args:
        row (dict): A dictionary containing a 'conversations' list with entries
                   to be modified.

    Returns:
        dict: Modified dictionary with standardized conversation roles.
    """
    modified_conversation = []
    for entry in row["conversations"]:
        modified_entry = {
            "role": "user"
            if entry["from"] == "human"
            else "assistant"
            if entry["from"] == "gpt"
            else entry["from"],
            "content": entry["value"],
        }
        modified_conversation.append(modified_entry)

    return {"conversations": modified_conversation}

dataset = load_dataset(config["dataset"]["name"])
print(dataset)
dataset=dataset["train"]
print(dataset[0]["conversations"])



tokenizer = AutoTokenizer.from_pretrained(
    config["tokenizer"]["path"],
    use_fast=True
)
print(tokenizer.__class__.__name__)
tokenizer.pad_token = tokenizer.eos_token
assert tokenizer.chat_template is not None, "Tokenizer has no chat template"



print(dataset[0]["conversations"])
dataset = dataset.map(
    modify_conversation_keys,
    batched=False
)

print(dataset[0]["conversations"])
def tokenize_conversation(example):
    """
    Tokenizes a conversation example using the appropriate chat template.
    
    Applies model-specific chat templates (Llama or Qwen3) and tokenizes the
    resulting text. For Qwen3 models, optionally handles thinking/reasoning tokens.
    
    Args:
        example (dict): Dictionary containing a 'conversations' list with
                       standardized role/content format.
    
    Returns:
        dict: Dictionary containing:
            - input_ids: Tokenized input sequence
            - labels: Same as input_ids (for language modeling)
            - length: Length of the tokenized sequence
            - attention_mask: Mask of ones (all tokens are attended to)
    
    Raises:
        ValueError: If an unsupported model_type is specified in the config.
    """
    # apply_chat_template returns the formatted string
    if config["tokenizer"]["model_type"] == "llama":
        tokenized_text = tokenizer.apply_chat_template(example["conversations"])

    elif config["tokenizer"]["model_type"] == "qwen3":
  
        #text = tokenizer.apply_chat_template(example["conversations"],add_generation_prompt=False)
        tokenized_text = tokenizer.apply_chat_template(example["conversations"], add_generation_prompt=False)

    else:
        raise ValueError(f"Model type {config['tokenizer']['model_type']} not supported")
    return {"input_ids": tokenized_text["input_ids"], "labels": tokenized_text["input_ids"], "length": len(tokenized_text["input_ids"]), "attention_mask": tokenized_text["attention_mask"]}
columns_to_keep = ["input_ids", "labels", "length", "attention_mask"]

tokenized_dataset = dataset.map(tokenize_conversation,    batched=False,
    remove_columns=["conversations"])
columns_to_remove = [col for col in tokenized_dataset.column_names if col not in columns_to_keep]
tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
print(tokenized_dataset[0])
print(tokenized_dataset)

tokenized_dataset.save_to_disk(config["dataset"]["output_path"])
ids=tokenized_dataset[0]["input_ids"]
print(ids)
print(tokenized_dataset[0]["labels"])
print(len(ids))
print(tokenizer.decode(ids))

