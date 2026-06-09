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
    for entry in row["conversation"]:
        for key, value in entry.items():
            if key == 'assistant':
                role = 'assistant'
                content = value
            elif key == 'human':
                role = 'user'
                content = value
            else:
                print(f"not support key: {key}")
                continue
            modified_entry = {
                "role": role,
                "content": content,
            }
            modified_conversation.append(modified_entry)

    return {"conversations": modified_conversation}




data_path = config["dataset"]["name"]
print(f"Loading dataset from {data_path}")
dataset = load_dataset("json", data_files=f"{data_path}/*.jsonl")
dataset=dataset["train"]


tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer"]["path"],
            use_fast=True
        )
tokenizer.pad_token = tokenizer.eos_token





dataset = dataset.map(
    modify_conversation_keys,
    batched=False
)

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

        assert tokenizer.chat_template is not None, "Tokenizer has no chat template"

        text = tokenizer.apply_chat_template(example["conversations"], tokenize=False)
        tokenized_text = tokenizer(text)

    elif config["tokenizer"]["model_type"] == "qwen3":

        assert tokenizer.chat_template is not None, "Tokenizer has no chat template"
  
        #text = tokenizer.apply_chat_template(example["conversations"],add_generation_prompt=False)
        text = tokenizer.apply_chat_template(example["conversations"], add_generation_prompt=False, tokenize=False)
        tokenized_text = tokenizer(text)
    elif config["tokenizer"]["model_type"] == "deepseek4":
        from encoding_dsv4 import encode_messages, parse_message_from_completion_text
        text = encode_messages(example["conversations"], thinking_mode="chat")
        tokenized_text = tokenizer(text)

    else:
        raise ValueError(f"Model type {config['tokenizer']['model_type']} not supported")
    return {"input_ids": tokenized_text["input_ids"], "labels": tokenized_text["input_ids"], "length": len(tokenized_text["input_ids"]), "attention_mask": tokenized_text["attention_mask"]}
columns_to_keep = ["input_ids", "labels", "length", "attention_mask"]

tokenized_dataset = dataset.map(tokenize_conversation,    batched=False, num_proc=16, 
    remove_columns=["conversations"])
columns_to_remove = [col for col in tokenized_dataset.column_names if col not in columns_to_keep]
tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)

tokenized_dataset.save_to_disk(config["dataset"]["output_path"])
ids=tokenized_dataset[0]["input_ids"]
print(ids)
print(tokenized_dataset[0]["labels"])
print(len(ids))
print(tokenizer.decode(ids[0]))

