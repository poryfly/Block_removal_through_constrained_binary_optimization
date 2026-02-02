import argparse
import yaml

import torch
import torch.nn.functional as F
from transformers import TrainingArguments
from transformers import BitsAndBytesConfig
from datasets import load_from_disk
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers.data.data_collator import DataCollatorForLanguageModeling
from accelerate import Accelerator
import torch

accelerator = Accelerator()
class CustomSFTTrainer(SFTTrainer):
    """
    Custom Supervised Fine-Tuning Trainer with Knowledge Distillation.
    
    Extends SFTTrainer to support knowledge distillation from a teacher model
    to a student model. Uses KL divergence loss between teacher and student
    logits with temperature scaling.
    
    Args:
        teacher_model: The teacher model used for knowledge distillation.
        kd_temperature (float): Temperature for softening logits in distillation. Defaults to 1.0.
        kd_alpha (float): Weight for the knowledge distillation loss. Defaults to 1.0.
        *args: Additional positional arguments passed to SFTTrainer.
        **kwargs: Additional keyword arguments passed to SFTTrainer.
    """
    def __init__(
        self,
        teacher_model,
        kd_temperature=1.0,
        kd_alpha=1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

 
        self.teacher = self.accelerator.prepare(teacher_model.eval())
 
        self.teacher.requires_grad_(False)

        self.kd_temperature = kd_temperature
        self.kd_alpha = kd_alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Computes the knowledge distillation loss between student and teacher models.
        
        The loss is computed as KL divergence between student and teacher logits,
        scaled by temperature and weighted by kd_alpha. Padding tokens are masked
        out of the loss computation.
        
        Args:
            model: The student model being trained.
            inputs: Dictionary of input tensors (input_ids, attention_mask, etc.).
            return_outputs (bool): If True, returns both loss and model outputs.
            **kwargs: Additional keyword arguments.
        
        Returns:
            If return_outputs is False: The computed loss tensor.
            If return_outputs is True: Tuple of (loss, student_outputs).
        """
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_logits = teacher_outputs.logits

        T = self.kd_temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        kl = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
            log_target=False,
        ).sum(dim=-1)

        if "attention_mask" in inputs:
            mask = inputs["attention_mask"].float()
            kl = kl * mask
            kl_loss = kl.sum() / mask.sum()
        else:
            kl_loss = kl.mean()

        kl_loss = kl_loss * (T ** 2)
        loss = self.kd_alpha * kl_loss

        return (loss, student_outputs) if return_outputs else loss

parser = argparse.ArgumentParser(description="Finetune a model.")
parser.add_argument("-config_file", type=str, help="Path to the config file")
args = parser.parse_args()
with open(args.config_file, "r") as file:
    config = yaml.safe_load(file)
task_type="CAUSAL_LM"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)
model=AutoModelForCausalLM.from_pretrained(config["model"]["path"], torch_dtype=config["model"]["dtype"])
if model.config.model_type == "llama":
    print("Using llama teacher model")
    teacher_model=AutoModelForCausalLM.from_pretrained(config["teacher_model"]["path"], torch_dtype=config["teacher_model"]["dtype"], quantization_config=bnb_config)
else:
    teacher_model=AutoModelForCausalLM.from_pretrained(config["teacher_model"]["path"], torch_dtype=config["teacher_model"]["dtype"], quantization_config=bnb_config)
tokenizer=AutoTokenizer.from_pretrained(config["model"]["path"])

dataset = load_from_disk(config["dataset"]["path"])

split_ds = dataset.train_test_split(
    test_size=config["dataset"]["test_size"],
    seed=config["dataset"]["seed"],
    shuffle=config["dataset"]["shuffle"]
)
dataset = split_ds["train"]
eval_dataset = split_ds["test"]
print(dataset)
print(eval_dataset)
tokenizer.pad_token = tokenizer.eos_token
training_args = SFTConfig(
    output_dir=config["training"]["output_dir"],
    learning_rate=config["training"]["learning_rate"],
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
    num_train_epochs=config["training"]["num_train_epochs"],
    weight_decay=config["training"]["weight_decay"],
    eval_strategy="steps",
    lr_scheduler_type= config["training"]["lr_scheduler_type"],
    save_strategy="steps",
    logging_strategy="steps",    
    warmup_steps=config["training"]["warmup_steps"],
    max_grad_norm=config["training"]["max_grad_norm"],
    logging_steps=config["training"]["logging_steps"],              # adjust as needed
    save_steps=config["training"]["save_steps"],
    eval_steps=config["training"]["eval_steps"],
    packing=config["training"]["packing"],
    max_length=config["training"]["max_length"],
    dataloader_num_workers=4,
    gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
    )
trainer = CustomSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    teacher_model=teacher_model,
)

trainer.train()
