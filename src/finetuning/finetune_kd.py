import argparse
import os
import yaml

import torch
import torch.nn.functional as F
from transformers import TrainingArguments
from datasets import load_from_disk
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers.data.data_collator import DataCollatorForLanguageModeling

# Optimize CPU inference for teacher model (multi-concurrency)
# 192 logical cores / 8 FSDP processes = 24 cores per process
# intra_op = 20: threads for parallelizing matrix ops (leave 4 cores for OS/inter-op)
# inter_op = 2:  threads for independent op-level parallelism (attention + FFN overlap)
torch.set_num_threads(20)
torch.set_num_interop_threads(2)
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
        kl_chunk_size=128,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Freeze teacher model (bf16 on CPU, no torch.compile — Inductor C++ codegen
        # has a known bug that generates invalid variable references on CPU+bf16+reduce-overhead).
        # bf16 is preferred over 4-bit on CPU: no dequantization overhead, better throughput.
        # CPU parallelism is handled globally via torch.set_num_threads(20) + set_num_interop_threads(2).
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model = teacher_model.eval()
        self.teacher = teacher_model

        self.kd_temperature = kd_temperature
        self.kd_alpha = kd_alpha
        self.kl_chunk_size = kl_chunk_size

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Computes the loss for knowledge distillation training.
        
        During training: computes chunked KL divergence between student and
        teacher logits with temperature scaling, along the seq_len dimension
        to reduce peak GPU memory usage.
        
        During evaluation: returns the student model's own cross-entropy loss
        to avoid expensive teacher inference (reduces eval time from hours
        to minutes).
        
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

        # Eval mode: return student's own CE loss (skip teacher inference)
        if not model.training:
            loss = student_outputs.loss
            return (loss, student_outputs) if return_outputs else loss

        # Training mode: full KD with chunked KL computation
        student_logits = student_outputs.logits

        # Teacher forward on CPU; use no_grad to skip autograd graph.
        # Remove labels from teacher inputs — KD only needs logits, not teacher's CE loss
        # (computing CE loss allocates ~0.8 GiB for vocab-softmax, wasted memory).
        # Move all input tensors to CPU since teacher model is on CPU.
        with torch.no_grad():
            teacher_inputs = {
                k: v.to("cpu") if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
                if k != "labels"
            }
            teacher_outputs = self.teacher(**teacher_inputs)
            teacher_logits = teacher_outputs.logits
        T = self.kd_temperature
        chunk_size = self.kl_chunk_size
        seq_len = student_logits.shape[1]

        # Chunked KL computation along seq_len dimension to reduce peak memory
        total_kl = torch.tensor(0.0, device=student_logits.device)
        total_mask = torch.tensor(0.0, device=student_logits.device)

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)

            s_chunk = student_logits[:, chunk_start:chunk_end]
            t_chunk = teacher_logits[:, chunk_start:chunk_end].to(student_logits.device)


            log_probs = F.log_softmax(s_chunk / T, dim=-1)
            probs = F.softmax(t_chunk / T, dim=-1)

            kl_chunk = F.kl_div(
                log_probs, probs, reduction="none", log_target=False
            ).sum(dim=-1)

            if "attention_mask" in inputs:
                mask_chunk = inputs["attention_mask"][:, chunk_start:chunk_end].float()
                kl_chunk = kl_chunk * mask_chunk
                total_kl = total_kl + kl_chunk.sum()
                total_mask = total_mask + mask_chunk.sum()
            else:
                total_kl = total_kl + kl_chunk.sum()
                total_mask = total_mask + kl_chunk.numel()

        kl_loss = total_kl / total_mask

        kl_loss = kl_loss * (T ** 2)
        loss = self.kd_alpha * kl_loss

        return (loss, student_outputs) if return_outputs else loss

parser = argparse.ArgumentParser(description="Finetune a model.")
parser.add_argument("-config_file", type=str, help="Path to the config file")
args = parser.parse_args()
with open(args.config_file, "r") as file:
    config = yaml.safe_load(file)
task_type="CAUSAL_LM"
model=AutoModelForCausalLM.from_pretrained(config["model"]["path"], torch_dtype=config["model"]["dtype"])
# Load teacher model with bf16 precision on CPU (no quantization)
# bf16 is faster than 4-bit on CPU because there's no dequantization overhead
# With 1.5TB system memory, 8 teachers × 16.4 GiB = 131.2 GiB is easily manageable
# CPU parallelism is configured globally via torch.set_num_threads(20) + set_num_interop_threads(2)
teacher_model = AutoModelForCausalLM.from_pretrained(
    config["teacher_model"]["path"],
    torch_dtype=torch.bfloat16,
    trust_remote_code=(model.config.model_type != "llama"),
    device_map="cpu",  # Keep teacher on CPU to avoid GPU OOM
)
print(f"Teacher model loaded in bf16 on CPU, dtype={next(teacher_model.parameters()).dtype}")
tokenizer=AutoTokenizer.from_pretrained(config["model"]["path"])

dataset = load_from_disk(config["dataset"]["path"])
dataset = dataset.remove_columns(["labels"])

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
    max_seq_length=config["training"]["max_length"],
    gradient_checkpointing=True,
    dataloader_num_workers=8,
    gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
    )
trainer = CustomSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    teacher_model=teacher_model,
)

teacher_model.eval()  # already frozen in CustomSFTTrainer.__init__
print(f"Teacher ready on device: {next(teacher_model.parameters()).device}")

# Resume from checkpoint: auto-detect or use specified path
resume_from = config["training"].get("resume_from_checkpoint", None)
if resume_from is None:
    output_dir = config["training"]["output_dir"]
    if os.path.isdir(output_dir):
        checkpoints = [
            d for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            resume_from = os.path.join(output_dir, latest)

if resume_from:
    print(f"Resuming training from checkpoint: {resume_from}")
    trainer.train(resume_from_checkpoint=resume_from)
else:
    print("Starting training from scratch")
    trainer.train()
