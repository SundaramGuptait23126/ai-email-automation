"""
Production Fine-Tuning Script for SOTA SLMs (Small Language Models)
Supports:
- Qwen/Qwen2.5-1.5B-Instruct
- meta-llama/Llama-3.2-1B-Instruct
- HuggingFaceTB/SmolLM2-1.7B-Instruct

Uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA.
Can be run on Google Colab (Free T4 GPU), RunPod, or local NVIDIA GPU.
"""

import os
import json
from pathlib import Path

# Instructions for 1-Click Free GPU execution
COLAB_INSTRUCTIONS = """
# ==============================================================================
# HOW TO RUN ON GOOGLE COLAB / RUNPOD (FREE GPU):
# 1. Open Google Colab (https://colab.research.google.com)
# 2. Select Runtime > Change Runtime Type > T4 GPU
# 3. Paste and run:
#    !pip install -q unsloth transformers datasets peft accelerate trl bitsandbytes
#    !python finetune_sota_slm.py
# ==============================================================================
"""

def prepare_hf_dataset(jsonl_path: str):
    """Converts email_slm_dataset.jsonl into chat template format for SOTA models."""
    from datasets import Dataset

    chat_data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            chat_data.append({
                "messages": [
                    {"role": "system", "content": "You are an autonomous AI Email Triage and Automation Agent. Analyze incoming emails, determine triage metadata (category, priority, sentiment, actions), and generate a polite, professional draft reply in structured JSON."},
                    {"role": "user", "content": f"{item['instruction']}\n\nIncoming Email:\n{item['input']}"},
                    {"role": "assistant", "content": item['output']}
                ]
            })

    return Dataset.from_list(chat_data)

def run_peft_training(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: str = "./sota_email_slm_lora",
    epochs: int = 3,
    batch_size: int = 2
):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        print("To run SOTA model fine-tuning, install: pip install transformers peft datasets accelerate")
        return

    print(f"Loading Base SOTA SLM: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_path = Path(__file__).parent.parent / "data" / "email_slm_dataset.jsonl"
    dataset = prepare_hf_dataset(str(data_path))

    def format_prompts(batch):
        texts = [tokenizer.apply_chat_template(conv, tokenize=False) for conv in batch["messages"]]
        return tokenizer(texts, padding="max_length", truncation=True, max_length=512)

    tokenized_dataset = dataset.map(format_prompts, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False # Set to True on CUDA GPUs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("Starting LoRA fine-tuning...")
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Successfully fine-tuned and saved LoRA adapter to {output_dir}")

if __name__ == "__main__":
    print(COLAB_INSTRUCTIONS)
