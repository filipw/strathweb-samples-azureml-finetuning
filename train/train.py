import argparse
import os
import torch
from datasets import load_dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Path to the training data.")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to the evaluation data.")
    parser.add_argument("--model_output", type=str, required=True, help="Path to save the trained model.")
    parser.add_argument("--save_merged_model", type=bool, default=False, help="Merge LoRA adapters and save the full model.")
    args = parser.parse_args()

    # --- 1. Load Datasets ---
    print("Loading datasets...")
    train_dataset = load_dataset('json', data_files=args.train_data, split='train')
    eval_dataset = load_dataset('json', data_files=args.eval_data, split='train')

    # --- 2. Pre-format the dataset with a System Prompt ---
    def format_dataset(example):
        system_prompt = "You are a helpful assistant that extracts PII from text and returns it as a JSON object."
        
        example['text'] = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{example['prompt']}<|end|>\n<|assistant|>\n{example['completion']}<|end|>"
        return example

    print("Formatting datasets...")
    train_dataset = train_dataset.map(format_dataset)
    eval_dataset = eval_dataset.map(format_dataset)

    # --- 3. Model and Tokenizer Configuration ---
    model_id = "microsoft/Phi-4-mini-instruct"
    print(f"Loading base model: {model_id}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model.tokenizer = tokenizer

    # --- 4. PEFT (LoRA) Configuration ---
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- 5. Training Configuration (SFTConfig) ---
    training_args = SFTConfig(
        output_dir=os.path.join(args.model_output, "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        optim="adamw",
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        dataset_text_field="text",
        max_length=2048,
        logging_steps=50,
        bf16=True,
        tf32=True,
        warmup_ratio=0.03,
        disable_tqdm=False,
        report_to="none",
        evaluation_strategy="epoch",
    )

    # --- 6. Initialize Trainer ---
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    # --- 7. Start Training ---
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # --- 8. Evaluation ---
    print("Running evaluation...")
    tokenizer.padding_side = "left"
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(eval_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    print("Evaluation finished. Metrics saved.")

    # --- 9. Save Model ---
    os.makedirs(args.model_output, exist_ok=True)

    if args.save_merged_model:
        print("Merging adapters and saving the full model...")
        adapter_tmp_dir = os.path.join(args.model_output, "tmp_adapters")
        trainer.model.save_pretrained(adapter_tmp_dir)
        
        model = AutoPeftModelForCausalLM.from_pretrained(adapter_tmp_dir, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
        merged_model = model.merge_and_unload()
        
        merged_model.save_pretrained(args.model_output, safe_serialization=True)
        print(f"Saved merged model to: {args.model_output}")
    else:
        print(f"Saving PEFT model adapters to: {args.model_output}")
        trainer.model.save_pretrained(args.model_output)

    tokenizer.save_pretrained(args.model_output)
    print("Model and tokenizer saved successfully.")


if __name__ == "__main__":
    main()