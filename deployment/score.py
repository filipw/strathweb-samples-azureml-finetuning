import os
import json
import pandas as pd
from transformers import AutoTokenizer, Phi3ForCausalLM
import torch

def init():
    """
    This function is called once when the container is initialized.
    """
    global model, tokenizer, system_prompt
    
    # Define the system prompt once during initialization
    system_prompt = "You are a helpful assistant that extracts PII from text and returns it as a JSON object."
    
    base_path = os.getenv("AZUREML_MODEL_DIR")
    model_path = os.path.join(base_path, "model_output")
    
    print(f"Loading model from: {model_path}")
    
    model = Phi3ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 
    print("Model and tokenizer loaded successfully.")


def run(mini_batch):
    """
    This function processes each file and writes the clean, raw model output to a JSONL file.
    """
    output_dir = os.environ.get("AZUREML_BI_OUTPUT_PATH")
    output_file_path = os.path.join(output_dir, "predictions.jsonl")
    
    processed_files = []

    with open(output_file_path, 'w') as out_f:
        for file_path in mini_batch:
            print(f"Processing file: {file_path}")
            with open(file_path, 'r') as in_f:
                for line in in_f:
                    data = json.loads(line)
                    prompt_text = data["prompt"]

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_text}
                    ]
                    
                    inputs = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_tensors="pt"
                    ).to(model.device)

                    input_token_length = inputs.shape[1]
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs,
                            max_new_tokens=512,
                            do_sample=False,
                            eos_token_id=tokenizer.eos_token_id
                        )
                    
                    generated_tokens = outputs[0, input_token_length:]
                    assistant_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

                    # Write the clean prompt and completion to the output file
                    result = {"prompt": prompt_text, "completion": assistant_response}
                    out_f.write(json.dumps(result) + '\n')
            
            processed_files.append({"file_processed": file_path, "status": "success"})

    return pd.DataFrame(processed_files)

init()