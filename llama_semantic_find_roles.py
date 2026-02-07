import torch
import json
import os
import re
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

def run_role_extraction(block_text):
    system_prompt = (
        "Identify the relationship of the two people speaking in this chat. "
        "Ensure the relationship is written inside square brackets: [Role 1, Role 2]. "
        "Provide a brief explaination for your selection."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": block_text},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt",
        return_dict=True 
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150,
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id 
        )
    
    return tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

def process_chat_file(file_path):
    """Legge il file, lo divide in blocchi e avvia l'estrazione ruoli."""
    if not os.path.exists(file_path):
        print(f"Errore: File {file_path} non trovato.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split basato sul delimitatore ###
    blocks = [b.strip() for b in content.split("###") if b.strip()]

    print(f"\n--- Estrazione Ruoli: {file_path} ({len(blocks)} blocchi) ---")

    for i, block in enumerate(blocks):
        print(f"\n>>> Analizzando Blocco {i+1}/{len(blocks)}...")
        
        risposta = run_role_extraction(block)
        
        # Estrazione ruoli tramite Regex: cerca il contenuto tra parentesi quadre
        tags = re.findall(r"\[(.*?)\]", risposta)
        roles = tags[0] if tags else "Unknown"
        
        print(f"Ruoli identificati: {roles}")
        print(f"Spiegazione Llama:\n{risposta.strip()}")
        
        # Salvataggio nel log
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "file": file_path,
            "block_index": i,
            "roles": roles,
            "full_response": risposta.strip(),
            "block_text": block
        }
        
        with open("log_roles.jsonl", "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")


print("\n--- Sistema Estrazione Ruoli Attivo ---")
while True:
    user_input = input("\nInserisci il nome del file in 'chat/' (es: 01.txt), 'all' per 1-10, o 'exit': ").strip().lower()
    
    if user_input == 'exit': 
        break
    
   
    if user_input == 'all':
        files_to_process = [f"{str(i).zfill(2)}.txt" for i in range(1, 11)]
        print(f"Avvio analisi sequenziale di {len(files_to_process)} file...")
    else:
        
        files_to_process = [user_input]

    
    for file_name in files_to_process:
        full_path = os.path.join("chat", file_name)
        
        if not os.path.exists(full_path):
            print(f"Salto: {file_name} non trovato in 'chat/'.")
            continue

        try:
            process_chat_file(full_path)
        except Exception as e:
            print(f"Errore durante l'elaborazione di {file_name}: {e}")

print("Sessione terminata.")