import torch
import json
import os
import re
import time  # Importato per il calcolo del tempo
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

def run_pii_extraction(block_text, block_idx):
    """Carica le istruzioni e seleziona la riga corretta ogni 100 blocchi."""
    
    with open("system_instructions.txt", "r", encoding="utf-8") as f:
        # legge le righe eliminando quelle vuote
        all_instructions = [line.strip() for line in f.readlines() if line.strip()]
    
    # calcola quale istruzione usare: 
    # blocchi 0-99 -> indice 0, 100-199 -> indice 1, ecc.
    instr_index = (block_idx // 100) % len(all_instructions)
    system_prompt = all_instructions[instr_index]
    
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
            max_new_tokens=300, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id 
        )
    
    risposta = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    
    return risposta, system_prompt

def process_chat_file(file_path, instruction_idx):
    """Legge il file e usa una specifica riga di istruzioni."""
    if not os.path.exists(file_path):
        print(f"Errore: File {file_path} non trovato.")
        return

    with open("system_instructions.txt", "r", encoding="utf-8") as f:
        all_instructions = [line.strip() for line in f if line.strip()]
    
    # seleziona l'istruzione specifica passata dal ciclo
    system_prompt = all_instructions[instruction_idx % len(all_instructions)]

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = [b.strip() for b in content.split("###") if b.strip()]

    print(f"\n--- Estrazione PII: {file_path} (Istruzione riga {instruction_idx + 1}) ---")

    for i, block in enumerate(blocks):
        print(f">>> Blocco {i+1}/{len(blocks)}")
        
        start_time = time.time()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": block},
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        
        risposta = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        end_time = time.time()
        
        block_duration = round(end_time - start_time, 2)
        tags = re.findall(r"\[(.*?)\]", risposta)
        pii_found = tags[0] if tags else "None"
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "file": file_path,
            "instruction_row": instruction_idx + 1,
            "PII": pii_found,
            "processing_time": block_duration,
            "system_instruction": system_prompt,
            "full_response": risposta,
            "block_text": block
        }
        
        with open("llama_semantic_PII.jsonl", "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")

print("\n--- Sistema Estrazione PII Attivo ---")
while True:
    file_name = input("\nInserisci il nome del file, 'all' per 01-10 su TUTTE le istruzioni, o 'exit': ")
    
    if file_name.lower() == 'exit': 
        break
    
    if file_name.lower() == 'all':
        # legge quante istruzioni ci sono
        with open("system_instructions.txt", "r", encoding="utf-8") as f:
            num_instr = len([line for line in f if line.strip()])
        
        # doppio ciclo for (ciascuna istruzione per i 10 files)
        for inst_idx in range(num_instr):
            print(f"\n=== INIZIO CICLO CON ISTRUZIONE {inst_idx + 1} ===")
            for i in range(1, 11):
                current_file = f"{i:02d}.txt"
                full_path = os.path.join("chat", current_file)
                process_chat_file(full_path, inst_idx)
    else:
        # singolo file
        full_path = os.path.join("chat", file_name)
        process_chat_file(full_path, 0)

print("Sessione terminata.")