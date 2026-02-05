import pandas as pd
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re
import time
import os

FULL_ANALYSIS = False 
SAMPLE_SIZE = 100
OUTPUT_FILENAME = 'pii_analysis_results.jsonl' 
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

try:
    df = pd.read_csv('campione_enron.csv')
except FileNotFoundError:
    print("ERRORE: File 'campione_enron.csv' non trovato.")
    exit()

if not FULL_ANALYSIS:
    email_samples = df.head(SAMPLE_SIZE).copy()
else:
    email_samples = df.copy()

fields_to_include = ['subject', 'from', 'to', 'cc', 'bcc', 'date', 'body', 'file_name']

# contenuto testuale con le etichette dei campi per dare contesto al modello
def format_email_full(row):
    parts = []
    for field in fields_to_include:
        val = str(row[field]).strip() if pd.notna(row[field]) else ""
        if val:
            # format "Campo: Valore" come aiuto al modello
            parts.append(f"{field.upper()}: {val}")
    return "\n".join(parts)

# applichiamo la formattazione
email_texts = email_samples.apply(format_email_full, axis=1).tolist()

# caricamento del modello
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Modello {MODEL_ID} caricato.")
except Exception as e:
    print(f"Errore nel caricamento del modello: {e}")
    exit()

def generate_pii_prompt(email_content):
    system_prompt = (
        "Extract into a valid JSON from Enron emails full names of at least two words (e.g. Name Surname), personal emails and phone numbers.\n"
        "If a category is empty return a empty list []. Extract only entities explicitly present in the text.\n"
        "Do not add any comments or additional text either inside or outside the JSON structure."
    )

    json_schema = {
        "names": [],
        "emails": [],
        "phones": []
    }
    
    user_prompt = (
        f"Analyze the following email and extract PII according to this JSON schema:\n"
        f"{json.dumps(json_schema, indent=2)}\n\n"
        f"### EMAIL CONTENT ###\n"
        f"{email_content}\n"
        f"### END EMAIL CONTENT ###\n\n"
        f"JSON RESPONSE:"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# estrazione senza filtri
def extract_robust_json_keep_all(response, email_index, start_time):
    end_time = time.time()
    elapsed_seconds = round(end_time - start_time, 2)
    
    try:
        # estrazione del blocco JSON con regex
        matches = re.findall(r'\{[\s\S]*?\}', response)
        raw_output = json.loads(matches[0]) if matches else json.loads(response.strip())
        
        # inizializzazione con i metadati obbligatori
        pii_data = {
            'email_index': email_index,
            'seconds': elapsed_seconds
        }
        
        # copia tutto il contenuto di raw_output in pii_data
        if isinstance(raw_output, dict):
            pii_data.update(raw_output)
        
        return pii_data

    except Exception as e:
        # nel caso non si riesca a prelevare il json
        return {
            'email_index': email_index,
            'seconds': elapsed_seconds,
            'error': f'JSON_DECODE_FAILED: {str(e)}',
            'raw_response': response
        }
        
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

MAX_CHUNK_LEN = 2500

with open(OUTPUT_FILENAME, 'a') as f:
    start_index = 0
    if os.path.exists(OUTPUT_FILENAME):
        try:
            with open(OUTPUT_FILENAME, 'r') as check_f:
                lines = check_f.readlines()
                if lines:
                    last_data = json.loads(lines[-1].strip())
                    start_index = last_data.get('email_index', -1) + 1
        except:
            start_index = 0

    for i in range(start_index, len(email_samples)):
        email_content = email_texts[i]
        
        # convertiamo in token per misurare la lunghezza reale
        tokens = tokenizer.encode(email_content, add_special_tokens=False)
        
        # dividiamo i token in pezzi da MAX_CHUNK_LEN
        chunks = [tokens[j:j + MAX_CHUNK_LEN] for j in range(0, len(tokens), MAX_CHUNK_LEN)]
        
        num_chunks = len(chunks)
        if num_chunks == 1:
            print(f"Analizzo email {i}...")
        else:
            print(f"Analizzo l'email {i} chunk {chunk_idx + 1}/{num_chunks}...")

        for chunk_idx, chunk_tokens in enumerate(chunks):
            start_time = time.time()
            chunk_text = tokenizer.decode(chunk_tokens)
            
            prompt = generate_pii_prompt(chunk_text) 
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1516,
                    eos_token_id=terminators,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, 
                )

            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            pii_data = extract_robust_json_keep_all(response, i, start_time)
            # Aggiungiamo info sul chunk per chiarezza (opzionale)
            pii_data['chunk'] = f"{chunk_idx + 1}/{len(chunks)}"

            f.write(json.dumps(pii_data) + '\n')
            f.flush() 
            
        if num_chunks == 1:
            print(f"Completata email {i} in {pii_data.get('seconds')}s")
        else:
            print(f"Completata email {i} chunk {chunk_idx + 1}/{num_chunks} in {pii_data.get('seconds')}s")


print(f"\nAnalisi completata. Risultati in: {OUTPUT_FILENAME}")

