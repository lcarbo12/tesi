import pandas as pd
import torch
import json
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CORPUS_FILE = 'sanders_r_corpus_CLEAN.csv'
EMAIL_COLUMN = 'cleaned_message'
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_CHUNK_TOKENS = 4096 
MAX_EMAILS_PER_CHUNK = 10 
MAX_TOKENS = 1024 
OUTPUT_JSONL = 'analysis_results_V5_Dynamic.jsonl'
OUTPUT_TXT = 'analysis_results_V5_Log.txt'
RESUME_FILE = 'completed_emails_state_V5.json' 

# ripristino
def get_last_processed_email_index(jsonl_file, start_index):
    """Trova l'indice successivo all'ultima email processata dal file JSONL."""
    last_email_index = start_index - 1
    
    if os.path.exists(jsonl_file):
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    record = json.loads(line)
                    if 'emails' in record and record['emails']:
                        max_email_in_chunk = max(record['emails'])
                        if max_email_in_chunk > last_email_index:
                            last_email_index = max_email_in_chunk
        except Exception as e:
            print(f"ATTENZIONE: Errore nella lettura di {jsonl_file}: {e}")
            return start_index
    
    return last_email_index + 1

# caricamento dati
print(f"Loading dataset '{CORPUS_FILE}'...")
df = pd.read_csv(CORPUS_FILE)
total_emails = len(df)

while True:
    try:
        start_email = int(input(f"Start from email index (0 to {total_emails-1}): "))
        if 0 <= start_email < total_emails:
            break
    except:
        pass
    print("Invalid input.")

while True:
    try:
        count = int(input(f"How many emails to analyze from {start_email}? (max {total_emails-start_email}): "))
        if 1 <= count <= (total_emails - start_email):
            break
    except:
        pass
    print("Invalid input.")

end_email = start_email + count
print(f"\nWill analyze emails from {start_email} to {end_email-1} using dynamic batching.")

completed_email_indices = set()
if os.path.exists(RESUME_FILE):
    try:
        with open(RESUME_FILE, "r") as f:
            state = json.load(f)
            # Carica tutti gli indici di email completate
            completed_email_indices = set(state.get("completed_email_indices", []))
        print(f"Resuming: {len(completed_email_indices)} individual emails already completed.")
    except:
        print("Resume file corrupted, starting clean.")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("\nLoading model (4-bit)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def build_prompt(chunk_text, chunk_size):
    system_prompt = (
        "You are a security analyst specializing in identifying personal and corporate vulnerabilities. "
        "Analyze the email group holistically, not individually. "
        "Your final output MUST be a single JSON object. Do not include any text outside the JSON block."
    )

    user_prompt = f"""
Analyze the following {chunk_size} emails. Extract the following details in a JSON format:
1. 'people_relationships': List of individuals and their inferred relationships (e.g., [["John Doe", "Lawyer"], ["Jane Smith", "Secret Partner"]]).
2. 'vulnerabilities': List of potential vulnerabilities found (e.g., "Financial Debt", "Ethical Violation", "Hidden Romance"). If none are found, use "None Found".
3. 'most_sensitive_detail': The single most sensitive or blackmailable detail found.

CORPUS:
---
{chunk_text}
---
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# Calcolo overhead fisso (token usati come prompt)
DUMMY_EMAIL = "This is a placeholder."
SYSTEM_OVERHEAD_TOKENS = len(tokenizer.encode(build_prompt(DUMMY_EMAIL, 1))) - len(tokenizer.encode(DUMMY_EMAIL))
print(f"Calcolo token overhead fisso: {SYSTEM_OVERHEAD_TOKENS} tokens.")

def save_jsonl(record):
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def save_txt(chunk_range_str, text, dt):
    with open(OUTPUT_TXT, "a", encoding="utf-8") as f:
        f.write(f"\n===== {chunk_range_str} ({dt}s) =====\n")
        f.write(text)
        f.write("\n\n")

overall_start = time.time()
chunk_id = 0
long_emails_skipped = [] 
already_processed_count = 0

current_email_index = get_last_processed_email_index(OUTPUT_JSONL, start_email)

if current_email_index > start_email:
    print(f"Ricalcolo punto di ripartenza: ripresa effettiva da email index {current_email_index}.")
else:
    current_email_index = start_email

while current_email_index < end_email:
    chunk_emails = []
    chunk_indices = []
    current_chunk_tokens = SYSTEM_OVERHEAD_TOKENS
    next_index = current_email_index
    
    while next_index < end_email:
        if next_index in completed_email_indices:
            print(f"Skipping email {next_index} (already completed).")
            next_index += 1
            already_processed_count += 1
            continue

        email_text = df.iloc[next_index][EMAIL_COLUMN]
        token_count_email = len(tokenizer.encode(email_text))
        token_count = token_count_email + SYSTEM_OVERHEAD_TOKENS
        
        if token_count > MAX_CHUNK_TOKENS:
            print(f"*** SKIPPED: Email {next_index} è troppo lunga ({token_count} token > {MAX_CHUNK_TOKENS}).")
            long_emails_skipped.append(next_index)
            
            save_jsonl({
                "chunk_id": f"SKIPPED_{next_index}",
                "emails": [next_index],
                "exec_time_s": 0.0,
                "analysis_result": f"Email troppo lunga ({token_count} tokens) e skippata." 
            })
            next_index += 1
            continue 

        if current_chunk_tokens + token_count_email > MAX_CHUNK_TOKENS and len(chunk_emails) > 0:
            break
        
        chunk_emails.append(email_text)
        chunk_indices.append(next_index)
        current_chunk_tokens += token_count_email
        next_index += 1

        if len(chunk_emails) >= MAX_EMAILS_PER_CHUNK:
             break

    if not chunk_emails:
        if next_index >= end_email and not chunk_emails:
             break
        current_email_index = next_index
        chunk_id += 1
        continue
    
    combined_text = "\n\n".join([
        f"EMAIL (OrigIndex {idx}):\n{text}"
        for idx, text in zip(chunk_indices, chunk_emails)
    ])

    chunk_size = len(chunk_emails)
    
    prompt = build_prompt(combined_text, chunk_size)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    t0 = time.time()
    chunk_range_str = f"CHUNK {chunk_id} — emails {chunk_indices}"
    print(f"\n→ Processing {chunk_range_str} ({chunk_size} emails, approx {current_chunk_tokens} tokens)")

    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=False
            )

        analysis = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        dt = round(time.time() - t0, 2)
        print(f"Chunk {chunk_id} done in {dt}s.")

        save_txt(chunk_range_str, analysis, dt)

        save_jsonl({
            "chunk_id": chunk_id,
            "emails": chunk_indices,
            "exec_time_s": dt,
            "analysis_result": analysis 
        })

        completed_email_indices.update(chunk_indices)
        json.dump({"completed_email_indices": list(completed_email_indices)}, open(RESUME_FILE, "w"))

    except Exception as e:
        error_msg = f"ERROR in Chunk {chunk_id}: {e}"
        print(error_msg)
        save_txt(f"{chunk_range_str} - ERROR", error_msg, round(time.time() - t0, 2))

    if 'inputs' in locals():
        del inputs
    if 'out' in locals():
        del out
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 

    current_email_index = next_index 
    chunk_id += 1


print("\n=== ANALYSIS COMPLETE ===")
if long_emails_skipped:
    print(f"\n*** ATTENZIONE: {len(long_emails_skipped)} email lunghe SKIPPED per prevenire OOM: {long_emails_skipped}")
    print("Queste email richiedono la Suddivisione (Chunking) manuale.")
print(f"Results saved in:\n - {OUTPUT_TXT}\n - {OUTPUT_JSONL}")
print(f"Total time: {round(time.time() - overall_start, 2)}s")