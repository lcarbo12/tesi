import pandas as pd
import json
import time
import re
from presidio_analyzer import AnalyzerEngine

INPUT_FILE = "campione_enron.csv"
OUTPUT_FILE = "pii_analysis_results_presidio.jsonl"
SAMPLE_SIZE = 100 

BLACKLIST = {
    "Original Message", "Sent", "Subject", "From", "To", "Cc", "Bcc", 
    "Forwarded", "Dear", "Regards", "Thank", "Thanks", "Hello", "Hi",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August", 
    "September", "October", "November", "December", "Enron", "Houston", "ECT",
    "Agreement", "Contract", "United States", "North America", "Information",
    "PM", "AM", "Fax", "Phone"
}

def deep_clean(text):
    if not text: return ""
    text = re.sub(r'[\n\t\r]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_valid_name(text):
    text = text.split('/')[0].strip()
    text = re.sub(r"['’]s$", "", text)
    text = re.sub(r"[^\w\s]$", "", text)
    clean_txt = deep_clean(text)
    
    if len(clean_txt) < 3: return False
    if clean_txt in BLACKLIST: return False
    if any(char.isdigit() for char in clean_txt): return False

    words = clean_txt.split()
    if len(words) < 2: return False

    return clean_txt

def is_personal_email(email_text):
    standard_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(standard_pattern, email_text))

def process_email_presidio(analyzer, text, email_index):
    start_time = time.time()
    
    if not isinstance(text, str):
        text = ""

    entities_to_find = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]
    results = analyzer.analyze(text=text, entities=entities_to_find, language='en')

    extracted = {
        "names": set(),
        "emails": set(),
        "phones": set()
    }

    for res in results:
        entity_text = text[res.start:res.end].strip()
        
        if res.entity_type == "PERSON":
            cleaned_name = is_valid_name(entity_text)
            if cleaned_name:
                extracted["names"].add(cleaned_name)
        
        elif res.entity_type == "EMAIL_ADDRESS":
            if is_personal_email(entity_text):
                extracted["emails"].add(entity_text.lower())
        
        elif res.entity_type == "PHONE_NUMBER":
            clean_phone = ''.join(filter(str.isdigit, entity_text))
            if len(clean_phone) >= 7:
                extracted["phones"].add(entity_text)

    result_json = {
        "email_index": email_index,
        "seconds": round(time.time() - start_time, 4),
        "names": sorted(list(extracted["names"])),
        "emails": sorted(list(extracted["emails"])),
        "phones": sorted(list(extracted["phones"]))
    }

    return result_json

def main():
    print("Inizializzazione Presidio Analyzer...")
    try:
        analyzer = AnalyzerEngine()
    except Exception as e:
        print(f"Errore inizializzazione: {e}")
        return

    try:
        df = pd.read_csv(INPUT_FILE)
        
        target_columns = [col for col in df.columns if col != 'message_id']
        
        combined_text = df[target_columns].fillna('').astype(str).agg(' '.join, axis=1)
        texts = combined_text.tolist()
        
        print(f"Inizio elaborazione di {min(len(texts), SAMPLE_SIZE)} record (campi: {', '.join(target_columns)})...")
        
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for idx, text in enumerate(texts[:SAMPLE_SIZE]):
                if idx % 20 == 0: print(f"Analisi riga {idx}...")
                
                result = process_email_presidio(analyzer, text, idx)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        print(f"Completato! File salvato in: {OUTPUT_FILE}")
        
    except FileNotFoundError:
        print(f"Errore: Il file {INPUT_FILE} non esiste.")
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")

if __name__ == "__main__":
    main()