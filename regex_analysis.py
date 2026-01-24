import re
import json
import time
import pandas as pd

EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PHONE_REGEX = re.compile(r'(\(?\b\d{3}\)?[-.\s/]?\d{3}[-.\s/]?\d{4}\b|\b\d{3}[-.\s/]?\d{4}\b|\bx\d{4,5}\b)')
NAME_REGEX = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b')

BLACKLIST = {
    "Original Message", "Sent", "Subject", "From", "To", "Cc", "Bcc", 
    "Forwarded", "Dear", "Regards", "Thank", "Thanks", "Hello", "Hi",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August", 
    "September", "October", "November", "December", "Enron", "Houston", "ECT",
    "Agreement", "Contract", "United States", "North America", "Information"
}

def deep_clean(text):
    if not text: return ""
    text = re.sub(r'[\n\t\r]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_valid_name(name_text):
    cleaned = deep_clean(name_text)
    if cleaned in BLACKLIST or len(cleaned) < 3:
        return False
    return True


def process_email(text, email_index):
    start_time = time.time()
    
    if not isinstance(text, str):
        text = ""

    emails = EMAIL_REGEX.findall(text)
    phones = PHONE_REGEX.findall(text)
 
    raw_names = NAME_REGEX.findall(text)
    names = [deep_clean(n) for n in raw_names if is_valid_name(n)]

    result = {
        "email_index": email_index,
        "seconds": round(time.time() - start_time, 4),
        "names": sorted(list(set(names))),
        "emails": sorted(list(set([deep_clean(e).lower() for e in emails]))),
        "phones": sorted(list(set([deep_clean(p) for p in phones])))
    }
    return result

def main():
    input_file = "campione_enron.csv"
    output_file = "pii_analysis_results_regex.jsonl"
    
    try:
        df = pd.read_csv(input_file)
        
        all_cols = ["subject", "from", "to", "cc", "bcc", "date", "body", "file_name"]
        target_columns = [col for col in all_cols if col in df.columns]
        
        print(f"Analisi sui campi: {', '.join(target_columns)}")
        
        df_combined = df[target_columns].fillna('').astype(str).agg(' '.join, axis=1)
        
        print(f"Inizio elaborazione...")
        
        with open(output_file, "w", encoding="utf-8") as f:
            for idx, text in enumerate(df_combined[:100]):
                result = process_email(text, idx)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        print(f"Completato! Risultati salvati in: {output_file}")
        
    except FileNotFoundError:
        print(f"Errore: Il file {input_file} non è stato trovato.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

if __name__ == "__main__":
    main()