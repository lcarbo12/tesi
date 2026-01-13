import re
import json
import time
import pandas as pd


# Email sia user@domain.com che Lotus Notes User/HOU/ECT@ECT
EMAIL_REGEX = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b' 
    r'|'
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*/[A-Z0-9/]+(?:@?[A-Z]*)\b'
)

# Numeri di telefono
PHONE_REGEX = re.compile(
    r'(\(?\b\d{3}\)?[-.\s/]?\d{3}[-.\s/]?\d{4}\b|\b\d{3}[-.\s/]?\d{4}\b|\bx\d{4,5}\b)'
)

# Soldi
MONEY_REGEX = re.compile(
    r'\$\s?\d+(?:,\d{3})*(?:\.\d+)?(?:\s?(?:million|billion|trillion))?\b', 
    re.IGNORECASE
)

# Nomi
NAME_REGEX = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b')

# Ruoli aziendali
JOB_ROLE_REGEX = re.compile(
    r'\b(Director|Manager|President|Counsel|Assistant|Analyst|Trader|Engineer|Clerk|Vice President|VP|Senior VP|CEO|COO|CFO)\b',
    re.IGNORECASE
)


# Parole da ignorare perché catturate spesso erroneamente come Nomi
BLACKLIST = {
    "Original Message", "Sent", "Subject", "From", "To", "Cc", "Bcc", 
    "Forwarded", "Dear", "Regards", "Thank", "Thanks", "Hello", "Hi",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August", 
    "September", "October", "November", "December", "Enron", "Houston", "ECT",
    "Agreement", "Contract", "United States", "North America", "Information"
}

def deep_clean(text):
    """Rimuove newline, tabulazioni e spazi multipli da una stringa."""
    if not text: return ""
    text = re.sub(r'[\n\t\r]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_valid_name(name_text):
    """Verifica se il nome estratto è sensato o se è un falso positivo."""
    cleaned = deep_clean(name_text)
    # Filtra se è nella blacklist, troppo corto o se contiene solo keyword Enron
    if cleaned in BLACKLIST or len(cleaned) < 3:
        return False
    # Filtra nomi che iniziano con termini di sistema
    if any(cleaned.startswith(word) for word in ["From:", "To:", "Subject:"]):
        return False
    return True



def process_email(text, email_index):
    start_time = time.time()
    
    if not isinstance(text, str):
        text = ""

    # Estrazione con Regex
    emails = EMAIL_REGEX.findall(text)
    phones = PHONE_REGEX.findall(text)
    financial = MONEY_REGEX.findall(text)
    job_roles = JOB_ROLE_REGEX.findall(text)
    
    # Nomi: estrazione + pulizia + filtraggio
    raw_names = NAME_REGEX.findall(text)
    names = [deep_clean(n) for n in raw_names if is_valid_name(n)]

    # Costruzione del risultato JSON (ancora da convertire)
    result = {
        "email_index": email_index,
        "names": sorted(list(set(names))),
        "emails": sorted(list(set([deep_clean(e) for e in emails]))),
        "phones": sorted(list(set([deep_clean(p) for p in phones]))),
        "financial": sorted(list(set([deep_clean(f) for f in financial]))),
        "job_roles_ids": sorted(list(set([deep_clean(j) for j in job_roles]))),
        "seconds": round(time.time() - start_time, 4)
    }
    return result



def main():
    input_file = "campione_enron.csv"
    output_file = "output_regex_v2.jsonl"
    
    try:
        df = pd.read_csv(input_file)
        column_name = "body" if "body" in df.columns else df.columns[0]
        texts = df[column_name].tolist()
        
        print(f"Inizio elaborazione di {min(len(texts), 100)} email...")
        
        with open(output_file, "w", encoding="utf-8") as f:
            for idx, text in enumerate(texts[:100]):
                result = process_email(text, idx)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        print(f"Completato! Risultati salvati in: {output_file}")
        
    except FileNotFoundError:
        print(f"Errore: Il file {input_file} non è stato trovato.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

if __name__ == "__main__":
    main()