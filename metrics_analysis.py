import pandas as pd
import json
import re

def normalize_name(name):
    if not isinstance(name, str): return ""
    
    # minuscolo e pulizia spazi
    name = name.lower().strip()
    
    # rimuove iniziali puntate o singole tipo p. o p
    name = re.sub(r'\b[a-z]\.?\s+', '', name)
    
    # tokenizzazione e solo parole con almeno 2 lettere
    tokens = re.findall(r'\b\w{2,}\b', name)
    
    # ordinamento alfabetico per gestire nome e cognome
    tokens.sort()
    
    return " ".join(tokens)

def normalize_phone(phone):
    # estrae solo le cifre
    if not phone: return ""
    return re.sub(r'\D', '', str(phone))

def load_jsonl(file_path, model_name):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                
                normalized_names = {normalize_name(n) for n in item.get("names", []) if normalize_name(n)}
                normalized_emails = {e.lower().strip() for e in item.get("emails", []) if e}
                normalized_phones = {normalize_phone(p) for p in item.get("phones", []) if normalize_phone(p)}
                
                data.append({
                    "email_index": item["email_index"],
                    f"{model_name}_time": item.get("seconds", 0),
                    f"{model_name}_names": normalized_names,
                    f"{model_name}_emails": normalized_emails,
                    f"{model_name}_phones": normalized_phones
                })
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"ATTENZIONE: File {file_path} non trovato.")
        return pd.DataFrame()

df_regex = load_jsonl('pii_analysis_results_regex.jsonl', 'regex')
df_presidio = load_jsonl('pii_analysis_results_presidio.jsonl', 'presidio')
df_llama = load_jsonl('pii_analysis_results_llama_cleaned.jsonl', 'llama')
df_gemini = load_jsonl('gemini_with_index_modified.jsonl', 'gemini')

df = df_regex.merge(df_presidio, on="email_index") \
             .merge(df_llama, on="email_index") \
             .merge(df_gemini, on="email_index")

def get_scores(model_prefix, gold_prefix, category):
    model_col = f"{model_prefix}_{category}"
    gold_col = f"{gold_prefix}_{category}"
    
    results = []
    for _, row in df.iterrows():
        model_set = row[model_col]
        gold_set = row[gold_col]
        
        tp = len(model_set.intersection(gold_set))
        fp = len(model_set - gold_set)
        fn = len(gold_set - model_set)
        
        if not gold_set and not model_set:
            prec, rec, f1 = 1.0, 1.0, 1.0
        else:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            
        results.append({'prec': prec, 'rec': rec, 'f1': f1})
    
    return pd.DataFrame(results).mean()

categories = ['names', 'emails', 'phones']
models = ['regex', 'presidio', 'llama']
final_report = []

for m in models:
    for c in categories:
        scores = get_scores(m, 'gemini', c)
        final_report.append({
            'Modello': m,
            'Categoria': c,
            'Precision': round(scores['prec'], 4),
            'Recall': round(scores['rec'], 4),
            'F1-Score': round(scores['f1'], 4)
        })

report_df = pd.DataFrame(final_report)
print("--- ANALISI QUANTITATIVA PII (Ground Truth: Gemini) ---")
print(report_df.to_string(index=False))

print("\n--- TEMPI MEDI DI ESECUZIONE (secondi) ---")
tempi_cols = [c for c in df.columns if '_time' in c]
print(df[tempi_cols].mean().round(4))