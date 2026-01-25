import json
import re

def clean_and_fix_json(raw_response):
    """Tenta di estrarre e pulire il JSON dalle risposte che hanno generato errore."""
    try:
        # Trova il blocco tra parentesi graffe
        match = re.search(r'\{[\s\S]*\}', raw_response)
        if not match:
            return None
        content = match.group(0)
        # Rimuove commenti del tipo // o parentesi esplicative (causa degli errori)
        content = re.sub(r'//.*', '', content)
        content = re.sub(r'\s*\([^)]*\)', '', content)
        return json.loads(content)
    except:
        return None

def process_jsonl(input_file, output_file):
    combined_data = {}

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line.strip())
            idx = entry['email_index']

            # Se c'è un errore, proviamo a recuperare i dati dal raw_response
            if 'error' in entry and 'raw_response' in entry:
                fixed = clean_and_fix_json(entry['raw_response'])
                if fixed:
                    entry.update(fixed)
                    entry.pop('error')
                    entry.pop('raw_response')

            if idx not in combined_data:
                combined_data[idx] = {
                    'email_index': idx,
                    'seconds': 0.0,
                    'names': set(),
                    'emails': set(),
                    'phones': set()
                }

            # Somma i secondi
            combined_data[idx]['seconds'] = round(combined_data[idx]['seconds'] + entry.get('seconds', 0), 2)

            # Unisci le liste eliminando i duplicati
            for field in ['names', 'emails', 'phones']:
                if field in entry and isinstance(entry[field], list):
                    for item in entry[field]:
                        if item and str(item).strip():
                            val = item.strip()
                            # MODIFICA: se il campo è 'emails', converti in minuscolo
                            if field == 'emails':
                                val = val.lower()
                            combined_data[idx][field].add(val)

    # Scrittura del file finale
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx in sorted(combined_data.keys()):
            # Riconvertiamo i set in liste per il JSON
            final_entry = combined_data[idx]
            final_entry['names'] = list(final_entry['names'])
            final_entry['emails'] = list(final_entry['emails'])
            final_entry['phones'] = list(final_entry['phones'])
            
            f.write(json.dumps(final_entry, ensure_ascii=False) + '\n')

# Esecuzione
process_jsonl('pii_analysis_results_llama.jsonl', 'pii_analysis_results_llama_cleaned.jsonl')
print("Elaborazione completata. Creato file: pii_analysis_results_new7_cleaned.jsonl")