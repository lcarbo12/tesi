import pandas as pd
import re
import os

INPUT_FILE = 'sanders-r_corpus.csv'
OUTPUT_FILE = 'sanders_r_corpus_CLEAN.csv'
MESSAGE_COLUMN = 'message' 

def clean_email_text_for_semantic_analysis(raw_text):

    if not isinstance(raw_text, str):
        return ""
    
    content = raw_text.strip()
    
    # Pulizia metadati e header di sistema (non rilevanti per l'analisi semantica)
    content = re.sub(r'^\s*(Message-ID|Mime-Version|Content-Type|Content-Transfer-Encoding|X-Folder|X-Origin|X-FileName):.*$', 
                     '', content, flags=re.MULTILINE | re.IGNORECASE)
    
    # Rimozione campi X- secondari (cc, bcc, etc.)
    content = re.sub(r'^\s*X-(cc|bcc|From|To|Origin|FileName|Mailer|User-Agent|Priority|UID|Status|User|Ref|Attachment|Keywords):.*$', 
                     '', content, flags=re.MULTILINE | re.IGNORECASE)
    
    # Strip dei separatori di inoltro standard Enron
    content = re.sub(r'^\s*-+\s*Forwarded by.*$', '', content, flags=re.MULTILINE | re.IGNORECASE)
    
    # Rimozione info di contatto e firme (URL, numeri di telefono)
    content = re.sub(r'^\s*Office:\s*\(?\d+\)?.*$', '', content, flags=re.MULTILINE) 
    content = re.sub(r'^\s*www\..*$', '', content, flags=re.MULTILINE) 
    
    # Rimozione separatori grafici (es. stringhe con << >>)
    content = re.sub(r'^[<]+.*[>]+$', '', content, flags=re.MULTILINE) 
    
    # Normalizzazione newline ed eccessi di spazi bianchi
    content = re.sub(r'\n{3,}', '\n\n', content).strip()
    
    return content

if __name__ == "__main__":
    print(f"Esecuzione script di cleaning su: {INPUT_FILE}")
    
    try:
        df = pd.read_csv(INPUT_FILE)
        
        if MESSAGE_COLUMN not in df.columns:
            print(f"Colonna '{MESSAGE_COLUMN}' mancante. Campi rilevati: {df.columns.tolist()}")
            exit()
            
        print(f"Processando {len(df)} entry...")
        
        # Applico la pulizia e genero il dataset finale
        df['cleaned_message'] = df[MESSAGE_COLUMN].apply(clean_email_text_for_semantic_analysis)
        
        # Export dei soli campi necessari per la tesi
        output_df = df[['file', 'cleaned_message']] 
        output_df.to_csv(OUTPUT_FILE, index=False)
        
        print(f"Completato. Dataset salvato in: {OUTPUT_FILE}")
        
        # Verifica rapida dell'output
        print("\nLog - Primo record processato:")
        print("-" * 40)
        print(output_df['cleaned_message'].iloc[0][:300] + "...")
        print("-" * 40)

    except FileNotFoundError:
        print(f"Errore: File {INPUT_FILE} non trovato.")
    except Exception as e:

        print(f"Errore durante l'esecuzione: {e}")
