import json
import matplotlib.pyplot as plt
import numpy as np

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def are_entities_equal(ent1, ent2):
    # Verifica se le entità contengono le stesse parole, ignorando l'ordine
    words1 = set(str(ent1).lower().strip().split())
    words2 = set(str(ent2).lower().strip().split())
    return words1 == words2 and len(words1) > 0

def calculate_metrics(pred_list, true_list):
    # Versione aggiornata con "Smart Match" per gestire l'ordine delle parole (es. Nome Cognome)
    preds = [str(x).strip() for x in pred_list]
    trues = [str(x).strip() for x in true_list]
    
    tp = 0
    remaining_trues = list(trues)
    
    for p in preds:
        for t in remaining_trues:
            if are_entities_equal(p, t):
                tp += 1
                remaining_trues.remove(t)
                break
    
    fp = len(preds) - tp
    fn = len(trues) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 1.0
    
    return precision, recall, f1

# Caricamento dati dai file
try:
    gemini_data = load_jsonl('pii_analysis_result_gemini.jsonl')
    human_data = load_jsonl('pii_analysis_result_human.jsonl')
except FileNotFoundError as e:
    print(f"Errore: Assicurati che i file .jsonl siano nella cartella. {e}")
    exit()

# Elaborazione metriche
categories = ["names", "emails", "phones"]
results = {cat: {"p": [], "r": [], "f1": []} for cat in categories}

for g, h in zip(gemini_data, human_data):
    for cat in categories:
        p, r, f1 = calculate_metrics(g.get(cat, []), h.get(cat, []))
        results[cat]["p"].append(p)
        results[cat]["r"].append(r)
        results[cat]["f1"].append(f1)

# Calcolo Medie
final_metrics = {cat: [np.mean(results[cat]["p"]), 
                        np.mean(results[cat]["r"]), 
                        np.mean(results[cat]["f1"])] for cat in categories}

# Stampa valori numerici in output
print("\n" + "="*50)
print("RIEPILOGO METRICHE MEDIE (GEMINI VS HUMAN GT)")
print("="*50)
print(f"{'Categoria':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 50)
for cat, values in final_metrics.items():
    print(f"{cat.capitalize():<12} | {values[0]:<10.4f} | {values[1]:<10.4f} | {values[2]:<10.4f}")
print("="*50 + "\n")

# Generazione Grafico
labels = ['Precision', 'Recall', 'F1-Score']
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width, final_metrics["names"], width, label='Names', color='#4285F4', edgecolor='black')
rects2 = ax.bar(x, final_metrics["emails"], width, label='Emails', color='#EA4335', edgecolor='black')
rects3 = ax.bar(x + width, final_metrics["phones"], width, label='Phones', color='#FBBC05', edgecolor='black')

ax.set_ylabel('Punteggio (0.0 - 1.0)')
ax.set_title('Validazione Gemini vs Human Ground Truth\n(Analisi campionaria Dataset Enron)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.1)
ax.legend(title="Categorie PII")
ax.grid(axis='y', linestyle='--', alpha=0.6)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.show()

# Stampa Tabella di Confronto
print(f"{'Email':<7} | {'Tipo':<8} | {'Estratto da Gemini':<40} | {'Estratto da Umano':<40}")
print("-" * 105)

for i, (g, h) in enumerate(zip(gemini_data, human_data)):
    for cat in ["names", "emails", "phones"]:
        g_list = [str(x) for x in g.get(cat, [])]
        h_list = [str(x) for x in h.get(cat, [])]
        
        if g_list or h_list:
            g_str = ", ".join(g_list[:3]) + ("..." if len(g_list)>3 else "")
            h_str = ", ".join(h_list[:3]) + ("..." if len(h_list)>3 else "")
            print(f"#{i:<6} | {cat:<8} | {g_str:<40} | {h_str:<40}")
    print("-" * 105)