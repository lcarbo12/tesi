import json
import matplotlib.pyplot as plt

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def are_entities_equal(ent1, ent2):
    w1 = set(str(ent1).lower().strip().split())
    w2 = set(str(ent2).lower().strip().split())
    return w1 == w2 and len(w1) > 0

# Caricamento dati
try:
    gemini_data = load_jsonl('pii_analysis_result_gemini.jsonl')
    human_data = load_jsonl('pii_analysis_result_human.jsonl')
except FileNotFoundError:
    print("Errore: File .jsonl non trovati.")
    exit()

# Raccolta solo delle discrepanze
discrepancy_rows = []
categories = ["names", "emails", "phones"]

for i, (g, h) in enumerate(zip(gemini_data, human_data)):
    g_piis = []
    for cat in categories: g_piis.extend(g.get(cat, []))
    
    h_piis = []
    for cat in categories: h_piis.extend(h.get(cat, []))
    
    combined = list(h_piis)
    for gp in g_piis:
        if not any(are_entities_equal(gp, hp) for hp in combined):
            combined.append(gp)
            
    for pii in combined:
        found_by_g = "✔" if any(are_entities_equal(pii, gp) for gp in g_piis) else ""
        found_by_h = "✔" if any(are_entities_equal(pii, hp) for hp in h_piis) else ""
        
        if found_by_g == "" or found_by_h == "":
            discrepancy_rows.append([f"Mail #{i}: {pii}", found_by_g, found_by_h])

# Creazione Tabella Grafica
if not discrepancy_rows:
    print("Nessuna discrepanza trovata!")
else:
    num_rows = len(discrepancy_rows)
    dynamic_height = max(6, num_rows * 0.5) 

    fig, ax = plt.subplots(figsize=(12, dynamic_height))
    ax.axis('off')

    column_labels = ['Identified PII (Discrepancy)', 'Gemini', 'Human']
    
    table = ax.table(cellText=discrepancy_rows, 
                    colLabels=column_labels, 
                    loc='center', 
                    cellLoc='center',
                    colWidths=[0.45, 0.275, 0.275]) 

    table.auto_set_font_size(False)
    table.set_fontsize(12) 
    table.scale(1, 2.5)    

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#d9534f") 
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
        elif col == 0:
            cell.get_text().set_ha('left')
            cell.get_text().set_position((0.02, 0.5))

    plt.title("PII Discrepancy Analysis: Gemini vs Human", fontsize=14, pad=20)
    plt.tight_layout()

    plt.savefig('pii_discrepancies_final_45.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Tabella generata con {num_rows} discrepanze.")