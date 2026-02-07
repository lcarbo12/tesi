import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

def ask_llama_final_relationship(combined_text):
    system_prompt = (
        "You are an expert linguistic analyst. I will provide you with several partial analyses of a conversation. "
        "Your job is to synthesize them and provide a final, definitive description of the relationship "
        "between the two speakers and the overall nature of their interaction. "
        "Start your response with this line: [Role 1, Role 2]"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Synthesize these partial analyses into one final report:\n\n{combined_text}"},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=500, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id 
        )
    
    input_length = inputs.shape[1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

def main():
    log_file = "log_roles.jsonl"
    if not os.path.exists(log_file):
        print("Errore: log_roles.jsonl non trovato.")
        return

    # caricamento dati dal log
    data_by_file = {}
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            # percorso 'chat\\01.txt' prendendo solo il nome finale
            fname = os.path.basename(entry["file"]) 
            if fname not in data_by_file:
                data_by_file[fname] = []
            data_by_file[fname].append(entry["full_response"])

    user_input = input("\nInserisci numero file (es: 1) o 'all': ").strip().lower()
    
    if user_input == 'all':
        targets = sorted(data_by_file.keys())
    else:
        target_file = f"{user_input.zfill(2)}.txt"
        if target_file in data_by_file:
            targets = [target_file]
        else:
            print(f"File {target_file} non trovato nel log.")
            return

    # elaborazione con Llama
    os.makedirs("relationships", exist_ok=True)
    for t in targets:
        print(f"\n>>> Generazione relazione finale per: {t}...")
        combined_responses = "\n---\n".join(data_by_file[t])
        
        # unione delle risposte
        final_report = ask_llama_final_relationship(combined_responses)
        
        # scrittura su file nella sottocartella relationships
        output_name = f"{t.split('.')[0]}_relationship.txt"
        output_path = os.path.join("relationships", output_name)
        
        with open(output_path, "w", encoding="utf-8") as out_f:
            out_f.write(final_report.strip())
        
        print(f"Completato: {output_path}")

if __name__ == "__main__":
    main()