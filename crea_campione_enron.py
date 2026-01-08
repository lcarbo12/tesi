import pandas as pd
from datasets import load_dataset

dataset = load_dataset("corbt/enron-emails")
df = pd.DataFrame(dataset['train'])

# Campione da 1000 mail
df_sample = df.sample(n=1000, random_state=42)
df_sample.to_csv("campione_enron.csv", index=False)
print("Tutto ok")