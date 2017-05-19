import pickle
import json
import numpy as np

MAX_ENTRIES = 600

# Load pkl data
with open('w2v.pkl', 'rb') as f:
    ans_embed = pickle.load(f)
with open('fqsp.pkl', 'rb') as f:
    ans_id, ans_question, ans_body = pickle.load(f)
    
# Write to JSON
with open('../js/fqsp.json', 'w') as f:
    json.dump({
        'words': [a[:100] for a in ans_body[:MAX_ENTRIES]],
        'vecs': ans_embed[:MAX_ENTRIES].tolist()
    }, f)