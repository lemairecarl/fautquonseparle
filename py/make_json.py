import pickle
import json
import numpy as np

MAX_ENTRIES = 2000

# Load pkl data
#with open('w2v.pkl', 'rb') as f:
#    ans_embed = pickle.load(f)
ans_tsne = np.loadtxt('fqsp_tsne_p20i1000.tsv', delimiter='\t')

with open('fqsp.pkl', 'rb') as f:
    _, ans_question, ans_body = pickle.load(f)

# raw ans_question: 4-17
ans_question = np.array(ans_question)
ans_question -= 4  # --> [0, 13] (10 + oublie qqch + 3 inconnues)

# Remove invalid classes (already done for ans_tsne)
valid_idx = ans_question <= 9
ans_question = ans_question[valid_idx]
ans_body = [ans_txt for i, ans_txt in enumerate(ans_body) if valid_idx.ravel()[i]]
num_classes = 10
assert len(np.unique(ans_question)) == num_classes

# Write to JSON
with open('../fqsp.json', 'w') as f:
    json.dump({
        'words': [a[:100] for a in ans_body[:MAX_ENTRIES]],
        'vecs': ans_tsne[:MAX_ENTRIES].tolist(),
        'q': ans_question[:MAX_ENTRIES].tolist()
    }, f)