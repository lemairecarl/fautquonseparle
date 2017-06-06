import pickle
import json
import numpy as np

MAX_ENTRIES = 999999

# Load pkl data
#with open('w2v.pkl', 'rb') as f:
#    ans_embed = pickle.load(f)
ans_tsne = np.loadtxt('fqsp_tsne_p20i1000.tsv', delimiter='\t')
ans_tsne = ans_tsne[:, ::-1]  # meilleur pour paysage

with open('fqsp.pkl', 'rb') as f:
    ans_id, ans_question, ans_body = pickle.load(f)

# raw ans_question: 4-17
ans_question = np.array(ans_question)
ans_question -= 4  # --> [0, 13] (10 + oublie qqch + 3 inconnues)
ans_id = np.array(ans_id)

# Remove invalid classes (already done for ans_tsne)
valid_idx = ans_question <= 9
ans_question = ans_question[valid_idx]
ans_id = ans_id[valid_idx]
ans_body = [ans_txt for i, ans_txt in enumerate(ans_body) if valid_idx.ravel()[i]]
num_classes = 10
assert len(np.unique(ans_question)) == num_classes

# Créer versions courtes pour aperçu
ans_court = []
longueur_apercu = 80
for texte in ans_body:
    if len(texte) > longueur_apercu:
        texte = texte[:longueur_apercu] + '…'
    ans_court.append(texte.replace('\n', ' ').replace('\r', ''))

# Write to JSON
with open('../fqsp.json', 'w') as f:
    json.dump({
        'court': ans_court[:MAX_ENTRIES],
        'long': ans_body[:MAX_ENTRIES],
        'vecs': ans_tsne[:MAX_ENTRIES].tolist(),
        'q': ans_question[:MAX_ENTRIES].tolist(),
        'id': ans_id[:MAX_ENTRIES].tolist()
    }, f)