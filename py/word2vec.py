import pickle
import re
import numpy as np
import sklearn.feature_extraction.text
from sklearn.metrics.pairwise import cosine_similarity


DIGITS = re.compile("[0-9]", re.UNICODE)


def load_stopwords(filename):
    stopwords = []
    with open(filename, 'r') as f:
        for ligne in f.readlines():
            stopwords.append(ligne.strip())
    return stopwords


def load_answers():
    ans_id = []
    ans_question = []
    ans_body = []
    with open('fautquonseparle.txt', 'r', encoding='utf-8') as f:
        for ligne in f.readlines():
            if ligne.strip() == '':
                continue
            ligne = ligne.replace('\\', '')
            if '\t' in ligne:
                q_id, q_answer, q_body = ligne.split('\t')
                ans_id.append(int(q_id))
                ans_question.append(int(q_answer))
                ans_body.append(q_body.strip())
            else:
                # Assumer que la reponse continue
                ans_body[-1] += '\n' + ligne.strip()
    return ans_id, ans_question, ans_body


def load_embeddings():
    # Load embeddings
    with open('embeddings/polyglot-fr.pkl', 'rb') as f:
        words, embeddings = pickle.load(f, encoding='latin1')
    
    # Special tokens
    Token_ID = {"<UNK>": 0, "<S>": 1, "</S>": 2, "<PAD>": 3}
    ID_Token = {v: k for k, v in Token_ID.items()}
    
    # Map words to indices and vice versa
    word_id = {w: i for (i, w) in enumerate(words)}
    id_word = dict(enumerate(words))
    
    return words, word_id, id_word, embeddings


def print_nonzero_words(repr):
    # Get words that are active in the tfidf repr
    kept_idx = np.argwhere(np.ravel(repr))
    if len(kept_idx) == 0:
        return False
    for x in np.ravel(kept_idx):
        print(str(np.ravel(repr)[x]) + ' ' + id_word[x])
    return True


def embed_phrase(phrase):
    counts = vectorizer.transform([phrase])
    counts = counts.toarray().ravel()
    mots = np.argwhere(counts).ravel()
    ans_embed = np.zeros((embeddings.shape[1],))
    for m in mots:
        ans_embed += embeddings[m]
    return ans_embed


def embed_counts_matrix():
    ans_embed = np.zeros((ans_counts.shape[0], embeddings.shape[1]))
    for i, counts_vec in enumerate(ans_counts):
        mots = np.argwhere(counts_vec).ravel()
        for m in mots:
            ans_embed[i] += embeddings[m]
    return ans_embed


def simil(w1, w2):
    return float(cosine_similarity([w1], [w2]).ravel())


def print_similar_to_query(query):
    # Transform query
    query_embed = embed_phrase(query)
    if np.linalg.norm(query_embed, ord=1) < 1e-4:
        print('Mots inconnus')
        return
    
    # Compute all similarities
    simil = cosine_similarity([query_embed], ans_embed, dense_output=True)  # --> (1, num_docs)
    assert simil.shape == (1, ans_embed.shape[0])
    simil = simil.ravel()
    print('Resultats: ' + str(np.sum(simil > 0.1)))
    matches = np.argsort(simil)[::-1]
    for m in matches[:10]:
        if simil[m] < 0.01:
            break
        print('----------')
        print('Simil:', simil[m])
        print(ans_body[m])


stopwords = load_stopwords('stopwords-fr.txt')
words, word_id, id_word, embeddings = load_embeddings()
ans_id, ans_question, ans_body = load_answers()

# Texts to matrix of representations
vectorizer = sklearn.feature_extraction.text.CountVectorizer(encoding='utf-8', stop_words=stopwords,
                                                             vocabulary=word_id)
ans_counts = vectorizer.transform(ans_body)
print('Embedding answers...')
ans_embed = embed_counts_matrix()

# Save embeddings
with open('w2v.pkl', 'wb') as f:
    pickle.dump(ans_embed, f, pickle.HIGHEST_PROTOCOL)

# Save fqsp data
with open('fqsp.pkl', 'wb') as f:
    o = (ans_id, ans_question, ans_body)
    pickle.dump(o, f, pickle.HIGHEST_PROTOCOL)

while True:
    print('')
    q = input('Requête: ')
    print_similar_to_query(q)
