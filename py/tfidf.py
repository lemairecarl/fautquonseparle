import numpy as np
import sklearn.feature_extraction.text
from sklearn.metrics.pairwise import cosine_similarity

def load_stopwords(filename):
    stopwords = []
    with open(filename, 'r') as f:
        for ligne in f.readlines():
            stopwords.append(ligne.strip())
    return stopwords


def load_answers():
    questions_id = []
    questions_answer = []
    questions_body = []
    with open('fautquonseparle.txt', 'r', encoding='utf-16') as f:
        for ligne in f.readlines():
            if ligne.strip() == '':
                continue
            ligne = ligne.replace('\\', '')
            if '\t' in ligne:
                q_id, q_answer, q_body = ligne.split('\t')
                questions_id.append(int(q_id))
                questions_answer.append(int(q_answer))
                questions_body.append(q_body.strip())
            else:
                # Assumer que la reponse continue
                questions_body[-1] += '\n' + ligne.strip()
    return questions_id, questions_answer, questions_body

stopwords = load_stopwords('stopwords-fr.txt')

questions_id, questions_answer, questions_body = load_answers()

D = 2048

# Texts to matrix of representations
print('\nAnalyzing corpus')
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(stop_words=stopwords)
qbody_repr = vectorizer.fit_transform(questions_body)
qbody_repr = qbody_repr.toarray()
index_to_word = {}
for word, idx in vectorizer.vocabulary_.items():
    index_to_word[idx] = word

# Keep best words
print('\nFinding best words')
word_importance = np.linalg.norm(qbody_repr, axis=0, ord=2)
best_words = np.argsort(word_importance)[::-1][:D]
# num_show = 4
# for widx in best_words[:num_show]:
#     print("{:<5} {:<20} {}".format(widx, index_to_word[widx], word_importance[widx]))

# Reduce dimensionality
doc_repr = qbody_repr[:, best_words]


def print_nonzero_words(repr):
    # Get words that are active in the tfidf repr
    kept_idx = np.argwhere(np.ravel(repr))
    if len(kept_idx) == 0:
        return False
    for x in np.ravel(kept_idx):
        print(str(np.ravel(repr)[x]) + ' ' + index_to_word[best_words[x]])
    return True


# Contient 'privatisation'
# q_index = questions_id.index(5089)
# q_repr = doc_repr[q_index]
# print('')
# print(questions_body[q_index])
# print_nonzero_words(q_repr)

def print_similar_to_query(query):
    # Transform query
    print('\nTransforming query: ' + query)
    query_repr = vectorizer.transform([query]).toarray()
    q_prime = query_repr[:, best_words]
    print('Words kept:')
    if not print_nonzero_words(q_prime):
        print('Essayez avec des mots plus communs.\n')
        return
    
    
    # Compute all similarities
    print('\nComputing similarities')
    simil = cosine_similarity(q_prime, doc_repr, dense_output=True)  # --> (1, num_docs)
    assert simil.shape == (1, doc_repr.shape[0])
    simil = simil.ravel()
    print('Resultats: ' + str(np.sum(simil > 0.1)))
    matches = np.argsort(simil)[::-1]
    for m in matches[:10]:
        if simil[m] < 0.001:
            break
        print('----------')
        print('Simil:', simil[m])
        #print('element 60 (love):', doc_repr[m, 60])
        #print_nonzero_words(doc_repr[m])
        #print('=====')
        print(questions_body[m])
        #print('----------')
        

print('')
while True:
    q = input('Requête: ')
    print_similar_to_query(q)