import pickle
import re
import numpy as np
import sklearn.feature_extraction.text
from sklearn.metrics.pairwise import cosine_similarity

DIGITS = re.compile("[0-9]", re.UNICODE)


class Embedder:
    def load_stopwords(self, filename):
        stopwords = []
        with open(filename, 'r') as f:
            for ligne in f.readlines():
                stopwords.append(ligne.strip())
        return stopwords
    
    def load_answers(self):
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
    
    def load_embeddings(self):
        # Load embeddings
        with open('../embeddings/polyglot-fr.pkl', 'rb') as f:
            words, embeddings = pickle.load(f, encoding='latin1')
        
        # Special tokens
        Token_ID = {"<UNK>": 0, "<S>": 1, "</S>": 2, "<PAD>": 3}
        ID_Token = {v: k for k, v in Token_ID.items()}
        
        # Map words to indices and vice versa
        word_id = {w: i for (i, w) in enumerate(words)}
        id_word = dict(enumerate(words))
        
        return words, word_id, id_word, embeddings
    
    def print_nonzero_words(self, repr):
        # Get words that are active in the tfidf repr
        kept_idx = np.argwhere(np.ravel(repr))
        if len(kept_idx) == 0:
            return False
        for x in np.ravel(kept_idx):
            print(str(np.ravel(repr)[x]) + ' ' + self.id_word[x])
        return True
    
    def embed_phrase(self, phrase):
        counts = self.vectorizer.transform([phrase])
        counts = counts.toarray().ravel()
        mots = np.argwhere(counts).ravel()
        ans_embed = np.zeros((self.embeddings.shape[1],))
        for m in mots:
            ans_embed += self.embeddings[m]
        return ans_embed
    
    def embed_counts_matrix(self):
        ans_embed = np.zeros((self.ans_counts.shape[0], self.embeddings.shape[1]))
        for i, counts_vec in enumerate(self.ans_counts):
            mots = np.argwhere(counts_vec).ravel()
            for m in mots:
                ans_embed[i] += self.embeddings[m]
        return ans_embed
    
    def simil(self, w1, w2):
        return float(cosine_similarity([w1], [w2]).ravel())
    
    def print_similar_to_query(self, query):
        # Transform query
        query_embed = self.embed_phrase(query)
        if np.linalg.norm(query_embed, ord=1) < 1e-4:
            print('Mots inconnus')
            return
        
        # Compute all similarities
        simil = cosine_similarity([query_embed], self.ans_embed, dense_output=True)  # --> (1, num_docs)
        assert simil.shape == (1, self.ans_embed.shape[0])
        simil = simil.ravel()
        print('Resultats: ' + str(np.sum(simil > 0.1)))
        matches = np.argsort(simil)[::-1]
        for m in matches[:10]:
            if simil[m] < 0.01:
                break
            print('----------')
            print('Simil:', simil[m])
            print(self.ans_body[m])
    
    def __init__(self):
        stopwords = self.load_stopwords('stopwords-fr.txt')
        words, word_id, self.id_word, self.embeddings = self.load_embeddings()
        ans_id, ans_question, self.ans_body = self.load_answers()
        
        # Texts to matrix of representations
        self.vectorizer = sklearn.feature_extraction.text.CountVectorizer(encoding='utf-8', stop_words=stopwords,
                                                                     vocabulary=word_id)
        self.ans_counts = self.vectorizer.transform(self.ans_body)
        print('Embedding answers...')
        self.ans_embed = self.embed_counts_matrix()
        
        # Save embeddings
        #with open('w2v.pkl', 'wb') as f:
        #    pickle.dump(self.ans_embed, f, pickle.HIGHEST_PROTOCOL)
        
        # Save fqsp data
        #with open('fqsp.pkl', 'wb') as f:
        #    o = (ans_id, ans_question, self.ans_body)
        #    pickle.dump(o, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    embedder = Embedder()
    
    while True:
        print('')
        q = input('Requête: ')
        embedder.print_similar_to_query(q)
