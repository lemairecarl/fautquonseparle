import pickle
import re
from collections import defaultdict

import numpy as np
import sklearn.feature_extraction.text
from sklearn.metrics.pairwise import cosine_similarity
from unidecode import unidecode
from tqdm import tqdm

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
        #self.vectorizer = sklearn.feature_extraction.text.CountVectorizer(encoding='utf-8', stop_words=stopwords,
        #                                                                  vocabulary=word_id)
        #self.ans_counts = self.vectorizer.transform(self.ans_body)
        self.vectorizer = sklearn.feature_extraction.text.CountVectorizer(encoding='utf-8', stop_words=stopwords)
        self.ans_counts = self.vectorizer.fit_transform(self.ans_body)
        id_to_word = self.vectorizer.get_feature_names()
        word_counts = np.array(np.sum(self.ans_counts, axis=0)).squeeze()
        
        print('Enlever nombres')
        contains_digit_re = re.compile(r'\d')
        for i1, w1 in enumerate(id_to_word):
            if contains_digit_re.search(w1):
                word_counts[i1] = 0
        
        print('Singulier-pluriel')
        # Combiner singulier-pluriel. Lorsqu'un mot avec un 's' existe sans 's', combiner avec la version singulier.
        sub_pluriel = {}  # clef: a remplacer; valeur: remplacer par
        n_cas = 0
        for i1, w1 in enumerate(id_to_word):
            if w1[-1] != 's':
                continue
            for i2, w2 in enumerate(id_to_word):
                if w1[:-1] == w2:
                    sub_pluriel[i1] = i2
                    word_counts[i2] += word_counts[i1]
                    word_counts[i1] = 0
                    n_cas += 1
        print("Cas: " + str(n_cas))
        
        print("Remplacer les accents absurdes")
        # Remplacer les accents absurdes þ_
        wtf = dict(zip('õīęėěūá', 'ôîeéêûâ'))
        wtf['ľ'] = ''
        wtf['ď'] = ''
        wtf['ß'] = ''
        wtf['æ'] = 'ae'
        wtf['þ'] = ''
        wtf['_'] = ''
        wtf['œ'] = 'oe'
        sub_accents_exotiques = {}
        n_cas = 0
        for i1, w1 in enumerate(id_to_word):
            if any(l in w1 for l in wtf):
                new_w = None
                for old, new in wtf.items():
                    new_w = w1.replace(old, new)
                if new_w in self.vectorizer.vocabulary_:
                    new_i = self.vectorizer.vocabulary_[new_w]
                    sub_accents_exotiques[i1] = new_i
                    word_counts[new_i] += word_counts[i1]
                else:
                    word_counts[i1] = -1
                    print('Suppr: ' + w1 + ' (' + new_w + ')')
                word_counts[i1] = 0
                n_cas += 1
        print("Cas: " + str(n_cas))
        
        print("Vérifier...")
        # Vérifier
        for i1, w1 in enumerate(id_to_word):
            if word_counts[i1] == 0:
                continue
            racine = w1[:-1] if w1[-1] == 's' else w1
            assert not any(i not in 'abcdefghijklmnopqrstuvwxyzéèàùçâîêôû1234567890' for i in racine[:-1])

        print('Oubli accents')
        # Oubli d'accent
        accents_exceptions = ['cote', 'eleve', 'eleve' 'forme', 'marque', 'prive', 'necessite']
        sub_accents = {}
        uni_words = defaultdict(list)
        dictionnaire_sans_accent = {unidecode(k): v for k, v in word_id.items()}
        for i1, w1 in enumerate(id_to_word):
            if word_counts[i1] == 0:  # or w1 in word_id:
                continue
            uw1 = unidecode(w1)
            if uw1 in accents_exceptions or (uw1[-1] == 's' and uw1[:-1] in accents_exceptions):
                continue
            uni_words[uw1].append(w1)

        log = []
        for uniw, duplicates in uni_words.items():
            if len(duplicates) == 1:
                #del uni_words[uniw]  # TODO Pas de substitution a faire
                continue
            # Verifier qui na pas dembedding
            has_emb = [w in word_id for w in duplicates]
            if all(has_emb):
                continue  # Tout le monde a un embedding! c'est la fete
            num_emb = sum(has_emb)
            ont_emb = [i for i, he in enumerate(has_emb) if he]
            pas_emb = [i for i, he in enumerate(has_emb) if not he]
            # TODO METTRE DANS  sub_accents !!!!
            if num_emb == 1:
                # Subst. tous par un
                gagnant = ont_emb[0]
                log.append('{:<20} remplace1 {}\n'.format(duplicates[gagnant], str(duplicates)))
            else:
                # Choisir le mot qui va remplacer ceux qui n'en ont pas
                if uniw in dictionnaire_sans_accent:
                    gagnant = dictionnaire_sans_accent[uniw]
                    print('{:<20} remplace! {}\n'.format(self.id_word[gagnant],
                                                         str([duplicates[i] for i in pas_emb])))
                    # TODO INTEGRER CECI
                else:
                    if num_emb > 0:
                        gagnant = ont_emb[0]
                    else:
                        gagnant = 0
                    log.insert(0, '{:<20} remplace0 {}\n'.format(duplicates[gagnant],
                                                                 str([duplicates[i] for i in pas_emb])))
        with open('oubli-accent.txt', 'w') as f:
            for l in log:
                f.write(l)
        
        print('Avant: ' + str(len(word_counts)))
        sorted_idx = np.argsort(word_counts)
        sorted_counts = word_counts[sorted_idx]
        non_zero = sorted_counts > 0
        del sorted_counts
        sorted_idx = sorted_idx[non_zero]
        print('Après: ' + str(len(sorted_idx)))
        
        with open('mots.txt', 'w') as f:
            for m in sorted_idx:
                f.write('{:<3} {}\n'.format(word_counts[m], id_to_word[m]))

        # print('Embedding answers...')
        # self.ans_embed = self.embed_counts_matrix()
        
        # Save embeddings
        #with open('w2v.pkl', 'wb') as f:
        #    pickle.dump(self.ans_embed, f, pickle.HIGHEST_PROTOCOL)
        
        # Save fqsp data
        #with open('fqsp.pkl', 'wb') as f:
        #    o = (ans_id, ans_question, self.ans_body)
        #    pickle.dump(o, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    embedder = Embedder()
    
    #while True:
    #    print('')
    #    q = input('Requête: ')
    #    embedder.print_similar_to_query(q)
