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
        words, word_id, self.id_word, self.embeddings = self.load_embeddings()  # TODO renommer!!!!
        ans_id, ans_question, self.ans_body = self.load_answers()
        
        # Texts to matrix of representations
        #self.vectorizer = sklearn.feature_extraction.text.CountVectorizer(encoding='utf-8', stop_words=stopwords,
        #                                                                  vocabulary=word_id)
        #self.ans_counts = self.vectorizer.transform(self.ans_body)
        self.vectorizer = sklearn.feature_extraction.text.CountVectorizer(encoding='utf-8', stop_words=stopwords)
        self.ans_counts = self.vectorizer.fit_transform(self.ans_body)
        id_to_word = self.vectorizer.get_feature_names()
        word_counts = np.array(np.sum(self.ans_counts, axis=0)).squeeze()
        # TODO combiner toutes les substitutions
        print('Enlever nombres')
        contains_digit_re = re.compile(r'\d')
        for i1, w1 in enumerate(id_to_word):
            if contains_digit_re.search(w1):
                word_counts[i1] = 0
        
        print('Singulier-pluriel')
        # Combiner singulier-pluriel. Lorsqu'un mot avec un 's' existe sans 's', combiner avec la version singulier.
        sub_pluriel = {}  # clef: a remplacer; valeur: remplacer par
        n_cas = 0
        mots_avec_s = []
        for i, w in enumerate(id_to_word):
            if w[-1] == 's':
                mots_avec_s.append(i)
        for i in mots_avec_s:
            w = id_to_word[i]
            new_i = self.vectorizer.vocabulary_.get(w[:-1], None)
            if new_i is not None:
                sub_pluriel[i] = new_i
                word_counts[new_i] += word_counts[i]
                word_counts[i] = 0
                n_cas += 1
        print("Cas: " + str(n_cas))
        
        print("Remplacer les accents absurdes")
        # Remplacer les accents absurdes
        wtf = dict(zip('õīęėěūáůñ', 'ôîeéêûâûn'))
        wtf.update({
            'ľ': '',
            'ď': '',
            'ß': '',
            'þ': '',
            '_': '',
            'œ': 'oe',
            'æ': 'ae',
        })
        sub_accents_exotiques = {}
        n_cas = 0
        for i1, w1 in enumerate(id_to_word):
            if any(l in wtf for l in w1):
                new_w = None
                for old, new in wtf.items():
                    new_w = w1.replace(old, new)
                new_i = self.vectorizer.vocabulary_.get(new_w, None)
                if new_i is not None:
                    sub_accents_exotiques[i1] = new_i
                    word_counts[new_i] += word_counts[i1]
                else:
                    print('Suppr: ' + w1 + ' (' + new_w + ')')
                word_counts[i1] = 0
                n_cas += 1
        print("Cas: " + str(n_cas))
        
        print("Vérifier...")
        # Vérifier s'il y a encore des mots avec des accents exotiques
        for i1, w1 in enumerate(id_to_word):
            if word_counts[i1] == 0:
                continue
            if any(l not in 'abcdefghijklmnopqrstuvwxyzéèàùçâîêôûï1234567890' for l in w1):
                raise Exception('Un mot a un accent exotique: ' + w1)
        print('OK!')

        print('Oubli accents')
        # Oubli d'accent
        accents_exceptions_generales = {  # TODO ajouter aux substitutions
            'eleve': 'élève',
            'eleves': 'élèves'
        }
        accents_exceptions_specifiques = {  # TODO ajouter aux substitutions
            'necessites': 'nécessités',
            'nécessites': 'nécessités',
            'necessités': 'nécessités',
        }
        sub_accents = {}
        sub_embed = {}  # TODO index_mon_vocab --> index_embed
        uni_words = defaultdict(list)
        dictionnaire_sans_accent = {unidecode(k): v for k, v in word_id.items()}
        for i1, w1 in enumerate(id_to_word):
            if word_counts[i1] == 0:
                continue
            uw1 = unidecode(w1)
            if uw1 in accents_exceptions_generales:
                new_w = accents_exceptions_generales[uw1]
                new_i = self.vectorizer.vocabulary_[new_w]
                sub_accents[i1] = new_i
                word_counts[i1] = 0
                continue
            if w1 in accents_exceptions_specifiques:
                new_w = accents_exceptions_specifiques[w1]
                new_i = self.vectorizer.vocabulary_[new_w]
                sub_accents[i1] = new_i
                word_counts[i1] = 0
                continue
            # Si le mot a un embedding, pas de problème.
            if w1 in word_id:
                continue
            # Si un embedding a le meme unicode, on lui donne cet embedding.
            new_emb_i = dictionnaire_sans_accent.get(uw1, None)
            if new_emb_i is not None:
                sub_embed[i1] = new_emb_i
                word_counts[i1] = 0
                continue
            uni_words[uw1].append(i1)

        log = []
        for uniw, duplicates in uni_words.items():
            if len(duplicates) == 1:
                # Pas de substitution a faire
                continue
            
            graphies = [id_to_word[i] for i in duplicates]
            le_remplacant = max(graphies)  # Celui qui a le plus d'accent! On aime ça les accents!
            le_remplacant_i = self.vectorizer.vocabulary_[le_remplacant]
            les_remplaces = []
            for i in duplicates:
                if i == le_remplacant_i:
                    continue
                sub_accents[i] = le_remplacant_i
                word_counts[le_remplacant_i] += word_counts[i]
                word_counts[i] = 0
                les_remplaces.append(id_to_word[i])
            log.append('{:<20} remplace {}\n'.format(le_remplacant,
                                                     str(' '.join(les_remplaces))))
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
