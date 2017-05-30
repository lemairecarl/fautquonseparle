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
    def __init__(self):
        stopwords = self.load_stopwords('stopwords-fr.txt')
        emb_words, self.emb_to_id, self.id_to_emb, self.embeddings = self.load_embeddings()  # emb: mot ayant un embedding
        ans_id, ans_question, self.ans_body = self.load_answers()
        
        # Constituer un vocabulaire initial (contenant des fautes d'orthographe et des pluriels)
        self.vectorizer = sklearn.feature_extraction.text.CountVectorizer(encoding='utf-8', stop_words=stopwords)
        self.ans_counts = self.vectorizer.fit_transform(self.ans_body)
        self.id_to_word = self.vectorizer.get_feature_names()
        
        # Réduire le vocabulaire
        self.substitutions = {}
        self.mots_restants = dict(enumerate(self.id_to_word))
        self.retirer_nombres()
        self.fusionner_pluriel()
        self.remplacer_accents_exotiques()
        sub_embed = self.substitution_accents()
        
        # Convertir en matrice dense avant de faire les substitutions
        self.ans_counts = self.ans_counts.todense()
        
        print('Effectuer les substitutions')
        # Effectuer les substitutions
        for ans_i in tqdm(range(self.ans_counts.shape[0])):
            for old, new in self.substitutions.items():
                self.ans_counts[ans_i, new] += self.ans_counts[ans_i, old]
                self.ans_counts[ans_i, old] = 0
        
        # Effectuer les substitution orthographiques
        sub_corr = self.vocab_correction()
        self.substitution_orthographe(sub_corr)  # Modifie ans_counts directement
        
        # Faire le pont entre les deux vocabulaires. id_voc_dataset -> id_voc_embed
        self.data_to_emb = {}
        for m in self.mots_restants:
            emb_i = sub_embed.get(m, None)  # D'abord traiter les substitution mot_dataset -> mot_embed
            if emb_i is None:
                emb_i = self.emb_to_id.get(self.id_to_word[m], 0)  # 0 correspond a l'embedding nul (vecteur nul)
            self.data_to_emb[m] = emb_i
            
        # Enlever les mots retirés
        mots_enleves = np.array([m for m in range(len(self.id_to_word)) if m not in self.mots_restants])
        self.ans_counts[:, mots_enleves] = 0
        
        self.analyser_vocabulaire()
        
        print("Faire l'embedding des réponses...")
        self.ans_embed = self.embed_counts_matrix()
        
        # Save embeddings
        with open('w2v.pkl', 'wb') as f:
            pickle.dump(self.ans_embed, f, pickle.HIGHEST_PROTOCOL)

    def vocab_correction(self):
        # Charger corrections orthographiques
        with open('correction.txt', 'r') as f:
            lignes = list(f.readlines())
    
        sub_corr = {}
        new_word_id = sorted(self.vectorizer.vocabulary_.values())[-1] + 1
        num_new_words = 0
    
        # Batir dict de substitution, et ajouter les mots au dict vocab
        for l in lignes:
            old, new = l.strip().split(',')
            new_words = new.split(' ')
            new_ids = []
            if new_words != ['']:
                for w in new_words:
                    wid = self.vectorizer.vocabulary_.get(w, None)
                    if wid is None:
                        self.vectorizer.vocabulary_[w] = new_word_id
                        wid = new_word_id
                        new_word_id += 1
                        num_new_words += 1
                        self.ajouter(wid, w)
                    new_ids.append(wid)
            sub_corr[self.vectorizer.vocabulary_[old]] = new_ids

        # Mettre a jour relation inverse
        self.id_to_word = self.vectorizer.get_feature_names()
    
        # Ajouter les nouveaux mots a ans_counts
        nw, nf = self.ans_counts.shape
        new_counts = np.zeros((nw, nf + num_new_words), dtype=np.int8)
        new_counts[:, :nf] = self.ans_counts
        self.ans_counts = new_counts
    
        return sub_corr

    def substitution_orthographe(self, sub_corr):
        for old, new in sub_corr.items():
            ans_touches = np.argwhere(self.ans_counts[:, old] != 0)
            c = self.ans_counts[ans_touches, old]
            self.ans_counts[ans_touches, old] = 0
            self.ans_counts[ans_touches, new] = c  # new est une liste
     
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
        
        embeddings[0] = np.zeros((embeddings.shape[1],))  # Utiliser comme embedding nul
        
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
            print(str(np.ravel(repr)[x]) + ' ' + self.id_to_emb[x])
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
            mots = np.argwhere(counts_vec.flat).ravel()
            for m in mots:
                emb_i = self.data_to_emb.get(m, None)
                if emb_i is None:
                    raise Exception('Cant find emb corresp to valid word {} {}'.format(m, self.id_to_word[m]))
                else:
                    ans_embed[i, :] += self.embeddings[emb_i, :]
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
    
    def substituer(self, old, new):
        # 1. Échanger partout où 'old' est un remplacant
        a_echanger = [x for x in self.substitutions if self.substitutions[x] == old]
        for orig in a_echanger:
            self.substitutions[orig] = new
        # 2. Ajouter la substitution
        self.substitutions[old] = new
        # 3. Modifiers mots restants
        self.retirer(old)
    
    def retirer(self, old):
        self.mots_restants.pop(old, None)
    
    def ajouter(self, i, w):
        self.mots_restants[i] = w
    
    def iter_mots_restants(self):
        """Permet d'iterer sur le dict en le modifiant"""
        keys = list(self.mots_restants.keys())
        for k in keys:
            yield k, self.mots_restants[k]
    
    def retirer_nombres(self):
        print('Retirer nombres')
        contains_digit_re = re.compile(r'\d')
        for i1, w1 in self.iter_mots_restants():
            if contains_digit_re.search(w1):
                self.retirer(i1)
    
    def fusionner_pluriel(self):
        print('Fusionner pluriel avec singulier')
        # Combiner singulier-pluriel. Lorsqu'un mot avec un 's' existe sans 's', combiner avec la version singulier.
        n_cas = 0
        mots_avec_s = []
        for i, w in self.iter_mots_restants():
            if w[-1] == 's':
                mots_avec_s.append(i)
        for i in mots_avec_s:
            w = self.id_to_word[i]
            new_i = self.vectorizer.vocabulary_.get(w[:-1], None)
            if new_i is not None:
                self.substituer(i, new_i)
                n_cas += 1
        print("Cas: " + str(n_cas))
     
    def analyser_vocabulaire(self):
        word_counts = np.array(np.sum(self.ans_counts, axis=0)).squeeze()
        nb_mots_suppr = np.sum(word_counts == 0)
        sorted_i = np.argsort(word_counts)[nb_mots_suppr:]  # Exclure les mots supprimés
        nouv_mots = []
        with open('mots.txt', 'w') as f:
            f.write('Mots avant: {}\n'.format(len(self.id_to_word)))
            f.write('Mots apres: {}\n'.format(len(self.mots_restants)))
            for m in sorted_i:
                indic_nouv = ''
                if self.data_to_emb[m] == 0:
                    indic_nouv = 'NOUV'
                    nouv_mots.append(m)
                f.write('{:<3} {:<20} {}\n'.format(word_counts[m], self.id_to_word[m], indic_nouv))
        with open('nouv.txt', 'w') as f:
            for m in nouv_mots:
                f.write('{:<4} {:<3} {}\n'.format(m, word_counts[m], self.mots_restants[m]))

    def substitution_accents(self):
        print('Oubli accents')
        # Oubli d'accent
        accents_exceptions_generales = {
            'eleve': 'élève',
            'eleves': 'élèves'
        }
        accents_exceptions_specifiques = {
            'necessites': 'nécessités',
            'nécessites': 'nécessités',
            'necessités': 'nécessités',
        }
        sub_embed = {}  # index_mon_vocab --> index_embed
        uni_words = defaultdict(list)
        dictionnaire_sans_accent = {unidecode(k): v for k, v in self.emb_to_id.items()}
        for i1, w1 in self.iter_mots_restants():
            uw1 = unidecode(w1)
            if uw1 in accents_exceptions_generales:
                new_w = accents_exceptions_generales[uw1]
                new_i = self.vectorizer.vocabulary_[new_w]
                self.substituer(i1, new_i)
                continue
            if w1 in accents_exceptions_specifiques:
                new_w = accents_exceptions_specifiques[w1]
                new_i = self.vectorizer.vocabulary_[new_w]
                self.substituer(i1, new_i)
                continue
            # Si le mot a un embedding, pas de problème.
            if w1 in self.emb_to_id:
                continue
            # Si un embedding a le meme unicode, on lui donne cet embedding.
            new_emb_i = dictionnaire_sans_accent.get(uw1, None)
            if new_emb_i is not None:
                sub_embed[i1] = new_emb_i
                self.retirer(i1)
                continue
            uni_words[uw1].append(i1)
        log = []
        for uniw, duplicates in uni_words.items():
            if len(duplicates) == 1:
                # Pas de substitution a faire
                continue
        
            graphies = [self.id_to_word[i] for i in duplicates]
            le_remplacant = max(graphies)  # Celui qui a le plus d'accent! On aime ça les accents!
            le_remplacant_i = self.vectorizer.vocabulary_[le_remplacant]
            les_remplaces = []
            for i in duplicates:
                if i == le_remplacant_i:
                    continue
                self.substituer(i, le_remplacant_i)
                les_remplaces.append(self.id_to_word[i])
            log.append('{:<20} remplace {}\n'.format(le_remplacant,
                                                     str(' '.join(les_remplaces))))
        with open('oubli-accent.txt', 'w') as f:
            for l in log:
                f.write(l)
    
        return sub_embed

    def verifier_accents_exotiques(self):
        print("Vérifier...")
        # Vérifier s'il y a encore des mots avec des accents exotiques
        for i1, w1 in self.iter_mots_restants():
            if any(l not in 'abcdefghijklmnopqrstuvwxyzéèàùçâîêôûï1234567890' for l in w1):
                raise Exception('Un mot a un accent exotique: ' + w1)
        print('OK!')

    def remplacer_accents_exotiques(self):
        print("Remplacer les accents exotiques")
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
        n_cas = 0
        for i1, w1 in self.iter_mots_restants():
            if any(l in wtf for l in w1):
                new_w = None
                for old, new in wtf.items():
                    new_w = w1.replace(old, new)
                new_i = self.vectorizer.vocabulary_.get(new_w, None)
                if new_i is not None:
                    self.substituer(i1, new_i)
                else:
                    self.retirer(i1)
                    print('Supprimé: ' + w1 + ' (' + new_w + ')')
                n_cas += 1
        print("Cas: " + str(n_cas))

        self.verifier_accents_exotiques()


if __name__ == '__main__':
    embedder = Embedder()
    
    #while True:
    #    print('')
    #    q = input('Requête: ')
    #    embedder.print_similar_to_query(q)
