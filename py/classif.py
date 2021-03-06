import pickle
import numpy as np
from collections import defaultdict
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.utils import to_categorical


def load_data(batch_size=32):
    # Load fqsp data
    with open('fqsp.pkl', 'rb') as f:
        ans_id, ans_question, ans_body = pickle.load(f)
    
    with open('dualrepr.pkl', 'rb') as f:
        ans_embed, ans_counts = pickle.load(f)

    ans_counts = np.array(ans_counts.todense())

    # raw ans_question: 4-17
    ans_question = np.array(ans_question)
    ans_question -= 4  # --> [0, 13] (10 + oublie qqch + 3 inconnues)
    
    # Remove invalid classes
    valid_idx = ans_question <= 9
    ans_question = ans_question[valid_idx]
    ans_embed = ans_embed[valid_idx]
    num_classes = 10
    assert len(np.unique(ans_question)) == num_classes
    
    # Make complete dual data for producing vectors
    dual_data = [ans_embed, ans_counts]
    
    # Shuffle the stuff
    rand_idx = np.arange(len(ans_question))
    np.random.shuffle(rand_idx)
    ans_embed = ans_embed[rand_idx]
    ans_question = ans_question[rand_idx]
    ans_counts = ans_counts[rand_idx]
    
    # Convert to categorical
    ans_question_cat = to_categorical(ans_question, num_classes)
    
    n_total = len(ans_question)
    n_train = int(n_total * 0.6)
    n_val = int(n_train * 0.1)
    n_test = n_total - n_train - n_val
    print('N train, val, test: {} {} {}'.format(n_train, n_val, n_test))
    
    data = {
        'X_train': [ans_embed[:n_train], ans_counts[:n_train]],
        'y_train': ans_question_cat[:n_train],
        'X_val': [ans_embed[n_train:n_train + n_val], ans_counts[n_train:n_train + n_val]],
        'y_val': ans_question_cat[n_train:n_train + n_val],
        'X_test': [ans_embed[-n_test:], ans_counts[-n_test:]],
        'y_test': ans_question_cat[-n_test:]
    }
    
    #for k, v in data.items():
    #    print(k, v.shape)
    
    # Vérifier qu'il n'y a pas d'overlap
    # print('Vérifier overlap...')
    # def verif_overlap(a, b):
    #     nb_cas = 0
    #     for i, x1 in enumerate(a):
    #         print('\r' + str(i), end='')
    #         for x2 in b:
    #             if np.all(x1 == x2):
    #                 nb_cas += 1
    #                 break
    #     print('')
    #     return nb_cas
    # n_cas_val = verif_overlap(data['X_train'], data['X_val'])
    # n_cas_test = verif_overlap(data['X_train'], data['X_test'])
    # print("Cas d'overlap: Val {}  Test {}".format(n_cas_val, n_cas_test))
    #
    # print('Bin count train: ', np.bincount(np.argmax(data['y_train'], axis=1)))
    # print('Bin count val:   ', np.bincount(np.argmax(data['y_val'], axis=1)))
    # print('Bin count test:  ', np.bincount(np.argmax(data['y_test'], axis=1)))
    
    embed_train_dict = defaultdict(list)
    count_train_dict = defaultdict(list)
    for x1, x2, y in zip(ans_embed, ans_counts, ans_question):
        embed_train_dict[y].append(x1)
        count_train_dict[y].append(x2)
    
    def gen():
        while True:
            x_batch, y_batch = [], []
            for c in np.random.choice(num_classes, size=batch_size):
                idx = np.random.choice(len(embed_train_dict[c]))
                x_batch.append(embed_train_dict[c][idx])
                y_batch.append(to_categorical(c, num_classes).ravel())
            yield np.array(x_batch), np.array(y_batch)
            
    def gen_dual():
        while True:
            x1_batch, x2_batch, y_batch = [], [], []
            for c in np.random.choice(num_classes, size=batch_size):
                idx = np.random.choice(len(embed_train_dict[c]))
                x1_batch.append(embed_train_dict[c][idx])
                x2_batch.append(count_train_dict[c][idx])
                y_batch.append(to_categorical(c, num_classes).ravel())
            yield [np.array(x1_batch), np.array(x2_batch)], np.array(y_batch)
    
    return data, num_classes, gen, gen_dual, dual_data


def train(model: Sequential, data, gen=None, verbose=1, epochs=10):
    print('Training...')
    if gen:
        history = model.fit_generator(gen(), 100, epochs=epochs, verbose=verbose,
                                      validation_data=(data['X_val'], data['y_val']))
    else:
        history = model.fit(data['X_train'], data['y_train'],
                            validation_data=(data['X_val'], data['y_val']),
                            verbose=verbose)
    
    return history


def reembed(dual_data, model):
    pass