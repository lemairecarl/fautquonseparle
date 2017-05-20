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
    
    with open('w2v.pkl', 'rb') as f:
        ans_embed = pickle.load(f)

    # raw ans_question: 4-17
    ans_question = np.array(ans_question)
    ans_question -= 4  # --> [0, 13] (10 + oublie qqch + 3 inconnues)
    
    # Remove invalid classes
    valid_idx = ans_question <= 9
    ans_question = ans_question[valid_idx]
    ans_embed = ans_embed[valid_idx]
    num_classes = 10
    assert len(np.unique(ans_question)) == num_classes
    
    # Shuffle the stuff
    rand_idx = np.arange(len(ans_question))
    np.random.shuffle(rand_idx)
    ans_embed = ans_embed[rand_idx]
    ans_question = ans_question[rand_idx]
    
    # Convert to categorical
    ans_question_cat = to_categorical(ans_question, num_classes)
    
    n_total = len(ans_question)
    n_train = int(n_total * 0.6)
    n_val = int(n_train * 0.1)
    n_test = n_total - n_train - n_val
    print('N train, val, test: {} {} {}'.format(n_train, n_val, n_test))
    
    data = {
        'X_train': ans_embed[:n_train],
        'y_train': ans_question_cat[:n_train],
        'X_val': ans_embed[n_train:n_train + n_val],
        'y_val': ans_question_cat[n_train:n_train + n_val],
        'X_test': ans_embed[-n_test:],
        'y_test': ans_question_cat[-n_test:]
    }
    
    for k, v in data.items():
        print(k, v.shape)
    
    # Vérifier qu'il n'y a pas d'overlap
    print('Vérifier overlap...')
    def verif_overlap(a, b):
        nb_cas = 0
        for i, x1 in enumerate(a):
            print('\r' + str(i), end='')
            for x2 in b:
                if np.all(x1 == x2):
                    nb_cas += 1
                    break
        print('')
        return nb_cas
    n_cas_val = verif_overlap(data['X_train'], data['X_val'])
    n_cas_test = verif_overlap(data['X_train'], data['X_test'])
    print("Cas d'overlap: Val {}  Test {}".format(n_cas_val, n_cas_test))
    
    print('Bin count train: ', np.bincount(np.argmax(data['y_train'], axis=1)))
    print('Bin count val:   ', np.bincount(np.argmax(data['y_val'], axis=1)))
    print('Bin count test:  ', np.bincount(np.argmax(data['y_test'], axis=1)))
    
    train_dict = defaultdict(list)
    for x, y in zip(ans_embed, ans_question):
        train_dict[y].append(x)
    
    def gen():
        while True:
            x_batch, y_batch = [], []
            for c in np.random.choice(num_classes, size=batch_size):
                idx = np.random.choice(len(train_dict[c]))
                x_batch.append(train_dict[c][idx])
                y_batch.append(to_categorical(c, num_classes).ravel())
            yield np.array(x_batch), np.array(y_batch)
    
    return data, num_classes, gen
    

def build_model(num_classes):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(64,)))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile('adam', 'categorical_crossentropy', metrics=['acc'])
    return model


def train(model: Sequential, data, gen=None, verbose=1, epochs=10):
    print('Training...')
    if gen:
        print('Using gen')
        history = model.fit_generator(gen(), 100, epochs=epochs, verbose=verbose,
                                      validation_data=(data['X_val'], data['y_val']))
    else:
        print('Using array')
        history = model.fit(data['X_train'], data['y_train'],
                            validation_data=(data['X_val'], data['y_val']),
                            verbose=verbose)
    return np.max(history.history['val_acc'])


if __name__ == '__main__':
    data, num_classes, gen = load_data()
    
    model = build_model(num_classes)
    
    val_acc = train(model, data, gen)
    print(val_acc)
