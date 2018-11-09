from keras.layers import Embedding, Flatten, Dense, Input
from keras.models import Model
import pickle as pkl
import numpy as np


def load_part_dataset(filename):
    f = open(filename, 'rb')
    partition = pkl.load(f)
    f.close()
    X = partition['dataset']
    y = partition['labels']
    return X, y


file_example = 'trainset_part_ind0.pkl'
X, y = load_part_dataset(file_example)
part_size, feature_size = X.shape
label_size = y.shape[1]

input_bytes = Input(shape=(feature_size, ))
emb = Embedding(input_dim=257, output_dim=8,
                input_length=feature_size)(input_bytes)
flatten = Flatten()(emb)
sm = Dense(label_size, activation='softmax')(flatten)
model = Model(inputs=input_bytes, outputs=sm)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
num_parts = 2
for i in np.random.random_integers(0, num_parts - 1, num_parts):
    X, y = load_part_dataset('trainset_part_ind%d.pkl' % i)
    model.fit(X, y, batch_size=20, epochs=2)
