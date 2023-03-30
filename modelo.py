import json , random , nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.layers import LSTM
from tensorflow.keras.optimizers import legacy as legacy_optimizer





nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
intents_file = open('ungs_dataset.json', encoding='utf8')
intents_data = json.load(intents_file)


words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

for intents in intents_data['intents']:
    for patterns in intents['patterns']:
        word_list = nltk.word_tokenize(patterns)
        words.extend(word_list)
        documents.append((word_list, intents['tag']))
        if intents['tag'] not in classes:
            classes.append(intents['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

optimizer = legacy_optimizer.Adam(lr=0.0001, decay=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(np.array(train_x), np.array(train_y), epochs=3000, batch_size=5, verbose=1)

model.save('universidad_chatbot_model.h5', save_format='h5')
print('Modelo de chatbot de la universidad creado.')