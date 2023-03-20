import telebot
import json
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.layers import LSTM
from tensorflow.keras.optimizers import legacy as legacy_optimizer


nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
intents_file = open('preguntas_respuestas_bot.json', encoding='utf8')
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
model.add(LSTM(128, input_shape=(len(train_x[0]), 1)))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

optimizer = legacy_optimizer.Adam(lr=0.001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(np.array(train_x), np.array(train_y), epochs=3000, batch_size=32, verbose=1)

model.save('universidad_chatbot_model.h5', save_format='h5')

print('Modelo de chatbot de la universidad creado.')


# Cargar el archivo con las preguntas y respuestas
intents = json.load(open('preguntas_respuestas_bot.json'))

# Crear el lematizador
lemmatizer = WordNetLemmatizer()

# Crear un objeto de Telegram bot
bot = telebot.TeleBot("5924283235:AAH2aWjFTHkR1cwGe6132h9U4Eo4EjXjnZM")

# Función para preprocesar el texto del usuario
def preprocess_text(text):  
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

# Función para hacer una predicción con el modelo de Keras
def predict_intent(text):
    processed_text = preprocess_text(text)
    bag_of_words = [0] * len(words)
    for w in processed_text:
        for i, word in enumerate(words):
            if word == w:
                bag_of_words[i] = 1
    bag_of_words = np.array(bag_of_words)
    result = model.predict(np.array([bag_of_words]))[0]
    return result

# Función para obtener la respuesta adecuada
def get_response(prediction):
    max_index = np.argmax(prediction)
    if prediction[max_index] < 0.6:
        return "Lo siento, no entiendo lo que estás diciendo."
    intent = intents['intents'][max_index]
    if intent['tag'] == 'saludos':
        response = random.choice(intent['responses'])
    elif intent['tag'] == 'informacion_general':
        response =  random.choice(intent['responses'])
    elif intent['tag'] == 'programas_estudio':
        response = random.choice(intent['responses'])
    elif intent['tag'] == 'admission':
        response = "Para información de admisión, por favor visite nuestro sitio web"
    else:
        response = "Lo siento, no puedo responder esa pregunta en este momento."
    return response

# Función para manejar los mensajes del usuario
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    text = message.text
    prediction = predict_intent(text)
    response = get_response(prediction)
    bot.reply_to(message, response)
    


# Ejecutar el bot
bot.polling()
