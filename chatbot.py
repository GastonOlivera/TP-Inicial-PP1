import telebot , json , random , nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from keras.layers import LSTM
from tensorflow.keras.optimizers import legacy as legacy_optimizer
from nltk.tokenize import word_tokenize
from fuzzywuzzy import process , fuzz

 



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

# Cargar el archivo con las preguntas y respuestas
intents = json.load(open('ungs_dataset.json', encoding='utf-8'))

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
    return np.array(bag_of_words)

# Función para obtener la respuesta adecuada
def get_response(prediction,text_predict):
    result = model.predict(np.array([prediction]))
    max_index = np.argmax(result)
    confiar = result[0][result.argmax()]
    if confiar < 0.7:
        return "Lo siento, no entiendo lo que estás diciendo."
    tag = intents['intents'][max_index]
    response = "aca seteo algo"
    for intent in intents['intents']:
        tagActual= "".join(intent['tag'])
        if tagActual == tag['tag'] :
            response = get_best_response(text_predict , intent ['patterns'], intent['responses'])
    return response





def get_best_response(message, patterns, responses):
    # Encuentra el patrón con el mejor match con el mensaje del usuario
    if len(responses) == 1:
       return responses[0]
   
    message = " ".join(message)
    
    tokenized_patterns = [word_tokenize(pattern.lower()) for pattern in patterns]
    best_match = process.extractOne(message, tokenized_patterns,  scorer=fuzz.token_set_ratio)[0]
    best_match = " ".join(best_match).rstrip('?')
    if best_match[-1] == " ":
        best_match = best_match[:-1]
    
    patterns_lowercase = [pattern.lower().rstrip('?') for pattern in patterns]
    # Encuentra el índice del mejor match
    index = patterns_lowercase.index(best_match)
    
    # Devuelve la respuesta correspondiente al índice encontrado
    return responses[index]




 

# Función para manejar los mensajes del usuario
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    text = message.text
    text_predict= preprocess_text(text)
    prediction = predict_intent(text)
    response = get_response(prediction,text_predict)
    bot.reply_to(message, response)
    

# Ejecutar el bot
bot.polling()
