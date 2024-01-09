from flask import Flask, request
from flask_cors import CORS, cross_origin
from flask.helpers import send_from_directory
import numpy as np
from tensorflow import keras
from sklearn.feature_extraction.text import CountVectorizer
import unicodedata

app = Flask(__name__, static_folder='frontend/build', static_url_path='')
CORS(app)

@app.route('/')
def hello():
    return "hello"


'''
ML Functions
'''

# clean the name
def process_name(name):
    normalized_string = unicodedata.normalize('NFD', name)
    latin_characters = ''.join(
        char if unicodedata.category(char) == 'Mn' else char
        for char in normalized_string
    )
    
    converted_string = latin_characters.encode('ascii', 'ignore').decode('utf-8')
    processed_name = converted_string.replace(" ", "").replace("-", "")
    processed_name = processed_name.upper()
    return processed_name


# convert softmax to race
def get_race(predicted_softmax):
    predicted_class_numeric = []
    for each in predicted_softmax:
        max_value = np.argmax(each) # it finds the indices of the maximum value
        predicted_class_numeric.append(max_value)
    
    predicted_race = []
    for each in predicted_class_numeric:
        if each == 0:
            predicted_race.append('likely of Asian origin.')
        if each == 1:
            predicted_race.append('likely of African origin.')
        if each == 2:
            predicted_race.append('likely of Hispanic origin.')
        if each == 3:
            predicted_race.append('likely of underrepresented ethnic origin.')
        if each == 4:
            predicted_race.append('likely of European origin.')
    return predicted_race


# convert lastname to numeric matrix
def preprocess_x(lastnames):
    # Convert the list to a 2D list. 
    # Each last name (a string) will be converted to a list of characters
    lastnames_2d = []
    for each in lastnames:
        char_list = list(each)
        lastnames_2d.append(char_list)
    
    # create a vocabulary (upper case alphabets) to be used as tokesn for vectorizer
    upper_case_alphabets = [chr(i) for i in range(65, 91)]
    
    # vectorizer using uppercase alphabets and binary
    vectorizer = CountVectorizer(analyzer='char', binary=True, lowercase=False, vocabulary=upper_case_alphabets)

    # get an array of matrices whose dimension is (35, 26) meaning the dimension of array is (num of examples, 35, 26)
    list_matrices = []
    for name in lastnames_2d:
        target_shape = (35, 26) # target shape of each matrix
        padded_array = np.zeros(target_shape) # this is for padding
    
        numeric_data = vectorizer.transform(name) 
        numeric_array = numeric_data.toarray()
        padded_array[:numeric_array.shape[0], :] = numeric_array
        list_matrices.append(padded_array)

    preprocessed_x = np.array(list_matrices)

    return preprocessed_x

# Give a list of last names. 
# This function returns a list of predicted race using the above functions
def predict_race(name_list, model, model_type='simple'):
    name_processed = preprocess_x(name_list)
    predict = model.predict(name_processed)
    res = get_race(predict)

    predict_res = {}
    for i in range(len(name_list)):
        predict_res[name_list[i]] = res[i]
    
    return predict_res


# Load model
model = keras.models.load_model('model.h5')


@app.route('/api', methods = ['POST'])
@cross_origin()
def submit_name():
    name = request.json['name']
    name_list = [process_name(name)]
    predicted_race = predict_race(name_list, model=model)
    race = predicted_race[name_list[0]]
    return {"name": race}


if __name__ == '__main__':
    app.run()
