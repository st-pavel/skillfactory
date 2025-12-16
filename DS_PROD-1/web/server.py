from flask import Flask, request, jsonify
import pickle
import numpy as np

# загружаем модель из файла
with open('models/model.pkl', 'rb') as pkl_file:
    model = pickle.load(pkl_file)

# создаём приложение
app = Flask(__name__)

@app.route('/')
def index():
    msg = "Test message. The server is running"
    return msg

@app.route('/predict', methods=['POST'])
def predict():
    # Получаем данные из запроса
    features = request.json
    
    # Преобразуем в numpy array и меняем размерность
    features = np.array(features).reshape(1, 4)
    
    # Делаем предсказание
    prediction = model.predict(features)
    
    # Извлекаем число из результата (prediction[0])
    result = prediction[0]
    
    # Возвращаем результат в формате JSON
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run('localhost', 5000)