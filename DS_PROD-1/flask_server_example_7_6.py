import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Загружаем модель при запуске приложения
with open('model.pkl', 'rb') as pkl_file:
    model = pickle.load(pkl_file)

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
    print("Server is starting...")
    app.run(port=5000)
