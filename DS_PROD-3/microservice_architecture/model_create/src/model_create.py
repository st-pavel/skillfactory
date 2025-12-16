import os
import sys

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
import pickle

# Загружаем датасет о диабете
X, y = load_diabetes(return_X_y=True)
# Инициализируем модель линейной регрессии
regressor = LinearRegression()
# Обучаем модель
regressor.fit(X,y)

# Производим сериализацию и записываем результат в файл формата pkl
output_path = '/home/pavel/IDE/skillfactory/DS_PROD-3/microservice_architecture/model/src/myfile.pkl'
with open(output_path, 'wb') as output:
    pickle.dump(regressor, output)
print(f'Model saved to {output_path}')

