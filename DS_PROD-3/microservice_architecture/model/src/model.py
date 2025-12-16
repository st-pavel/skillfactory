import pika
import pickle
import numpy as np
import json

# Абсолютный путь к файлу модели для надежности
pkl_path = '/home/pavel/IDE/skillfactory/DS_PROD-3/microservice_architecture/model/src/myfile.pkl'
# Читаем файл с сериализованной моделью
with open(pkl_path, 'rb') as pkl_file:
    regressor = pickle.load(pkl_file)

# Создаём подключение к серверу на локальном хосте:
connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

# Объявляем очередь features
channel.queue_declare(queue='features')
# Объявляем очередь y_pred
channel.queue_declare(queue='y_pred')

# Создаём функцию callback для обработки данных из очереди features
def callback(ch, method, properties, body):
    print(f'Получен вектор признаков {body}')
    
    # 1. Десериализуем сообщение
    features = json.loads(body)
    
    # 2. Приводим к нужному формату (1, n_features)
    shaped_features = np.array(features).reshape(1, -1)
    
    # 3. Делаем предсказание
    prediction = regressor.predict(shaped_features)[0]
    
    # 4. Публикуем предсказание в очередь y_pred
    channel.basic_publish(exchange='',
                          routing_key='y_pred',
                          body=json.dumps(prediction))
    
    # 5. Выводим сообщение в консоль
    print(f'Предсказание {prediction} отправлено в очередь y_pred')

# Извлекаем сообщение из очереди features
# on_message_callback показывает, какую функцию вызвать при получении сообщения
channel.basic_consume(
    queue='features',
    on_message_callback=callback,
    auto_ack=True
)
print('...Ожидание сообщений, для выхода нажмите CTRL+C')

# Запускаем режим ожидания прихода сообщений
channel.start_consuming()