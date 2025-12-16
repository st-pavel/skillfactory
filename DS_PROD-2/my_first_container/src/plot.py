import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
 
# Создаём бесконечный цикл
while True:
    # Блок обработки исключений на случай неверного ввода
    try:
        # Считываем среднее с клавиатуры 
        mean = input('Введите среднее для нормального распределения:')
        # Пытаемся преобразовать введённую строку в число float
        mean = float(mean)
        # Если код преобразования во float отработал, выходим из цикла
        break
    # Если возникло исключение, выводим предупреждение об ошибке
    except:
        print('Это не число!')
# Создаём бесконечный цикл
while True:
    # Блок обработки исключений на случай неверного ввода
    try:
        # Считываем стандартное отклонение с клавиатуры
        deviation = input('Введите стандартное отклонение для нормального распределения:')
        # Пытаемся преобразовать введённую строку в число float
        deviation = float(deviation)
        # Если код преобразования во float отработал, выходим из цикла
        break
    # Если возникло исключение, выводим предупреждение об ошибке
    except:
        print('Это не число!')
# Генерируем стандартное нормальное распределение
distribution_n1 = np.random.normal(0,1,1000)
# Генерируем нормальное распределение с параметрами, введёнными пользователем
# Дополнительно умножаем результат на 2            
distribution_n2 = np.random.normal(mean,deviation,1000)*2      
 
sns_plot  = sns.histplot(distribution_n1, kde=True, color="orange")
sns_plot  = sns.histplot(distribution_n2, kde=True, color="skyblue")

# Получаем абсолютный путь к папке, где лежит скрипт (src)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Формируем путь к папке output
output_dir = os.path.join(current_dir, 'output')
# Создаем папку, если её вдруг нет (хорошая практика)
os.makedirs(output_dir, exist_ok=True)
# Сохраняем график по надежному пути
plt.savefig(os.path.join(output_dir, 'plot.png'))


#plt.savefig('./output/plot.png')
 
print('Файл успешно сохранен')