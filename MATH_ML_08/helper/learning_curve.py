import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection 


def plot_learning_curve(model, X, y, cv, scoring="f1", ax=None, title=""):
    """
    Строит кривую обучения для заданной модели машинного обучения.

    Кривая обучения показывает зависимость метрики качества модели от размера
    обучающей выборки как на тренировочных, так и на валидационных данных.

    Parameters
    ----------
    model : estimator object
        Модель машинного обучения, реализующая методы fit и predict.
    X : array-like of shape (n_samples, n_features)
        Матрица признаков.
    y : array-like of shape (n_samples,)
        Целевая переменная.
    cv : int или cross-validation generator
        Определяет стратегию кросс-валидации.
        cv может быть целым числом для определения количества фолдов
        или объектом, реализующим split метод.
    scoring : string, default="f1"
        Метрика качества для оценки модели.
        Допустимые значения: 'accuracy', 'f1', 'precision', 'recall' и др.
    ax : matplotlib.axes.Axes, optional (default=None)
        Объект осей matplotlib для построения графика.
        Если None, создается новый график.
    title : str, default=""
        Заголовок графика.

    Returns
    -------
    matplotlib.axes.Axes
        Объект осей с построенной кривой обучения.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import KFold
    >>> model = LogisticRegression()
    >>> cv = KFold(n_splits=5)
    >>> plot_learning_curve(model, X, y, cv, scoring='accuracy', title='Logistic Regression')
    """
    # Вычисляем координаты для построения кривой обучения
    train_sizes, train_scores, valid_scores = model_selection.learning_curve(
        estimator=model,  # модель
        X=X,  # матрица наблюдений X
        y=y,  # вектор ответов y
        cv=cv,  # кросс-валидатор
        scoring=scoring,  # метрика
    )
    # Вычисляем среднее значение по фолдам для каждого набора данных
    train_scores_mean = np.mean(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    # Если координатной плоскости не было передано, создаём новую
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))  # фигура + координатная плоскость
    # Строим кривую обучения по метрикам на тренировочных фолдах
    ax.plot(train_sizes, train_scores_mean, label="Train")
    # Строим кривую обучения по метрикам на валидационных фолдах
    ax.plot(train_sizes, valid_scores_mean, label="Valid")
    # Даём название графику и подписи осям
    ax.set_title("Learning curve: {}".format(title))
    ax.set_xlabel("Train data size")
    ax.set_ylabel("Score")
    # Устанавливаем отметки по оси абсцисс
    ax.xaxis.set_ticks(train_sizes)
    # Устанавливаем диапазон оси ординат
    ax.set_ylim(0, 1)
    # Отображаем легенду
    ax.legend()


def plot_probabilities_2d(X, y, model):
    """
    Визуализация вероятности класса 1 для двумерного признакового пространства.

    Args:
    X : array-like of shape (n_samples, n_features)
        Матрица признаков.
    y : array-like of shape (n_samples,)
        Целевая переменная.
    model : estimator object
        Модель машинного обучения, реализующая методы fit и predict для предсказания вероятности класса 1.
    """
    
    #Генерируем координатную сетку из всех возможных значений для признаков

    #Результат работы функции - два массива xx1 и xx2, которые образуют координатную сетку
    xx1, xx2 = np.meshgrid(
        np.arange(X.iloc[:, 0].min()-1, X.iloc[:, 0].max()+1, 0.1),
        np.arange(X.iloc[:, 1].min()-1, X.iloc[:, 1].max()+1, 0.1)
    )
    #Вытягиваем каждый из массивов в вектор-столбец — reshape(-1, 1)
    #Объединяем два столбца в таблицу с помощью hstack
    X_net = np.hstack([xx1.reshape(-1, 1), xx2.reshape(-1, 1)])
    #Предсказывает вероятность для всех точек на координатной сетке
    #Нам нужна только вероятность класса 1
    probs = model.predict_proba(X_net)[:, 1]
    #Переводим столбец из вероятностей в размер координатной сетки
    probs = probs.reshape(xx1.shape)
    #Создаём фигуру и координатную плоскость
    fig, ax = plt.subplots(figsize = (10, 5))
    #Рисуем тепловую карту вероятностей
    contour = ax.contourf(xx1, xx2, probs, 100, cmap='bwr')
    #Рисуем разделяющую плоскость — линия, где вероятность равна 0.5.
    bound = ax.contour(xx1, xx2, probs, [0.5], linewidths=2, colors='black');
    #Добавляем цветовую панель 
    colorbar = fig.colorbar(contour)
    #Накладываем поверх тепловой карты диаграмму рассеяния
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y, palette='seismic', ax=ax)
    #Даём графику название
    ax.set_title('Scatter Plot with Decision Boundary');
    #Смещаем легенду в верхний левый угол вне графика
    ax.legend(bbox_to_anchor=(-0.05, 1))