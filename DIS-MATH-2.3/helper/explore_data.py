# Импорт библиотек
import numpy as np # для работы с массивами
import pandas as pd # для работы с DataFrame 
import seaborn as sns # библиотека для визуализации статистических данных
import matplotlib.pyplot as plt # для построения графиков

#%matplotlib inline

from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Настройки для более красивых графиков - исправленная версия
sns.set_theme(style="whitegrid")  # Используем встроенный стиль seaborn
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

CORRELATION_MATRIX_THRESHOLD = 0.4

# Если нужен конкретный стиль matplotlib, можно использовать:
# plt.style.use('default')  # или другие доступные стили

def explore_data(df: pd.DataFrame) -> None:
    """
    Проводит комплексный разведывательный анализ данных
    
    Parameters:
    -----------
    df : pd.DataFrame
        Датафрейм для анализа
    """
    
    # 1. Общая информация о данных
    print("1. Общая информация о данных:")
    print("\nРазмерность датафрейма:", df.shape)
    print("\nТипы данных:\n", df.dtypes)
    display("Описательная статистика:", df.describe())
    
    # 2. Проверка пропущенных значений
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("\nПропущенные значения:\n", missing_values[missing_values > 0])
        
        # Визуализация пропущенных значений
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
        plt.title('Тепловая карта пропущенных значений')
        plt.show()
    
    # 3. Распределение числовых признаков
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Гистограммы и ящики с усами
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.ravel()
    
    for idx, col in enumerate(numeric_cols):
        # Гистограмма
        sns.histplot(data=df, x=col, kde=True, ax=axes[idx])
        axes[idx].set_title(f'Распределение {col}')
        
        # Добавляем информацию о скосе и куртозисе
        skew = stats.skew(df[col].dropna())
        kurt = stats.kurtosis(df[col].dropna())
        axes[idx].text(0.05, 0.95, f'Скос: {skew:.2f}\nКуртозис: {kurt:.2f}', 
                      transform=axes[idx].transAxes, 
                      bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # 4. Корреляционный анализ
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm',
                center=0)
    plt.title('Корреляционная матрица')
    plt.show()
    
    # 5. Попарные графики рассеяния для наиболее коррелированных признаков
    # Находим пары с высокой корреляцией
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i,j]) > CORRELATION_MATRIX_THRESHOLD:  # порог корреляции
                high_corr_pairs.append((
                    correlation_matrix.columns[i], 
                    correlation_matrix.columns[j],
                    correlation_matrix.iloc[i,j]
                ))
    
    # Строим графики рассеяния для высококоррелированных пар
    if high_corr_pairs:
        n_pairs = len(high_corr_pairs)
        n_cols = 2
        n_rows = (n_pairs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_pairs > 1:
            axes = axes.ravel()
        
        for idx, (col1, col2, corr) in enumerate(high_corr_pairs):
            ax = axes[idx] if n_pairs > 1 else axes
            sns.scatterplot(data=df, x=col1, y=col2, ax=ax)
            ax.set_title(f'Корреляция: {corr:.2f}')
        
        plt.tight_layout()
        plt.show()
    
    # 6. Выявление выбросов через z-score
    z_scores = pd.DataFrame()
    for col in numeric_cols:
        z_scores[col] = np.abs(stats.zscore(df[col]))
    
    print("\nКоличество выбросов (z-score > 3) по признакам:")
    print((z_scores > 3).sum())
    
    # 7. Boxplots для выявления выбросов

    # plt.figure(figsize=(15, 6))
    # df[numeric_cols].boxplot()
    # plt.xticks(rotation=45)
    # plt.title('Диаграммы размаха для числовых признаков')
    # plt.show()
    
    plt.figure(figsize=(15, 6))
    
    # Нормализуем данные
    normalized_data = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    
    sns.boxplot(data=normalized_data)
    plt.xticks(rotation=45)
    plt.title('Диаграммы размаха (нормализованные данные)')
    plt.ylabel('Стандартизованные значения')
    plt.grid(True, alpha=0.3)
    plt.show()