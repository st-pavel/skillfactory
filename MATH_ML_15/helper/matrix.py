import numpy as np #для матричных вычислений
import pandas as pd #для анализа и предобработки данных

def check_matrix_multiplication(A: np.ndarray, B: np.ndarray) -> tuple:
    """
    Проверяет возможность умножения матриц и определяет правильный порядок.
    
    Parameters:
    -----------
    A, B : np.ndarray
        Матрицы для проверки
        
    Returns:
    --------
    tuple
        (можно ли умножить, порядок умножения, размерность результата)
        Порядок умножения: 'AB' или 'BA' или None если умножение невозможно
    """
    
    # Получаем размерности матриц
    a_rows, a_cols = A.shape
    b_rows, b_cols = B.shape
    
    # Проверяем оба варианта умножения
    can_multiply_AB = a_cols == b_rows
    can_multiply_BA = b_cols == a_rows
    
    if can_multiply_AB and can_multiply_BA:
        # Если возможны оба варианта, возвращаем оба
        return True, ['AB', 'BA'], [(a_rows, b_cols), (b_rows, a_cols)]
    elif can_multiply_AB:
        return True, 'AB', (a_rows, b_cols)
    elif can_multiply_BA:
        return True, 'BA', (b_rows, a_cols)
    else:
        return False, None, None

def multiply_matrices(A: np.ndarray, B: np.ndarray) -> tuple:
    """
    Выполняет умножение матриц в правильном порядке, если это возможно.
    
    Parameters:
    -----------
    A, B : np.ndarray
        Матрицы для умножения
        
    Returns:
    --------
    tuple
        (результат умножения, использованный порядок)
        Если умножение невозможно, возвращает (None, None)
    """
    
    can_multiply, order, result_shape = check_matrix_multiplication(A, B)
    
    if not can_multiply:
        print(f"Умножение невозможно!")
        print(f"Размерности матриц: A{A.shape}, B{B.shape}")
        return None, None
    
    if isinstance(order, list):
        print(f"Возможны оба варианта умножения:")
        print(f"A×B -> размерность {result_shape[0]}")
        print(f"B×A -> размерность {result_shape[1]}")
        return [A @ B, B @ A], order
    
    result = B @ A if order == 'BA' else A @ B
    print(f"Выполнено умножение в порядке {order}")
    print(f"Размерность результата: {result.shape}")
    
    return result, order

# Пример использования:
if __name__ == "__main__":
    # Создаем тестовые матрицы
    A = np.array([[1, 9, 8, 5], 
                  [3, 6, 3, 2], 
                  [3, 3, 3, 3], 
                  [0, 2, 5, 9], 
                  [4, 4, 1, 2]])
    
    B = np.array([[1, -1, 0, 1, 1], 
                  [-2, 0, 2, -1, 1]])
    
    # Проверяем возможность умножения
    result, order = multiply_matrices(A, B)
    
    if result is not None:
        print("\nРезультат:")
        print(result)