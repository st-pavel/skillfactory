

# Введите свое решение ниже
def matrix_sum(matrix1, matrix2):
    #проверка на совпадение размерности матриц
    if len(matrix1)==len(matrix2):
        for i in range(len(matrix1)):
            if len(matrix1[i])!=len(matrix2[i]):
                print('Error! Matrices dimensions are different!')
                return None
    else:
        print('Error! Matrices dimensions are different!')
        return None
    result_matrix = matrix1[:]
    for row in range(len(result_matrix)):
        for i in range(len(result_matrix[row])):
            result_matrix[row][i]+= matrix2[row][i]
    return result_matrix


matrix_example = [
          [1, 5, 4],
          [4, 2, -2],
          [7, 65, 88]
]

print(matrix_sum(matrix1=matrix_example, matrix2=matrix_example))
# [[2, 10, 8], [8, 4, -4], [14, 130, 176]]


matrix1_example = [
          [1, 5, 4],
          [4, 2, -2]
]

matrix2_example = [
          [10, 15, 43],
          [41, 2, -2],
          [7, 5, 7]
]

print(matrix_sum(matrix1=matrix1_example, matrix2=matrix2_example))
# Error! Matrices dimensions are different!
# None


matrix1_example = [
          [1, 5, 4, 5],
          [4, 2, -2, -5],
          [4, 2, -2, -5]
]

matrix2_example = [
          [10, 15, 43],
          [41, 2, -2],
          [7, 5, 7]
]

print(matrix_sum(matrix1=matrix1_example, matrix2=matrix2_example))
# Error! Matrices dimensions are different!
# None