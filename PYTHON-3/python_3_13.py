a = 1
b = 1
i = 2

n = 100

fibonacci_list = [1,1]

# while i< n:
#     b = a + b
#     a = b - a
#     i+=1
#     fibonacci_list.append(b)

while i< n:
    b, a = a + b, b
    i+=1
    fibonacci_list.append(b)



print(fibonacci_list)
