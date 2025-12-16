# расчет совершенных чисел до заданного предела по формуле, связанной с простыми числами Мерсенна

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def find_perfect_numbers(limit):
    perfect_numbers = []
    p = 2
    while True:
        mersenne_prime = 2**p - 1
        if mersenne_prime > limit:
            break
        if is_prime(mersenne_prime):
            perfect_number = 2**(p-1) * mersenne_prime
            if perfect_number <= limit:
                perfect_numbers.append(perfect_number)
        
        # Отображение процента выполнения
        progress = (mersenne_prime / limit) * 100
        print(f"Progress: {progress:.2f}%", end='\r')
        
        p += 1
    return perfect_numbers

# Установите предел для поиска совершенных чисел
limit = 1_000_000_000_000_000_0000

# Найдите и выведите совершенные числа до заданного предела
perfect_numbers = find_perfect_numbers(limit)
print(f"\nСовершенные числа до {limit}: {perfect_numbers}")

#Совершенные числа до 10000000000000000000: [6, 28, 496, 8128, 33550336, 8589869056, 137438691328, 2305843008139952128]