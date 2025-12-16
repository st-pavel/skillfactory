import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell
import os

nb_path = '/home/pavel/IDE/skillfactory/DS_PROD-4/DS_PROD-4_001.ipynb'
if not os.path.exists(nb_path):
    print(f"Notebook not found at {nb_path}")
    exit(1)

nb = nbformat.read(nb_path, as_version=4)

# Determine where to edit. We appended the previous solution at the end.
# We will just append the "Corrected" version or overwrite the last cell if it looks like the previous task.
# Simpler: just append a new cell clarifying the use of 'alternative'.

md_source = "### Решение Задания 5.2 (Уточнение)\n\nИспользуем параметр `alternative='smaller'`, чтобы проверить гипотезу о том, что конверсия в группе А меньше, чем в группе B."
nb.cells.append(new_markdown_cell(md_source))

code_source = """from statsmodels.stats.proportion import proportions_ztest

# Рассчитываем количество успехов (транзакций) и наблюдений (посетителей)
successes_a = sample_a['transactions'].sum()
nobs_a = sample_a['cid'].count()

successes_b = sample_b['transactions'].sum()
nobs_b = sample_b['cid'].count()

print(f"Группа A: Успехи={successes_a}, Наблюдения={nobs_a}, Конв={successes_a/nobs_a:.4f}")
print(f"Группа B: Успехи={successes_b}, Наблюдения={nobs_b}, Конв={successes_b/nobs_b:.4f}")

# Z-тест с параметром alternative='smaller'
alpha = 0.05
stat, p_value = proportions_ztest(count=[successes_a, successes_b], nobs=[nobs_a, nobs_b], alternative='smaller')

print(f"Z-статистика: {stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < alpha:
    print("Отвергаем нулевую гипотезу: Конверсия в группе A статистически значимо меньше, чем в группе B.")
    print("Вывод: Вариант B лучше.")
else:
    print("Не удалось отвергнуть нулевую гипотезу.")
    print("Вывод: Нет оснований считать, что вариант B лучше варианта A (варианты равнозначны).")
"""
nb.cells.append(new_code_cell(code_source))

nbformat.write(nb, nb_path)
print("Notebook updated.")
