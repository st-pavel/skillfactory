import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell
import os

nb_path = '/home/pavel/IDE/skillfactory/DS_PROD-4/DS_PROD-4_002.ipynb'
if not os.path.exists(nb_path):
    print(f"Notebook not found at {nb_path}")
    exit(1)

nb = nbformat.read(nb_path, as_version=4)

md_source = "### Решение Задания 6.9\n\nРасчет выборочной пропорции и доверительного интервала для отзывов."
nb.cells.append(new_markdown_cell(md_source))

code_source = """import scipy.stats as st
import math

n = 189
positive = 132

# Выборочная пропорция
p_hat = positive / n
print(f"Выборочная пропорция: {p_hat:.3f}")

# 90% Доверительный интервал
gamma = 0.90
alpha = 1 - gamma
z_crit = abs(st.norm.ppf(alpha/2)) # z-критическое

margin_of_error = z_crit * math.sqrt((p_hat * (1 - p_hat)) / n)

lower = (p_hat - margin_of_error) * 100
upper = (p_hat + margin_of_error) * 100

print(f"90% Доверительный интервал: [{lower:.2f}%, {upper:.2f}%]")
"""
nb.cells.append(new_code_cell(code_source))

nbformat.write(nb, nb_path)
print("Notebook updated.")
