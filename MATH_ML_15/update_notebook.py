
import json
import os

notebook_path = '/home/pavel/IDE/skillfactory/MATH_ML_15/DST_MATH_ML_15_unit6.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The code to inject
code_lines = [
    "### Решение - Задание 6.2\n",
    "\n",
    "# Создаем матрицу взаимодействий\n",
    "users_items_pivot_matrix_df = interactions_train_df.pivot(index='personId', \n",
    "                                                          columns='contentId', \n",
    "                                                          values='eventStrength').fillna(0)\n",
    "\n",
    "# Преобразуем в массив numpy\n",
    "ratings_matrix = users_items_pivot_matrix_df.values\n",
    "\n",
    "# Находим среднее арифметическое\n",
    "global_mean = np.mean(ratings_matrix)\n",
    "print(f\"Global Mean: {global_mean}\")\n",
    "print(f\"Rounded Global Mean: {global_mean:.3f}\")"
]

# Find the last cell. If it is the one we want to modify (contains "### Решение - Задание 6.2"), we update it.
# Otherwise we append a new one.

cells = nb['cells']
if cells:
    last_cell = cells[-1]
    # Check if this is the target cell (heuristically)
    # The previous view showed the last cell had source: ["### Решение - Задание 6.2\n"]
    if last_cell['cell_type'] == 'code' and len(last_cell['source']) > 0 and "### Решение - Задание 6.2" in last_cell['source'][0]:
         last_cell['source'] = code_lines
         print("Updated existing last cell.")
    else:
        # Append new cell
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code_lines
        }
        cells.append(new_cell)
        print("Appended new cell.")
else:
     # Append new cell if empty (unlikely)
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code_lines
    }
    cells.append(new_cell)
    print("Appended new cell to empty notebook.")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook saved successfully.")
