import json

nb_path = r"c:\Users\stepu\OneDrive\IDE\skillfactory\Блок 8\MATH&ML-13. Временные ряды. Часть II\project Ghana GDP redo.ipynb"
out_path = r"c:\Users\stepu\OneDrive\IDE\skillfactory\Блок 8\MATH&ML-13. Временные ряды. Часть II\notebook_code.py"

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    code_lines = []
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            code_lines.append(source)
            code_lines.append("\n\n")

    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines(code_lines)
    
    print(f"Code extracted to {out_path}")

except Exception as e:
    print(f"Error: {e}")
