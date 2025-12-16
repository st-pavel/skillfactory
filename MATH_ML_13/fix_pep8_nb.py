import json
import re

nb_path = r"c:\Users\stepu\OneDrive\IDE\skillfactory\Блок 8\MATH&ML-13. Временные ряды. Часть II\project Ghana GDP redo.ipynb"

# Standard library modules
std_libs = {'os', 'sys', 'warnings', 'pathlib', 'json', 're', 'datetime', 'time', 'math', 'random'}

def is_std_lib(line):
    match = re.match(r'^(?:import|from)\s+(\w+)', line)
    if match:
        module = match.group(1)
        return module in std_libs
    return False

def fix_line(line):
    # Fix comments: #comment -> # comment
    if '#' in line:
        # Avoid changing #! or encoding lines or strings containing #
        # Simple heuristic: if # is followed by non-space and not part of a string (hard to detect perfectly without parser)
        # We will just look for # followed by a letter
        line = re.sub(r'#([a-zA-Zа-яА-Я])', r'# \1', line)
    
    # Fix whitespace around = for plt.rcParams
    if 'plt.rcParams' in line and '=' in line:
        line = re.sub(r'(plt\.rcParams\[[^\]]+\])\s*=\s*(.+)', r'\1 = \2', line)
        
    return line

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    imports_std = []
    imports_third = []
    
    # First pass: collect and remove imports
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                stripped = line.strip()
                # Check for import lines (not indented)
                if (stripped.startswith('import ') or stripped.startswith('from ')) and not line.startswith(' '):
                    if is_std_lib(stripped):
                        imports_std.append(line)
                    else:
                        imports_third.append(line)
                else:
                    new_source.append(line)
            cell['source'] = new_source

    # Sort imports (simple sort)
    imports_std.sort()
    imports_third.sort()
    
    # Create import cell content
    import_cell_source = []
    if imports_std:
        import_cell_source.extend(imports_std)
        import_cell_source.append('\n')
    if imports_third:
        import_cell_source.extend(imports_third)
    
    # Ensure newlines at end of import lines if missing
    import_cell_source = [line if line.endswith('\n') else line + '\n' for line in import_cell_source]

    # Create new cell
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": import_cell_source
    }
    
    # Insert at top
    data['cells'].insert(0, new_cell)
    
    # Second pass: fix formatting and empty lines
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            empty_lines_count = 0
            for line in cell['source']:
                fixed_line = fix_line(line)
                
                if fixed_line.strip() == '':
                    empty_lines_count += 1
                else:
                    # If we had empty lines, add one (unless it's the start)
                    if empty_lines_count > 0 and new_source:
                        new_source.append('\n')
                    empty_lines_count = 0
                    new_source.append(fixed_line)
            
            # Remove trailing newline if exists (optional, but cleaner)
            if new_source and new_source[-1].strip() == '':
                new_source.pop()
                
            cell['source'] = new_source

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    
    print(f"Notebook processed and saved to {nb_path}")

except Exception as e:
    print(f"Error: {e}")
