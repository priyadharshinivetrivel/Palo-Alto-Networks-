import os
import json

# Read scratch_prompt.json
with open('scratch_prompt.json', 'r', encoding='utf-8-sig') as f:
    line = f.readline()
    data = json.loads(line)

content = data.get('content', '')

# Extract CSV content starting with Age,Attrition
csv_start = content.find("Age,Attrition,BusinessTravel")
if csv_start != -1:
    csv_data = content[csv_start:]
    lines = csv_data.splitlines()
    clean_lines = []
    for l in lines:
        l_str = l.strip()
        if l_str and (l_str.startswith("Age,Attrition") or (l_str[0].isdigit() and ',' in l_str)):
            clean_lines.append(l_str)
        elif clean_lines and not (l_str and l_str[0].isdigit()):
            break

    os.makedirs('data', exist_ok=True)
    with open('data/employee_attrition.csv', 'w', encoding='utf-8') as out:
        out.write("\n".join(clean_lines))
    print(f"Extracted {len(clean_lines)-1} rows of employee attrition dataset to data/employee_attrition.csv.")
else:
    print("CSV header not found in prompt json.")
