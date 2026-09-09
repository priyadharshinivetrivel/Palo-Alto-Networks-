import json
import os

path = r'C:\Users\vpriy\.gemini\antigravity-ide\brain\d9de630d-30d6-4be4-a8c7-fe02d414d33d\.system_generated\logs\transcript_full.jsonl'

found_csv = False
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        if 'Age,Attrition' in line:
            obj = json.loads(line)
            content = ""
            if isinstance(obj.get('content'), str):
                content = obj['content']
            elif isinstance(obj.get('content'), list):
                content = "".join([str(c) for c in obj['content']])
            else:
                content = str(obj)

            idx = content.find("Age,Attrition")
            if idx != -1:
                csv_text = content[idx:]
                lines = csv_text.splitlines()
                valid_lines = []
                for l in lines:
                    l = l.strip()
                    if l.startswith("Age,Attrition") or (l and l[0].isdigit() and ',' in l):
                        valid_lines.append(l)
                    elif valid_lines and not (l and l[0].isdigit()):
                        break
                
                os.makedirs('data', exist_ok=True)
                with open('data/employee_attrition.csv', 'w', encoding='utf-8') as out:
                    out.write('\n'.join(valid_lines))
                print(f"Extracted {len(valid_lines)-1} rows into data/employee_attrition.csv")
                found_csv = True
                break

if not found_csv:
    print("CSV not found in transcript_full.jsonl")
