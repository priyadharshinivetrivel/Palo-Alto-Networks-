import json
import os

path = r'C:\Users\vpriy\.gemini\antigravity-ide\brain\d9de630d-30d6-4be4-a8c7-fe02d414d33d\.system_generated\logs\transcript_full.jsonl'

with open(path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if '41,1,Travel_Rarely,1102' in line:
            print(f"Found line {i}")
            # Search for header
            header_idx = line.find("Age,Attrition,BusinessTravel")
            if header_idx != -1:
                sub = line[header_idx:]
                # Split by escaped newlines or actual newlines
                raw_lines = sub.replace('\\n', '\n').split('\n')
                csv_rows = []
                for row in raw_lines:
                    row_clean = row.strip().strip('"').strip("'")
                    if row_clean.startswith("Age,Attrition") or (row_clean and row_clean[0].isdigit() and ',' in row_clean):
                        csv_rows.append(row_clean)
                    elif csv_rows and not (row_clean and row_clean[0].isdigit()):
                        break
                
                os.makedirs('data', exist_ok=True)
                with open('data/employee_attrition.csv', 'w', encoding='utf-8') as out:
                    out.write('\n'.join(csv_rows))
                print(f"Successfully extracted {len(csv_rows)-1} rows into data/employee_attrition.csv")
                break
