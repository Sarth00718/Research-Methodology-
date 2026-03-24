import zipfile
from collections import Counter
import sys

with zipfile.ZipFile('B. Disease Grading.zip', 'r') as z:
    with z.open('B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv') as f:
        content = f.read().decode('utf-8')
    with z.open('B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv') as f:
        content2 = f.read().decode('utf-8')

train_grades = []
for line in content.strip().split('\n')[1:]:
    parts = line.strip().split(',')
    if len(parts) >= 2 and parts[1].strip().isdigit():
        train_grades.append(int(parts[1].strip()))

test_grades = []
for line in content2.strip().split('\n')[1:]:
    parts = line.strip().split(',')
    if len(parts) >= 2 and parts[1].strip().isdigit():
        test_grades.append(int(parts[1].strip()))

train_counter = Counter(train_grades)
test_counter = Counter(test_grades)
grade_names = {0: 'No DR', 1: 'Mild NPDR', 2: 'Moderate NPDR', 3: 'Severe NPDR', 4: 'Proliferative DR'}

with open('dataset_stats.txt', 'w', encoding='utf-8') as out:
    out.write('=== TRAINING SET ===\n')
    out.write(f'Total images: {len(train_grades)}\n')
    for g in sorted(train_counter):
        out.write(f'  Grade {g} ({grade_names[g]}): {train_counter[g]} images ({100*train_counter[g]/len(train_grades):.1f}%)\n')
    
    out.write('\n=== TESTING SET ===\n')
    out.write(f'Total images: {len(test_grades)}\n')
    for g in sorted(test_counter):
        out.write(f'  Grade {g} ({grade_names[g]}): {test_counter[g]} images ({100*test_counter[g]/len(test_grades):.1f}%)\n')
    
    all_grades = train_grades + test_grades
    all_counter = Counter(all_grades)
    out.write(f'\n=== COMBINED DATASET ===\n')
    out.write(f'Total images: {len(all_grades)}\n')
    for g in sorted(all_counter):
        out.write(f'  Grade {g} ({grade_names[g]}): {all_counter[g]} images ({100*all_counter[g]/len(all_grades):.1f}%)\n')

print('Written to dataset_stats.txt')
