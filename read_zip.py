import zipfile
from collections import Counter

with zipfile.ZipFile('B. Disease Grading.zip', 'r') as z:
    names = z.namelist()

# Write to output file
with open('zip_contents.txt', 'w', encoding='utf-8') as f:
    f.write(f'Total entries: {len(names)}\n\n')
    f.write('=== ALL ENTRIES ===\n')
    for n in names:
        f.write(n + '\n')
    
    # Extension count
    exts = Counter()
    for n in names:
        last_part = n.split('/')[-1]
        if '.' in last_part:
            exts[last_part.rsplit('.',1)[-1]] += 1
    f.write('\n=== Extension counts ===\n')
    for k,v in exts.items():
        f.write(f'  .{k}: {v}\n')

print('Written to zip_contents.txt')
