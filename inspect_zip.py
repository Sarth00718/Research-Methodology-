import zipfile
from PIL import Image
import io
import pandas as pd
import json

zip_path = r'd:\RMS\dataset.zip'

info = {'structure': {}, 'features': {}}

def summarize_structure(d, max_items=5):
    summary = {}
    for k, v in d.items():
        if isinstance(v, dict):
            if v:
                keys = list(v.keys())
                if len(keys) > max_items:
                    v_summary = {k: "file" if v[k] == "file" else summarize_structure(v[k], max_items) for k in keys[:max_items]}
                    v_summary[f"... and {len(keys) - max_items} more files/dirs"] = ""
                    summary[k] = v_summary
                else:
                    summary[k] = summarize_structure(v, max_items)
            else:
                summary[k] = v
        else:
            summary[k] = v
    return summary


try:
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_list = z.namelist()
        
        # Build raw structure
        raw_structure = {}
        for f in file_list:
            parts = f.split('/')
            curr = raw_structure
            for i, part in enumerate(parts):
                if not part: continue
                if i == len(parts) - 1 and not f.endswith('/'):
                    curr[part] = "file"
                else:
                    if part not in curr:
                        curr[part] = {}
                    curr = curr[part]

        info['structure'] = summarize_structure(raw_structure)

        # Analyze features
        image_checked = False
        csv_checked = False
        for f in file_list:
            if f.endswith('.csv') and not csv_checked:
                with z.open(f) as csv_file:
                    try:
                        df = pd.read_csv(csv_file, nrows=5)
                        info['features'][f'csv_preview ({f})'] = {
                            'columns': list(df.columns),
                            'shape': "unknown (use full read for shape)"
                        }
                        csv_checked = True
                    except Exception as e:
                        info['features'][f'csv_error ({f})'] = str(e)
            if (f.lower().endswith('.jpg') or f.lower().endswith('.png') or f.lower().endswith('.jpeg')) and not image_checked:
                with z.open(f) as img_file:
                    try:
                        img = Image.open(img_file)
                        info['features'][f'image_preview ({f})'] = {
                            'format': img.format,
                            'size': img.size,
                            'mode': img.mode
                        }
                        image_checked = True
                    except Exception as e:
                        info['features'][f'image_error ({f})'] = str(e)

except Exception as e:
    info['error'] = str(e)

print(json.dumps(info, indent=2))
