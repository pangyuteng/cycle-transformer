
import os
import sys
import json
from pathlib import Path

root = sys.argv[1]
file_list = [str(x) for x in Path(root).rglob("*")]
file_list = [x for x in file_list if "DICOMDIR" not in x and not os.path.isdir(x)]

mydict = {}
category_list = ["ARTERIAL","NATIVE","VENOUS"]
for category in category_list:
    if category not in mydict.keys():
        mydict[category] = []
    for file_path in file_list:
        if category in file_path:
            mydict[category].append(file_path)

with open('ct_data.json','w') as f:
    f.write(json.dumps(mydict, indent=4, sort_keys=True))

"""
python gen_ct_json.py /radraid/pteng-public/Coltea-Lung-CT-100W/Coltea-Lung-CT-100W

"""