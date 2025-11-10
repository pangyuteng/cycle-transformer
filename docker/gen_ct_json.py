
import os
import sys
import json
import pandas as pd
from pathlib import Path

root = sys.argv[1]

train_csv_file = sys.argv[2]
df = pd.read_csv(train_csv_file)

file_list = [str(x) for x in Path(root).rglob("*")]
file_list = [x for x in file_list if "DICOMDIR" not in x and not os.path.isdir(x)]

mydict = {}
category_list = ["ARTERIAL","NATIVE","VENOUS"]
for category in category_list:
    if category not in mydict.keys():
        mydict[category] = []
    for file_path in file_list:
        if any([x in file_path for x in df.train.to_list()]) is False:
            continue
        if category in file_path:
            mydict[category].append(file_path)

with open('ct_data_train.json','w') as f:
    f.write(json.dumps(mydict, indent=4, sort_keys=True))

"""

python gen_ct_json.py \
    /radraid/pteng-public/Coltea-Lung-CT-100W/Coltea-Lung-CT-100W \
    /radraid/pteng-public/Coltea-Lung-CT-100W/train_data.csv

cp ct_data_train.json /radraid/pteng-public/Coltea-Lung-CT-100W

"""