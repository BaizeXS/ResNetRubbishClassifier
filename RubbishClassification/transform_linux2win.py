import json


with open('annotations.json', 'r') as f:
    obj = json.load(f)
    for imgs in obj.values():
        for i in range(len(imgs)):
            imgs[i] = imgs[i].replace('/', '\\')

with open('train.json', 'r') as f:
    obj = json.load(f)
    for item in obj:
        item['path'] = item['path'].replace('/', '\\')

with open('val.json', 'r') as f:
    obj = json.load(f)
    for item in obj:
        item['path'] = item['path'].replace('/', '\\')
