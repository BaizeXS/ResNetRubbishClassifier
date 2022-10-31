import json


with open('annotations.json', 'r') as f:
    obj = json.load(f)
    for imgs in obj.values():
        for i in range(len(imgs)):
            imgs[i] = imgs[i].replace('\\', '/')
    with open('new_annotations.json', 'w') as nf:
        json.dump(obj, f)
    print("Annotation successfully transformed.")

with open('train.json', 'r') as f:
    obj = json.load(f)
    for item in obj:
        item['path'] = item['path'].replace('\\', '/')
    with open('new_train.json', 'w') as nf:
        json.dump(obj, nf)
    print("Train successfully transformed.")

with open('val.json', 'r') as f:
    obj = json.load(f)
    for item in obj:
        item['path'] = item['path'].replace('\\', '/')
    with open('new_val.json', 'w') as nf:
        json.dump(obj, nf)
    print("Val successfully transformed.")
