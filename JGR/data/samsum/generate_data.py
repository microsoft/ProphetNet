import json


def generate_data(data):
    new_data = []
    for d in data:
        new_sample = {}
        new_sample['target'] = d['target']
        dialog = d['dialogue']
        new_sample['source'] = dialog
        new_data.append(new_sample)
            
    return new_data


    
with open('raw_data/train.json', encoding = 'utf-8') as f:
    train_data = json.load(f)

new_train_data = generate_data(train_data)

with open('train_data.json','w', encoding='utf-8') as f:
    json.dump(new_train_data, f)



with open('raw_data/val.json', encoding = 'utf-8') as f:
    dev_data = json.load(f)

new_dev_data = generate_data(dev_data)

with open('dev_data.json','w', encoding='utf-8') as f:
    json.dump(new_dev_data, f)



with open('raw_data/test.json', encoding = 'utf-8') as f:
   test_data = json.load(f)


new_test_data = generate_data(test_data)

with open('test_data.json','w', encoding='utf-8') as f:
    json.dump(new_test_data, f)