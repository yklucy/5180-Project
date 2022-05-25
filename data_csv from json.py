import json
import pandas as pd
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# read train data and output to the csv file
with open("/Volumes/KUN/Python/CSI5180Project/ATIS_dataset-master/data/standard_format/rasa/train.json","r") as f:
    data = json.load(f)

#get the items - text, intent, entities
core_data = data['rasa_nlu_data']['common_examples']

# build array [['text','intent']]
atis_data = np.empty(shape=[0,2],dtype=str)

for i in range(0,len(core_data)):
    atis_data = np.append(atis_data,[[core_data[i]['text'],core_data[i]['intent']]],axis=0)

atis_data = pd.DataFrame(atis_data)
#atis_data.columns = ['text','intent']

atis_data.to_csv("/Volumes/KUN/Python/CSI5180Project/atis_train.csv",index=0, header=0)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# read test data and output to the csv file
with open("/Volumes/KUN/Python/CSI5180Project/ATIS_dataset-master/data/standard_format/rasa/test.json","r") as f:
    test_data = json.load(f)

#get the items - text, intent, entities
core_test_data = test_data['rasa_nlu_data']['common_examples']

# build array [['text','intent']]
atis_test_data = np.empty(shape=[0,2],dtype=str)

for i in range(0,len(core_test_data)):
    atis_test_data = np.append(atis_test_data,[[core_test_data[i]['text'],core_test_data[i]['intent']]],axis=0)

atis_test_data = pd.DataFrame(atis_test_data)
#atis_test_data.columns = ['text','intent']

atis_test_data.to_csv("/Volumes/KUN/Python/CSI5180Project/atis_test.csv",index=0, header=0)