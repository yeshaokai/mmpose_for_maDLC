import pandas as pd
import pickle
import json

def extract_uncropped_name(filename):

    f = filename.split('/')[-1]
    video_source = filename.split('/')[-2]
    video_source = video_source.replace('_cropped','')
    
    image_format = f.split('.')[-1]
    image_prefix = f.split('c')[0]
    new_name = video_source+'_'+image_prefix+'.'+image_format
    return new_name


csv_path='CollectedData_Daniel.csv'
all_data = pd.read_csv(csv_path)

for shuffle in [0,1,2]:

    docu_path = 'Documentation_data-MultiMouse_95shuffle{}.pickle'.format(shuffle)

    f = open(docu_path,'rb')
    a = pickle.load(f)

    train_indices = a[1]
    test_indices = a[2]



    data = all_data.iloc[3:,0].to_numpy()


    train_data = data[train_indices]
    test_data = data[test_indices]


    train_data_set = set()
    test_data_set = set()
    for e in test_data:
        test_data_set.add(extract_uncropped_name(e))
    for e in train_data:
        train_data_set.add(extract_uncropped_name(e))

    print ('train dataset')
    #print (train_data_set)
    print (len(train_data_set))

    print ('test dataset')
    #print (test_data_set)
    print (len(test_data_set))


    ret_obj = {}

    ret_obj['train_data'] = list(train_data_set)
    ret_obj['test_data'] = list(test_data_set)


    with open('3mouse_shuffule{}.json'.format(shuffle),'w') as f:
        json.dump(ret_obj,f)
