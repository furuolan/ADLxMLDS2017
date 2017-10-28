
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense,GRU,Dropout, Activation,LSTM,Embedding
from keras.utils import np_utils, to_categorical
from keras.optimizers import Adam, SGD, Adadelta
from keras.callbacks import EarlyStopping
from keras import models

modelname = sys.argv[1]
datapath = sys.argv[2]
csvpath = sys.argv[3]


# In[4]:

map39 = pd.read_csv(datapath+"phones/48_39.map",header=None,delimiter='\t')
_48to39 = dict()
for char48 in map39[0]:
    _48to39[char48] = map39.loc[map39[0]==char48][1].tolist()[0]


# In[6]:

from sklearn.preprocessing import LabelBinarizer
import pickle as pk
with open('40class.pkl', 'rb') as f:
    lb = pk.load(f)


# In[7]:

def sp(x):
    all_id = x.split('_')
    sentenceid = all_id[0]+'_'+all_id[1]
    frameid = all_id[2]
    final = [sentenceid,frameid]
    return final


# In[10]:

with open(modelname+".json",'r') as f:
    json = f.read()

model = models.model_from_json(json)
model.load_weights(modelname+".hdf5")


# In[11]:

phone=pd.read_csv(datapath+"48phone_char.map",delimiter='\t',header=None)
phone.columns=["phone","id","char"]


# In[12]:

phonetochar = dict()
for pho in phone['phone']:
    phonetochar[pho] = phone.loc[phone['phone']==pho]["char"].tolist()[0]


# In[13]:

test = pd.read_csv(datapath+"mfcc/test.ark",delimiter='\t',header=None)


# In[14]:

test_data=pd.DataFrame(test[0].apply(lambda x: x.split(' ')).values.tolist())


# In[15]:

test_id = pd.DataFrame(test_data[0].apply(sp).values.tolist())


# In[16]:

test_id.columns=["instance","frame"]


# In[17]:

test_data=test_data.drop(0,axis=1)


# In[18]:

test_final = pd.concat([test_id,test_data],axis=1)


# In[19]:

test_final['frame']=test_final['frame'].astype(int)


# In[20]:

test_final_group=test_final.groupby('instance')


# In[21]:

test_data=test_final_group.apply(lambda x: x.iloc[:,2:41].values.tolist())


# In[22]:

instance_list = test_data.index.tolist()


# In[23]:

test_data_pad = np.zeros((len(test_data),777,39))
index = 0
for inst in test_data:
    test_data_pad[index,:len(inst),:] = inst
    index+=1
    


# In[24]:

test_data_pad=test_data_pad.astype(np.float32,order='C')
result = model.predict_classes(test_data_pad,verbose=1)


# In[26]:

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(lb.classes_)


# In[27]:

result_df = pd.DataFrame(le.inverse_transform(result))
result_df = result_df.replace(to_replace='zzz',value='sil')

result_transformed=result_df.applymap(lambda x: phonetochar[x])


# In[29]:

all_list = []
for indx in range(result_transformed.shape[0]):   
    temp_list = []
    for item in result_transformed.iloc[indx,:]:
        if len(temp_list) == 0:
            temp_list.append(item)

        elif len(temp_list) > 0:
            if  temp_list[-1] != item:
                temp_list.append(item)
    all_list.append(''.join(temp_list).strip('L'))


# In[30]:

final_result = pd.concat([pd.DataFrame(instance_list),pd.DataFrame(all_list)],axis=1)
final_result.columns=['id','phone_sequence']
final_result.to_csv(csvpath,index=None)



