
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import sys

# In[2]:

from keras.models import Sequential
from keras.layers import Dense,GRU,Dropout, Activation,LSTM,Embedding
from keras.utils import np_utils, to_categorical
from keras.optimizers import Adam, SGD, Adadelta
from keras.callbacks import EarlyStopping


datapath = sys.argv[1]
outputmodel = sys.argv[2]

# In[4]:

train = pd.read_csv(datapath+"mfcc/train.ark",delimiter='\t',header=None)
label = pd.read_csv(datapath+"label/train.lab",delimiter=',',header=None)
map39 = pd.read_csv(datapath+"phones/48_39.map",header=None,delimiter='\t')
_48to39 = dict()
for char48 in map39[0]:
    _48to39[char48] = map39.loc[map39[0]==char48][1].tolist()[0]


# In[7]:

label[1]=label[1].map(lambda x: _48to39[x])
label.columns=[0,40]


# In[9]:

train_data=pd.DataFrame(train[0].apply(lambda x: x.split(' ')).values.tolist())
train_all=train_data.join(label.set_index(0),on=[0],how='inner')


# In[12]:

class39=train_all[40].unique().tolist()
class39.append('zzz')


# In[13]:

from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb.fit(class39) 
import pickle as pk
with open('40class.pkl', 'wb') as f:
    pk.dump(lb, f)

# In[15]:

def sp(x):
    all_id = x.split('_')
    sentenceid = all_id[0]+'_'+all_id[1]
    frameid = all_id[2]
    final = [sentenceid,frameid]
    return final


# In[16]:

train_id = pd.DataFrame(train_all[0].apply(sp).values.tolist())
train_id.columns=["instance","frame"]


# In[18]:

train_all=train_all.drop(0,axis=1)


# In[19]:

final = pd.concat([train_id,train_all],axis=1)
final['frame']=final['frame'].astype(int)


# In[21]:

train_data=final.groupby('instance').apply(lambda x: x.iloc[:,2:41].values.tolist())


# In[22]:

train_instance_list=train_data.index.tolist()


# In[23]:

labeldigit=lb.transform(final[40])
all_label=pd.concat([final[['instance','frame']],pd.DataFrame(labeldigit)],axis=1)
all_label['frame']=all_label['frame'].astype(int)
train_label=all_label.groupby('instance').apply(lambda x: x.iloc[:,2:].values)


# In[26]:

count=list()
for temp in train_data:
    count.append(len(temp))
count.sort()
max_frame = count[len(count)-1]

# In[27]:

train_data_pad = np.zeros((len(train_data),max_frame,39))
index = 0
for inst in train_data:
    train_data_pad[index,:len(inst),:] = inst
    index+=1


# In[28]:

train_data_pad = train_data_pad.astype(np.float32,order='C')


# In[29]:

train_label_pad=np.zeros((len(train_label),max_frame,40))
index = 0
for inst in train_label:
    train_label_pad[index,:,39] = 1
    train_label_pad[index,:len(inst),:] = inst
    index+=1


# In[30]:

train_label_pad = train_label_pad.astype(np.float32,order='C')

# # padding

# In[118]:

from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.core import Masking,RepeatVector
from keras.layers.recurrent import SimpleRNN,LSTM
from keras.layers.convolutional import AtrousConvolution1D
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D,Flatten,GaussianNoise, Reshape,UpSampling1D,UpSampling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers import Merge
from keras.regularizers import l2
from keras.layers.convolutional import ZeroPadding1D
from keras.layers import GaussianDropout, Cropping1D


# In[34]:

model = Sequential()
model.add(GaussianNoise(0.3,input_shape=(777,39)))
model.add(Convolution1D(200,4,padding='causal',dilation_rate=1, activation='selu'))
model.add(Dropout(0.3))
model.add(Convolution1D(150,4,padding='causal',dilation_rate=2, activation='selu'))
model.add(Dropout(0.3))
model.add(Convolution1D(120,3,padding='causal',dilation_rate=4, activation='selu'))
model.add(MaxPooling1D(3))

model.add(Convolution1D(80,2,padding='causal', activation='selu'))
model.add(Dropout(0.3))

model.add(Bidirectional(GRU(200,implementation=2,return_sequences=True,
                             activation='tanh',recurrent_activation='sigmoid',recurrent_dropout=0.2,dropout=0.3)))
model.add(Bidirectional(GRU(200,implementation=2,return_sequences=True,
                             activation='tanh',recurrent_activation='sigmoid',recurrent_dropout=0.2,dropout=0.3)))
model.add(Dense(150,activation="selu"))
model.add(UpSampling1D(3))
model.add(Dropout(0.3))
model.add(Dense(100,activation="selu"))
model.add(Dropout(0.3))
model.add(Dense(40,activation="softmax"))
model.compile(loss='categorical_crossentropy',optimizer="adam", metrics=['accuracy'])
model.summary()

# In[257]:

json = model.to_json()
with open(outputmodel+".json",'w') as f:
    f.write(json)


# In[258]:

earlystopping = EarlyStopping(monitor='val_loss', patience = 10, verbose=1, mode='min')
checkpoint = ModelCheckpoint(filepath=outputmodel+".hdf5",
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_loss',
                             mode='min')


# In[259]:

batch_size = 32
epochs = 1000

model.fit(train_data_pad, train_label_pad,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1,
          callbacks=[earlystopping,checkpoint])


