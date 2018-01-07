import tensorflow as tf
import numpy as np
import model
import pickle
from os.path import join
import h5py
import scipy.misc
import random
import json
import os
import pandas as pd
import sys

testdata=pd.read_csv(sys.argv[2],header= None)

from sklearn.preprocessing import LabelBinarizer
import pickle as pk
with open('class5.pkl', 'rb') as f:
    lb = pk.load(f)

classnum = len(lb.classes_)
caption=testdata[1].tolist()

vectors = np.zeros((len(caption),classnum))
for i,tags in enumerate(caption):
    for tag in lb.classes_.tolist():
        if(tags.find(tag)!=-1):
            vectors[i,:] += lb.transform([tag]).reshape(len(lb.classes_))

batch_size = 64
epochs = 500
image_size = 64
z_dim = 30
caption_vector_length = classnum
data_dir = "./"
save_every = 50
n_images = 5


model_options = {
        'z_dim' : z_dim,
        't_dim' : 20,
        'batch_size' : n_images,
        'image_size' : 64,
        'gf_dim' : 64,
        'df_dim' : 64,
        'gfc_dim' : 1024,
        'caption_vector_length' : classnum
    }
model_path = sys.argv[1]

gan = model.GAN(model_options)
_, _, _, _, _ = gan.build_model()
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, model_path)

input_tensors, outputs = gan.build_generator()

caption_vectors = vectors
caption_image_dic = {}

np.random.seed(200)

for cn, caption_vector in enumerate(caption_vectors):

    caption_images = []
    z_noise = np.random.uniform(-1, 1, [n_images, z_dim])
    caption = [ caption_vector[0:caption_vector_length] ] * n_images

    [ gen_image ] = sess.run( [ outputs['generator'] ], 
        feed_dict = {
            input_tensors['t_real_caption'] : caption,
            input_tensors['t_z'] : z_noise,
        } )

    caption_images = [gen_image[i,:,:,:] for i in range(0, n_images)]
    caption_image_dic[ cn ] = caption_images

samplepath='samples/'+sys.argv[3]+'_{}_{}.jpg'

for cn in range(0, len(caption_vectors)):
    for i, im in enumerate( caption_image_dic[ cn ] ):
        scipy.misc.imsave( samplepath.format(cn+1,i+1) , im)




