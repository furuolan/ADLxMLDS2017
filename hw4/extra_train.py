import pandas as pd
import re
import tensorflow as tf
import numpy as np
import sys
import argparse
import pickle
from os.path import join
import h5py
import scipy.misc
import random
import json
import os
import shutil
import skimage
import skimage.io
import skimage.transform

tags = pd.read_csv("tags_clean.csv",header = None)
tags[1] = tags[1].apply(lambda x: x.split('\t'))
tags2=pd.read_csv("extra_tags.csv",index_col=0)

def finding(x):
    new_tags = list()
    max_hair_vote = 0
    max_eyes_vote = 0
    hair_tag = ""
    eyes_tag = ""
    for tag in x[1]:
        if(tag.find(' hair')!=-1):
            if(tag.find('long')==-1 and tag.find('short')==-1 and tag.find('pubic')==-1 and tag.find('damage')==-1):
                temp = tag.split(':')
                if(max_hair_vote < int(temp[1])):
                    max_hair_vote = int(temp[1])
                    hair_tag = temp[0]
        elif(tag.find(' eyes')!=-1):
            if(tag.find('11')==-1 and tag.find('bicolored')==-1 ):
                temp = tag.split(':')
                if(max_eyes_vote < int(temp[1])):
                    max_eyes_vote = int(temp[1])
                    eyes_tag = temp[0]
    hair_tag2 = ""
    eyes_tag2 = ""
    for tag2 in tags2.iloc[int(x[0]),:]:
        if(tag2.find(' hair')!=-1):
            hair_tag2 = tag2
        if(tag2.find(' eyes')!=-1):
            eyes_tag2 = tag2
    if(hair_tag==""):
        hair_tag = hair_tag2
    if(eyes_tag==""):
        eyes_tag = eyes_tag2
    new_tags.append(hair_tag)
    new_tags.append(eyes_tag)
    return new_tags

new_tags=tags.apply(finding,axis=1)
eyes=new_tags[1].unique()
hairs=new_tags[0].unique()
alltags=eyes.tolist()+hairs.tolist()

from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb.fit(alltags)

import pickle as pk
with open('class5.pkl', 'wb') as f:
    pk.dump(lb, f)

classnum = len(lb.classes_)

def bagofword(x):
    vector = np.zeros(classnum)
    vector += lb.transform([x[0]]).reshape(classnum)
    vector += lb.transform([x[1]]).reshape(classnum)
    return vector

vectors=lb.transform(new_tags[0].values)+lb.transform(new_tags[1].values)

png_list=list()
for idx in range(vectors.shape[0]):
    png_list.append("faces/"+str(idx)+".jpg")

png_list = np.array(png_list)


def load_training_data():
    indexs=list(range(vectors.shape[0]))
    random.shuffle(indexs) 
    return {
        'image_list' : png_list[indexs],
        'captions' : vectors[indexs],
        'data_length' : vectors.shape[0]
    }

def save_for_vis(data_dir, real_images, generated_images, image_files, epoch, batch):
    for i in range(0, real_images.shape[0]):
        fake_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
        fake_images_255 = (generated_images[i,:,:,:])
        scipy.misc.imsave('samples/fake_image_{}_epoch{}_batch{}.jpg'.format(i,epoch,batch), fake_images_255)


def get_training_batch(batch_no, batch_size, image_size, z_dim,caption_vector_length, split, loaded_data = None):
    real_images = np.zeros((batch_size, 64, 64, 3))
    wrong_images = np.zeros((batch_size, 64, 64, 3))
    captions = np.zeros((batch_size, caption_vector_length))
    wrong_captions = np.zeros((batch_size, caption_vector_length))
    cnt = 0
    image_files = []
    for i in range(batch_no * batch_size, batch_no * batch_size + batch_size):
        idx = i % len(loaded_data['image_list'])
        image_file =  loaded_data['image_list'][idx]
        image_array = skimage.io.imread(image_file)
        image_array = skimage.transform.resize(image_array, (64,64))
        real_images[cnt,:,:,:] = image_array
        
        # Improve this selection of wrong image
        wrong_image_id = random.randint(0,len(loaded_data['image_list'])-1)
        wrong_image_file =  loaded_data['image_list'][wrong_image_id]
        wrong_image_array = skimage.io.imread(wrong_image_file)
        wrong_image_array = skimage.transform.resize(wrong_image_array, (64,64))
        wrong_images[cnt, :,:,:] = wrong_image_array

        captions[cnt,:] = loaded_data['captions'][idx]
        
        wrong_captions_id = random.randint(0,len(loaded_data['image_list'])-1)
        wrong_captions[cnt,:] = loaded_data['captions'][wrong_captions_id]
        
        image_files.append( image_file )
        cnt += 1

    z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
    return real_images, wrong_images, captions, wrong_captions, z_noise, image_files

z_dim = 30
model_options = {
        'z_dim' : z_dim,
        't_dim' : 20,
        'batch_size' : 64,
        'image_size' : 64,
        'gf_dim' : 64,
        'df_dim' : 64,
        'gfc_dim' : 1024,
        'caption_vector_length' : classnum
    }

resume_model = False
model_path = "Data/extra_Models/latest_model_temp.ckpt"

import model
gan = model.GAN(model_options)
input_tensors, variables, loss, outputs, checks = gan.build_model()

with tf.variable_scope(tf.get_variable_scope(), reuse=False):
    d_optim = tf.train.AdamOptimizer(0.0002, beta1 = 0.5).minimize(loss['d_loss'], var_list=variables['d_vars'])
    g_optim = tf.train.AdamOptimizer(0.0002, beta1 = 0.5).minimize(loss['g_loss'], var_list=variables['g_vars'])
sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

saver = tf.train.Saver()
if resume_model:
    saver.restore(sess, model_path)

batch_size = 64
epochs = 500
image_size = 64

caption_vector_length = classnum
data_dir = "./"
save_every = 100

caption=["aqua hair,aqua eyes","red hair,green eyes","blue eyes,green hair","white hair,red eyes"]
val_vectors = np.zeros((len(caption),classnum))
for i,tags in enumerate(caption):
    val_vectors[i,:] = bagofword(tags.split(','))


# In[28]:


def predict(val_vectors,n_images,epoch,z_noise):
    caption_vectors = val_vectors
    caption_image_dic = {}
    for cn, caption_vector in enumerate(caption_vectors):

        caption_images = []
        
        caption = [ caption_vector[0:caption_vector_length] ] * n_images
        caption_noie=np.random.normal(loc=np.mean(np.array(caption)),scale=np.std(np.array(caption)),
                                      size=np.array(caption).shape)
        [ gen_image ] = sess.run( [ outputs['generator'] ], 
            feed_dict = {
                input_tensors['t_real_caption'] : caption+caption_noie,
                input_tensors['t_z'] : z_noise,
            } )

        caption_images = [gen_image[i,:,:,:] for i in range(0, n_images)]
        caption_image_dic[ cn ] = caption_images
        print("Generated", cn)

    for cn in range(0, len(caption_vectors)):
        caption_images = []
        for i, im in enumerate( caption_image_dic[ cn ] ):
            caption_images.append( im )
            caption_images.append( np.zeros((64, 5, 3)) )
        combined_image = np.concatenate( caption_images[0:-1], axis = 1 )
        scipy.misc.imsave( 'samples/image_{}_epoch{}.jpg'.format(cn,epoch) , combined_image)

fix_noise = np.random.uniform(-1, 1, [batch_size, z_dim])

d_min_loss = sys.float_info.max
g_min_loss = sys.float_info.max
for i in range(epochs):
    batch_no = 0
    loaded_data = load_training_data()
    
    d_avg_loss = 0
    g_avg_loss = 0
    while batch_no*batch_size < loaded_data['data_length']:
        real_images, wrong_images, caption_vectors, wrong_caption_vectors, z_noise, image_files = get_training_batch(batch_no, batch_size, 
            image_size, z_dim, caption_vector_length, 'train', loaded_data)
        
        noise = np.random.normal(loc=np.mean(caption_vectors),scale=np.std(caption_vectors),size=caption_vectors.shape)
        wrong_noise = np.random.normal(loc=np.mean(caption_vectors),scale=np.std(caption_vectors),size=caption_vectors.shape)
        # DISCR UPDATE
        check_ts = [ checks['d_loss1'] , checks['d_loss2'], checks['d_loss3'], checks['d_loss4']]
        _, d_loss, gen, d1, d2, d3, d4 = sess.run([d_optim, loss['d_loss'], outputs['generator']] + check_ts,
            feed_dict = {
                input_tensors['t_real_image'] : real_images,
                input_tensors['t_wrong_image'] : wrong_images,
                input_tensors['t_real_caption'] : noise+caption_vectors,
                input_tensors['t_wrong_caption'] : wrong_noise+wrong_caption_vectors,
                input_tensors['t_z'] : z_noise,
            })
        
        print("d1", d1)
        print("d2", d2)
        print("d3", d3)
        print("d4", d4)
        print("D", d_loss)
        
        # GEN UPDATE
        _, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
            feed_dict = {
                input_tensors['t_real_image'] : real_images,
                input_tensors['t_wrong_image'] : wrong_images,
                input_tensors['t_real_caption'] : noise+caption_vectors,
                input_tensors['t_wrong_caption'] : wrong_noise+wrong_caption_vectors,
                input_tensors['t_z'] : z_noise,
            })

        # GEN UPDATE TWICE, to make sure d_loss does not go to 0
        _, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
            feed_dict = {
                input_tensors['t_real_image'] : real_images,
                input_tensors['t_wrong_image'] : wrong_images,
                input_tensors['t_real_caption'] : noise+caption_vectors,
                input_tensors['t_wrong_caption'] : wrong_noise+wrong_caption_vectors,
                input_tensors['t_z'] : z_noise,
            })
        
        print("LOSSES", d_loss, g_loss, batch_no, i, len(loaded_data['image_list'])/ batch_size)
        batch_no += 1
        d_avg_loss+=d_loss
        g_avg_loss+=g_loss
        if (batch_no % save_every)== 0:
            print("Saving Model")
            #save_for_vis(data_dir, real_images, gen, image_files, i, batch_no)
            save_path = saver.save(sess, "Data/extra_Models/latest_model_temp.ckpt")
    #predict(val_vectors,64,i,fix_noise)      
    if  i%10 == 0:
        save_path = saver.save(sess, "Data/extra_Models/model_after_epoch_{}.ckpt".format(i))
        d_min_loss = d_avg_loss
        g_min_loss = g_avg_loss




