
# coding: utf-8

# In[1]:


from keras import backend as K
import tensorflow as tf
from GlobalLSE import GlobalLSEPooling2D
import os
import pickle as pkl
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#config = tf.ConfigProto(device_count = {'CPU' : 2, 'GPU': 1})
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


# In[2]:


import numpy as np
import pandas as pd
import sys

# In[3]:

# In[10]:


# In[11]:


# In[13]:


from sklearn.preprocessing import MultiLabelBinarizer


# In[14]:

with open("mlb.pkl",'rb') as r:
    mlb = pkl.load(r)

mlb.classes_


# In[23]:

from keras import backend as K
from keras.engine import Layer
from keras.utils.generic_utils import get_custom_objects
from keras.utils.conv_utils import normalize_data_format

if K.backend() == 'theano':
    import theano as K_BACKEND
else:
    import tensorflow as K_BACKEND

class SubPixelUpscaling(Layer):
    """ Sub-pixel convolutional upscaling layer based on the paper "Real-Time Single Image
    and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network"
    (https://arxiv.org/abs/1609.05158).
    This layer requires a Convolution2D prior to it, having output filters computed according to
    the formula :
        filters = k * (scale_factor * scale_factor)
        where k = a user defined number of filters (generally larger than 32)
              scale_factor = the upscaling factor (generally 2)
    This layer performs the depth to space operation on the convolution filters, and returns a
    tensor with the size as defined below.
    # Example :
    ```python
        # A standard subpixel upscaling block
        x = Convolution2D(256, 3, 3, padding='same', activation='relu')(...)
        u = SubPixelUpscaling(scale_factor=2)(x)
        [Optional]
        x = Convolution2D(256, 3, 3, padding='same', activation='relu')(u)
    ```
        In practice, it is useful to have a second convolution layer after the
        SubPixelUpscaling layer to speed up the learning process.
        However, if you are stacking multiple SubPixelUpscaling blocks, it may increase
        the number of parameters greatly, so the Convolution layer after SubPixelUpscaling
        layer can be removed.
    # Arguments
        scale_factor: Upscaling factor.
        data_format: Can be None, 'channels_first' or 'channels_last'.
    # Input shape
        4D tensor with shape:
        `(samples, k * (scale_factor * scale_factor) channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, k * (scale_factor * scale_factor) channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, k channels, rows * scale_factor, cols * scale_factor))` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows * scale_factor, cols * scale_factor, k channels)` if data_format='channels_last'.
    """

    def __init__(self, scale_factor=2, data_format=None, **kwargs):
        super(SubPixelUpscaling, self).__init__(**kwargs)

        self.scale_factor = scale_factor
        self.data_format = normalize_data_format(data_format)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        y = K_BACKEND.depth_to_space(x, self.scale_factor, self.data_format)
        return y

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            b, k, r, c = input_shape
            return (b, k // (self.scale_factor ** 2), r * self.scale_factor, c * self.scale_factor)
        else:
            b, r, c, k = input_shape
            return (b, r * self.scale_factor, c * self.scale_factor, k // (self.scale_factor ** 2))

    def get_config(self):
        config = {'scale_factor': self.scale_factor,
                  'data_format': self.data_format}
        base_config = super(SubPixelUpscaling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({'SubPixelUpscaling': SubPixelUpscaling})


# In[24]:


'''DenseNet models for Keras.
# Reference
- [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
- [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326.pdf)
'''

import warnings

from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils.layer_utils import convert_all_kernels_in_model, convert_dense_weights_data_format
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import decode_predictions
import keras.backend as K

# from subpixel import SubPixelUpscaling

DENSENET_121_WEIGHTS_PATH = r'https://github.com/titu1994/DenseNet/releases/download/v3.0/DenseNet-BC-121-32.h5'
DENSENET_161_WEIGHTS_PATH = r'https://github.com/titu1994/DenseNet/releases/download/v3.0/DenseNet-BC-161-48.h5'
DENSENET_169_WEIGHTS_PATH = r'https://github.com/titu1994/DenseNet/releases/download/v3.0/DenseNet-BC-169-32.h5'
DENSENET_121_WEIGHTS_PATH_NO_TOP = r'https://github.com/titu1994/DenseNet/releases/download/v3.0/DenseNet-BC-121-32-no-top.h5'
DENSENET_161_WEIGHTS_PATH_NO_TOP = r'https://github.com/titu1994/DenseNet/releases/download/v3.0/DenseNet-BC-161-48-no-top.h5'
DENSENET_169_WEIGHTS_PATH_NO_TOP = r'https://github.com/titu1994/DenseNet/releases/download/v3.0/DenseNet-BC-169-32-no-top.h5'

def preprocess_input(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.
    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.
    # Returns
        Preprocessed tensor.
    """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        if x.ndim == 3:
            # 'RGB'->'BGR'
            x = x[::-1, ...]
            # Zero-center by mean pixel
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68
        else:
            x = x[:, ::-1, ...]
            x[:, 0, :, :] -= 103.939
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]
        # Zero-center by mean pixel
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    x *= 0.017 # scale values

    return x


def DenseNet(input_shape=None, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1, nb_layers_per_block=-1,
             bottleneck=False, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, subsample_initial_block=False,
             include_top=True, weights=None, input_tensor=None,
             classes=10, activation='softmax'):
    '''Instantiate the DenseNet architecture,
        optionally loading weights pre-trained
        on CIFAR-10. Note that when using TensorFlow,
        for best performance you should set
        `image_data_format='channels_last'` in your Keras config
        at ~/.keras/keras.json.
        The model and the weights are compatible with both
        TensorFlow and Theano. The dimension ordering
        convention used by the model is the one
        specified in your Keras config file.
        # Arguments
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(32, 32, 3)` (with `channels_last` dim ordering)
                or `(3, 32, 32)` (with `channels_first` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            depth: number or layers in the DenseNet
            nb_dense_block: number of dense blocks to add to end (generally = 3)
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters. -1 indicates initial
                number of filters is 2 * growth_rate
            nb_layers_per_block: number of layers in each dense block.
                Can be a -1, positive integer or a list.
                If -1, calculates nb_layer_per_block from the network depth.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
            bottleneck: flag to add bottleneck blocks in between dense blocks
            reduction: reduction factor of transition blocks.
                Note : reduction value is inverted to compute compression.
            dropout_rate: dropout rate
            weight_decay: weight decay rate
            subsample_initial_block: Set to True to subsample the initial convolution and
                add a MaxPool2D before the dense blocks are added.
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: one of `None` (random initialization) or
                'imagenet' (pre-training on ImageNet)..
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
            activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                Note that if sigmoid is used, classes must be 1.
        # Returns
            A Keras model instance.
        '''

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `cifar10` '
                         '(pre-training on CIFAR-10).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as ImageNet with `include_top`'
                         ' as true, `classes` should be 1000')

    if activation not in ['softmax', 'sigmoid']:
        raise ValueError('activation must be one of "softmax" or "sigmoid"')

    if activation == 'sigmoid' and classes != 1:
        raise ValueError('sigmoid activation can only be used when classes = 1')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=8,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = __create_dense_net(classes, img_input, include_top, depth, nb_dense_block,
                           growth_rate, nb_filter, nb_layers_per_block, bottleneck, reduction,
                           dropout_rate, weight_decay, subsample_initial_block, activation)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='densenet')

    # load weights
    if weights == 'imagenet':
        weights_loaded = False

        if (depth == 121) and (nb_dense_block == 4) and (growth_rate == 32) and (nb_filter == 64) and                 (bottleneck is True) and (reduction == 0.5) and (dropout_rate == 0.0) and (subsample_initial_block):
            if include_top:
                weights_path = get_file('DenseNet-BC-121-32.h5',
                                        DENSENET_121_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='a439dd41aa672aef6daba4ee1fd54abd')
            else:
                weights_path = get_file('DenseNet-BC-121-32-no-top.h5',
                                        DENSENET_121_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='55e62a6358af8a0af0eedf399b5aea99')
            model.load_weights(weights_path)
            weights_loaded = True

        if (depth == 161) and (nb_dense_block == 4) and (growth_rate == 48) and (nb_filter == 96) and                 (bottleneck is True) and (reduction == 0.5) and (dropout_rate == 0.0) and (subsample_initial_block):
            if include_top:
                weights_path = get_file('DenseNet-BC-161-48.h5',
                                        DENSENET_161_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='6c326cf4fbdb57d31eff04333a23fcca')
            else:
                weights_path = get_file('DenseNet-BC-161-48-no-top.h5',
                                        DENSENET_161_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='1a9476b79f6b7673acaa2769e6427b92')
            model.load_weights(weights_path)
            weights_loaded = True

        if (depth == 169) and (nb_dense_block == 4) and (growth_rate == 32) and (nb_filter == 64) and                 (bottleneck is True) and (reduction == 0.5) and (dropout_rate == 0.0) and (subsample_initial_block):
            if include_top:
                weights_path = get_file('DenseNet-BC-169-32.h5',
                                        DENSENET_169_WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='914869c361303d2e39dec640b4e606a6')
            else:
                weights_path = get_file('DenseNet-BC-169-32-no-top.h5',
                                        DENSENET_169_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='89c19e8276cfd10585d5fadc1df6859e')
            model.load_weights(weights_path)
            weights_loaded = True

        if weights_loaded:
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)

            if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')

            print("Weights for the model were loaded successfully")

    return model


def DenseNetFCN(input_shape, nb_dense_block=5, growth_rate=16, nb_layers_per_block=4,
                reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, init_conv_filters=48,
                include_top=True, weights=None, input_tensor=None, classes=1, activation='softmax',
                upsampling_conv=128, upsampling_type='deconv'):
    '''Instantiate the DenseNet FCN architecture.
        Note that when using TensorFlow,
        for best performance you should set
        `image_data_format='channels_last'` in your Keras config
        at ~/.keras/keras.json.
        # Arguments
            nb_dense_block: number of dense blocks to add to end (generally = 3)
            growth_rate: number of filters to add per dense block
            nb_layers_per_block: number of layers in each dense block.
                Can be a positive integer or a list.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
            reduction: reduction factor of transition blocks.
                Note : reduction value is inverted to compute compression.
            dropout_rate: dropout rate
            init_conv_filters: number of layers in the initial convolution layer
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: one of `None` (random initialization) or
                'cifar10' (pre-training on CIFAR-10)..
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(32, 32, 3)` (with `channels_last` dim ordering)
                or `(3, 32, 32)` (with `channels_first` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
            activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                Note that if sigmoid is used, classes must be 1.
            upsampling_conv: number of convolutional layers in upsampling via subpixel convolution
            upsampling_type: Can be one of 'upsampling', 'deconv' and
                'subpixel'. Defines type of upsampling algorithm used.
            batchsize: Fixed batch size. This is a temporary requirement for
                computation of output shape in the case of Deconvolution2D layers.
                Parameter will be removed in next iteration of Keras, which infers
                output shape of deconvolution layers automatically.
        # Returns
            A Keras model instance.
    '''

    if weights not in {None}:
        raise ValueError('The `weights` argument should be '
                         '`None` (random initialization) as no '
                         'model weights are provided.')

    upsampling_type = upsampling_type.lower()

    if upsampling_type not in ['upsampling', 'deconv', 'subpixel']:
        raise ValueError('Parameter "upsampling_type" must be one of "upsampling", '
                         '"deconv" or "subpixel".')

    if input_shape is None:
        raise ValueError('For fully convolutional models, input shape must be supplied.')

    if type(nb_layers_per_block) is not list and nb_dense_block < 1:
        raise ValueError('Number of dense layers per block must be greater than 1. Argument '
                         'value was %d.' % (nb_layers_per_block))

    if activation not in ['softmax', 'sigmoid']:
        raise ValueError('activation must be one of "softmax" or "sigmoid"')

    if activation == 'sigmoid' and classes != 1:
        raise ValueError('sigmoid activation can only be used when classes = 1')

    # Determine proper input shape
    min_size = 2 ** nb_dense_block

    if K.image_data_format() == 'channels_first':
        if input_shape is not None:
            if ((input_shape[1] is not None and input_shape[1] < min_size) or
                    (input_shape[2] is not None and input_shape[2] < min_size)):
                raise ValueError('Input size must be at least ' +
                                 str(min_size) + 'x' + str(min_size) + ', got '
                                                                       '`input_shape=' + str(input_shape) + '`')
        else:
            input_shape = (classes, None, None)
    else:
        if input_shape is not None:
            if ((input_shape[0] is not None and input_shape[0] < min_size) or
                    (input_shape[1] is not None and input_shape[1] < min_size)):
                raise ValueError('Input size must be at least ' +
                                 str(min_size) + 'x' + str(min_size) + ', got '
                                                                       '`input_shape=' + str(input_shape) + '`')
        else:
            input_shape = (None, None, classes)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = __create_fcn_dense_net(classes, img_input, include_top, nb_dense_block,
                               growth_rate, reduction, dropout_rate, weight_decay,
                               nb_layers_per_block, upsampling_conv, upsampling_type,
                               init_conv_filters, input_shape, activation)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='fcn-densenet')

    return model


def DenseNetImageNet121(input_shape=None,
                        bottleneck=True,
                        reduction=0.5,
                        dropout_rate=0.0,
                        weight_decay=1e-4,
                        include_top=True,
                        weights='imagenet',
                        input_tensor=None,
                        classes=1000,
                        activation='softmax'):
    return DenseNet(input_shape, depth=121, nb_dense_block=4, growth_rate=32, nb_filter=64,
                    nb_layers_per_block=[6, 12, 24, 16], bottleneck=bottleneck, reduction=reduction,
                    dropout_rate=dropout_rate, weight_decay=weight_decay, subsample_initial_block=True,
                    include_top=include_top, weights=weights, input_tensor=input_tensor,
                    classes=classes, activation=activation)


def DenseNetImageNet169(input_shape=None,
                        bottleneck=True,
                        reduction=0.5,
                        dropout_rate=0.0,
                        weight_decay=1e-4,
                        include_top=True,
                        weights='imagenet',
                        input_tensor=None,
                        classes=1000,
                        activation='softmax'):
    return DenseNet(input_shape, depth=169, nb_dense_block=4, growth_rate=32, nb_filter=64,
                    nb_layers_per_block=[6, 12, 32, 32], bottleneck=bottleneck, reduction=reduction,
                    dropout_rate=dropout_rate, weight_decay=weight_decay, subsample_initial_block=True,
                    include_top=include_top, weights=weights, input_tensor=input_tensor,
                    classes=classes, activation=activation)


def DenseNetImageNet201(input_shape=None,
                        bottleneck=True,
                        reduction=0.5,
                        dropout_rate=0.0,
                        weight_decay=1e-4,
                        include_top=True,
                        weights=None,
                        input_tensor=None,
                        classes=1000,
                        activation='softmax'):
    return DenseNet(input_shape, depth=201, nb_dense_block=4, growth_rate=32, nb_filter=64,
                    nb_layers_per_block=[6, 12, 48, 32], bottleneck=bottleneck, reduction=reduction,
                    dropout_rate=dropout_rate, weight_decay=weight_decay, subsample_initial_block=True,
                    include_top=include_top, weights=weights, input_tensor=input_tensor,
                    classes=classes, activation=activation)


def DenseNetImageNet264(input_shape=None,
                        bottleneck=True,
                        reduction=0.5,
                        dropout_rate=0.0,
                        weight_decay=1e-4,
                        include_top=True,
                        weights=None,
                        input_tensor=None,
                        classes=1000,
                        activation='softmax'):
    return DenseNet(input_shape, depth=201, nb_dense_block=4, growth_rate=32, nb_filter=64,
                    nb_layers_per_block=[6, 12, 64, 48], bottleneck=bottleneck, reduction=reduction,
                    dropout_rate=dropout_rate, weight_decay=weight_decay, subsample_initial_block=True,
                    include_top=include_top, weights=weights, input_tensor=input_tensor,
                    classes=classes, activation=activation)


def DenseNetImageNet161(input_shape=None,
                        bottleneck=True,
                        reduction=0.5,
                        dropout_rate=0.0,
                        weight_decay=1e-4,
                        include_top=True,
                        weights='imagenet',
                        input_tensor=None,
                        classes=1000,
                        activation='softmax'):
    return DenseNet(input_shape, depth=161, nb_dense_block=4, growth_rate=48, nb_filter=96,
                    nb_layers_per_block=[6, 12, 36, 24], bottleneck=bottleneck, reduction=reduction,
                    dropout_rate=dropout_rate, weight_decay=weight_decay, subsample_initial_block=True,
                    include_top=include_top, weights=weights, input_tensor=input_tensor,
                    classes=classes, activation=activation)


def __conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1e-4):
    ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)

    if bottleneck:
        inter_channel = nb_filter * 4  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

        x = Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)

    x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                  grow_nb_filters=True, return_concat_list=False):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: flag to decide to allow number of filters to grow
        return_concat_list: return the list of feature maps along with the actual output
    Returns: keras tensor with nb_layers of conv_block appended
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x_list = [x]

    for i in range(nb_layers):
        cb = __conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
        x_list.append(cb)

        x = concatenate([x, cb], axis=concat_axis)

        if grow_nb_filters:
            nb_filter += growth_rate

    if return_concat_list:
        return x, nb_filter, x_list
    else:
        return x, nb_filter


def __transition_block(ip, nb_filter, compression=1.0, weight_decay=1e-4):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D
    Args:
        ip: keras tensor
        nb_filter: number of filters
        compression: calculated as 1 - reduction. Reduces the number of feature maps
                    in the transition block.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)
    x = Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x


def __transition_up_block(ip, nb_filters, type='deconv', weight_decay=1E-4):
    ''' SubpixelConvolutional Upscaling (factor = 2)
    Args:
        ip: keras tensor
        nb_filters: number of layers
        type: can be 'upsampling', 'subpixel', 'deconv'. Determines type of upsampling performed
        weight_decay: weight decay factor
    Returns: keras tensor, after applying upsampling operation.
    '''

    if type == 'upsampling':
        x = UpSampling2D()(ip)
    elif type == 'subpixel':
        x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay),
                   use_bias=False, kernel_initializer='he_normal')(ip)
        x = SubPixelUpscaling(scale_factor=2)(x)
        x = Conv2D(nb_filters, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay),
                   use_bias=False, kernel_initializer='he_normal')(x)
    else:
        x = Conv2DTranspose(nb_filters, (3, 3), activation='relu', padding='same', strides=(2, 2),
                            kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(ip)

    return x


def __create_dense_net(nb_classes, img_input, include_top, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1,
                       nb_layers_per_block=-1, bottleneck=False, reduction=0.0, dropout_rate=None, weight_decay=1e-4,
                       subsample_initial_block=False, activation='softmax'):
    ''' Build the DenseNet model
    Args:
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        depth: number or layers
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters. Default -1 indicates initial number of filters is 2 * growth_rate
        nb_layers_per_block: number of layers in each dense block.
                Can be a -1, positive integer or a list.
                If -1, calculates nb_layer_per_block from the depth of the network.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
        bottleneck: add bottleneck blocks
        reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay rate
        subsample_initial_block: Set to True to subsample the initial convolution and
                add a MaxPool2D before the dense blocks are added.
        subsample_initial:
        activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                Note that if sigmoid is used, classes must be 1.
    Returns: keras tensor with nb_layers of conv_block appended
    '''

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, 'reduction value must lie between 0.0 and 1.0'

    # layers in each dense block
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == (nb_dense_block), 'If list, nb_layer is used as provided. '                                                    'Note that list size must be (nb_dense_block)'
        final_nb_layer = nb_layers[-1]
        nb_layers = nb_layers[:-1]
    else:
        if nb_layers_per_block == -1:
            assert (depth - 4) % 3 == 0, 'Depth must be 3 N + 4 if nb_layers_per_block == -1'
            count = int((depth - 4) / 3)

            if bottleneck:
                count = count // 2

            nb_layers = [count for _ in range(nb_dense_block)]
            final_nb_layer = count
        else:
            final_nb_layer = nb_layers_per_block
            nb_layers = [nb_layers_per_block] * nb_dense_block

    # compute initial nb_filter if -1, else accept users initial nb_filter
    if nb_filter <= 0:
        nb_filter = 2 * growth_rate

    # compute compression factor
    compression = 1.0 - reduction

    # Initial convolution
    if subsample_initial_block:
        initial_kernel = (7, 7)
        initial_strides = (2, 2)
    else:
        initial_kernel = (3, 3)
        initial_strides = (1, 1)

    x = Conv2D(nb_filter, initial_kernel, kernel_initializer='he_normal', padding='same',
               strides=initial_strides, use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)

    if subsample_initial_block:
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, bottleneck=bottleneck,
                                     dropout_rate=dropout_rate, weight_decay=weight_decay)
        # add transition_block
        x = __transition_block(x, nb_filter, compression=compression, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    # The last dense_block does not have a transition_block
    x, nb_filter = __dense_block(x, final_nb_layer, nb_filter, growth_rate, bottleneck=bottleneck,
                                 dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    if include_top:
        x = Dense(nb_classes, activation=activation)(x)

    return x


def __create_fcn_dense_net(nb_classes, img_input, include_top, nb_dense_block=5, growth_rate=12,
                           reduction=0.0, dropout_rate=None, weight_decay=1e-4,
                           nb_layers_per_block=4, nb_upsampling_conv=128, upsampling_type='upsampling',
                           init_conv_filters=48, input_shape=None, activation='deconv'):
    ''' Build the DenseNet model
    Args:
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay
        nb_layers_per_block: number of layers in each dense block.
            Can be a positive integer or a list.
            If positive integer, a set number of layers per dense block.
            If list, nb_layer is used as provided. Note that list size must
            be (nb_dense_block + 1)
        nb_upsampling_conv: number of convolutional layers in upsampling via subpixel convolution
        upsampling_type: Can be one of 'upsampling', 'deconv' and 'subpixel'. Defines
            type of upsampling algorithm used.
        input_shape: Only used for shape inference in fully convolutional networks.
        activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                    Note that if sigmoid is used, classes must be 1.
    Returns: keras tensor with nb_layers of conv_block appended
    '''

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    if concat_axis == 1:  # channels_first dim ordering
        _, rows, cols = input_shape
    else:
        rows, cols, _ = input_shape

    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, 'reduction value must lie between 0.0 and 1.0'

    # check if upsampling_conv has minimum number of filters
    # minimum is set to 12, as at least 3 color channels are needed for correct upsampling
    assert nb_upsampling_conv > 12 and nb_upsampling_conv % 4 == 0, 'Parameter `upsampling_conv` number of channels must '                                                                     'be a positive number divisible by 4 and greater '                                                                     'than 12'

    # layers in each dense block
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == (nb_dense_block + 1), 'If list, nb_layer is used as provided. '                                                        'Note that list size must be (nb_dense_block + 1)'

        bottleneck_nb_layers = nb_layers[-1]
        rev_layers = nb_layers[::-1]
        nb_layers.extend(rev_layers[1:])
    else:
        bottleneck_nb_layers = nb_layers_per_block
        nb_layers = [nb_layers_per_block] * (2 * nb_dense_block + 1)

    # compute compression factor
    compression = 1.0 - reduction

    # Initial convolution
    x = Conv2D(init_conv_filters, (7, 7), kernel_initializer='he_normal', padding='same', name='initial_conv2D',
               use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    nb_filter = init_conv_filters

    skip_list = []

    # Add dense blocks and transition down block
    for block_idx in range(nb_dense_block):
        x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate,
                                     weight_decay=weight_decay)

        # Skip connection
        skip_list.append(x)

        # add transition_block
        x = __transition_block(x, nb_filter, compression=compression, weight_decay=weight_decay)

        nb_filter = int(nb_filter * compression)  # this is calculated inside transition_down_block

    # The last dense_block does not have a transition_down_block
    # return the concatenated feature maps without the concatenation of the input
    _, nb_filter, concat_list = __dense_block(x, bottleneck_nb_layers, nb_filter, growth_rate,
                                              dropout_rate=dropout_rate, weight_decay=weight_decay,
                                              return_concat_list=True)

    skip_list = skip_list[::-1]  # reverse the skip list

    # Add dense blocks and transition up block
    for block_idx in range(nb_dense_block):
        n_filters_keep = growth_rate * nb_layers[nb_dense_block + block_idx]

        # upsampling block must upsample only the feature maps (concat_list[1:]),
        # not the concatenation of the input with the feature maps (concat_list[0].
        l = concatenate(concat_list[1:], axis=concat_axis)

        t = __transition_up_block(l, nb_filters=n_filters_keep, type=upsampling_type, weight_decay=weight_decay)

        # concatenate the skip connection with the transition block
        x = concatenate([t, skip_list[block_idx]], axis=concat_axis)

        # Dont allow the feature map size to grow in upsampling dense blocks
        x_up, nb_filter, concat_list = __dense_block(x, nb_layers[nb_dense_block + block_idx + 1], nb_filter=growth_rate,
                                                     growth_rate=growth_rate, dropout_rate=dropout_rate,
                                                     weight_decay=weight_decay, return_concat_list=True,
                                                     grow_nb_filters=False)

    if include_top:
        x = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', use_bias=False)(x_up)

        if K.image_data_format() == 'channels_first':
            channel, row, col = input_shape
        else:
            row, col, channel = input_shape

        x = Reshape((row * col, nb_classes))(x)
        x = Activation(activation)(x)
        x = Reshape((row, col, nb_classes))(x)
    else:
        x = x_up

    return x


# In[25]:


classes = len(mlb.classes_)


# In[31]:


def weighted_binary_crossentropy(y_true, y_pred):
    _epsilon = K.epsilon()
    P_ratio  = K.mean(y_true, axis=0, keepdims=True)
    P_ratio  = K.clip(P_ratio, _epsilon, 1. - _epsilon) # To prevent overflow.

    y_pred = K.clip(y_pred, _epsilon, 1. - _epsilon)    # To prevent overflow.
    P_ce   = - y_true * K.log(y_pred) / P_ratio
    N_ce   = - (1. - y_true) * K.log(1. - y_pred) / (1. - P_ratio)

    return K.sum(P_ce + N_ce, axis=-1)


# In[32]:


K.set_learning_phase(1)
model = model = DenseNetImageNet121(input_shape=(224,224,1), include_top=False, weights=None)

# x = model.output

x = model.layers[-2].output
x = GlobalLSEPooling2D(r=10)(x)
x = Dense(classes, name='fc')(x)
x = Activation('sigmoid', name='prob')(x)
model = Model(model.input, x, name='densenet')
model.summary()


# In[33]:


from keras import backend as K
    
thresh = 0.5

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.

    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.

    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.

    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.

    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)

fscore = f1score = fmeasure


# In[34]:


def sum_binary_crossentropy(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


# In[35]:


# In[36]:


#parallel_model = multi_gpu_model(model, gpus=2)
#with tf.device('/gpu:1'):
model.compile(optimizer='adam', loss=weighted_binary_crossentropy, metrics=[recall,precision,f1score])


# In[37]:

# In[38]:


batch_size = 32


# In[39]:


import cv2


# Subtract mean pixel and multiple by scaling constant 
# Reference: https://github.com/shicai/DenseNet-Caffe
def shicai(im):
    im[:,:,0] = (im[:,:,0] - 103.94) * 0.017
    im[:,:,1] = (im[:,:,1] - 116.78) * 0.017
    im[:,:,2] = (im[:,:,2] - 123.68) * 0.017
    return im

# # train_x = shicai(train_x)
# val_x = shicai(val_x)
# test_x = shicai(test_x)


# run all above

# In[52]:


from keras import models
model.load_weights("best.hdf5")


# In[53]:


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


# In[54]:


layer_name=[layer.name for layer in model.layers]


# In[56]:


class_weights = model.layers[-2].get_weights()[0]


# In[57]:


final_conv_layer = get_output_layer(model, 'activation_121')


# In[58]:


get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-2].output])



# In[61]:


mlb_to_val = dict()
for name in mlb.classes_:
    mlb_to_val[name] = name

# In[108]:


import judger_medical as judger


# In[109]:


imgs = judger.get_file_names()
f = judger.get_output_file_object()


# In[110]:


# In[113]:


judge_x = np.empty((len(imgs),224,224,1),dtype='float16')

for i,file_name in enumerate(imgs):
    img = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(224,224),interpolation=cv2.INTER_NEAREST)
    judge_x[i] = img.reshape((224,224,1))


# In[114]:


judge_x = shicai(judge_x)


# In[115]:


heatmap_thresh = 0.5
result_thresh = 0.5
result = model.predict(judge_x)
[conv_outputs, predictions] = get_output([judge_x])
for id in range(len(judge_x)):
    predict_classes = np.where(result[id]>result_thresh)[0]
    conv_outputs_class = conv_outputs[id, :, :, :]
    f.write(('%s %d\n'%(imgs[id], len(predict_classes))).encode())
    for idx in predict_classes:
        classname = mlb.classes_[idx]
        
        #Create the class activation map.
        cam = np.zeros(dtype = np.float32, shape = conv_outputs_class.shape[0:2])
        for i, w in enumerate(class_weights[:, idx]):
            cam +=  w * conv_outputs_class[:, :, i]
        cam = cam/np.max(cam)
        cam = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam < heatmap_thresh)] = 0
        resized_heatmap = cv2.resize(heatmap,(1024,1024))

        ret,thresh = cv2.threshold(cv2.cvtColor(resized_heatmap,cv2.COLOR_BGR2GRAY),10,10,10)
        x, y, w, h = cv2.boundingRect(thresh)
        f.write(('%s %f %f %f %f\n'%(classname, x, y, w, h)).encode())
    


# In[116]:


score, err = judger.judge()
if err is not None:  # in case we failed to judge your submission
    print(err)


# In[ ]:




