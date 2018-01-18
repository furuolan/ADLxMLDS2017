from keras import backend as K
from keras.layers.pooling import _GlobalPooling2D


class GlobalLSEPooling2D(_GlobalPooling2D):
    """Global Log-Sum-Exp pooling operation for spatial data.
    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`
    # Output shape
        2D tensor with shape:
        `(batch_size, channels)`
    """
    def __init__(self, r, **kwargs):
        super(GlobalLSEPooling2D, self).__init__(**kwargs)
        self.r = r

    def call(self, inputs):
        if self.data_format == 'channels_last':
            axis = [1, 2]
        else:
            axis = [2, 3]

        r = K.constant(self.r, K.floatx())

        gMax = K.max(inputs, axis=axis, keepdims=True)  # (None, 1, 1, n_channels)
        Exp  = K.exp(r * (inputs - gMax))
        SE   = K.mean(Exp, axis=axis)
        LSE  = K.log(SE) / r
        for i in reversed(axis):
            gMax = K.squeeze(gMax, axis=i)              # (None, n_channels)

        return gMax + LSE


# Aliases

GlobalLSEPool2D = GlobalLSEPooling2D
