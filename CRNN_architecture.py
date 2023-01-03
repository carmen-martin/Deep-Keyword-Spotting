import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform

def KWS_CRNN_model(input_shape, n_outputs, n_recurrents=2, dropout=0, norm=None):
    """
    Arguments:
    :param input_shape: shape of the data of the dataset

    :returns Model: a tf.keras.Model() instance
    """

    X_input = tf.keras.Input(input_shape)
    n_frames = input_shape[0]
    
    # CONV -> Recurrent -> Dense
    # Convolution layer
    X = tf.keras.layers.Conv2D(filters=32, kernel_size=(20,5), strides = (8,2), 
                               padding = 'same', name='First_Conv')(X_input)
    #Dropout
    if dropout > 0.0:
        X = tf.keras.layers.Dropout(rate = dropout)(X)
    #BatchNorm
    if norm:
        X = tf.keras.layers.BatchNormalization()(X)
    
    # Bidirectional Recurrent layers
    # Adjust input shape
    h = tf.shape(X)[1]
    w = tf.shape(X)[2]
    c = tf.shape(X)[3]
    X = tf.reshape(X, [-1, h*w, c])
    
    for nr in range(n_recurrents):
        X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True, name=f'gru{nr}'))(X)
       
    # Dense layer
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(64, activation='relu', name='Dense')(X)  
    
    # Softmax
    X = tf.keras.layers.Dense(n_outputs, activation='softmax', name='Softmax')(X)
    
    # MODEL
    model = Model(inputs = X_input, outputs = X, name='KWS_CNN')
    
    return model