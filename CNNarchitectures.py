import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform

def KWS_CNN_model(input_shape, n_outputs, dropout=None, norm=None):
    """
    Arguments:
    :param input_shape: shape of the data of the dataset

    :returns Model: a tf.keras.Model() instance
    """

    X_input = tf.keras.Input(input_shape)
    n_frames = input_shape[0]
    
    # CONV -> pooling -> CONV -> lin -> Dense?
    # First convolution
    X = tf.keras.layers.Conv2D(filters=94, kernel_size=(int(2*n_frames/3),8), strides = (1,1), 
                               padding = 'same', name='First_Conv')(X_input)
    #Dropout
    if dropout > 0.0:
        X = tf.keras.layers.Dropout(rate = dropout)(X)
    #BatchNorm
    if norm:
        X = tf.keras.layers.BatchNormalization()(X)
    
    #Pooling on time and frequency
    X = tf.keras.layers.MaxPooling2D(pool_size=(2,3), strides=(2,3), padding='valid', name='MaxPooling')(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    #Second convolution
    X = tf.keras.layers.Conv2D(filters=94, kernel_size=(int(n_frames/5),4), strides = (1,1), 
                               padding = 'same', name='Second_Conv')(X)
    #Dropout
    if dropout > 0.0:
        X = tf.keras.layers.Dropout(rate = dropout)(X)
    #BatchNorm
    if norm:
        X = tf.keras.layers.BatchNormalization()(X)
 
    X = tf.keras.layers.Activation('relu')(X)
    
    # Linear layer
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(32, name='Linear')(X)
    
    # Dense layer
    X = tf.keras.layers.Dense(128, activation='relu', name='Dense1')(X)
    X = tf.keras.layers.Dense(128, activation='relu', name='Dense2')(X)
    X = tf.keras.layers.Dense(128, activation='relu', name='Dense3')(X)  
    
    # Softmax
    X = tf.keras.layers.Dense(n_outputs, activation='softmax', name='Softmax')(X)
    
    # MODEL
    model = Model(inputs = X_input, outputs = X, name='KWS_CNN')
    
    return model