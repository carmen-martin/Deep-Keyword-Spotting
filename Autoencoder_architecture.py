import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform


def dcrnn_mse(x, z, y, z_hat):
    return 0.5 * (tf.keras.losses.MeanSquaredError(name='reconstruction_error')(x, y) 
        + tf.keras.losses.MeanSquaredError(name='time_sequence_error')(z, z_hat))


def build_dcrnn_autoencoder(img_shape, code_size, dropout=0, norm=None):
    # Follow best CNN architecture
    inp = tf.keras.Input(img_shape)
    #print(f'Encoder input {inp.shape.as_list() = }')
    # Encoder
    n_frames = img_shape[0]
    #   CNN + MaxPooling layers
    x = tf.keras.layers.Conv2D(filters=94, kernel_size=(int(2*n_frames/3),8), strides=(1,1),
                                       padding='same', name='First_Conv')(inp)
    if dropout > 0.0:
        x = tf.keras.layers.Dropout(rate=dropout)(x)
    if norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,3), strides=(2,3), padding='valid', name='MaxPooling')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(int(n_frames/5),4), strides=(1,1), 
                                       padding='same', name='Second_Conv')(x)
    if dropout > 0.0:
        x = tf.keras.layers.Dropout(rate=dropout)(x)
    if norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    _, h, w, c = x.shape.as_list()
    #print(f'Encoder CNN output {x.shape.as_list() = }')
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(code_size * 2, activation='tanh')(x)
    code = tf.keras.layers.Dense(code_size, activation='tanh')(x)
    code = tf.reshape(code, [-1, code_size, 1])
    encoder = Model(inputs=inp, outputs=code, name='Enconder')
    #print(f'RNN input {code.shape.as_list() = }')
    # RNN layer
    rnn = tf.keras.layers.GRU(code_size, name='gru')
    rnn_output = rnn(code)
    #print(f'RNN output {rnn_output.shape.as_list() = }')
    # Decoder
    inp = tf.keras.layers.Flatten()(rnn_output)
    x = tf.keras.layers.Dense(code_size * 2, activation='tanh')(inp)
    x = tf.keras.layers.Dense(h * w * c, activation='tanh')(x)
    x = tf.reshape(x, [-1, h, w, c])
    #    ConvTranspose layers
    #print(f'Decoder CNNT input {x.shape.as_list() = }')
    x = tf.keras.layers.Conv2DTranspose(filters=94, kernel_size=(int(n_frames/5),4), strides=(1,1), 
                                       padding='same', name='First_ConvT')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(int(2*n_frames/3),8), strides=(1,1),
                                       padding='same', name='Second_ConvT')(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 3), interpolation='nearest')(x)
    #print(f'Decoder output {x.shape.as_list() = }')
    decoder = Model(inputs=inp, outputs=x, name='Decoder')
    # Autoencoder
    inp = tf.keras.Input(img_shape)
    code = encoder(inp)
    time_evolved_code = rnn(code)
    reconstruction = decoder(time_evolved_code)
    autoencoder = tf.keras.Model(inputs=inp, outputs=reconstruction)
    # Intermediate layer loss function: https://stackoverflow.com/a/62454812
    autoencoder.add_loss(dcrnn_mse(inp, code, reconstruction, time_evolved_code))
    return encoder, decoder, autoencoder


def build_img_dcrnn_autoencoder(img_shape, code_size, dropout=0, norm=None):
    inp = tf.keras.Input(img_shape)
    #print(f'Encoder input {inp.shape.as_list() = }')
    # Encoder
    n_frames = img_shape[0]
    #   CNN + MaxPooling layers
    x = tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 4), strides=(1,1),
                                       padding='same', name='First_Conv')(inp)
    if dropout > 0.0:
        x = tf.keras.layers.Dropout(rate=dropout)(x)
    if norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(1, 4), strides=(1,1), 
                                       padding='same', name='Second_Conv')(x)
    if dropout > 0.0:
        x = tf.keras.layers.Dropout(rate=dropout)(x)
    if norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 4), strides=(1,1), 
                                       padding='same', name='Third_Conv')(x)
    if dropout > 0.0:
        x = tf.keras.layers.Dropout(rate=dropout)(x)
    if norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 4), strides=(1,1), 
                                       padding='same', name='Fourth_Conv')(x)
    if dropout > 0.0:
        x = tf.keras.layers.Dropout(rate=dropout)(x)
    if norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    _, h, w, c = x.shape.as_list()
    #print(f'Encoder CNN output {x.shape.as_list() = }')
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(code_size * 2, activation='tanh')(x)
    code = tf.keras.layers.Dense(code_size, activation='tanh')(x)
    code = tf.reshape(code, [-1, code_size, 1])
    encoder = Model(inputs=inp, outputs=code, name='Enconder')
    #print(f'RNN input {code.shape.as_list() = }')
    # RNN layer
    rnn = tf.keras.layers.GRU(code_size, name='gru')
    rnn_output = rnn(code)
    #print(f'RNN output {rnn_output.shape.as_list() = }')
    # Decoder
    inp = tf.keras.layers.Flatten()(rnn_output)
    x = tf.keras.layers.Dense(code_size * 2, activation='tanh')(inp)
    x = tf.keras.layers.Dense(h * w * c, activation='tanh')(x)
    x = tf.reshape(x, [-1, h, w, c])
    #    ConvTranspose layers
    #print(f'Decoder CNNT input {x.shape.as_list() = }')
    x = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(1, 4), strides=(1,1), 
                                       padding='same', name='First_ConvT')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(1, 4), strides=(1,1), 
                                       padding='same', name='Second_ConvT')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=(1, 4), strides=(1,1), 
                                       padding='same', name='Third_ConvT')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(1, 4), strides=(1,1),
                                       padding='same', name='Fourth_ConvT')(x)
    #x = tf.keras.layers.UpSampling2D(size=(2, 3), interpolation='nearest')(x)
    #print(f'Decoder output {x.shape.as_list() = }')
    decoder = Model(inputs=inp, outputs=x, name='Decoder')
    # Autoencoder
    inp = tf.keras.Input(img_shape)
    code = encoder(inp)
    time_evolved_code = rnn(code)
    reconstruction = decoder(time_evolved_code)
    autoencoder = tf.keras.Model(inputs=inp, outputs=reconstruction)
    # Intermediate layer loss function: https://stackoverflow.com/a/62454812
    autoencoder.add_loss(dcrnn_mse(inp, code, reconstruction, time_evolved_code))
    return encoder, decoder, autoencoder


def build_crnn_autoencoder(img_shape, code_size, dropout=0, norm=None, n_recurrents=2):
    # Follow best CNN architecture
    inp = tf.keras.Input(img_shape)
    #print(f'Encoder input {inp.shape.as_list() = }')
    # Encoder
    n_frames = img_shape[0]
    #   CNN + MaxPooling layers
    x = tf.keras.layers.Conv2D(filters=94, kernel_size=(int(2*n_frames/3),8), strides=(1,1),
                                       padding='same', name='First_Conv')(inp)
    if dropout > 0.0:
        x = tf.keras.layers.Dropout(rate=dropout)(x)
    if norm:
        x = f.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,3), strides=(2,3), padding='valid', name='MaxPooling')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(int(n_frames/5),4), strides=(1,1), 
                                       padding='same', name='Second_Conv')(x)
    if dropout > 0.0:
        x = tf.keras.layers.Dropout(rate=dropout)(x)
    if norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    #print(f'Encoder CNN output {x.shape.as_list() = }')
    #   RNN layers
    _, h, w, c = x.shape.as_list()
    x = tf.reshape(x, [-1, h*w, c])
    #print(f'Encoder RNN intput {x.shape.as_list() = }')
    for nr in range(n_recurrents):
        x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True, name=f'gru{nr}'))(x)
    #print(f'Encoder RNN output {x.shape.as_list() = }')
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(code_size)(x)
    encoder = Model(inputs=inp, outputs=x, name='Enconder')
    # Decoder
    inp = tf.keras.Input((code_size,))
    x = tf.keras.layers.Dense(h * w * c, activation='tanh')(inp)
    #    RNN layers
    x = tf.reshape(x, [-1, h*w, c])
    #print(f'Decoder RNN intput {x.shape.as_list() = }')
    for nr in range(n_recurrents):
        x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32 // n_recurrents, return_sequences=True, name=f'gru{nr}'))(x)
    #print(f'Decoder RNN output {x.shape.as_list() = }') 
    x = tf.reshape(x, [-1, h, w, c])
    #    ConvTranspose layers
    #print(f'Decoder CNNT input {x.shape.as_list() = }')
    x = tf.keras.layers.Conv2DTranspose(filters=94, kernel_size=(int(n_frames/5),4), strides=(1,1), 
                                       padding='same', name='First_ConvT')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(int(2*n_frames/3),8), strides=(1,1),
                                       padding='same', name='Second_ConvT')(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 3), interpolation='nearest')(x)
    #print(f'Decoder output {x.shape.as_list() = }')
    decoder = Model(inputs=inp, outputs=x, name='Decoder')
    # Autoencoder
    inp = tf.keras.Input(img_shape)
    reconstruction = decoder(encoder(inp))
    autoencoder = tf.keras.Model(inputs=inp, outputs=reconstruction)
    return encoder, decoder, autoencoder


def build_autoencoder(imgs_shape, code_size, model_type, dropout=0, norm=None, n_recurrents=2):
    if model_type == 'crnn':
        return build_crnn_autoencoder(imgs_shape, code_size, dropout=dropout, norm=norm, n_recurrents=n_recurrents)
    elif model_type == 'dcrnn':
        return build_dcrnn_autoencoder(imgs_shape, code_size, dropout=dropout, norm=norm)
    elif model_type == 'img_dcrnn':
        return build_img_dcrnn_autoencoder(imgs_shape, code_size, dropout=dropout, norm=norm)
    else:
        raise ValueError('Invalid autoencoder model type')
    
