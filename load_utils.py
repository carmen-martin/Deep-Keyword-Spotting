import numpy as np
np.random.seed(1234)

from os.path import join as pjoin
import numpy as np
from python_speech_features import mfcc, delta, logfbank
import scipy.io.wavfile as wav

def load_dataset(directory, categories, train_size=0.85, winlen=0.025, winstep=0.01, numcep=13, nfilt=26):
    '''
    Loads all the dataset from the given directory and divides it in test and training sets. 
    The label set provides a different label for each category.
    
    :param directory: path to the folder where dataset is stored
    :param categories: array of keywords wanted for loading
    :param train_size: percentage of data used for training (1-train_size will be used for testing). Default is 0.85 (85%)
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    
    :returns: 4 numpy arrays - X_train, Y_train, X_test, Y_test
    '''
    
    dataset = np.load(pjoin(directory, f"{categories[0]}_processed_audio_features_{int(winlen*1000)}ms_{int(winstep*1000)}ms_{nfilt}_{numcep}.npy"))
    temp1 =  np.shape(dataset)[2]
    labels = np.zeros(temp1)
    for idx, key in enumerate(categories[1:]):
        dataset = np.dstack((dataset, np.load(pjoin(directory, f"{key}_processed_audio_features_{int(winlen*1000)}ms_{int(winstep*1000)}ms_{nfilt}_{numcep}.npy")))) 
        temp2 = np.shape(dataset)[2]
        labels = np.append(labels, (idx+1)*np.ones(temp2-temp1))
        temp1 = temp2

    # Randomly permute the data
    permutation = np.random.permutation(len(labels))

    X = dataset[:,:,permutation]
    Y = labels[permutation]

    # Split in training and test sets
    print(labels, len(labels), train_size)
    m_training = int(train_size*len(labels))

    X_train = X[:,:,:m_training]
    Y_train = Y[:m_training]
    X_test = X[:,:, m_training:]
    Y_test = Y[m_training:]
    
    return X_train, Y_train, X_test, Y_test


def load_dataset_keywords(directory, keywords, categories, frames=99, train_size=0.85, winlen=0.025, winstep=0.01, numcep=13, nfilt=26):
    '''
    Loads all the dataset from the given directory and divides it in test and training sets.
    Only the keywords will be set as output categories, and another category "not a keyword" will englobe all other words.
    The label set provides a different label for each category.
    
    :param directory: path to the folder where dataset is stored
    :param keywords: array of keywords we want to learn
    :param categories: array of words of which audios will be loaded
    :param train_size: percentage of data used for training (1-train_size will be used for testing). Default is 0.85 (85%)
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    
    :returns: 4 numpy arrays - X_train, Y_train, X_test, Y_test
    '''
    
    dataset = np.zeros((frames, 39, 1))
    temp1 =  np.shape(dataset)[2]
    labels = []
    lab = 1
    for word in categories:
        if word in keywords:
            dataset = np.dstack((dataset, np.load(pjoin(directory, f"{word}_processed_audio_features_{int(winlen*1000)}ms_{int(winstep*1000)}ms_{nfilt}_{numcep}.npy")))) 
            temp2 = np.shape(dataset)[2]
            labels = np.append(labels, lab*np.ones(temp2-temp1))
            temp1 = temp2
            lab += 1 
        else:
            dataset = np.dstack((dataset, np.load(pjoin(directory, f"{word}_processed_audio_features_{int(winlen*1000)}ms_{int(winstep*1000)}ms_{nfilt}_{numcep}.npy")))) 
            temp2 = np.shape(dataset)[2]
            labels = np.append(labels, np.zeros(temp2-temp1)) # maybe change to -1
            temp1 = temp2
    
    dataset = dataset[:,:,1:]

    # Randomly permute the data
    permutation = np.random.permutation(len(labels))

    X = dataset[:,:,permutation]
    Y = labels[permutation]

    # Split in training and test sets
    m_training = int(train_size*len(labels))

    X_train = X[:,:,:m_training]
    Y_train = Y[:m_training]
    X_test = X[:,:, m_training:]
    Y_test = Y[m_training:]
    
    return X_train, Y_train, X_test, Y_test


def rawtofeatures(audio_file, window_len=0.025, window_step=0.02, num_coefs = 13, num_filt=26, frames=50):
    # Load file
    samplerate, signl = wav.read(audio_file)
    
    # MFCCs
    mfcc_feat = mfcc(signal=signl, samplerate=samplerate, winlen=window_len, winstep=window_step,
                     numcep=num_coefs, nfilt=num_filt, nfft=512, lowfreq=0, appendEnergy=True)
    # Delta coefs
    d_mfcc_feat = delta(mfcc_feat, 2)
    # Delta-Delta coefs
    d2_mfcc_feat = delta(d_mfcc_feat, 2)
    # save audio features in desired order
    audio_features = np.concatenate((mfcc_feat[:,1:], d_mfcc_feat[:,1:], d2_mfcc_feat[:,1:],
                                    mfcc_feat[:,:1], d_mfcc_feat[:,:1], d2_mfcc_feat[:,:1]), axis=1)

    # if the audio length is smaller than 1s -> pad with 0s to get enough frames to stack
    if len(audio_features) < frames: 
        audio_features = np.concatenate((file_features, np.zeros(((frame-len(file_features)), 39))), axis=0)
    return audio_features