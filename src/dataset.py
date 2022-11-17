import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def read_record_mfcc(record, sample_rate = 16000, number_of_classes=12, train_fix=100, sequential=False, sequence_length=10):

    audio = tf.cast(record['audio'], tf.float32)
    label = tf.cast(record['label'], tf.int32)

    # calculating parameters for performing STFT
    frame_length = tf.cast(tf.cast(sample_rate, tf.float16) * 0.025, tf.int32)
    frame_step = tf.cast(tf.cast(sample_rate, tf.float16) * 0.01, tf.int32)

    # extracting STFT and generating spectrograms
    stft = tf.signal.stft(tf.squeeze(audio), frame_length=frame_length, frame_step=frame_step, fft_length=512)
    spectrogram = tf.abs(stft)

    # converting the spectrograms to mel-scaled spectrograms
    num_spectrogram_bins = tf.shape(spectrogram)[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 257
    
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrograms = tf.reshape(mel_spectrograms, ( tf.concat([tf.shape(spectrogram)[:-1], tf.shape(linear_to_mel_weight_matrix)[-1:]], axis=0)))
    
    # extracting MFCCs
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :num_spectrogram_bins]

    
    # ZeroPadding the mfcc
    zeros = tf.zeros(shape=(tf.constant(train_fix, dtype=tf.int32), tf.shape(mfccs)[1]), dtype=tf.dtypes.float32)
    mfccs = tf.concat([mfccs, zeros], axis=0)
    mfccs = mfccs[:tf.constant(train_fix, dtype=tf.int32)]

    # Fixing the shape of the spectrogram
    if sequential:
        mfccs = tf.reshape(mfccs, (sequence_length, -1, tf.shape(mfccs)[1]))
    mfccs = tf.expand_dims(mfccs, -1)

    label_one_hot = tf.one_hot(label, number_of_classes)
    
    return mfccs, label_one_hot


def read_record_stft(record, sample_rate = 16000, number_of_classes=12, train_fix=100, sequential=False, sequence_length=10):

    audio = tf.cast(record['audio'], tf.float32)
    label = tf.cast(record['label'], tf.int32)

    # calculating parameters for performing STFT
    frame_length = tf.cast(tf.cast(sample_rate, tf.float16) * 0.025, tf.int32)
    frame_step = tf.cast(tf.cast(sample_rate, tf.float16) * 0.01, tf.int32)

    # extracting STFT and generating spectrograms
    stft = tf.signal.stft(tf.squeeze(audio), frame_length=frame_length, frame_step=frame_step, fft_length=512)
    spectrogram = tf.abs(stft)

    # ZeroPadding the spectrograms
    zeros = tf.zeros(shape=(tf.constant(train_fix, dtype=tf.int32), tf.shape(spectrogram)[1]), dtype=tf.dtypes.float32)
    spectrogram = tf.concat([spectrogram, zeros], axis=0)
    spectrogram = spectrogram[:tf.constant(train_fix, dtype=tf.int32)]

    # Fixing the shape of the spectrogram
    if sequential:
        spectrogram = tf.reshape(spectrogram, (sequence_length, -1, tf.shape(spectrogram)[1]))
    spectrogram = tf.expand_dims(spectrogram, -1)

    label_one_hot = tf.one_hot(label, number_of_classes)
    
    return spectrogram, label_one_hot


def read_file_mfcc(file_name, label, number_of_classes=35, train_fix=120, sequential=False, sequence_length=10):
    # reading the content of the wav file and converting it to raw signal and extracting the sample rate
    raw_input = tf.io.read_file(file_name)
    wav, sample_rate = tf.audio.decode_wav(raw_input, desired_channels=1)

    # calculating parameters for performing STFT
    frame_length = tf.cast(tf.cast(sample_rate, tf.float16) * 0.025, tf.int32)
    frame_step = tf.cast(tf.cast(sample_rate, tf.float16) * 0.01, tf.int32)
    
    # extracting STFT and generating spectrograms
    stft = tf.signal.stft(tf.squeeze(wav), frame_length=frame_length, frame_step=frame_step, fft_length=512)
    spectrogram = tf.abs(stft)

    # converting the spectrograms to mel-scaled spectrograms
    num_spectrogram_bins = tf.shape(spectrogram)[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 257
    
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrograms = tf.reshape(mel_spectrograms, ( tf.concat([tf.shape(spectrogram)[:-1], tf.shape(linear_to_mel_weight_matrix)[-1:]], axis=0)))
    
    # extracting MFCCs
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :num_spectrogram_bins]

    
    # ZeroPadding the mfcc
    zeros = tf.zeros(shape=(tf.constant(train_fix, dtype=tf.int32), tf.shape(mfccs)[1]), dtype=tf.dtypes.float32)
    mfccs = tf.concat([mfccs, zeros], axis=0)
    mfccs = mfccs[:tf.constant(train_fix, dtype=tf.int32)]

    # Fixing the shape of the spectrogram
    if sequential:
        mfccs = tf.reshape(mfccs, (sequence_length, -1, tf.shape(mfccs)[1]))
    mfccs = tf.expand_dims(mfccs, -1)

    label_one_hot = tf.one_hot(label, number_of_classes)
    
    return mfccs, label_one_hot


def read_file_stft(file_name, label, number_of_classes=35, train_fix=120, sequential=False, sequence_length=10):
    # reading the content of the wav file and converting it to raw signal and extracting the sample rate
    raw_input = tf.io.read_file(file_name)
    wav, sample_rate = tf.audio.decode_wav(raw_input, desired_channels=1)

    # calculating parameters for performing STFT
    frame_length = tf.cast(tf.cast(sample_rate, tf.float16) * 0.025, tf.int32)
    frame_step = tf.cast(tf.cast(sample_rate, tf.float16) * 0.01, tf.int32)
    
    # extracting STFT and generating spectrograms
    stft = tf.signal.stft(tf.squeeze(wav), frame_length=frame_length, frame_step=frame_step, fft_length=512)
    spectrogram = tf.abs(stft)

    # ZeroPadding the spectrogram
    zeros = tf.zeros(shape=(tf.constant(train_fix, dtype=tf.int32), tf.shape(spectrogram)[1]), dtype=tf.dtypes.float32)
    spectrogram = tf.concat([spectrogram, zeros], axis=0)
    spectrogram = spectrogram[:tf.constant(train_fix, dtype=tf.int32)]

    # Fixing the shape of the spectrogram
    if sequential:
        spectrogram = tf.reshape(spectrogram, (sequence_length, -1, tf.shape(spectrogram)[1]))
    spectrogram = tf.expand_dims(spectrogram, -1)

    label_one_hot = tf.one_hot(label, number_of_classes)
    
    return spectrogram, label_one_hot


def load_train_tfds(
                    tfds_name = 'speech_commands', 
                    number_of_classes=12, 
                    train_fix=120,
                    sequential=False,
                    sequence_length=10,
                    use_stft=True,
                    batch_size=64):

    """
        This function loads and returns the train and validation split of tensorflow dataset specified by its name.

        inputs: 
            tfds_name :  the name of the dataset from the list of the existing datasets in tfds
            number_of_classes : the number of classes for the labeling of the dataset 
            train_fix : the fixed size of the training samples that are going to be fed to the DNN. Directly linked to the duration of the recording. (In milliseconds)
            sequential : Wether or not generate a sequential data from the input
            sequence_length : The length of the sequence when the sequential mode is selected
            use_stft: wether or not use stft as the extracted feature. use_stft=False : Use MFCC
            batch_size : the size of each batch used in training the DNN model for 1 step. 

        return: 
            [train_set, train_steps, validation_set, validation_steps, number_of_classes] : the tensorflow dataset instances and training/validation steps for training and validation sets as well as number of classes 

    """
    
    # check sequential compatiblity
    if sequential and train_fix%sequence_length!=0:
        raise ValueError('parameter sequence_length is not compatible with the input data, select a sequence_lengthfrom these values {values}'.format(values=[i for i in range(1, 1+train_fix//2) if train_fix%i==0]))


    # loading the tfds dataset
    train_set, validation_set = tfds.load(tfds_name, split=['train', 'validation'])

    # calculating training steps and validation steps
    train_steps = 1 + train_set.cardinality().numpy()//batch_size
    
    validation_steps = 1 + validation_set.cardinality().numpy()//batch_size
    
    # generating function wrappers for feature extraction 
    if use_stft:
        func = lambda x: read_record_stft(x, number_of_classes=number_of_classes, train_fix=train_fix, sequential=sequential, sequence_length=sequence_length)
    else:
        func = lambda x: read_record_mfcc(x, number_of_classes=number_of_classes, train_fix=train_fix, sequential=sequential, sequence_length=sequence_length)

    # mapping the wrapped feature exctraction functions to trainset and validation set
    train_set = train_set.map(func).shuffle(batch_size*10).batch(batch_size).repeat()
    validation_set = validation_set.map(func).shuffle(batch_size*10).batch(batch_size).repeat()

    input_shape = (sequence_length, train_fix//sequence_length, 257, 1) if sequential else (train_fix, 257, 1)

    return train_set, train_steps, validation_set, validation_steps, input_shape, number_of_classes


def load_train_from_file(
                    training_meta_file = '../Data/meta_data/speech_commands/training_list.txt',
                    validation_meta_file = '../Data/meta_data/speech_commands/validation_list.txt',
                    data_path = '../Data/speech_commands_v0.02',
                    train_fix=120,
                    sequential=False,
                    sequence_length=10,
                    use_stft=True,
                    batch_size=64):
    """
        This function generates a tensorflow dataset using meta_data files.

        arguments: 
            training_meta_file :  the file containing the training list of the dataset each sample is formatted as: "path/to/file.wav    label"
            validation_meta_file : the file containing the validation list of the dataset each sample is formatted as: "path/to/file.wav    label"
            data_path: the physical address of the root of the dataset
            train_fix : the fixed size of the training samples that are going to be fed to the DNN. Directly linked to the duration of the recording. (In milliseconds)
            sequential : Wether or not generate a sequential data from the input
            sequence_length : The length of the sequence when the sequential mode is selected
            use_stft: wether or not use stft as the extracted feature. use_stft=False : Use MFCC
            batch_size : the size of each batch used in training the DNN model for 1 step. 

        return: 
            [train_set, train_steps, validation_set, validation_steps, number_of_classes] : the tensorflow dataset instances and training/validation steps for training and validation sets as well as number of classes

    """
    # check sequential compatiblity
    if sequential and train_fix%sequence_length!=0:
        raise ValueError('parameter sequence_length is not compatible with the input data, select a sequence_lengthfrom these values {values}'.format(values=[i for i in range(1, 1+train_fix//2) if train_fix%i==0]))

    # reading the meta_files
    training_list = open(training_meta_file,'r').readlines()
    train_files = [os.path.join(data_path, x.split(' ')[0].strip()) for x in training_list]
    train_labels = [x.split(' ')[-1].strip() for x in training_list]


    validation_list = open(validation_meta_file,'r').readlines()
    validation_files = [os.path.join(data_path, x.split(' ')[0].strip()) for x in validation_list]
    validation_labels = [x.split(' ')[-1].strip() for x in validation_list]


    # converting the labels to indexes
    if np.all(np.unique(train_labels) == np.unique(validation_labels)):
        label_set = np.unique(train_labels)    
    else: 
        raise ValueError('training and validation labels are not equal')

    train_labels = [label_set.searchsorted(x) for x in train_labels]
    validation_labels = [label_set.searchsorted(x) for x in validation_labels]

        
    # creating tensorflow datasets from the lists
    train_set = tf.data.Dataset.from_tensor_slices((train_files, train_labels)).shuffle(batch_size*10)
    
    validation_set = tf.data.Dataset.from_tensor_slices((validation_files, validation_labels)).shuffle(batch_size*10)


    # calculating training steps and validation steps
    train_steps = 1 + train_set.cardinality().numpy()//batch_size
    
    validation_steps = 1 + validation_set.cardinality().numpy()//batch_size


    # generating function wrappers for feature extraction 
    if use_stft:
        func = lambda x, y: read_file_stft(x, y, number_of_classes=len(label_set), train_fix=train_fix, sequential=sequential, sequence_length=sequence_length)
    else:
        func = lambda x, y: read_file_mfcc(x, y, number_of_classes=len(label_set), train_fix=train_fix, sequential=sequential, sequence_length=sequence_length)

    # mapping the wrapped feature exctraction functions to trainset and validation set
    train_set = train_set.map(func).shuffle(batch_size*10).batch(batch_size).repeat()
    validation_set = validation_set.map(func).shuffle(batch_size*10).batch(batch_size).repeat()

    input_shape = (sequence_length, train_fix//sequence_length, 257, 1) if sequential else (train_fix, 257, 1)

    return train_set, train_steps, validation_set, validation_steps, input_shape, len(label_set)


def load_test_tfds(
                    tfds_name = 'speech_commands', 
                    number_of_classes=12, 
                    train_fix=120,
                    sequential=False,
                    sequence_length=10,
                    use_stft=True,
                    batch_size=64):

    """
        This function loads and returns the test split of tensorflow dataset specified by its name.

        inputs: 
            tfds_name :  the name of the dataset from the list of the existing datasets in tfds
            number_of_classes : the number of classes for the labeling of the dataset 
            train_fix : the fixed size of the training samples that are going to be fed to the DNN. Directly linked to the duration of the recording. (In milliseconds)
            sequential : Wether or not generate a sequential data from the input
            sequence_length : The length of the sequence when the sequential mode is selected
            use_stft: wether or not use stft as the extracted feature. use_stft=False : Use MFCC
            batch_size : the size of each batch used in training the DNN model for 1 step. 

        return: 
            [test_set, test_steps] : the tensorflow dataset instances and training/validation steps for training and validation sets as well as number of classes 

    """
    
    # check sequential compatiblity
    if sequential and train_fix%sequence_length!=0:
        raise ValueError('parameter sequence_length is not compatible with the input data, select a sequence_lengthfrom these values {values}'.format(values=[i for i in range(1, 1+train_fix//2) if train_fix%i==0]))


    # loading the tfds dataset
    test_set = tfds.load(tfds_name, split='test')

    # calculating training steps and validation steps
    test_steps = 1 + test_set.cardinality().numpy()//batch_size
    
    
    # generating function wrappers for feature extraction 
    if use_stft:
        func = lambda x: read_record_stft(x, number_of_classes=number_of_classes, train_fix=train_fix, sequential=sequential, sequence_length=sequence_length)
    else:
        func = lambda x: read_record_mfcc(x, number_of_classes=number_of_classes, train_fix=train_fix, sequential=sequential, sequence_length=sequence_length)

    # mapping the wrapped feature exctraction functions to test set
    test_set = test_set.map(func).shuffle(batch_size*10).batch(batch_size)

    input_shape = (sequence_length, train_fix//sequence_length, 257, 1) if sequential else (train_fix, 257, 1)

    return test_set, test_steps, input_shape, number_of_classes

def load_test_from_file(
                    testing_meta_file = '../Data/meta_data/speech_commands/testing_list.txt',
                    data_path = '../Data/speech_commands_v0.02',
                    train_fix=120,
                    sequential=False,
                    sequence_length=10,
                    use_stft=True,
                    batch_size=64):
    """
        This function generates a tensorflow dataset using meta_data files.

        arguments: 
            training_meta_file :  the file containing the training list of the dataset each sample is formatted as: "path/to/file.wav    label"
            validation_meta_file : the file containing the validation list of the dataset each sample is formatted as: "path/to/file.wav    label"
            data_path: the physical address of the root of the dataset
            train_fix : the fixed size of the training samples that are going to be fed to the DNN. Directly linked to the duration of the recording. (In milliseconds)
            sequential : Wether or not generate a sequential data from the input
            sequence_length : The length of the sequence when the sequential mode is selected
            use_stft: wether or not use stft as the extracted feature. use_stft=False : Use MFCC
            batch_size : the size of each batch used in training the DNN model for 1 step. 

        return: 
            [train_set, train_steps, validation_set, validation_steps, number_of_classes] : the tensorflow dataset instances and training/validation steps for training and validation sets as well as number of classes

    """
    # check sequential compatiblity
    if sequential and train_fix%sequence_length!=0:
        raise ValueError('parameter sequence_length is not compatible with the input data, select a sequence_lengthfrom these values {values}'.format(values=[i for i in range(1, 1+train_fix//2) if train_fix%i==0]))

    # reading the meta_files
    test_list = open(testing_meta_file,'r').readlines()
    test_files = [os.path.join(data_path, x.split(' ')[0].strip()) for x in test_list]
    test_labels = [x.split(' ')[-1].strip() for x in test_list]


    # converting the labels to indexes
    label_set = np.unique(test_labels)    

    test_labels = [label_set.searchsorted(x) for x in test_labels]

        
    # creating tensorflow datasets from the lists
    test_set = tf.data.Dataset.from_tensor_slices((test_files, test_labels)).shuffle(batch_size*10)


    # calculating training steps and validation steps
    test_steps = 1 + test_set.cardinality().numpy()//batch_size


    # generating function wrappers for feature extraction 
    if use_stft:
        func = lambda x, y: read_file_stft(x, y, number_of_classes=len(label_set), train_fix=train_fix, sequential=sequential, sequence_length=sequence_length)
    else:
        func = lambda x, y: read_file_mfcc(x, y, number_of_classes=len(label_set), train_fix=train_fix, sequential=sequential, sequence_length=sequence_length)

    # mapping the wrapped feature exctraction functions to trainset and validation set
    test_set = test_set.map(func).shuffle(batch_size*10).batch(batch_size)

    input_shape = (sequence_length, train_fix//sequence_length, 257, 1) if sequential else (train_fix, 257, 1)

    return test_set, test_steps, input_shape, len(label_set)


if __name__ == '__main__':
    test_set, test_steps, number_of_classes = load_test_from_file(sequential=True, sequence_length=10, use_stft=True)
    for spectrogram, label in test_set.take(test_steps):
        print(np.shape(spectrogram), np.shape(label), list(map(np.argmax, label)))
