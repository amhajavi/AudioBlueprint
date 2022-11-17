import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import dataset
import models

from model_profiler import model_profiler


sys.path.append('../')
# ===========================================
#        Parse the argument
# ===========================================
import argparse
parser = argparse.ArgumentParser()

# set up model configuration
parser.add_argument('--model', default='ResNet', choices=['LSTM', 'GRU', 'CRNN', 'MobileNet', 'MobileNetV2', 'MobileNetV3Small', 'ResNet', 'SEResNet', 'VGG_M'], type=str)
parser.add_argument('--sequential_model', default=False, action='store_true')
parser.add_argument('--sequence_length', default=10, type=int)


# set up dataset configuration.
parser.add_argument('--use_tfds', default=True ,action='store_true')
parser.add_argument('--tfds_name', default='speech_commands', type=str)
parser.add_argument('--use_stft', dest='use_stft', default=True ,action='store_true')
parser.add_argument('--use_mfcc', dest='use_stft' ,action='store_false')

parser.add_argument('--use_file', dest='use_tfds', default=True ,action='store_false')
parser.add_argument('--testing_meta_file', default='../Data/meta_data/speech_commands/testing_list.txt', type=str)
parser.add_argument('--data_path', default= '../Data/speech_commands_v0.02', type=str)
parser.add_argument('--batch_size', default=64, type=int)

global args
args = parser.parse_args()

def main():

    if args.use_tfds:
        # using tfds api to load and generate 
        test_set, test_steps, input_shape, number_of_classes = dataset.load_test_tfds(
                                                                                        tfds_name = args.tfds_name, 
                                                                                        number_of_classes=12, 
                                                                                        train_fix=120, 
                                                                                        sequential=args.sequential_model,
                                                                                        sequence_length=args.sequence_length,
                                                                                        use_stft=args.use_stft,
                                                                                        batch_size=args.batch_size)
    else:
        # load dataset from meta_data files
        test_set, test_steps, input_shape, number_of_classes = dataset.load_test_from_file(
                                                                                            testing_meta_file = args.testing_meta_file,
                                                                                            data_path = args.data_path,
                                                                                            train_fix=120,
                                                                                            sequential=args.sequential_model,
                                                                                            sequence_length=args.sequence_length,
                                                                                            use_stft=args.use_stft,
                                                                                            batch_size=args.batch_size)
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = models.create_model(args, input_shape, number_of_classes)
        if args.model not in ['MobileNetV3Small']:
            profile = model_profiler(model, args.batch_size)
            print('\n\n\n\n\n', profile, '\n\n\n\n\n')


    weights_file = os.path.join('../saved_models', args.model,'weights.h5')

    if os.path.exists(weights_file):
        model.load_weights(weights_file, by_name=True)
        print('model loaded successfully.')
    else:
        raise IOError('The model {} has not been trained yet please run the training for the model first'.format(args.model))


    predictions, ground_truth_labels = [], []

    metrics = {
            "Accuracy": tf.keras.metrics.Accuracy(),
            "Top2": tf.keras.metrics.TopKCategoricalAccuracy(k=2)

    }

    for batch, labels in test_set.take(test_steps):
        predictions = model.predict_on_batch(batch)
        metrics['Accuracy'].update_state(list(map(np.argmax, labels)), list(map(np.argmax, predictions)))
        metrics['Top2'].update_state( labels, predictions)

    print("The results for Accuracy and Top2 Accuracy of the model on the test set are: {}, {}".format(metrics['Accuracy'].result().numpy(), metrics['Top2'].result().numpy()))        
    

if __name__ == "__main__":
    main()