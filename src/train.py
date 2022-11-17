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
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--model', default='ResNet', choices=['LSTM', 'GRU', 'CRNN', 'MobileNet', 'MobileNetV2', 'MobileNetV3Small', 'ResNet', 'SEResNet', 'VGG_M'], type=str)
parser.add_argument('--sequential_model', default=False, action='store_true')
parser.add_argument('--sequence_length', default=10, type=int)


# set up dataset configuration.
parser.add_argument('--use_tfds', default=True ,action='store_true')
parser.add_argument('--tfds_name', default='speech_commands', type=str)

parser.add_argument('--use_file', dest='use_tfds', default=True ,action='store_false')
parser.add_argument('--training_meta_file', default='../Data/meta_data/speech_commands/training_list.txt', type=str)
parser.add_argument('--validation_meta_file', default='../Data/meta_data/speech_commands/validation_list.txt', type=str)
parser.add_argument('--data_path', default= '../Data/speech_commands_v0.02', type=str)

parser.add_argument('--use_stft', dest='use_stft', default=True ,action='store_true')
parser.add_argument('--use_mfcc', dest='use_stft' ,action='store_false')

parser.add_argument('--batch_size', default=64, type=int)


# set up learning rate, training loss and optimizer.
parser.add_argument('--epochs', default=56, type=int)
parser.add_argument('--lr', default=0.0001, type=float)

# set up TPU cload execution
parser.add_argument('--use_tpu', default=False, action='store_true')
parser.add_argument('--tpu_name', default='', type=str)



global args
args = parser.parse_args()


def main():

    if args.use_tfds:
        # using tfds api to load and generate 
        train_set, train_steps, validation_set, validation_steps, input_shape, number_of_classes = dataset.load_train_tfds(
                                                                                                tfds_name = args.tfds_name, 
                                                                                                number_of_classes=12, 
                                                                                                train_fix=120, 
                                                                                                sequential=args.sequential_model,
                                                                                                sequence_length=args.sequence_length,
                                                                                                use_stft=args.use_stft,
                                                                                                batch_size=args.batch_size)
    else:
        # load dataset from meta_data files
        train_set, train_steps, validation_set, validation_steps, input_shape, number_of_classes = dataset.load_train_from_file(
                                                                                                training_meta_file = args.training_meta_file,
                                                                                                validation_meta_file = args.validation_meta_file,
                                                                                                data_path = args.data_path,
                                                                                                train_fix=120,
                                                                                                sequential=args.sequential_model,
                                                                                                sequence_length=args.sequence_length,
                                                                                                use_stft=args.use_stft,
                                                                                                batch_size=args.batch_size)

    
    if args.use_tpu:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu_name)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    else:
        strategy = tf.distribute.MirroredStrategy()

    

    with strategy.scope():

        model = models.create_model(args, input_shape, number_of_classes)
        if args.model not in ['MobileNetV3Small']:
            profile = model_profiler(model, args.batch_size)
            print('\n\n\n\n\n', profile, '\n\n\n\n\n')

    if args.resume:
        model.load_weights(args.resume, by_name=True)
        print('model loaded successfully.')



    normal_lr = keras.callbacks.LearningRateScheduler(step_decay)

    tbcallbacks = keras.callbacks.TensorBoard(log_dir=os.path.join('../log', args.model),
                                              histogram_freq=0,
                                              write_graph=True,
                                              write_images=False,
                                              update_freq=args.batch_size * 16)

    checkpoint = keras.callbacks.ModelCheckpoint(
                    os.path.join('../saved_models', args.model,'weights.h5'),
                    monitor='val_acc',
                    mode='max',
                    save_best_only=True)

    callbacks = [checkpoint, normal_lr, tbcallbacks]

    model.fit(
        train_set,
        steps_per_epoch=train_steps,
        callbacks=callbacks,
        epochs=args.epochs,
        validation_data=validation_set,
        validation_steps=validation_steps)


def step_decay(epoch):
    '''
    The learning rate begins at 10^initial_power,
    and decreases by a factor of 10 every step epochs.
    '''
    half_epoch = args.epochs // 2
    stage1, stage2, stage3 = int(half_epoch * 0.5), int(half_epoch * 0.8), half_epoch
    stage4 = stage3 + stage1
    stage5 = stage4 + (stage2 - stage1)
    stage6 = args.epochs

    milestone = [stage1, stage2, stage3, stage4, stage5, stage6]
    gamma = [1.0, 0.1, 0.01, 1.0, 0.1, 0.01]

    lr = 0.005
    init_lr = args.lr
    stage = len(milestone)
    for s in range(stage):
        if epoch < milestone[s]:
            lr = init_lr * gamma[s]
            break
    print('Learning rate for epoch {} is {}.'.format(epoch + 1, lr))
    return float(lr)


if __name__ == '__main__':
    main()
