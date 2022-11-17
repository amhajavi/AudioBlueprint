# Speech Command Recognition

This is a framework for implementing and testing different Deep Neural Networks (DNNs) for speech command recognition from audio signal. This framework has been implemented using **Python3.10.7** and **Tensofrlow 2.10**. 

# Getting Started

The dependencies used in this project are all listed in "Requirements.txt" placed in the root of the project. Using any virtual environment such as `virtualenv`or `conda` is possible and encouraged. 

To install all the dependencies using `virtualenv` simple use:

    virtualenv -P path/to/python3 venv
    source venv/bin/activate
    pip install -r Requirements.txt
    

 
## Datasets and Folder Structure
We use `google speech command` dataset for training and testing the  models.

```
@article{speechcommandsv2,   
	author = { {Warden}, P.},    
	title = "{Speech Commands: A Dataset for Limited-Vocabulary 	Speech Recognition}",  
	journal = {ArXiv e-prints},  
	year = 2018,
	url = {https://arxiv.org/abs/1804.03209}
}
```

In this project 2 options are provided: 

 - **Using Tensorflow Datasets**: This options provides a cleaned and simplified version of the dataset including only 12 classes from the 35 classes available in the original version of the dataset. The recordings in this option are saved in the raw wav format. Selecting this option is possible by passing `--use_tfds` flag and providing the name of the dataset by using option `--tfds_name speech_commands` during training and testing of the models.  

 - **Using The Original Datasets**: This options provides the complete version of the dataset including all of the 35 classes available in the dataset. Selecting this option is possible by passing `--use_file` flag during training and testing of the models. The original dataset should be downloaded manually and the path the the root of the dataset should be passed to provided during the training and testing of the models by using the option `--data_path path/to/the/root/of/dataset`. Additionally the meta_data files for training, validation, and testing should be generated. An already generated version of these files are saved in `Data/speech_commands` folder under the names of `training_list.txt`, `validation_list.txt`, and `testing_list.txt`. The path to training_list and validation_list files should be given to the training script using the option `--training_meta_file` and `--validation_meta_file`. Default values for these options is set to the already generated lists. The path for testing_list file should be given to the testing script using the option `--testing_meta_file`. Default value for this option is also the already generated testing list. 

## Feature Extraction and Dataloader

The project provides two sets of feature extraction methods which are implemented in `src/datasets.py`:

 - **MFCC**: This method enables the framework to use  "Mel-frequency cepstral coefficients" as the main features for the input of the DNNs. This method can be selected using the option `--use_mfcc` during the training and testing of the models. It should be noted that both training and testing of the models should be done using the same method. The computation load of generating MFCCs are higher compared to the other options. 
 
 - **STFT**: This method enables the framework to use the initial "spectrograms" (output of the "Short-Term Fourier Transform") directly without bringing them to the mel-scale. This method can be selected using the option `--use_stft` during training and testing of the method.  It should be noted that both training and testing of the models should be done using the same method. The computational load of using STFT option is considerably lower than MFCC option. 

## Training DNNs
Traning the DNNs in this frame work is done using the script `src/train.py`. The command for training the model is:

    cd src
    python train.py
				    --model ['LSTM', 'GRU', 'CRNN', 'MobileNet', 'MobileNetV2', 'MobileNetV3Small', 'ResNet', 'SEResNet', 'VGG_M'] # For selecting one of the implemented models 
				    --resume [path]               # Provides the path to the pre-trained weights of the model from previous execution 
				    --sequential_model            # Wether the model uses recurrent architecture or not
				    --sequence_length [NUMBER]    # The length of the sequence if the --seqeuntial_model is selected
				    {--use_tfds|--use_file}       # Wether to use dataset provided by tfds or use files
				    --tfds_name                   # The name of the tfds dataset if --use_tfds is selected default value is "speech_commands"
				    --training_meta_file [path]   # The path to the training metadata file if --use_file is selected
				    --validation_meta_file [path] # The path to the validation metadata file if --use_file is selected
				    --data_path [path]            # The path to the root of the downloaded speech command dataset
				    {--use_stft|--use_mfcc}       # Using either STFT or MFCC for training of the models
				    --batch_size [NUMBER]         # The batch size used for training the models
				    --epochs                      # The number epochs the model should be trained for
				    --lr                          # The initial learning rate used for training the model 
				    --use_tpu                     # Wether or not to use TPU from google cload service (Should be used in the google cloud platform)
				    --tpu_name                    # The name of the TPU node (Should be used in google cload platform)

### Bonus Options

 - Training of the DNN models is logged using tensorboard toolkit and the logs are saved for each model inside the `log` folder. Accessing the logs and viewing the results is possible using the command: 
 `tensorboard --logdir log/[DNN_MODEL]`
 The list of the DNN models is provided below. 
 
 - The learning rate for training the models is adjusted using the `step_decay` function implemented in the `train.py`.  The `--lr` option is to set the initial learning rate for the training. 

 - While running the training script the generated model is analyzed for the memory-size that is required for deploying the model. The result of this analysis is presented in the console when running either of `train.py` or `test.py` scripts. 

 - While training the models, each model is evaluated on the validation set of the `google speech commands` dataset. At the end of the training the best weights based on this evaluation are saved in the `saved_models` folder under the name of the model itself. These weights can either be used in `--resume` option or they can be automatically used in `test.py`.

 - `--use_tpu` option is operable for executing the training script on google cloud platform and in this version of the framework only works with `--use_tfds` option. 

## DNNs 

All the implementations of the models can be found in `src/backbones` folder. The header for all of the models are attached to the models in `src/models.py` where the model is selected and created inside function `create_model`.  Here is the list of DNNs available in this framework:

 - **LSTM**: This model uses a `fully-connected` layer to generate embedding from the input features followed by  2 `bidirectional LSTM` layers which learn dependencies between these embeddings throughout time. The model is then followed by a `Self-Attention`. 
 - **GRU**: This model  uses a `fully-connected` layer to generate embedding from the input features followed by  2 `bidirectional GRU` layers which learn dependencies between these embeddings throughout time. The model is then followed by a `Self-Attention`. 
 - **CRNN**: This model  uses  4 `Convolutional` layers to generate embedding from the input features followed by  2 `bidirectional LSTM` layers which learn dependencies between these embeddings throughout time. The model is then followed by a `Self-Attention`.
 - **MobileNet**: The series of MobileNet models in this framework are directly adopted from the tensorflow/keras implementations of the MobileNet. These implementations can be found in: [https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/MobileNet](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/MobileNet)
 - **MobileNetV2**: The series of MobileNet models in this framework are directly adopted from the tensorflow/keras implementations of the MobileNet. These implementations can be found in: [https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/MobileNet](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/MobileNet)
 - **MobileNetV3small**: The series of MobileNet models in this framework are directly adopted from the tensorflow/keras implementations of the MobileNet. These implementations can be found in: [https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/MobileNet](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/MobileNet)
 - **ResNet**: This model is an adaptation of the model ResNet34. The model is implemented based on thin-ResNet34 model introduced in the paper below. The implementation for all of the components of the this model (network, ghostvlad aggregation, and etc.) can be found in both `src/backbones` and `src/tools` folders. 
```
@article{chung2020defence,
  title={In Defence of Metric Learning for Speaker Recognition},
  author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee-Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
  journal={Proc. Interspeech 2020},
  pages={2977--2981},
  year={2020}
}
```
 - **SEResNet**: This model is another adaptation of the ResNet34 model. The architecture of the the model is very similar to the thin-ResNet34 with the key difference that the `identity-blocks` of the thin-ResNet34 model is replaced with `squeeze-and-excitation blocks` from paper below. The implementation for all of the components of the this model (network, ghostvlad aggregation, and etc.) can be found in both `src/backbones` and `src/tools` folders. 
 ```
 @artivle{hu2018squeeze,
  title={Squeeze-and-excitation networks},
  author={Hu, Jie and Shen, Li and Sun, Gang},
  journal={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={7132--7141},
  year={2018}
}
```
-**VGG_M**: This model is a lightweight VGG model implemented from the details mentioned in paper below. 
```
@article{nagrani2017voxceleb,
  title={VoxCeleb: A Large-Scale Speaker Identification Dataset},
  author={Nagrani, Arsha and Chung, Joon Son and Zisserman, Andrew},
  journal={Proc. Interspeech 2017},
  pages={2616--2620},
  year={2017}
}
```
   

## Testing

The DNNs in this framework are tested on the test set of the `google speech commands` dataset. We use 2 metrics of `Accuracy` and `Top2Accuracy` for evaluating the performance the DNNs against this dataset. The command for testing the DNNs is:

```
cd src
    python test.py
				    --model ['LSTM', 'GRU', 'CRNN', 'MobileNet', 'MobileNetV2', 'MobileNetV3Small', 'ResNet', 'SEResNet', 'VGG_M'] # For selecting one of the implemented models 

				    --sequential_model            # Wether the model uses recurrent architecture or not
				    --sequence_length [NUMBER]    # The length of the sequence if the --seqeuntial_model is selected
				    {--use_tfds|--use_file}       # Wether to use dataset provided by tfds or use files
				    --tfds_name                   # The name of the tfds dataset if --use_tfds is selected default value is "speech_commands"
				    --training_meta_file [path]   # The path to the training metadata file if --use_file is selected
				    --validation_meta_file [path] # The path to the validation metadata file if --use_file is selected
				    --data_path [path]            # The path to the root of the downloaded speech command dataset
				    {--use_stft|--use_mfcc}       # Using either STFT or MFCC for training of the models
				    --batch_size [NUMBER]         # The batch size used for training the models		   
```

## Next Steps 
These are some of the possible next steps in development of this framework:

 - Introducing the option to select the number of frequency-bins (the number of bins in STFT or MFCC) to the training script to be able to perform a thorough analysis on the effect of the input on the performance. 
 - Including a more configurable model creation method to generate the models from YAML files to be able to tweak the models more easily.  This makes it easier to perform a grid search on different design parameters of the DNN architectures. 
- Currently the original version of the `google speech commands` dataset (available through the option `--use_file`) is not cleaned and there are some issues with consistency of the dataset itself. A possible next step would be to analyze the splits of the dataset and clean it thoroughly for better training of the models. 
