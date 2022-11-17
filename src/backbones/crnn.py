from tensorflow.keras.layers import Input ,LSTM, Reshape, Dense, TimeDistributed, Bidirectional, Flatten, BatchNormalization, Conv2D, GlobalAveragePooling2D

from keras_self_attention import SeqSelfAttention

from tensorflow.keras.models import Model

def crnn(input_dim = (10, 12, 257)):
	inputs = Input(shape=input_dim)
	cnn_embedding_1 = TimeDistributed(Conv2D(256, kernel_size=(2, 2), strides=(1,2), activation='relu', padding='valid'))(inputs)
	cnn_embedding_1 = TimeDistributed(BatchNormalization())(cnn_embedding_1)
	cnn_embedding_2 = TimeDistributed(Conv2D(256, kernel_size=(2, 4), strides=(1,2), activation='relu', padding='valid'))(cnn_embedding_1)
	cnn_embedding_2 = TimeDistributed(BatchNormalization())(cnn_embedding_2)
	cnn_embedding_3 = TimeDistributed(Conv2D(256, kernel_size=(2, 4), strides=(1,2), activation='relu', padding='valid'))(cnn_embedding_2)
	cnn_embedding_3 = TimeDistributed(BatchNormalization())(cnn_embedding_3)
	cnn_embedding_4 = TimeDistributed(Conv2D(256, kernel_size=(2, 4), strides=(1,2), activation='relu', padding='valid'))(cnn_embedding_3)
	cnn_embedding_4 = TimeDistributed(GlobalAveragePooling2D())(cnn_embedding_4)
	embedding = TimeDistributed(Dense(256, activation='relu'))(cnn_embedding_4)
	embedding = TimeDistributed(BatchNormalization())(embedding)
	first_bilstm = Bidirectional(LSTM(256, return_sequences=True, activation='relu'))(embedding)
	second_bilstm = Bidirectional(LSTM(256, return_sequences=True, activation='relu', dropout=0.2))(first_bilstm)
	self_attention = SeqSelfAttention(attention_activation='sigmoid')(second_bilstm)
	flattened = Flatten()(self_attention)

	return inputs, flattened

