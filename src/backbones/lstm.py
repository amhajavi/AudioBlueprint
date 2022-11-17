from tensorflow.keras.layers import Input, Reshape, LSTM, Dense, TimeDistributed, Bidirectional, Flatten, BatchNormalization

from keras_self_attention import SeqSelfAttention

from tensorflow.keras.models import Model

def lstm(input_dim = (10, 12, 257)):
    inputs = Input(shape=input_dim)
    end_flattened = Reshape((input_dim[0], -1))(inputs)
    embedding = TimeDistributed(Dense(256, activation='relu'))(end_flattened)
    embedding = TimeDistributed(BatchNormalization())(embedding)
    first_bilstm = Bidirectional(LSTM(256, return_sequences=True, activation='relu'))(embedding)
    second_bilstm = Bidirectional(LSTM(256, return_sequences=True, activation='relu', dropout=0.2))(first_bilstm)
    self_attention = SeqSelfAttention(attention_activation='sigmoid')(second_bilstm)
    flattened = Flatten()(self_attention)
    

    return inputs, flattened

if __name__ == '__main__':
	x, y = lstm()
	model = Model(x, y)
	model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
	model.summary()