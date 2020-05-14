from keras.models import Model, Sequential, load_model
from keras.layers import LSTM, Masking, Dense,  Bidirectional, Dropout, MaxPooling1D, Conv1D, Activation, Flatten, Input, Multiply
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Nadam

#recommend using masking: dna_mask_blstm for dna, aa_mask_blstm for amino acids
#note: the embedding layer is always named "lstm_2"

#amino acid, no masking
def original_blstm(num_classes, num_letters, sequence_length, embed_size=50):
	model = Sequential()
	model.add(Conv1D(input_shape=(sequence_length, num_letters), filters=320, kernel_size=26, padding="valid", activation="relu"))
	model.add(MaxPooling1D(pool_length=13, stride=13))
	model.add(Dropout(0.2))
	model.add(Bidirectional(LSTM(320, activation="tanh", return_sequences=True)))
	model.add(Dropout(0.5))
	#model.add(LSTM(num_classes, activation="softmax", name="AV"))
	model.add(LSTM(embed_size, activation="tanh"))
	model.add(Dense(num_classes, activation=None, name="AV"))
	model.add(Activation("softmax"))
	model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
	return model

#dna (extra layer to convert 3 nucleotides to amino acids), no masking
def dna_blstm(num_classes, num_letters, sequence_length, mask_length=None, embed_size=256):
	model = Sequential()
        model.add(Conv1D(input_shape=(sequence_length, num_letters), filters=26, kernel_size=3, strides=3, padding="valid", activation="relu"))
        model.add(Conv1D(filters=320, kernel_size=26, padding="valid", activation="relu"))
	model.add(MaxPooling1D(pool_length=13, stride=13))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(320, activation="tanh", return_sequences=True)))
        model.add(Dropout(0.5))
        #model.add(LSTM(num_classes, activation="softmax", name="AV"))
        model.add(LSTM(embed_size, activation="tanh"))
        model.add(Dense(num_classes, activation=None, name="AV"))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

def dna_mask_blstm(num_classes, num_letters, sequence_length, mask_length, embed_size=256):
	x = Input(shape=(sequence_length, num_letters))
	m = Input(shape=(mask_length, 1))
	conv1 = Conv1D(filters=26, kernel_size=3, strides=3, padding="valid", activation="relu")(x)
	conv2 = Conv1D(filters=320, kernel_size=26, padding="valid", activation="relu")(conv1)
	pool = MaxPooling1D(pool_length=13, stride=13)(conv2)
	masked = Multiply()([pool, m])
	masking = Masking(mask_value=0)(masked)
	drop1 = Dropout(0.2)(masking)
	blstm = Bidirectional(LSTM(320, activation="tanh", return_sequences=True))(drop1)
	drop2 = Dropout(0.5)(blstm)
	lstm = LSTM(embed_size, activation="tanh")(drop2)
	dense = Dense(num_classes, activation=None, name="AV")(lstm)
	out = Activation("softmax")(dense)
	model = Model(inputs=[x,m], outputs=out)
	model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
	return model

def aa_mask_blstm(num_classes, num_letters, sequence_length, mask_length, embed_size=256):
	x = Input(shape=(sequence_length, num_letters))
        m = Input(shape=(mask_length, 1))
        conv = Conv1D(filters=320, kernel_size=26, padding="valid", activation="relu")(x)
        pool = MaxPooling1D(pool_length=13, stride=13)(conv)
        masked = Multiply()([pool, m])
        masking = Masking(mask_value=0)(masked)
        drop1 = Dropout(0.2)(masking)
        blstm = Bidirectional(LSTM(320, activation="tanh", return_sequences=True))(drop1)
        drop2 = Dropout(0.5)(blstm)
        lstm = LSTM(embed_size, activation="tanh")(drop2)
        dense = Dense(num_classes, activation=None, name="AV")(lstm)
        out = Activation("softmax")(dense)
        model = Model(inputs=[x,m], outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model
