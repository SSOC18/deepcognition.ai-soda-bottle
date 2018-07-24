import keras
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.applications.vgg16 import VGG16
from keras.layers.core import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input
from keras.models import Model
from keras.regularizers import *


def get_model():
	aliases = {}
	Input_1 = Input(shape=(3, 480, 640), name='Input_1')
	VGG16_8_model = VGG16(include_top= False, input_tensor = Input_1)
	VGG16_8 = VGG16_8_model(Input_1)
	aliases['VGG16_8'] = VGG16_8_model.name
	num_layers = len(VGG16_8_model.layers)
	for i, layer in enumerate(VGG16_8_model.layers):
		if ((i * 100) / (num_layers - 1)) <= (100 - 0):
			layer.trainable = False
	Convolution2D_12 = Convolution2D(name='Convolution2D_12',nb_filter= 3,nb_row= 9,activation= 'relu' ,nb_col= 9)(VGG16_8)
	MaxPooling2D_13 = MaxPooling2D(name='MaxPooling2D_13')(Convolution2D_12)
	Flatten_19 = Flatten(name='Flatten_19')(MaxPooling2D_13)
	Dense_1 = Dense(name='Dense_1',output_dim= 512,activation= 'relu' )(Flatten_19)
	Dropout_1 = Dropout(name='Dropout_1',p= 0.3)(Dense_1)
	Dense_2 = Dense(name='Dense_2',output_dim= 512,activation= 'relu' )(Dropout_1)
	Dropout_2 = Dropout(name='Dropout_2',p= 0.3)(Dense_2)
	Dense_3 = Dense(name='Dense_3',output_dim= 8,activation= 'softmax' )(Dropout_2)

	model = Model([Input_1],[Dense_3])
	return aliases, model


from keras.optimizers import *

def get_optimizer():
	return Adadelta()

def is_custom_loss_function():
	return False

def get_loss_function():
	return 'categorical_crossentropy'

def get_batch_size():
	return 1

def get_num_epoch():
	return 10

def get_data_config():
	return '{"samples": {"test": 661, "training": 5292, "validation": 661, "split": 4}, "shuffle": false, "datasetLoadOption": "batch", "kfold": 1, "dataset": {"samples": 6615, "type": "public", "name": "Soda Bottles"}, "numPorts": 1, "mapping": {"Filename": {"port": "InputPort0", "options": {"Width": 28, "horizontal_flip": false, "Scaling": 1, "Augmentation": false, "vertical_flip": false, "shear_range": 0, "pretrained": "None", "Resize": false, "height_shift_range": 0, "Normalization": false, "width_shift_range": 0, "Height": 28, "rotation_range": 0}, "shape": "", "type": "Image"}, "Label": {"port": "OutputPort0", "options": {}, "shape": "", "type": "Categorical"}}}'