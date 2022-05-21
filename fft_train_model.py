# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 20:11:24 2021

@author: sanaalamgeer
"""
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[1], 'GPU') #gpus[0], gpus[1], gpus[2]
##########################################################################################################
#pip install --upgrade tf_fourier_features
import numpy as np
import keras
from keras.layers import Input, Conv2D, ELU, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Dropout, concatenate, add,  AdditiveAttention, Conv1D, MaxPooling1D, LSTM, Reshape
from tensorflow.keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import time
import matplotlib.pyplot as plt
from skimage import color
#from tf_fourier_features import FourierFeatureProjection
#from tf_fourier_features import FourierFeatureMLP

#%%
epochs = 2000
width, height = 81, 512
#%%
def convert_rg2gray(img):
	grayImg = color.rgb2gray(img)
	return grayImg

def generate_fft(img):
	f = np.fft.fft2(img)
	fshift = np.fft.fftshift(f)
	magnitude_spectrum = 20 * np.log(np.abs(fshift))
	return magnitude_spectrum

def get_model(width, height):
	stream1_input = Input(shape=(height, width, 1))
	#x = FourierFeatureProjection(gaussian_projection = 15, gaussian_scale = 1.0)(left_image)
	#conv1
	stream1=Conv2D(32, (3, 3), padding='same', name='conv1_left')(stream1_input)
	stream1=ELU()(stream1)
	stream1=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1_left')(stream1)
	#conv2
	stream1=Conv2D(32, (3, 3), padding='same', name='conv2_left')(stream1)
	stream1=ELU()(stream1)
	stream1=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2_left')(stream1)
	#conv3
	stream1=Conv2D(64, (3, 3), padding='same', name='conv3_left')(stream1)
	stream1=ELU()(stream1)
	#conv4
	stream1=Conv2D(64, (3, 3), padding='same', name='conv4_left')(stream1)
	stream1=ELU()(stream1)
	#conv5
	stream1=Conv2D(128, (3, 3), padding='same', name='conv5_left')(stream1)
	stream1=ELU()(stream1)
	stream1=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5_left')(stream1)
	#fc6
	stream1=Flatten()(stream1)
	stream1=Dense(512)(stream1)
	stream1=ELU()(stream1)
	stream1=Dropout(0.5)(stream1)
	#fc7
	stream1=Dense(512)(stream1)
	stream1=ELU()(stream1)
	stream1=Dropout(0.5)(stream1)
	
	###############################################################################
	#right image
	stream2_input = Input(shape=(height, width, 1))
	#x2 = FourierFeatureProjection(gaussian_projection = 15, gaussian_scale = 1.0)(right_image)
	#conv1
	stream2=Conv2D(32, (3, 3), padding='same', name='conv1_right')(stream2_input)
	stream2=ELU()(stream2)
	stream2=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1_right')(stream2)
	#conv2
	stream2=Conv2D(32, (3, 3), padding='same', name='conv2_right')(stream2)
	stream2=ELU()(stream2)
	stream2=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2_right')(stream2)
	#conv3
	stream2=Conv2D(64, (3, 3), padding='same', name='conv3_right')(stream2)
	stream2=ELU()(stream2)
	#conv4
	stream2=Conv2D(64, (3, 3), padding='same', name='conv4_right')(stream2)
	stream2=ELU()(stream2)
	#conv5
	stream2=Conv2D(128, (3, 3), padding='same', name='conv5_right')(stream2)
	stream2=ELU()(stream2)
	stream2=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5_right')(stream2)
	#fc6
	stream2=Flatten()(stream2)
	stream2=Dense(512)(stream2)
	stream2=ELU()(stream2)
	stream2=Dropout(0.5)(stream2)
	#fc7
	stream2=Dense(512)(stream2)
	stream2=ELU()(stream2)
	stream2=Dropout(0.5)(stream2)
	###############################################################################
	#concatenate1
	add_conv2 = concatenate([stream1, stream2])
	final_fusion = ELU()(add_conv2)
	#fc6
	final_fc = Flatten()(final_fusion)
	final_fc = Dense(512)(final_fc)
	final_fc = ELU()(final_fc)
	final_fc = Dropout(0.5)(final_fc)
	#fc7
	final_fc = Dense(512)(final_fc)
	final_fc = ELU()(final_fc)
	final_fc = Dropout(0.5)(final_fc)
	
	#concatenate3
	#fusion3_drop7 = concatenate([left_drop7, right_drop7, fusion2_drop7])
	#fc8
	final_fc = Dense(1024)(final_fc)
	#fc9
	predictions = Dense(1)(final_fc)
	
	model_all = keras.Model([stream1_input, stream2_input], predictions, name='all_model')
	model_all.summary()

def compile_model(model):
	sgd=SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1) #lr=0.0001
	model.compile(loss='mean_squared_error', optimizer=sgd)
	
	return model

def run_model(model, stream1_input, stream2_input, labels):
	# simple early stopping
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
	mc = ModelCheckpoint('model/fft_model.hdf5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

	#fitting model
	history = model.fit(x=[stream1_input, stream2_input], y=[labels], validation_split=0.2, batch_size=128, epochs=epochs, verbose=1, callbacks=[es, mc], shuffle=True)
	
	#saving history
	np.save('fft_model_history.npy',history.history)

	print('Training complete!')
#END#
