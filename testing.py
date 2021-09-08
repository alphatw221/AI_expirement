
from ResNets.resnet import resnet101, resnet50
from nfnets.dataset import makeDataset, makeDatasetFromDLS
import pathlib
import os
import tensorflow as tf
import cv2
import numpy as np

#python38_env\Scripts\python.exe testing.py 
data_path=r'D:\SynpowerLabelData\s017004tt1304-07a-dmc#s017004tt1304-07a-dmc[I001-M2012050090]_6#20210104#200'

train_imsize = 200
test_imsize = 200
batchSize=8
epoch=10

ds_train, ds_valid, ds_test, trainSize = makeDatasetFromDLS(
    dataPath=data_path,
    batchSize=batchSize,
    epochSize=epoch,
    trainImageSize=train_imsize,
    testImageSize=test_imsize,
    splitRatio=(9,1)
)

steps_per_epoch = int(trainSize / batchSize)
training_steps = (trainSize * epoch) / batchSize

with tf.device('/gpu:0'):

    
    input = tf.keras.layers.Input(shape=(200,200,6))
    out = resnet50(input)
    model=tf.keras.models.Model(input,out)
   
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                    loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        
    model.fit(
        ds_train,
        validation_data=ds_valid,
        epochs=epoch,
        steps_per_epoch=steps_per_epoch
    )