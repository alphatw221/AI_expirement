import argparse
from typing import Callable

import tensorflow as tf
import tensorflow_addons as tfa

from nfnets.nfnet import NFNet, nfnet_params
from nfnets.other import WarmUpSchedule
from Dataset.dataset import makeBasicDataset, makeDLS_Dataset, makeQiuDataset

import pathlib
import os
import datetime

LABEL_SMOOTHING=0.1     #二分類標籤0,1大約修飾成0.1,0.9  
LEARNING_RATE=0.1       #使用AGD 論文使用0.1
DROPRATE=0.2            #F0 架構 使用0.2
EMA_DECAY=0.99999       #matrics使用
CLIPPING_FACTOR=0.01     #論文建議0.01

VARIANT = "F0"
DATA_PATH = "xxxxx"
BATCH_SIZE=16
EPOCH=25

RAW_IMAGE_SIZE=200
RESIZE=nfnet_params[VARIANT]["train_imsize"]

class Cfg():
    def __init__(self):
        self.batch_size = BATCH_SIZE
        self.epoch = EPOCH
        self.raw_image_size = RAW_IMAGE_SIZE
        self.resize = RESIZE

def main(args):
    
    train_imsize = nfnet_params[VARIANT]["train_imsize"]
    test_imsize = nfnet_params[VARIANT]["test_imsize"]
    
    ds_train, ds_valid, ds_test, trainSize = makeDLS_Dataset(
        dataPaths=[r'C:\Users\AlphaLin\Desktop\s017004tt1304-07a-dmc#s017004tt1304-07a-dmc[I001-M2012050090]_6#20210104#200'],      # 台灣健鼎
        cfg=Cfg()
    )

    steps_per_epoch = int(trainSize / BATCH_SIZE)
    training_steps = (trainSize * EPOCH) / BATCH_SIZE
    
    
    
    model = NFNet(
        num_classes=2,
        variant=args.variant,
        drop_rate=DROPRATE,
        label_smoothing=LABEL_SMOOTHING,
        ema_decay=EMA_DECAY,              
        clipping_factor=CLIPPING_FACTOR
    )

    model.build((EPOCH, RESIZE, RESIZE, 3))  #batch_input_shape

    max_lr = LEARNING_RATE * BATCH_SIZE / 256

    lr_decayed_fn = tf.keras.experimental.CosineDecay(
        initial_learning_rate=max_lr,
        decay_steps=training_steps - 5 * steps_per_epoch,
    )
    lr_schedule = WarmUpSchedule(
        initial_learning_rate=max_lr,   #初始IR=LEARNING_RATE* batch_size/256   batch越大 lr越大
        decay_schedule_fn=lr_decayed_fn,
        warmup_steps=5 * steps_per_epoch,  #warmup_steps設為五個epoch #第五個epoch後使用cosineDecay #第五個以前使用 初始IR*warmup完成百分比
    )
    optimizer = tfa.optimizers.SGDW(
        learning_rate=lr_schedule, weight_decay=2e-5, momentum=0.9
    )
    
    model.compile(
        optimizer=optimizer,
        # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['categorical_accuracy']
        #TODO 自訂一TP TN FP FN metrics
    )

    log_dir = "logs/fit/" +"normal_r50_"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(
        ds_train,
        validation_data=ds_valid,
        epochs=EPOCH,
        steps_per_epoch=steps_per_epoch,
        callbacks=[tensorboard_callback],
    )

        

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    with tf.device('/gpu:0'):
        main()