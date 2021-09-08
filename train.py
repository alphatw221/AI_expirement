import argparse
from typing import Callable

import tensorflow as tf
import tensorflow_addons as tfa

from nfnets.nfnet import NFNet, nfnet_params
from nfnets.dataset import makeDataset, makeDatasetFromDLS
from nfnets.other import WarmUpSchedule

import pathlib
import os

LABEL_SMOOTHING=0.1     #二分類標籤0,1大約修飾成0.1,0.9  
LEARNING_RATE=0.1       #使用AGD 論文使用0.1
DROPRATE=0.2            #F0 架構 使用0.2
EMA_DECAY=0.99999       #matrics使用
CLIPPING_FACTOR=0.01     #論文建議0.01


#python38_env\Scripts\python.exe train.py --data_path=\\192.168.1.65\d\TrainCreate\Dataset\LianQiao_Green_Golden_OSP_25um\dataset20210903163210
#python38_env\Scripts\python.exe train.py --data_path=C:\Users\AlphaLin\Desktop\LianQiao\Dataset\LianQiao_Green_Golden_OSP_25um\dataset20210706200358
#python38_env\Scripts\python.exe train.py --data_path=C:\Users\AlphaLin\Desktop\LianQiao\Dataset\LianQiao_Green_Golden_OSP_25um\dataset20210706200358 --model_path=C:\Users\AlphaLin\Desktop\NFestNet\model\default.h5
#python38_env\Scripts\python.exe train.py --data_path=\\192.168.1.65\d\LIN\s017004tt1304-07a-dmc#s017004tt1304-07a-dmc[I001-M2012050090]_6#20210104#200
#python38_env\Scripts\python.exe train.py --data_path=D:\SynpowerLabelData\s017004tt1304-07a-dmc#s017004tt1304-07a-dmc[I001-M2012050090]_6#20210104#200
def parse_args():
    ap = argparse.ArgumentParser()

    #data_path
    ap.add_argument(
        "-dp",
        "--data_path",
        type=str,
        help="training data path",
        required=True
    )

    #model_path
    ap.add_argument(
        "-mp",
        "--model_path",
        default=None,
        type=str,
        help="model path",
    )

    ap.add_argument(
        "-v",
        "--variant",
        default="F0",
        type=str,
        help="model variant",
    )
    ap.add_argument(
        "-b",
        "--batch_size",
        default=8,
        type=int,
        help="train batch size",
    )
    ap.add_argument(
        "-n",
        "--num_epochs",
        default=5,
        type=int,
        help="number of training epochs",
    )
    
    return ap.parse_args()


def main(args):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    train_imsize = nfnet_params[args.variant]["train_imsize"]
    test_imsize = nfnet_params[args.variant]["test_imsize"]
    
    ds_train, ds_valid, ds_test, trainSize = makeDatasetFromDLS(
        dataPath=args.data_path,
        batchSize=args.batch_size,
        epochSize=args.num_epochs,
        trainImageSize=train_imsize,
        testImageSize=test_imsize
    )

    steps_per_epoch = int(trainSize / args.batch_size)
    training_steps = (trainSize * args.num_epochs) / args.batch_size
    
    
    with tf.device('/gpu:0'):
        print("Initialing Model...")
        model = NFNet(
            num_classes=2,
            variant=args.variant,
            drop_rate=DROPRATE,
            label_smoothing=LABEL_SMOOTHING,
            ema_decay=EMA_DECAY,              
            clipping_factor=CLIPPING_FACTOR
        )
        model.build((1, train_imsize, train_imsize, 6))  #batch_input_shape

        max_lr = LEARNING_RATE * args.batch_size / 256

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


        if args.model_path and os.path.isfile(args.model_path):
            print("Load model weights...")
            model.load_weights(args.model_path,by_name=False)
            print("Load model successfully.")
        
                
        model.fit(
            ds_train,
            validation_data=ds_valid,
            epochs=args.num_epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[tf.keras.callbacks.TensorBoard()],
        )

        filePath=args.model_path if args.model_path else './model/default.h5'

        dirName = os.path.dirname(filePath)
        if not os.path.exists(dirName):
            os.makedirs(dirName)

        for i in range(len(model.weights)):
            model.weights[i]._handle_name = str(i)   #weight name 保密
        model.save_weights(filePath,overwrite=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)