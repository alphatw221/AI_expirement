from ResNets.resnet import r50_subtract, resnet101, resnet50, r101_subtract
from Dataset.dataset import makeBasicDataset, makeDLS_Dataset, makeQiuDataset
import os
import tensorflow as tf
import cv2
import numpy as np
import random
import matplotlib.cm as cm
import datetime


BATCH_SIZE=16
EPOCH=25

RAW_IMAGE_SIZE=200
RESIZE=192

class Cfg():
    def __init__(self):
        self.batch_size = BATCH_SIZE
        self.epoch = EPOCH
        self.raw_image_size = RAW_IMAGE_SIZE
        self.resize = RESIZE

def main():

    ds_train, ds_valid, trainSize = makeBasicDataset(
        trainDir=r'C:\Users\AlphaLin\Desktop\baselin_train',             #自訂資料集
        valDir=r'C:\Users\AlphaLin\Desktop\baselin_val',
        cfg=Cfg()
    )

    # ds_train, ds_valid, ds_test, trainSize = makeDLS_Dataset(
    #     dataPaths=[r'C:\Users\AlphaLin\Desktop\s017004tt1304-07a-dmc#s017004tt1304-07a-dmc[I001-M2012050090]_6#20210104#200'],      # 台灣健鼎
    #     cfg=Cfg()
    # )

    # ds_train, ds_valid, ds_test, trainSize = makeQiuDataset(            # 阿丘
    #     trainDir=r'C:\Users\AlphaLin\Desktop\qiu_train',
    #     testDir=r'C:\Users\AlphaLin\Desktop\qiu_test',
    #     cfg=Cfg()
    # )


    steps_per_epoch = int(trainSize / BATCH_SIZE)
        
    input = tf.keras.layers.Input(shape=(RESIZE,RESIZE,3))
    # out = resnet101(input)
    out = resnet50(input)
    
    # input = tf.keras.layers.Input(shape=(RESIZE,RESIZE,6))
    # # out = r101_6channel(input)
    # out = r50_6channel(input)

    model=tf.keras.models.Model(input,out)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss='categorical_crossentropy', metrics=['categorical_accuracy'])
                    # loss='binary_crossentropy', metrics=['binary_accuracy'])

    model.load_weights('./model/normal_r50.h5')

    log_dir = "logs/fit/" +"normal_r50_"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(
        ds_train,
        validation_data=ds_valid,
        epochs=EPOCH,
        steps_per_epoch=steps_per_epoch,
        callbacks=[tensorboard_callback]
    )

    # 輸出熱像圖影像
    for d in ds_valid:
        heatMap=make_gradcam_heatmap(d[0],model,'res5c_out')
        heatMap=cv2.cvtColor(heatMap, cv2.COLOR_BGR2RGB)

        oneHotPrediction=model.predict(d[0])[0]
        predictClass=tf.math.argmax(oneHotPrediction,axis=0)
        labelClass=tf.math.argmax(d[1][0],axis=0)

        # originalImage = cv2.cvtColor(d[0][0][:,:,3:].numpy()*255, cv2.COLOR_BGR2RGB)
        originalImage = cv2.cvtColor(d[0][0].numpy()*255, cv2.COLOR_BGR2RGB)

        heatMapImage = cv2.addWeighted(originalImage, 1, heatMap, 0.3, 0)
        checkImage=cv2.hconcat([originalImage, heatMapImage])
        if labelClass.numpy()==predictClass.numpy():
            if labelClass.numpy()==0:
                cv2.imwrite('./result/TN/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.jpg', checkImage)
            else:
                cv2.imwrite('./result/TP/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.jpg', checkImage)
        else:
            if labelClass.numpy()==0:
                cv2.imwrite('./result/FP/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.jpg', checkImage)
            else:
                cv2.imwrite('./result/FN/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.jpg', checkImage)


# 產生熱像圖
def make_gradcam_heatmap(img, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    
    jet_colors = jet(np.arange(256))[:, :3]*255
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = cv2.resize(jet_heatmap,(img[0].shape[1], img[0].shape[0]))
    return jet_heatmap.astype(np.float32)

if __name__=='__main__':
    with tf.device('/gpu:0'):
        main()