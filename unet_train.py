from sys import getdefaultencoding
import tensorflow as tf
from tensorflow import keras
from keras_unet_collection import models
import numpy as np
import cv2
import random
import os

END_SIZE=400
EPOCH = 2 # number of epoches
STEP = 10 # number of batches per epoch   (step)
MAX_TOL = 100 # the max-allowed early stopping patience
MIN_DEL = 0 # the lowest acceptable loss value reduction 

def main():
    model = models.unet_plus_2d((None, None, 3), [64, 128, 256, 512], n_labels=6,
                                stack_num_down=2, stack_num_up=2,
                                activation='LeakyReLU', output_activation='Softmax', 
                                batch_norm=False, pool='max', unpool=False, deep_supervision=False, name='xnet')

    #如果使用Deep_supervision的話loss要自己定義 BCE + dice 因為輸出有維度 
    # model.compile(
        # loss=["binary_crossentropy","binary_crossentropy","binary_crossentropy","binary_crossentropy"], 
        # loss_weights=[0.1, 0.1, 0.1, 0.7], 
        # optimizer=keras.optimizers.Adam(learning_rate=3e-4))
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=3e-4))


    folderName=os.listdir('./ImageData')
    folderPath=[os.path.join('./ImageData',f) for f in folderName]
    
    tol = 0 # current early stopping patience
    stop=False
    for epoch in range(EPOCH):    
        if stop: break
        for fp in folderPath:
            image, label_image=getData(fp) 

            valid_position=0
            # ri = random.randint(0,65)
            # rj = random.randint(0,2)
            # valid_input=image[None,ri*END_SIZE:(ri+1)*END_SIZE,rj*END_SIZE:(rj+1)*END_SIZE,:]
            # valid_label=label_image[None,ri*END_SIZE:(ri+1)*END_SIZE,rj*END_SIZE:(rj+1)*END_SIZE,:]

            # if epoch == 0:
            #     # record = model.evaluate([valid_input,],[valid_label, valid_label, valid_label, valid_label])
            #     record = model.evaluate([valid_input,],[valid_label,])
            #     print('\tInitial loss = {}'.format(record))
            record = 0.5
            print('\tInitial loss = {}'.format(record))

            for i in range(132):
                if stop: break
                for j in range(6):
                    train_input=image[None,i*200:i*200+END_SIZE,j*200:j*200+END_SIZE,:]
                    train_label=label_image[None,i*200:i*200+END_SIZE,j*200:j*200+END_SIZE,:]
                    # loss_ = model.train_on_batch([train_input,], [train_label,train_label,train_label,train_label])
                    loss_ = model.train_on_batch([train_input,], [train_label,])

                    if np.mean(record) - np.mean(loss_) > MIN_DEL:
                        print('Validation performance is improved from {} to {}'.format(record, loss_))
                        record = loss_
                        tol = 0
                        
                    else:
                        print('Validation performance {} is NOT improved'.format(loss_))
                        tol += 1
                        if tol >= MAX_TOL:
                            print('Early stopping')
                            stop=True
                            break
                        else:
                            continue

            # for step in range(STEP):
            #     i=random.randint(0,image.shape[0]-END_SIZE-1)
            #     train_input=image[None,i:i+END_SIZE,:,:]
            #     train_label=label_image[None,i:i+END_SIZE,:,:]
            #     loss_ = model.train_on_batch([train_input,], [train_label,train_label,train_label,train_label])
            
           
            # record_temp = model.evaluate([valid_input,],[valid_label, valid_label, valid_label, valid_label])
            # record_temp = model.evaluate([valid_input,],[valid_label, ])

            # if np.mean(record) - np.mean(record_temp) > MIN_DEL:
            #     print('Validation performance is improved from {} to {}'.format(record, record_temp))
            #     record = record_temp
            #     tol = 0
                
            # else:
            #     print('Validation performance {} is NOT improved'.format(record_temp))
            #     tol += 1
            #     if tol >= MAX_TOL:
            #         print('Early stopping')
            #         stop=True
            #         break
            #     else:
            #         continue

    for i in range(len(model.weights)):
        model.weights[i]._handle_name = str(i)   #weight name 保密
    model.save_weights('unet++_400.h5',overwrite=False)


def getData(folderPath):
    via=cv2.imread(os.path.join(folderPath,'1.bmp'))[:,:,0,None].astype(bool).astype(np.uint8)
    pad=cv2.imread(os.path.join(folderPath,'2.bmp'))[:,:,0,None].astype(bool).astype(np.uint8)
    text=cv2.imread(os.path.join(folderPath,'3.bmp'))[:,:,0,None].astype(bool).astype(np.uint8)
    green1=cv2.imread(os.path.join(folderPath,'4.bmp'))[:,:,0,None].astype(bool).astype(np.uint8)
    green2=cv2.imread(os.path.join(folderPath,'5.bmp'))[:,:,0,None].astype(bool).astype(np.uint8)
    green3=cv2.imread(os.path.join(folderPath,'6.bmp'))[:,:,0,None].astype(bool).astype(np.uint8)

    label_image=np.concatenate([via,pad,text,green1,green2,green3],axis=2)
    image=cv2.imread(os.path.join(folderPath,'image.bmp'))/255

    label_image=cv2.resize(label_image, (1200, 26400 ) )
    image=cv2.resize(image, (1200, 26400 ) )

    return image, label_image

if __name__=='__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    with tf.device('/gpu:0'):
        main()