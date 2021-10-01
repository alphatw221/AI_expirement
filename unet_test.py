import tensorflow as tf
from tensorflow import keras
from keras_unet_collection import models
import numpy as np
import cv2
import os 
import time

END_SIZE=400
LAYER_NAME=['via','pad','text','green1','green2','green3']
def main():
    #1:via, 2:pad, 3:text, 4:green1, 5:green2, 6:green
    model = models.unet_plus_2d((None, None, 3), [64, 128, 256, 512], n_labels=6,
                                stack_num_down=2, stack_num_up=2,
                                activation='LeakyReLU', output_activation='Softmax', 
                                batch_norm=False, pool='max', unpool=False, deep_supervision=False, name='xnet')

    model.load_weights('unet++_400.h5')

    folderNames=os.listdir('./ImageData')
    folderPaths=[os.path.join('./ImageData',f) for f in folderNames]

    for folderPath in folderPaths:
        image=getData(folderPath)
        output=np.zeros((image.shape[0],image.shape[1],len(LAYER_NAME)))

        for i in range(66):
            for j in range(3):
                test_input=image[None,i*END_SIZE:(i+1)*END_SIZE,j*END_SIZE:(j+1)*END_SIZE,:]
                test_predict=model.predict(test_input)[0]
                print(test_predict.shape)
                output[i*END_SIZE:(i+1)*END_SIZE,j*END_SIZE:(j+1)*END_SIZE,:]=test_predict

        # for i in range(int(image.shape[0]/END_SIZE)):
        #     test_input=image[None,i*END_SIZE:i*END_SIZE+END_SIZE,:,:]
        #     test_predict=model.predict(test_input)[0]
        #     output[i*END_SIZE:i*END_SIZE+END_SIZE,:,:]=test_predict

        # if image.shape[0]%END_SIZE !=0:
        #     test_input=image[None,-1*END_SIZE:,:,:]  #尾端
        #     test_predict=model.predict(test_input)[0]
        #     output[-1*END_SIZE:,:,:]=test_predict

        
        newFolderName = os.path.basename(os.path.normpath(folderPath))
        resultPath=os.path.join('./result',newFolderName)
        if not os.path.exists(resultPath):
            os.makedirs(resultPath)

        cv2.imwrite(os.path.join(resultPath,'DS_image.bmp'),image*255)
        for i in range(output.shape[-1]):
            cv2.imwrite(os.path.join(resultPath,'DS_'+LAYER_NAME[i]+'.bmp'),output[:,:,i]*255)

def getData(folderPath):
    image=cv2.imread(os.path.join(folderPath,'image.bmp'))/255
    # image=cv2.resize(image, (END_SIZE, int(image.shape[0]/2) ) )
    # image=cv2.resize(image, (END_SIZE, 15484 ) )
    image=cv2.resize(image, (1200, 26400 ) )
    return image

if __name__=='__main__':
    start_time = time.time()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    with tf.device('/gpu:0'):
        main()
    
    print("--- %s seconds ---" % (time.time() - start_time))