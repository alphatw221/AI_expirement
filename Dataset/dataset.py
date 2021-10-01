import os
from resnet_training import EPOCH, RAW_IMAGE_SIZE
import tensorflow as tf
import cv2
import numpy as np
import random

def makeQiuDataset(trainDir, testDir, cfg):

    RAW_IMAGE_SIZE = cfg.raw_image_size
    RESIZE = cfg.resize
    BATCH_SIZE = cfg.batch_size
    EPOCH = cfg.epoch
    
    def load_file_train(goldPath, scanPath, label):#裡面的變數都是tensor

        goldImage = cv2.imread(goldPath.numpy().decode())
        goldImage=cv2.resize(goldImage,(RAW_IMAGE_SIZE,RAW_IMAGE_SIZE))
        goldImage = cv2.cvtColor(goldImage, cv2.COLOR_BGR2RGB)
        scanImage = cv2.imread(scanPath.numpy().decode())
        scanImage=cv2.resize(goldImage,(RAW_IMAGE_SIZE,RAW_IMAGE_SIZE))
        scanImage = cv2.cvtColor(scanImage, cv2.COLOR_BGR2RGB)

        rotate=random.randint(1,4)
        if rotate==1:
            goldImage = cv2.rotate(goldImage, cv2.ROTATE_90_COUNTERCLOCKWISE) 
            scanImage = cv2.rotate(scanImage, cv2.ROTATE_90_COUNTERCLOCKWISE) 
        elif rotate==2:
            goldImage = cv2.rotate(goldImage, cv2.cv2.ROTATE_90_CLOCKWISE) 
            scanImage = cv2.rotate(scanImage, cv2.cv2.ROTATE_90_CLOCKWISE) 
        elif rotate==3:
            goldImage = cv2.rotate(goldImage, cv2.ROTATE_180) 
            scanImage = cv2.rotate(scanImage, cv2.ROTATE_180) 
        else:
            pass

        image=np.concatenate((goldImage,scanImage),axis=2)
        # image = scanImage
        
        flip=random.randint(1,4)
        if flip==1:
            image=cv2.flip(image,1)
        elif flip ==2:
            image=cv2.flip(image,0)                 
        elif flip ==3:
            image=cv2.flip(image,-1)
        else:
            pass

        ri=random.randint(1,7)
        rj=random.randint(1,7)
        image = image[ri:ri+RESIZE,rj:rj+RESIZE,:]

        return image/255, np.eye(2)[ int( label.numpy() ) ]#oneHot

    def load_file_test(goldPath, scanPath, label):#裡面的變數都是tensor

        goldImage = cv2.imread(goldPath.numpy().decode())
        goldImage=cv2.resize(goldImage,(RAW_IMAGE_SIZE,RAW_IMAGE_SIZE))
        goldImage = cv2.cvtColor(goldImage, cv2.COLOR_BGR2RGB)
        scanImage = cv2.imread(scanPath.numpy().decode())
        scanImage=cv2.resize(goldImage,(RAW_IMAGE_SIZE,RAW_IMAGE_SIZE))
        scanImage = cv2.cvtColor(scanImage, cv2.COLOR_BGR2RGB)

        image=np.concatenate((goldImage,scanImage),axis=2)
        # image = scanImage
        
        image = image[4:4+RESIZE,4:4+RESIZE,:]

        return image/255, np.eye(2)[ int( label.numpy() ) ]#oneHot

    def mappable_fn_train(goldName, scanName, label):
        image,label = tf.py_function(func=load_file_train,inp=[goldName, scanName, label],Tout=[tf.float32,tf.float32])
        return image, label

    def mappable_fn_test(goldName, scanName, label):
        image,label = tf.py_function(func=load_file_test,inp=[goldName, scanName, label],Tout=[tf.float32,tf.float32])
        return image, label
    #真點母版
    NG_Gold=os.path.join(trainDir,"NG","GoldenImage") 
    NG_Gold=[os.path.join(NG_Gold,f) for f in os.listdir(NG_Gold)]
    #真點料版
    NG_Scan=os.path.join(trainDir,"NG","Scanimage")
    NG_Scan=[os.path.join(NG_Scan,f) for f in os.listdir(NG_Scan)] 
    #假點母版
    OK_Gold=os.path.join(trainDir,"OK","GoldenImage")
    OK_Gold=[os.path.join(OK_Gold,f) for f in os.listdir(OK_Gold)] 
    #假點料版
    OK_Scan=os.path.join(trainDir,"OK","Scanimage")
    OK_Scan=[os.path.join(OK_Scan,f) for f in os.listdir(OK_Scan)] 

    GoldFiles = (NG_Gold + OK_Gold)
    ScanFiles = (NG_Scan + OK_Scan)
    labels = [1]*len(NG_Scan) + [0]*len(OK_Scan) 

    datasetSize=len(labels)
    trainDataset=tf.data.Dataset.from_tensor_slices((GoldFiles,ScanFiles,labels)).shuffle(buffer_size=datasetSize)
    
    # trainSize = int( SPLIT_RATE * datasetSize)

    # valDataset=trainDataset.skip(trainSize)
    # trainDataset=trainDataset.take(trainSize)
    
    trainDataset=trainDataset.repeat(EPOCH).map(lambda x,y,z:mappable_fn_train(x, y, z)).batch(BATCH_SIZE)
    # valDataset=valDataset.map(lambda x,y,z:mappable_fn_train(x, y, z)).batch(BATCH_SIZE)



    #真點母版
    testNG_Gold=os.path.join(testDir,"NG","GoldenImage") 
    testNG_Gold=[os.path.join(testNG_Gold,f) for f in os.listdir(testNG_Gold)]
    #真點料版
    testNG_Scan=os.path.join(testDir,"NG","Scanimage")
    testNG_Scan=[os.path.join(testNG_Scan,f) for f in os.listdir(testNG_Scan)] 
    #假點母版
    testOK_Gold=os.path.join(testDir,"OK","GoldenImage")
    testOK_Gold=[os.path.join(testOK_Gold,f) for f in os.listdir(testOK_Gold)] 
    #假點料版
    testOK_Scan=os.path.join(testDir,"OK","Scanimage")
    testOK_Scan=[os.path.join(testOK_Scan,f) for f in os.listdir(testOK_Scan)] 

    testGoldFiles = (testNG_Gold + testOK_Gold)
    testScanFiles = (testNG_Scan + testOK_Scan)
    testlabels = [1]*len(testNG_Scan) + [0]*len(testOK_Scan) 

    testDatasetSize=len(testlabels)
    testDataset=tf.data.Dataset.from_tensor_slices((testGoldFiles,testScanFiles,testlabels)).shuffle(buffer_size=testDatasetSize)

    testDataset=testDataset.map(lambda x,y,z:mappable_fn_test(x, y, z)).batch(BATCH_SIZE)

    return trainDataset, testDataset, testDataset, datasetSize




def makeDLS_Dataset(dataPaths, cfg):

    RAW_IMAGE_SIZE = cfg.raw_image_size
    RESIZE = cfg.resize
    BATCH_SIZE = cfg.batch_size
    EPOCH = cfg.epoch

    def load_file(goldPath, scanPath, label, imageSize):#裡面的變數都是tensor

        goldImage = cv2.imread(goldPath.numpy().decode())
        goldImage = cv2.cvtColor(goldImage, cv2.COLOR_BGR2RGB)
        scanImage = cv2.imread(scanPath.numpy().decode())
        scanImage = cv2.cvtColor(scanImage, cv2.COLOR_BGR2RGB)
        image=np.concatenate((goldImage,scanImage),axis=2)
        # image = scanImage
        image=cv2.resize(image,(imageSize.numpy(),imageSize.numpy()))
        oneHot=np.eye(2)[ int( label.numpy() ) ]
        return image/255, oneHot
        
    def mappable_fn(goldName, scanName, label, imageSize):
        image,label = tf.py_function(func=load_file,inp=[goldName, scanName, label, imageSize],Tout=[tf.float32,tf.float32])
        
        return image, label

    GoldFiles=[]
    ScanFiles=[]
    labels=[]
    for dataPath in dataPaths:
        #真點母版
        TP_Gold=os.path.join(dataPath,"TP","GoldenImage") 
        FN_Gold=os.path.join(dataPath,"FN","GoldenImage")
        NG_Gold=[os.path.join(TP_Gold,f) for f in os.listdir(TP_Gold)] + [os.path.join(FN_Gold,f) for f in os.listdir(FN_Gold)]
        #真點料版
        TP_Scan=os.path.join(dataPath,"TP","Scanimage")
        FN_Scan=os.path.join(dataPath,"FN","Scanimage")
        NG_Scan=[os.path.join(TP_Scan,f) for f in os.listdir(TP_Scan)] + [os.path.join(FN_Scan,f) for f in os.listdir(FN_Scan)]
        #假點母版
        TN_Gold=os.path.join(dataPath,"TN","GoldenImage")
        FP_Gold=os.path.join(dataPath,"FP","GoldenImage")
        OK_Gold=[os.path.join(TN_Gold,f) for f in os.listdir(TN_Gold)] + [os.path.join(FP_Gold,f) for f in os.listdir(FP_Gold)]
        #假點料版
        TN_Scan=os.path.join(dataPath,"TN","Scanimage")
        FP_Scan=os.path.join(dataPath,"FP","Scanimage")
        OK_Scan=[os.path.join(TN_Scan,f) for f in os.listdir(TN_Scan)] + [os.path.join(FP_Scan,f) for f in os.listdir(FP_Scan)]
    
        GoldFiles+=(NG_Gold+OK_Gold)
        ScanFiles+=(NG_Scan+OK_Scan)
        labels+=( [1]*len(NG_Scan)+[0]*len(OK_Scan) )


    datasetSize=len(labels)
    dataset=tf.data.Dataset.from_tensor_slices((GoldFiles,ScanFiles,labels)).shuffle(buffer_size=datasetSize)
    
    trainSize = int( 0.8 * datasetSize)
    valSize = int(0.1 * datasetSize)

    trainDataset=dataset.take(trainSize)
    rest=dataset.skip(trainSize)
    valDataset=rest.take(valSize)
    testDataset=rest.skip(valSize)

    trainDataset=trainDataset.repeat(EPOCH).map(lambda x,y,z:mappable_fn(x, y, z, RESIZE)).batch(BATCH_SIZE)
    valDataset=valDataset.repeat(EPOCH).map(lambda x,y,z:mappable_fn(x, y, z, RESIZE)).batch(BATCH_SIZE)
    testDataset=testDataset.repeat(EPOCH).map(lambda x,y,z:mappable_fn(x, y, z, RESIZE)).batch(BATCH_SIZE)

    return trainDataset, valDataset, testDataset, trainSize


def makeBasicDataset(trainDir, valDir, cfg):
    RAW_IMAGE_SIZE = cfg.raw_image_size
    RESIZE = cfg.resize
    BATCH_SIZE = cfg.batch_size
    EPOCH = cfg.epoch

    def load_file_train(filePath, label):
        
        image = cv2.imread(filePath.numpy().decode())
        if(image.shape[0]!=RAW_IMAGE_SIZE or image.shape[1]!=RAW_IMAGE_SIZE*2):
            image=cv2.resize(image,(RAW_IMAGE_SIZE*2,RAW_IMAGE_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        scanImage = image[:,:RAW_IMAGE_SIZE,:]
        goldImage = image[:,RAW_IMAGE_SIZE:,:]

        rotate=random.randint(1,4)
        if rotate==1:
            goldImage = cv2.rotate(goldImage, cv2.ROTATE_90_COUNTERCLOCKWISE) 
            scanImage = cv2.rotate(scanImage, cv2.ROTATE_90_COUNTERCLOCKWISE) 
        elif rotate==2:
            goldImage = cv2.rotate(goldImage, cv2.cv2.ROTATE_90_CLOCKWISE) 
            scanImage = cv2.rotate(scanImage, cv2.cv2.ROTATE_90_CLOCKWISE) 
        elif rotate==3:
            goldImage = cv2.rotate(goldImage, cv2.ROTATE_180) 
            scanImage = cv2.rotate(scanImage, cv2.ROTATE_180) 
        else:
            pass

        # image=np.concatenate((goldImage,scanImage),axis=2)
        image = scanImage
        
        flip=random.randint(1,4)
        if flip==1:
            image=cv2.flip(image,1)
        elif flip ==2:
            image=cv2.flip(image,0)                 
        elif flip ==3:
            image=cv2.flip(image,-1)
        else:
            pass

        ri=random.randint(1,7)
        rj=random.randint(1,7)
        image = image[ri:ri+RESIZE,rj:rj+RESIZE,:]

        return image/255, np.eye(2)[ int( label.numpy() ) ]#oneHot
    

    def mappable_fn_train(filePath, label):
        image, label = tf.py_function(func=load_file_train,inp=[filePath, label],Tout=[tf.float32,tf.float32])
        return image, label

    trainFiles=[]
    trainLabels=[]

    catFolders=os.listdir(trainDir)
    for catFolder in catFolders:
        NG_Dir=os.path.join(trainDir,catFolder,"NG") 
        NG_Path=[os.path.join(NG_Dir,fileName) for fileName in os.listdir(NG_Dir) ]

        OK_Dir=os.path.join(trainDir,catFolder,"OK",) 
        OK_Path=[os.path.join(OK_Dir,fileName) for fileName in os.listdir(OK_Dir) ]

        trainFiles+=NG_Path
        trainLabels+=[1]*len(NG_Path)
        trainFiles+=OK_Path
        trainLabels+=[0]*len(OK_Path)

    trainDatasetSize=len(trainLabels)
    trainDataset=tf.data.Dataset.from_tensor_slices((trainFiles,trainLabels)).shuffle(buffer_size=trainDatasetSize)
    trainDataset=trainDataset.repeat(EPOCH).map(lambda x,y:mappable_fn_train(x, y)).batch(BATCH_SIZE)


    def load_file_test(filePath, label):
        
        image = cv2.imread(filePath.numpy().decode())
        if(image.shape[0]!=RAW_IMAGE_SIZE or image.shape[1]!=RAW_IMAGE_SIZE*2):
            image=cv2.resize(image,(RAW_IMAGE_SIZE*2,RAW_IMAGE_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        scanImage = image[:,:RAW_IMAGE_SIZE,:]
        goldImage = image[:,RAW_IMAGE_SIZE:,:]

        # image=np.concatenate((goldImage,scanImage),axis=2)                        
        image = scanImage
        
        image = image[4:4+RESIZE,4:4+RESIZE,:]

        return image/255, np.eye(2)[ int( label.numpy() ) ]#oneHot
        
    def mappable_fn_test(filePath, label):
        image, label = tf.py_function(func=load_file_test,inp=[filePath, label],Tout=[tf.float32,tf.float32])
        return image, label

    valFiles=[]
    valLabels=[]

    catFolders=os.listdir(valDir)
    for catFolder in catFolders:
        NG_Dir=os.path.join(trainDir,catFolder,"NG") 
        NG_Path=[os.path.join(NG_Dir,fileName) for fileName in os.listdir(NG_Dir) ]

        OK_Dir=os.path.join(trainDir,catFolder,"OK",) 
        OK_Path=[os.path.join(OK_Dir,fileName) for fileName in os.listdir(OK_Dir) ]

        valFiles+=NG_Path
        valLabels+=[1]*len(NG_Path)
        valFiles+=OK_Path
        valLabels+=[0]*len(OK_Path)

    valDatasetSize=len(valLabels)
    valDataset=tf.data.Dataset.from_tensor_slices((valFiles,valLabels)).shuffle(buffer_size=valDatasetSize)
    valDataset=valDataset.map(lambda x,y:mappable_fn_test(x, y)).batch(BATCH_SIZE)

    return trainDataset, valDataset, trainDatasetSize