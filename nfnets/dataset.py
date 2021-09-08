import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def makeDatasetFromDLS(dataPath,batchSize=1,epochSize=1,trainImageSize=200,testImageSize=200,splitRatio=(8,1,1)):

    def load_file(goldPath, scanPath, label, imageSize):#裡面的變數都是tensor

        goldImage = cv2.imread(goldPath.numpy().decode())
        goldImage = cv2.cvtColor(goldImage, cv2.COLOR_BGR2RGB)
        scanImage = cv2.imread(scanPath.numpy().decode())
        scanImage = cv2.cvtColor(scanImage, cv2.COLOR_BGR2RGB)
        image=np.concatenate((goldImage,scanImage),axis=2)
        image=cv2.resize(image,(imageSize.numpy(),imageSize.numpy()))
        oneHot=np.eye(2)[ int( label.numpy() ) ]
        return image/255, oneHot
        
    def mappable_fn(goldName, scanName, label, imageSize):
        image,label = tf.py_function(func=load_file,inp=[goldName, scanName, label, imageSize],Tout=[tf.float32,tf.float32])
        image.set_shape((imageSize,imageSize,6))
        label.set_shape((2,))
        return image, label

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
    
    GoldFiles=NG_Gold+OK_Gold
    ScanFiles=NG_Scan+OK_Scan
    labels=[1]*len(NG_Scan)+[0]*len(OK_Scan)

    datasetSize=len(labels)
    dataset=tf.data.Dataset.from_tensor_slices((GoldFiles,ScanFiles,labels)).shuffle(buffer_size=datasetSize)
    
    trainSize = int( ( splitRatio[0]/sum(splitRatio) ) * datasetSize)
    valSize = int(( splitRatio[1]/sum(splitRatio) ) * datasetSize)

    trainDataset=dataset.take(trainSize)
    rest=dataset.skip(trainSize)
    valDataset=rest.take(valSize)
    testDataset=rest.skip(valSize)

    trainDataset=trainDataset.repeat(epochSize).map(lambda x,y,z:mappable_fn(x, y, z, trainImageSize)).batch(batchSize)
    valDataset=valDataset.repeat(epochSize).map(lambda x,y,z:mappable_fn(x, y, z, trainImageSize)).batch(batchSize)
    testDataset=testDataset.repeat(epochSize).map(lambda x,y,z:mappable_fn(x, y, z, testImageSize)).batch(batchSize)

    return trainDataset, valDataset, testDataset, trainSize



def makeDataset(dataPath,batchSize,epochSize,trainImageSize,testImageSize):

    def load_file(fileName,label,dataPath,imageSize):
        path=os.path.join(dataPath.numpy().decode(),label.numpy().decode(),fileName.numpy().decode())
        image=cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=cv2.resize(image,(imageSize.numpy(),imageSize.numpy()))
        oneHot=np.eye(2)[ int( label.numpy().decode() ) ]
        return image/255,oneHot
        
    def mappable_fn(fileName,label,dataPath,imageSize):
            image,label = tf.py_function(func=load_file,inp=[fileName,label,dataPath,imageSize],Tout=[tf.float32,tf.float32])
            image.set_shape((imageSize,imageSize,3))
            label.set_shape((2,))
            return image,label
    
        
        
    trainOkPath=os.path.join(dataPath,"train","0")
    trainNgPath=os.path.join(dataPath,"train","1")
    trainOkFileNames=os.listdir( trainOkPath )
    trainNgFileNames=os.listdir( trainNgPath )
    trainFileNames=trainOkFileNames+trainNgFileNames
    trainLabels=["0"]*len(trainOkFileNames)+["1"]*len(trainNgFileNames)
    trainDataset=tf.data.Dataset.from_tensor_slices((trainFileNames,trainLabels)).shuffle(buffer_size=len(trainLabels)).repeat(epochSize)
    trainDataset=trainDataset.map(lambda x,y:mappable_fn(x,y,os.path.join(dataPath,"train"),trainImageSize)).batch(batchSize)


    valOkPath=os.path.join(dataPath,"val","0")
    valNgPath=os.path.join(dataPath,"val","1")
    valOkFileNames=os.listdir(valOkPath)
    valNgFileNames=os.listdir(valNgPath)
    valFileNames=valOkFileNames+valNgFileNames
    valLabels=(["0"]*len(valOkFileNames))+["1"]*len(valNgFileNames)
    valDataset=tf.data.Dataset.from_tensor_slices((valFileNames,valLabels)).shuffle(buffer_size=len(valLabels)).repeat(epochSize)
    valDataset=valDataset.map(lambda x,y:mappable_fn(x,y,os.path.join(dataPath,"val"),trainImageSize)).batch(batchSize)


    testOkPath=os.path.join(dataPath,"test","0")
    testNgPath=os.path.join(dataPath,"test","1")
    testOkFileNames=os.listdir(testOkPath)
    testNgFileNames=os.listdir(testNgPath)
    testFileNames=testOkFileNames+testNgFileNames
    testLabels=(["0"]*len(testOkFileNames))+["1"]*len(testNgFileNames)
    testDataset=tf.data.Dataset.from_tensor_slices((testFileNames,testLabels)).shuffle(buffer_size=len(testLabels)).repeat(epochSize)
    testDataset=testDataset.map(lambda x,y:mappable_fn(x,y,os.path.join(dataPath,"test"),testImageSize)).batch(batchSize)
    
    return trainDataset,valDataset,testDataset