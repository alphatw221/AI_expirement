import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def makeTrainValDataset(dataPath,batchSize,imageSize):

    def load_file(fileName,label,dataPath,imageSize):
        path=os.path.join(dataPath.numpy().decode(),label.numpy().decode(),fileName.numpy().decode())
        image=cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=cv2.resize(image,(imageSize.numpy(),imageSize.numpy()))
        # oneHot=np.eye(2)[ int( label.numpy().decode() ) ]
        return image,[int(label.numpy())]
        
    def mappable_fn(fileName,label,dataPath,imageSize):
            image,label = tf.py_function(func=load_file,inp=[fileName,label,dataPath,imageSize],Tout=[tf.float32,tf.float32])
            image.set_shape((imageSize,imageSize,3))
            label.set_shape((1,))
            return image,label
    
        
        
    trainOkPath=os.path.join(dataPath,"train","0")
    trainNgPath=os.path.join(dataPath,"train","1")

    trainOkFileNames=os.listdir( trainOkPath )
    trainNgFileNames=os.listdir( trainNgPath )

    trainFileNames=trainOkFileNames+trainNgFileNames
    trainLabels=["0"]*len(trainOkFileNames)+["1"]*len(trainNgFileNames)
    
    trainDataset=tf.data.Dataset.from_tensor_slices((trainFileNames,trainLabels)).shuffle(buffer_size=len(trainLabels))
    trainDataset=trainDataset.map(lambda x,y:mappable_fn(x,y,os.path.join(dataPath,"train"),imageSize)).batch(batchSize)


    valOkPath=os.path.join(dataPath,"val","0")
    valNgPath=os.path.join(dataPath,"val","1")

    valOkFileNames=os.listdir(valOkPath)
    valNgFileNames=os.listdir(valNgPath)

    valFileNames=valOkFileNames+valNgFileNames
    valLabels=(["0"]*len(valOkFileNames))+["1"]*len(valNgFileNames)

    valDataset=tf.data.Dataset.from_tensor_slices((valFileNames,valLabels)).shuffle(buffer_size=len(valLabels))
    valDataset=valDataset.map(lambda x,y:mappable_fn(x,y,os.path.join(dataPath,"val"),imageSize)).batch(batchSize)

    return trainDataset,valDataset