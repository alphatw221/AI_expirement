import tensorflow as tf


def makeDataset(dataPath,batchSize):
    trainOkFileNames=os.listdir( os.path.join(dataPath,"train","0"))
    trainNgFileNames=os.listdir( os.path.join(dataPath,"train","1"))
    trainFileNames=trainOkFileNames+trainNgFileNames
    trainLabels=([0]*len(trainOkFileNames))+[1]*len(trainNgFileNames)
    valOkFileNames=os.listdir( os.path.join(dataPath,"val","0"))
    valNgFileNames=os.listdir( os.path.join(dataPath,"val","1"))
    valFileNames=valOkFileNames+valNgFileNames
    valLabels=([0]*len(valOkFileNames))+[1]*len(valNgFileNames)

    def load_files(filename,label):
        return plt.imread(filename.numpy().decode()),label

    trainDataset=tf.data.Dataset.from_tensor_slices(trainFileNames,trainLabels)
    trainDataset=trainDataset.map(lambda x,y:tf.py_function(load_files,[x,y],[tf.float32,tf.float32]),num_parallel_calls=2).batch(batchSize).prefetch(1)
    
    valDataset=tf.data.Dataset.from_tensor_slices(valFileNames,valLabels)
    valDataset=valDataset.map(lambda x,y:tf.py_function(load_files,[x,y],[tf.float32,tf.float32])).batch(batch_size)

    return trainDataset, valDataset