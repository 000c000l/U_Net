import numpy as np
import skimage as sk
import os
import keras as ke
from skimage import img_as_ubyte
import skimage.io as io
import skimage.transform as transform
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

def normalizeData(image,label):
    if np.max(image)>1:
        image=image/255
        label=label/255
        label[label>0.5]=1
        label[label<=0.5]=0
        return (image,label)
    return (image,label)
def fetchTestData(testDataSize,path):
    for i in range(testDataSize):
        image=io.imread(os.path.join(os.path.join(path,"test"),"%d.png"%i),as_gray=True)
        image=image/255
        image=transform.resize(image,(256,256))
        image=np.reshape(image,(1,)+image.shape)
        yield image
def fetchTrainData(path):
    preprocess=ke.preprocessing.image.ImageDataGenerator(fill_mode="nearest")
    trainDataImage=preprocess.flow_from_directory(  os.path.join(path,"train"),
                                                    classes=["image"],
                                                    class_mode=None,
                                                    color_mode="grayscale",
                                                    save_to_dir=None,
                                                    batch_size=3,
                                                    target_size=(256,256),
                                                    save_prefix="image",
                                                    seed=17)
    trainDataLabel=preprocess.flow_from_directory(  os.path.join(path,"train"),
                                                    classes=["label"],
                                                    class_mode=None,
                                                    color_mode="grayscale",
                                                    save_to_dir=None,
                                                    batch_size=3,
                                                    target_size=(256,256),
                                                    save_prefix="label",
                                                    seed=17)
    for i,j in zip(trainDataImage,trainDataLabel):
        yield normalizeData(i,j)
def adjustData(image,label):
    if(np.max(image) > 1):
        image = image / 255
        label = label /255
        label[label > 0.5] = 1
        label[label <= 0.5] = 0
    return (image,label)
def saveResult(path,file):
    for i,j in enumerate(file):
        j=j[:,:,0]
        sk.io.imsave(os.path.join(path,"%dPredicted.png"%i),img_as_ubyte(j))