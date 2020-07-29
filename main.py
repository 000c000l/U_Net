from mainModel import mainModel
from dataGeneration import adjustData,saveResult,normalizeData
import keras as ke
import skimage as sk
import skimage.io as io
import skimage.transform as transform
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np

import os

# importing model
model=mainModel(pretrainedWeights=None)
# importing train Data
preprocess=ke.preprocessing.image.ImageDataGenerator(fill_mode="nearest")
trainDataImage=preprocess.flow_from_directory(  "data/train",
                                                classes=["image"],
                                                class_mode=None,
                                                color_mode="grayscale",
                                                save_to_dir=None,
                                                batch_size=3,
                                                target_size=(256,256),
                                                save_prefix="image",
                                                seed=17)
trainDataLabel=preprocess.flow_from_directory(  "data/train",
                                                classes=["label"],
                                                class_mode=None,
                                                color_mode="grayscale",
                                                save_to_dir=None,
                                                batch_size=3,
                                                target_size=(256,256),
                                                save_prefix="label",
                                                seed=17)
data=zip(trainDataImage,trainDataLabel)
trainData=[]
for i,j in data:
    trainData.append(normalizeData(i,j))
trainData=np.asarray(trainData)
# importing test Data
testDataSize=10
testData=[]
for i in range(testDataSize):
    image=io.imread(os.path.join("data/test/","%d.png"%i),as_gray=True)
    image=image/255
    image=transform.resize(image,(256,256))
    image=np.reshape(image,(1,)+image.shape)
    testData.append(image)
testData=np.asarray(testData)
# training
model.fit_generator(trainData,steps_per_epoch=10,epochs=1,callbacks=[ModelCheckpoint("unet",monitor="loss",save_best_only=True)])
model.save("data/")
# testing
model=load_model("data/")
results=model.predict_generator(testData,testDataSize)
# saving the results of testing
saveResult("data/test",results)