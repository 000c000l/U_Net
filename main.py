from mainModel import mainModel
from dataGeneration import adjustData,saveResult,normalizeData,fetchTestData,fetchTrainData
import keras as ke
import skimage as sk
import skimage.io as io
import skimage.transform as transform
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np
import os

"""
    Tune the Hyperparameters
    path_to_data should contain the folder with following folders

    path_to_data/
        train/
            image/ 
                the images should be serialized and follow the naming convention %d.png 
            label/
                the corresponding labels for images and should have the same name to the corresponding images
        test/
            image/
                the images should be serialized and follow the naming convention %d.png 
"""
path_to_data="../data/"
steps_per_epoch=10
epochs=1
testDataSize=10

# importing model
model=mainModel(pretrainedWeights=None)
"""
    training part of the code
    Comment this part of the code if model is pretrained
"""
# importing train Data
trainData=fetchTrainData(path=path_to_data)
# training
model.fit_generator(trainData,steps_per_epoch=steps_per_epoch,epochs=epochs,callbacks=[ModelCheckpoint("unet",monitor="loss",save_best_only=True)])
model.save(path_to_data)
"""    
"""

"""
    Testing part of the code
"""
# importing test Data
testDataSize=10
testData=fetchTestData(testDataSize,path=path_to_data)
# testing
model=load_model(path_to_data)
results=model.predict(testData,testDataSize)
# saving the results of testing
saveResult(os.path.join(path_to_data,"test"),results)