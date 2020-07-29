import keras as ke
import numpy as np

# here each block of layers mean each segment or group of layers of U-Net
def mainModel(pretrainedWeights=None):

    layer1Input=ke.layers.Input((256,256,1))
    layer1Conv1=ke.layers.Conv2D(64,3,padding="same",activation="relu",kernel_initializer="he_normal")(layer1Input)
    layer1Conv2=ke.layers.Conv2D(64,3,padding="same",activation="relu",kernel_initializer="he_normal")(layer1Conv1)
    
    layer2Pool=ke.layers.MaxPooling2D(pool_size=(2,2))(layer1Conv2)
    layer2Conv1=ke.layers.Conv2D(128,3,padding="same",activation="relu",kernel_initializer="he_normal")(layer2Pool)
    layer2Conv2=ke.layers.Conv2D(128,3,padding="same",activation="relu",kernel_initializer="he_normal")(layer2Conv1)

    layer3Pool=ke.layers.MaxPooling2D(pool_size=(2,2))(layer2Conv2)
    layer3Conv1=ke.layers.Conv2D(256,3,padding="same",activation="relu",kernel_initializer="he_normal")(layer3Pool)
    layer3Conv2=ke.layers.Conv2D(256,3,padding="same",activation="relu",kernel_initializer="he_normal")(layer3Conv1)

    layer4Pool=ke.layers.MaxPooling2D(pool_size=(2,2))(layer3Conv2)
    layer4Conv1=ke.layers.Conv2D(512,3,padding="same",activation="relu",kernel_initializer="he_normal")(layer4Pool)
    layer4Conv2=ke.layers.Conv2D(512,3,padding="same",activation="relu",kernel_initializer="he_normal")(layer4Conv1)

    layer5Pool=ke.layers.MaxPooling2D(pool_size=(2,2))(layer4Conv2)
    layer5Conv1=ke.layers.Conv2D(1024,3,padding="same",activation="relu",kernel_initializer="he_normal")(layer5Pool)
    layer5Conv2=ke.layers.Conv2D(1024,3,padding="same",activation="relu",kernel_initializer="he_normal")(layer5Conv1)

    layer6Up=ke.layers.Conv2D(512,2,padding="same",activation="relu",kernel_initializer="he_normal")(ke.layers.UpSampling2D(size=(2,2))(layer5Conv2))
    layer6Add=ke.layers.concatenate([layer4Conv2,layer6Up],axis=3)
    layer6Conv1=ke.layers.Conv2D(512,3,padding="same",activation="relu",kernel_initializer="he_normal")(layer6Add)
    layer6Conv2=ke.layers.Conv2D(512,3,padding="same",activation="relu",kernel_initializer="he_normal")(layer6Conv1)

    layer7Up=ke.layers.Conv2D(256,2,padding="same",activation="relu",kernel_initializer="he_normal")(ke.layers.UpSampling2D(size=(2,2))(layer6Conv2))
    layer7Add=ke.layers.concatenate([layer3Conv2,layer7Up],axis=3)
    layer7Conv1=ke.layers.Conv2D(256,3,padding="same",activation="relu",kernel_initializer="he_normal")(layer7Add)
    layer7Conv2=ke.layers.Conv2D(256,3,padding="same",activation="relu",kernel_initializer="he_normal")(layer7Conv1)

    layer8Up=ke.layers.Conv2D(128,2,padding="same",activation="relu",kernel_initializer="he_normal")(ke.layers.UpSampling2D(size=(2,2))(layer7Conv2))
    layer8Add=ke.layers.concatenate([layer2Conv2,layer8Up],axis=3)
    layer8Conv1=ke.layers.Conv2D(128,3,padding="same",activation="relu",kernel_initializer="he_normal")(layer8Add)
    layer8Conv2=ke.layers.Conv2D(128,3,padding="same",activation="relu",kernel_initializer="he_normal")(layer8Conv1)

    layer9Up=ke.layers.Conv2D(64,2,padding="same",activation="relu",kernel_initializer="he_normal")(ke.layers.UpSampling2D(size=(2,2))(layer8Conv2))
    layer9Add=ke.layers.concatenate([layer1Conv2,layer9Up],axis=3)
    layer9Conv1=ke.layers.Conv2D(64,3,padding="same",activation="relu",kernel_initializer="he_normal")(layer9Add)
    layer9Conv2=ke.layers.Conv2D(64,3,padding="same",activation="relu",kernel_initializer="he_normal")(layer9Conv1)

    layer10Conv1=ke.layers.Conv2D(2,3,padding="same",activation="relu",kernel_initializer="he_normal")(layer9Conv2)
    layer10Conv2=ke.layers.Conv2D(1,1,activation="sigmoid")(layer10Conv1)

    model=ke.models.Model(layer1Input,layer10Conv2)
    model.compile(optimizer= ke.optimizers.Adam(learning_rate=1e-4),loss="binary_crossentropy",metrics=["accuracy"])

    if(pretrainedWeights):
        model.load_weights(pretrainedWeights)
    return model