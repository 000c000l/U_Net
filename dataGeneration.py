import numpy as np
import skimage as sk
import os
def normalizeData(image,label):
    if np.max(image)>1:
        image=image/255
        label=label/255
        label[label>0.5]=1
        label[label<=0.5]=0
        return (image,label)
    return (image,label)
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
        sk.io.imsave(os.path.join(path,"%dPredicted.png"%i),j)