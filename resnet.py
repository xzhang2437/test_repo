#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import IPython.core.display         
# setup output image format (Chrome works best)
IPython.core.display.set_matplotlib_formats("svg")
import matplotlib.pyplot as plt
import matplotlib
from numpy import *
from sklearn import *
from scipy import stats
from keras.preprocessing import image
import csv
import tensorflow as tf
random.seed(100)


# In[2]:


from itertools import islice 

def read_csv_data(fname):
    imagesID = []
    labels = []
    with open(fname, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in islice(reader, 1, None): 
            imagesID.append(row[0])
            if len(row) > 1:
                labels.append(row[1:])
                
    return (imagesID, labels)

def write_csv_kaggle_tags(fname, imageID, Yscores):
    # header
    tmp = [['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab']]   
    
    # add ID numbers for each Y, and usage if necessary
    for i in range(len(Yscores)):
        tmp2 = [imageID[i]]
        for t in range(4):
            tmp2.append(Yscores[i,t])
        
        tmp.append(tmp2)
        
    # write CSV file
    f = open(fname, 'w')
    writer = csv.writer(f)
    writer.writerows(tmp)
    f.close()
    
def load_images(imageID):
    images = empty((len(imageID), 224, 224, 3))
    for i, imageName in enumerate(imageID):
        myImage = image.load_img('./plant-pathology-2020-fgvc7/images/'+imageName+'.jpg', target_size=(224, 224))
        myImage = image.img_to_array(myImage) / 255.0
        images[i] = myImage
    return images


# In[ ]:


(trainImagesID, trainLabels) = read_csv_data("./plant-pathology-2020-fgvc7/train.csv")
# (testImagesID, _) = read_csv_data("./plant-pathology-2020-fgvc7/test.csv")

trainImages = load_images(trainImagesID)


# In[ ]:


trainY = empty((len(trainImages), 4))
for i in range(len(trainImages)):
    trainY[i] = array(trainLabels[i])
print(len(trainImages))
print(trainImages[0].size)
print(type(trainY[0][0]))


# In[ ]:


trainX, testX, trainY, testY = model_selection.train_test_split(trainImages, trainY, train_size=0.80, test_size=0.20, random_state=4487)


# In[ ]:


from tensorflow.keras.applications.resnet50 import ResNet50

model = ResNet50(
    weights=None,
    classes=4
)

model.compile(optimizer=tf.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(trainX, trainY, epochs=70, batch_size=128)


# In[ ]:


model.evaluate(testX, testY, batch_size=64)


# In[ ]:


# testImages = load_images(testImagesID)
predY = model.predict(testImages)


# In[ ]:


write_csv_kaggle_tags('resnet.csv',testImagesID, predY)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




