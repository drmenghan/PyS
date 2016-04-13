import pandas as pd # Dataframe
from sklearn.ensemble import RandomForestClassifier # Classification algorithm - random forest
from sklearn import metrics, grid_search
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
import numpy as np
import math
import random as rd
import pylab as pl
import matplotlib.pyplot as plt
#matplotlib inline
import os
os.chdir('c:\\TEMP')
pwd
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_data = train_df.drop('label',axis=1).values
train_labels = train_df['label'].values
test_data = test_df.values

print ("Loading finished.")
print ("Train size:", train_df.shape)
print ("Test size:", test_df.shape)

train_df.head(5)
train_images = []

for image in train_data:
    train_images.append(image.reshape(28,28))

train_images = np.array(train_images)

plt.figure(figsize=(20,10), dpi=600)
for i in range(10):
    plt.subplot(1,10,(i+1))
    print(train_labels[i],
    pl.imshow(train_images[i],cmap=pl.cm.gray_r))
pl.show()


clf = RandomForestClassifier()
clf.fit(train_data,train_labels)

predictions = clf.predict(test_data)
print ("Predicting finished.")

submission = pd.DataFrame({"ImageId": np.arange(1,28001),"Label": predictions})
submission.to_csv('./submission.csv',index=False)
print ("Submission created.")