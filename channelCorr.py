
# coding: utf-8

# In[1]:


import os
import sys
import glob


# In[2]:


import csv
import time
import librosa
import numpy as np
import pandas as pd
import scipy.io as sio  
from scipy.io import wavfile


# In[28]:


def data_process(folder_list):
    cnt = 0
    corrs = []
    labels = []
    emptyFile = []
    for index, sub_dir in enumerate(folder_list):
        
        #printf(str(index+1) + '-' + sub_dir)
        for filename in glob.glob(sub_dir):
            label = sub_dir.split('(')[1].split(')')[0]
            
            #printf(label)
            
            (rate, sig) = wavfile.read(filename)
            if len(sig) == 0:
                emptyFile.append(filename)
                continue
            corr = np.corrcoef(sig[:,0], sig[:,1])  #will return a corrcoef matrix
            #print(corr[0][1])
            corrs.append(corr[0][1])
            labels.append(label)
    # 根據底部偷跑一次算出來的030
    features = np.asarray(corrs, 'f')
    print('\n', features.shape)
    print('\n誰的鍋:', emptyFile)
    return np.array(features), np.array(labels)


# In[29]:


points = ['0', '15', '30']
height_list = ['20', '30', '40', '50', '60']
paths = []
# x-axis
for rx in points:
    # y-axis
    for ry in points:
        for rz in height_list:
            for ri in range(1, 5):
                # 記得自己改讀檔規則
                # my dir will be like '../wav/(0, 0, 20)-1/chunk/*.wav'
                folder = '{0}-{1}\({2}, {3}, {4})-{5}'.format(rx, ry, rx, ry, rz, ri)
                paths.append(os.path.join('..', 'wav', folder, 'chunk', '*.wav'))
print(len(paths), paths[0])


# In[30]:


feature, label = data_process(paths)


# In[31]:


onehot = []
for x in label:
    arr = []
    if(x == '0, 0, 20' or x == '0, 0, 30' or x == '0, 0, 40' or x == '0, 0, 50' or x == '0, 0, 60'):
        arr = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif(x == '0, 15, 20' or x == '0, 15, 30' or x == '0, 15, 40' or x == '0, 15, 50' or x == '0, 15, 60'):
        arr = [0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif(x == '0, 30, 20' or x == '0, 30, 30' or x == '0, 30, 40' or x == '0, 30, 50' or x == '0, 30, 60'):
        arr = [0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif(x == '15, 0, 20' or x == '15, 0, 30' or x == '15, 0, 40' or x == '15, 0, 50' or x == '15, 0, 60'):
        arr = [0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif(x == '15, 15, 20' or x == '15, 15, 30' or x == '15, 15, 40' or x == '15, 15, 50' or x == '15, 15, 60'):
        arr = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif(x == '15, 30, 20' or x == '15, 30, 30' or x == '15, 30, 40' or x == '15, 30, 50' or x == '15, 30, 60'):
        arr = [0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif(x == '30, 0, 20' or x == '30, 0, 30' or x == '30, 0, 40' or x == '30, 0, 50' or x == '30, 0, 60'):
        arr = [0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif(x == '30, 15, 20' or x == '30, 15, 30' or x == '30, 15, 40' or x == '30, 15, 50' or x == '30, 15, 60'):
        arr = [0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif(x == '30, 30, 20' or x == '30, 30, 30' or x == '30, 30, 40' or x == '30, 30, 50' or x == '30, 30, 60'):
        arr = [0, 0, 0, 0, 0, 0, 0, 0, 1]
    print(arr)
    onehot.append(arr)


# In[42]:


csvFile = 'featuredata-corr.csv'
with open(csvFile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    #feature = np.array(feature, dtype=np.float32)
    #print(feature.dtype)
    for val in feature:
        #print(val.dtype)
        writer.writerow([val])

