#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import cv2


# barutanにある20枚の画像を配列の形で読み込んでdatasに格納する

# In[ ]:


def add_ori(character):
    datas = []
    for file in sorted(glob.glob('/Users/shibaharanaoki/Documents/raspi/' + character + '/' + character + '_ori/*')):
        datas.append(cv2.imread(file))
    return datas


# datasにある20枚の画像（配列データ）からpic部分を取り出してbarutan_picに格納する（.pngとして）

# In[ ]:


def divide_and_pickup(character):
    datas = add_ori(character)
    s = 1
    for data in datas:
        pic = data[0:1000]#ここから前と同じ
        hei = data[1000:3000] 
        height = []
        for x in range(2000):
            for y in range(0, 1527, 2):
                a = hei[x][y][0] + 256 * hei[x][y+1][0]
                height.append(a)
        height = np.array(height)
        height = np.reshape(height, (1000, 1528))
        height = height.astype('uint16')
        pic = pic[10 : 660, 440 : 1120]
        height = height[10 : 660, 440 : 1120]#ここまで前と同じ
        cv2.imwrite('/Users/shibaharanaoki/Documents/raspi/' + character + '/' + character + '_pic/' + character + '_pic' + str(s) + '.png', pic)
        for i in range(650):
            for j in range(680):
                if height[i][j] > 12500:
                   height[i][j] -= 12500
                else:
                   height[i][j] = 0
        height = height.astype('float32')
        height /= 2**5
        cv2.imwrite('/Users/shibaharanaoki/Documents/raspi/' + character + '/' + character + '_height/' + character + '_height' + str(s) + '.png', height)
        print(s)#進捗確認
        s += 1


# In[ ]:


divide_and_pickup('barutan')


# In[ ]:


divide_and_pickup('metron')
divide_and_pickup('syodai')
divide_and_pickup('taiga')
divide_and_pickup('tarou')

