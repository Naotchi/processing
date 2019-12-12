#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras import layers, models, optimizers
import glob
import cv2
import numpy as np
import random
import math
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import EarlyStopping


# In[2]:


train_pic = []#(画像データnp.array, それが何のcharacterかint)というlen=2のtupleが並んだlistを作る.
train_height = []
test_pic = []
test_height = []


# In[3]:


def train_test_split(character):#characterごとに行う
    label = 0#barutan : 0 metron : 1 syodai : 2 taiga : 3 tarou : 4 とラベリングする
    if(character == 'metron'):
        label = 1
    elif(character == 'syodai'):
        label = 2
    elif(character == 'taiga'):
        label = 3
    elif(character == 'tarou'):
        label = 4
        
    allfiles = []
    for i in range(1, 79, 1):
        pic = cv2.imread('/Users/shibaharanaoki/Documents/raspi/cnn2_pic/' + character + '/' + character + '_pic' + str(i) + '.png', cv2.IMREAD_GRAYSCALE)
        pic = cv2.resize(pic, dsize=(256, 256))
        height = cv2.imread('/Users/shibaharanaoki/Documents/raspi/cnn2_height/' + character + '/' + character + '_height' + str(i) + '.png', cv2.IMREAD_GRAYSCALE)
        height = cv2.resize(height, dsize=(256, 256))
        allfiles.append((pic, height))
    random.shuffle(allfiles)
    length = len(allfiles)
    
    allfiles = np.array(allfiles)
    allfiles_pic = allfiles[:, 0, :, :]
    allfiles_height = allfiles[:, 1, :, :]
    th = math.floor(length * 0.8)#80%をtrain, 20%をtestに分割
    a = list(allfiles_pic)
    b = list(allfiles_height)
    for i in range(len(a)):
        a[i] = (a[i], label)
        b[i] = (b[i], label)
    train_pic.extend(a[:th])
    test_pic.extend(a[th:])
    train_height.extend(b[:th])
    test_height.extend(b[th:])


# In[4]:


train_test_split('barutan')#予め用意しておいたtrain, testというlistにbarutan分を追加する


# In[5]:


len(train_pic)


# In[6]:


print(train_pic[0][1])


# In[7]:


train_test_split('metron')
train_test_split('syodai')
train_test_split('taiga')
train_test_split('tarou')


# In[8]:


len(train_pic)


# train_picとtrain_heightを水増しする

# In[9]:


generator = ImageDataGenerator(#水増しの方法を指定する.
    rotation_range = 180,#-180から180のランダムな数を選んで回転させる
    width_shift_range = 0.1,#最大で全体の1割横移動させる
    height_shift_range = 0.1,#縦移動
    #brightness, zoom, shearは今回はなくていい？
    #https://qiita.com/takurooo/items/c06365dd43914c253240
    #↑に詳しく
)


# In[10]:


# dir_pic = '/Users/shibaharanaoki/Documents/raspi/cnn2_pic/aug_12_1'
# dir_height = '/Users/shibaharanaoki/Documents/raspi/cnn2_height/aug_12_1'
#生成した画像をいれるディレクトリ. labelは無視する.


# In[11]:


def draw_images(x, y, index, label):#引数に(水増し方法, 元データ)を取る関数を作る.この中で実際の水増しメソッドが実行される
  g_pic = generator.flow(x, seed=1)#このメソッドが勝手に配列→.pngの変換と指定したディレクトリへの保存をしてくれる
  g_height = generator.flow(y, seed=1)#https://keras.io/ja/preprocessing/image/
  for i in range(50):#5回生成するよ
    aug_pic = g_pic.next()
    aug_height = g_height.next()
    train_pic.append(((aug_pic[0, :, :, 0]), label))#nextで水増しメソッド実行
    train_height.append(((aug_height[0, :, :, 0]), label))


# In[12]:


def augment():
    for index in range(len(train_pic)):
        pic = train_pic[index][0]#画像をひとつ選んで
        pic = pic.reshape(256, 256, 1)
        height = train_height[index][0]
        height = height.reshape(256, 256, 1)
        pic = pic.reshape((1, ) + pic.shape)
        height = height.reshape((1, ) + height.shape)#次元をひとつ拡張(その次元の値は1). generator.flowが画像データを4次元で渡せって
        if not train_pic[index][1] == train_height[index][1]:
            print('errorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr\n' * 30)
            break
        label = train_pic[index][1]
        draw_images(pic, height, index, label)#上で定義したnextメソッドを含む関数を呼び出して実行.
        print('\r' + str(index), end='')#進捗確認


# In[13]:


augment()


# In[14]:


len(train_pic)


# In[15]:


train_pic[430][1]


# In[16]:


random.seed(0)
random.shuffle(train_pic)


# In[17]:


random.seed(0)
random.shuffle(train_height)


# In[18]:


random.seed(1)
random.shuffle(test_pic)


# In[19]:


random.seed(1)
random.shuffle(test_height)


# ここからはheightのみ使う. train, testそれぞれで画像データXとラベルyに分ける

# In[20]:


X_train = []
y_train = []
for i in train_height:
    X_train.append(i[0])
    y_train.append(i[1])
X_test = []
y_test = []
for i in test_height:
    X_test.append(i[0])
    y_test.append(i[1])


# In[21]:


X_train = np.array(X_train)#(len(train), 2)の2次元np.arrayに変換し0列目だけを取り出す
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)#次元を1つ追加（長さはモノクロなら1カラー画像なら3)
X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], X_train.shape[2], 1)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[22]:


X_train = X_train.astype('float32') / 255#規格化
X_test = X_test.astype('float32') / 255


# In[23]:


from keras.utils import np_utils#one-hot化
y_train = np_utils.to_categorical(y_train, num_classes=5)
y_test = np_utils.to_categorical(y_test, num_classes=5)


# In[24]:


y_train[4]


# In[25]:


np.save('/Users/shibaharanaoki/Documents/raspi/cnn2_height/X_train', X_train)
np.save('/Users/shibaharanaoki/Documents/raspi/cnn2_height/y_train', y_train)
np.save('/Users/shibaharanaoki/Documents/raspi/cnn2_height/X_test', X_test)
np.save('/Users/shibaharanaoki/Documents/raspi/cnn2_height/y_test', y_test)


# In[26]:


np.save('/Users/shibaharanaoki/Documents/raspi/cnn2_pic/X_train', X_train)
np.save('/Users/shibaharanaoki/Documents/raspi/cnn2_pic/y_train', y_train)
np.save('/Users/shibaharanaoki/Documents/raspi/cnn2_pic/X_test', X_test)
np.save('/Users/shibaharanaoki/Documents/raspi/cnn2_pic/y_test', y_test)

