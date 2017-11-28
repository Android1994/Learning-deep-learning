import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split #数据拆分

from keras.models import Sequential #序列模型
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop #优化器
from keras.utils import np_utils #编码器


def get_files(file_dir,flag):
    '''
    Args:
        file_dir: file directory
        flag: whether output random data or not
    Returns:
        images and labels
    '''
    classes={'edge','noedge'} #2 class
    
    images = []
    labels = []
    countEdge=0
    countNoEdge=0
    
    for index, name in enumerate(classes):
        class_path=file_dir+name+'/'
        for img_name in os.listdir(class_path):
            img_path=class_path+img_name
            im = Image.open(img_path).convert('L') #转换为灰度图，选项:1,L,P,RGB,RGBA,CMYK,YCbCr,I,F
            imageData = np.array(im)
            images.append(imageData)
            if name=='edge':
                labels.append(1)
                countEdge += 1 
            else:
                labels.append(0)
                countNoEdge += 1
    print('In ' + file_dir + ':\nthere are %d edges\nthere are %d noedges\n' %(countEdge, countNoEdge))
    
    #转换成np.array
    images_array = np.array(images)
    labels_array = np.array(labels)
    
    #打乱顺序
    index = [i for i in range(len(images_array))]    
    np.random.shuffle(index)   
    images_rand = images_array[index]  
    labels_rand = labels_array[index]
    
    if flag == 1:
        return images_rand, labels_rand
    else:
        return images_array, labels_array


#获取数据
X_train, y_train = get_files('TrainData/yangben/',1)
X_test, y_test = get_files('TestData/testYangben/',0)

#数据预处理
X_train = X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
X_test = X_test.reshape(X_test.shape[0],28,28,1).astype('float32')

y_train = np_utils.to_categorical(y_train,2) #矢量编码
y_test = np_utils.to_categorical(y_test,2)

#变量
BATCH_SIZE= 128 #一次喂给神经网络多少数据
NB_EPOCH = 100 #总共训练多少次

#model
model = Sequential()
#conv1
model.add(Conv2D(filters = 64, kernel_size=(3,3),strides = (1,1), padding='same', input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
#conv2
model.add(Conv2D(filters = 128, kernel_size=(3,3),strides = (1,1), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
#conv3
model.add(Conv2D(filters = 256, kernel_size=(3,3),strides = (1,1), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
#展平节点
model.add(Flatten())
#full connect
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(2,activation='softmax'))

#loss
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#training
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=NB_EPOCH,batch_size=BATCH_SIZE,verbose=1)

#testing
scores=model.evaluate(X_test,y_test,verbose=1)
print('Test score:', scores[0])
print('Test accuracy:', scores[1])

model.save('edge_model1_random.h5')
