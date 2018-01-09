from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.initializers import Constant
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint
from PIL import Image
import numpy as np
import argparse
import copy
import os
import cv2


nb_classes = 6  #21,6
# Bilinear interpolation (reference: https://github.com/warmspringwinds/tf-image-segmentation/blob/master/tf_image_segmentation/utils/upsampling.py)
def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = factor*2 - factor%2
    factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:filter_size, :filter_size]
    upsample_kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes),
                       dtype=np.float32)
    for i in xrange(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights

def fcn_32s():
    inputs = Input(shape=(None, None, 3))
    vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
    x = Conv2D(filters=nb_classes, 
               kernel_size=(1, 1))(vgg16.output)
    x = Conv2DTranspose(filters=nb_classes, 
                        kernel_size=(64, 64),
                        strides=(32, 32),
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer=Constant(bilinear_upsample_weights(32, nb_classes)))(x)
    model = Model(inputs=inputs, outputs=x)
    for layer in model.layers[:15]:
        layer.trainable = False
    return model

def load_image(path):
    img_org = Image.open(path)
    w, h = img_org.size
    img = img_org.resize(((w//32)*32, (h//32)*32))
    img = np.array(img, dtype=np.float32)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    return x

# def load_label(path):
#     img_org = Image.open(path)
#     w, h = img_org.size
#     img = img_org.resize(((w//32)*32, (h//32)*32))
#     img = np.array(img, dtype=np.uint8)
#     img[img==255] = 0
#     y = np.zeros((1, img.shape[0], img.shape[1], nb_classes), dtype=np.float32)
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             y[0, i, j, img[i][j]] = 1
#     return y

def convert_P(img_org):   ##6 classes
    w, h = img_org.size
    img_P = np.zeros((h,w),np.uint8)
    #print w,h
    for i in range(0,w):
        for j in range(0,h):
            #RGB,not BGR
            if(img_org.getpixel((i,j))[0] == 128 and img_org.getpixel((i,j))[1] == 0 and img_org.getpixel((i,j))[2] == 0):
                img_P[j][i] = 1
            elif(img_org.getpixel((i,j))[0] == 0 and img_org.getpixel((i,j))[1] == 128 and img_org.getpixel((i,j))[2] == 0):
                img_P[j][i] = 2
            elif(img_org.getpixel((i,j))[0] == 128 and img_org.getpixel((i,j))[1] == 128 and img_org.getpixel((i,j))[2] == 0):
                img_P[j][i] = 3
            elif(img_org.getpixel((i,j))[0] == 0 and img_org.getpixel((i,j))[1] == 0 and img_org.getpixel((i,j))[2] == 128):
                img_P[j][i] = 4
            elif(img_org.getpixel((i,j))[0] == 128 and img_org.getpixel((i,j))[1] == 0 and img_org.getpixel((i,j))[2] == 128):
                img_P[j][i] = 5
            else:
                img_P[j][i] = 0
    return img_P

def load_label(path):
    img_org = Image.open(path)
    w, h = img_org.size
    
    img = img_org.resize(((w//32)*32, (h//32)*32))
    
    img_p = convert_P(img)  ##转换成'索引模式(8位彩色图像)'
    y = np.zeros((1, img_p.shape[0], img_p.shape[1], nb_classes), dtype=np.float32)
    for i in range(img_p.shape[0]):
        for j in range(img_p.shape[1]):
            y[0, i, j, img_p[i][j]] = 1
            #print img[i][j]
    return y

def generate_arrays_from_file(path, image_dir, label_dir):
    while 1:
        f = open(path)
        for line in f:
            filename = line.rstrip('\n')
            path_image = os.path.join(image_dir, filename+'.jpg')
            path_label = os.path.join(label_dir, filename+'.png')
            x = load_image(path_image)
            y = load_label(path_label)
            yield (x, y)
        f.close()

        
# def model_predict(model, input_path, output_path):
#     img_org = Image.open(input_path)
#     w, h = img_org.size
#     img = img_org.resize(((w//32)*32, (h//32)*32))
#     img = np.array(img, dtype=np.float32)
#     x = np.expand_dims(img, axis=0)
#     x = preprocess_input(x)
#     pred = model.predict(x)
#     pred = pred[0].argmax(axis=-1).astype(np.uint8)
#     img = Image.fromarray(pred, mode='P')
#     img = img.resize((w, h))
#     palette_im = Image.open('palette.png')
#     img.palette = copy.copy(palette_im.palette)
#     img.save(output_path)

# parser = argparse.ArgumentParser()
# parser.add_argument('train_data')
# parser.add_argument('image_dir')
# parser.add_argument('label_dir')
# args = parser.parse_args()
# nb_data = sum(1 for line in open(args.train_data))

###########################################################  train  #######################
path_to_txt = "train.txt"
with open(path_to_txt,"r") as f:
    ls = f.readlines()
names = [l.rstrip('\n') for l in ls]
nb_data = len(names)         

image_dir = "train/"  ##720
label_dir = "label/"

# image_dir = "data/VOC2011/JPEGImages/"  ##768
# label_dir = "data/VOC2011/SegmentationClass/"
# img_size = 224

model = fcn_32s()
model.compile(loss="binary_crossentropy", optimizer='sgd')
# for epoch in range(100):
#     model.fit_generator(
#         generate_arrays_from_file(path_to_txt, image_dir, label_dir),
#         steps_per_epoch=nb_data, 
#         epochs=1)
     #model_predict(model, 'test.jpg', 'predict-{}.png'.format(epoch))
    
model.fit_generator(
        generate_arrays_from_file(path_to_txt, image_dir, label_dir),
        steps_per_epoch=nb_data, 
        epochs=100)
################################################################################################

################################################################  test    ######################
def draw_predict(img):   ##6 classes
    #print img
    img_R = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            #BGR,not RGB
            if(img[i][j] == 1):
                img_R[i][j][0] = 0
                img_R[i][j][1] = 0
                img_R[i][j][2] = 128
            elif(img[i][j] == 2):
                img_R[i][j][0] = 0
                img_R[i][j][1] = 128
                img_R[i][j][2] = 0
            elif(img[i][j] == 3):
                img_R[i][j][0] = 0
                img_R[i][j][1] = 128
                img_R[i][j][2] = 128
            elif(img[i][j] == 4):
                img_R[i][j][0] = 128
                img_R[i][j][1] = 0
                img_R[i][j][2] = 0
            elif(img[i][j] == 5):
                img_R[i][j][0] = 128
                img_R[i][j][1] = 0
                img_R[i][j][2] = 128
            else:
                img_R[i][j][0] = 0
                img_R[i][j][1] = 0
                img_R[i][j][2] = 0
    return img_R



##测试predict,预测每一小块
def small_predict(model, input_path):
    img_org = Image.open(input_path)
    w, h = img_org.size
    img = img_org.resize(((w//32)*32, (h//32)*32))
    img = np.array(img, dtype=np.float32)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)
    pred = pred[0].argmax(axis=-1).astype(np.uint8)

    img = Image.fromarray(pred, mode='P')
    img = img.resize((w, h))
    img = np.array(img, dtype=np.uint8)
    
    img_R = draw_predict(img)
    #cv2.imwrite(output_path,img_R)
    
    return img_R
    
#输出大图
def big_predict(model,test_path,predict_path):
    step_row = 115
    step_col = 128
    count1 = 1
    srcImg = cv2.imread(test_path)
    
    img_big_R = np.zeros((srcImg.shape[0],srcImg.shape[1],3),np.uint8)
    
    for y in range(0,srcImg.shape[0],step_row):
        for x in range(0,srcImg.shape[1],step_col):
            #输出需要预测的小块
            imgTemp = np.zeros((step_row,step_col,3),np.uint8)
            for i in range(0,step_row):
                for j in range(0,step_col):
                    imgTemp[i][j][0] = srcImg[y + i][x + j][0]
                    imgTemp[i][j][1] = srcImg[y + i][x + j][1]
                    imgTemp[i][j][2] = srcImg[y + i][x + j][2]
                    
            outName1 = ""
            outName1 = "pred_%d.jpg"%(count1)
            cv2.imwrite(outName1,imgTemp)
            count1+=1
            
            #预测小块
            img_small_R = small_predict(model, outName1)
            
            #填充大块
            for i in range(0,step_row):
                for j in range(0,step_col):
                    img_big_R[y + i][x + j][0] = img_small_R[i][j][0]
                    img_big_R[y + i][x + j][1] = img_small_R[i][j][1]
                    img_big_R[y + i][x + j][2] = img_small_R[i][j][2]
                    
            os.remove(outName1)
            
    cv2.imwrite(predict_path,img_big_R)


#model = load_model('fcn_test_Edge1_0.0756_epoch100.h5') #加载训练好的模型
print 'start'
for i in range(1,17):
    test_img = '/notebooks/FCN_test/genarateData/data1/src_jpg/%d.jpg'%(i)
    big_predict(model,test_img,'results/predict-%d.png'%(i))
print 'done'
