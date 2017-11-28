import cv2
import matplotlib.pyplot as plt
import numpy as np  
from keras.models import load_model  

img=cv2.imread('images/18.tif',cv2.IMREAD_GRAYSCALE)

#形态学
element = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))  
dilated = cv2.dilate(img,element)
eroded = cv2.erode(dilated,element)

#二值化
#thre1 = cv2.adaptiveThreshold(dilated,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,0)
ret,thre = cv2.threshold(dilated,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite("thre.png",thre)

#遍历图像
radius_left_up = 13
radius_right_down = 14
step = 1
bianChang = radius_left_up + radius_right_down + 1

model = load_model('edge_model1_random.h5') #加载训练好的模型
count=0
imgResult = np.zeros((thre.shape[0],thre.shape[1]),np.uint8) #生成一个空图像

for i in range(radius_left_up, thre.shape[0]-radius_right_down, step): #shape[0]行数
    for j in range(radius_left_up, thre.shape[1]-radius_right_down, step): #shape[1]列数
        if(thre[i,j] == 0):
            count+=1
            imgTemp = np.zeros((bianChang,bianChang),np.uint8) #生成一个空图像
            for y in range(bianChang):
                for x in range(bianChang):
                    imgTemp[y,x] = img[i+(y-radius_left_up),j+(x-radius_left_up)]
            X_image = imgTemp.reshape(1,28,28,1).astype('float32')
            preds = model.predict(X_image)
            preds=preds.reshape(2,1)
            if(preds[0] < preds[1]): #[0,1]->edge
                #print 'edge'
                imgResult[i,j]=255
            #else:
                #print 'noedge'
                    

print count #265194
cv2.imwrite("res.png",imgResult)
print 'done'
# plt.figure()
# plt.imshow(dilated)
# plt.figure()
# plt.imshow(thre)
# plt.figure()
# plt.imshow(dilated)
# #plt.title('My image')
# plt.show()


