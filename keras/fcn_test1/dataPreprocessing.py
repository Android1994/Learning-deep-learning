import cv2
import numpy as np

count = 1
step_row = 115
step_col = 128

print "start"

f=file("train.txt", "a+")
for n in range(1,17):
    srcPath = "/notebooks/FCN_test/genarateData/data1/src_img/%d.png"%(n)
    labelPath = "/notebooks/FCN_test/genarateData/data1/src_label/%d-label.png"%(n)

    srcImg = cv2.imread(srcPath)
    labelImg = cv2.imread(labelPath)

    ##将标签图片转换成8位彩色图,,,,BGR
    labelTemp = np.zeros((labelImg.shape[0],labelImg.shape[1],3),np.uint8)
    for y in range(0,labelImg.shape[0]):
        for x in range(0,labelImg.shape[1]):
            if(labelImg[y][x][0] == 0 and labelImg[y][x][1]==0 and labelImg[y][x][2]==255):
                labelTemp[y][x][0] = 0
                labelTemp[y][x][1] = 0
                labelTemp[y][x][2] = 128
            elif(labelImg[y ][x ][0] == 0 and labelImg[y ][x ][1]==255 and labelImg[y ][x ][2]==0):
                labelTemp[y][x][0] = 0
                labelTemp[y][x][1] = 128
                labelTemp[y][x][2] = 0
            elif(labelImg[y ][x ][0] == 255 and labelImg[y ][x ][1]==0 and labelImg[y ][x ][2]==0):
                labelTemp[y][x][0] = 0
                labelTemp[y][x][1] = 128
                labelTemp[y][x][2] = 128
            elif(labelImg[y ][x ][0] == 255 and labelImg[y ][x ][1]==255 and labelImg[y ][x ][2]==255):
                labelTemp[y][x][0] = 128
                labelTemp[y][x][1] = 0
                labelTemp[y][x][2] = 0
            elif(labelImg[y ][x ][0] == 0 and labelImg[y ][x ][1]==241 and labelImg[y ][x ][2]==255):
                labelTemp[y][x][0] = 128
                labelTemp[y][x][1] = 0
                labelTemp[y][x][2] = 128
            else:
                labelTemp[y][x][0] = 0
                labelTemp[y][x][1] = 0
                labelTemp[y][x][2] = 0
    cv2.imwrite("biglabel_8/label-%d.png"%(n),labelTemp)
    
    if(n != 12): ###留出第12张用于测试

        for y in range(0,srcImg.shape[0],step_row):
            for x in range(0,srcImg.shape[1],step_col):
                #输出训练样本
                imgTemp = np.zeros((step_row,step_col,3),np.uint8)
                for i in range(0,step_row):
                    for j in range(0,step_col):
                        imgTemp[i][j][0] = srcImg[y + i][x + j][0]
                        imgTemp[i][j][1] = srcImg[y + i][x + j][1]
                        imgTemp[i][j][2] = srcImg[y + i][x + j][2]
                outName1 = ""
                outName1 = "train/%d.jpg"%(count)
                cv2.imwrite(outName1,imgTemp)
                #train.txt
                new_context = "%d"%(count) + '\n'
                f.write(new_context)

                #输出标签样本
                label_small_Temp = np.zeros((step_row,step_col,3),np.uint8)
                for i in range(0,step_row):
                    for j in range(0,step_col):
                        label_small_Temp[i][j][0] = labelTemp[y + i][x + j][0]
                        label_small_Temp[i][j][1] = labelTemp[y + i][x + j][1]
                        label_small_Temp[i][j][2] = labelTemp[y + i][x + j][2]

                outName2 = ""
                outName2 = "label/%d.png"%(count)
                cv2.imwrite(outName2,label_small_Temp)
                
                count+=1

f.close()
print count
print "done"
