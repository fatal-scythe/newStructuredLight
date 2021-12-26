
import cv2
import os

FILEPATH = os.path.dirname(os.path.abspath(__file__))  # 获取当前py文件的路径

for i in range(1,11):
    ImagePath = FILEPATH + "/ob" + str(i) + '.jpg'
    print(ImagePath)
    simage = cv2.imread(ImagePath, cv2.IMREAD_GRAYSCALE)
    outPath = FILEPATH + "/gb" + str(i) + '.jpg'
    cv2.imwrite(outPath, simage)
