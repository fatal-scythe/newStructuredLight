import cv2
import os
import time
import numpy as np

# url = 'rtsp://admin:linux123@115.25.41.108/'
# cap = cv2.VideoCapture(url)
cap = cv2.VideoCapture(1)
print(cap.isOpened())
cap.set(3, 1920)  # 宽
cap.set(4, 1080)  # 高


# 给BGR的图，读取将红色变为白色，返回BGR
def _red_enhance(img: np.ndarray) -> np.ndarray:
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 周围红色条纹
    lower_hsv = np.array([151, 34, 200])  # hsv过滤范围最小值
    upper_hsv = np.array([180, 255, 255])  # hsv过滤范围最大值
    mask1 = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
    # 中心亮条纹
    lower_hsv = np.array([0, 0, 245])  # hsv过滤范围最小值
    upper_hsv = np.array([180, 255, 255])  # hsv过滤范围最大值
    mask2 = cv2.inRange(img_hsv, lower_hsv, upper_hsv)  # lower20===>0,upper200==>0,

    mask = cv2.add(mask1, mask2)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(3000)
    for i in range(img_bgr.shape[0]):
        for j in range(img_bgr.shape[1]):
            if mask[i][j] == 255:
                img_bgr[i][j] = np.array((255, 255, 255))

    # b = ma.masked_where(mask != 0, img_hsv)
    # b.set_fill_value(np.array((180, 255, 255), dtype=img.dtype))   # 将mask的白色部分直接给到b
    # img_hsv = b.filled()
    return img_bgr


# 最终结论，拍摄30张用来标定
def calib30():
    """总共拍摄30张
    a1~a20:无光条纹
    b1~b10:有光条纹"""
    floder_name = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    filedir = os.path.dirname(os.path.abspath(__file__)) + "\\" + floder_name + "\\"
    if not os.path.exists(filedir):
        os.mkdir(filedir)

    suffix = '.jpg'
    ia = ib = 0

    while True:
        ret, image = cap.read()
        image = cv2.rotate(image, cv2.ROTATE_180)
        image = cv2.resize(image, (960, 540), interpolation=cv2.INTER_NEAREST)  # 大小统一缩放为960,540
        cv2.imshow('allInOne', image)
        cv2.moveWindow('allInOne', 0, 0)

        k = cv2.waitKey(5) & 0xff
        if k == 27:  # ESC
            break
        elif k == ord('a'):  # 没有光的
            if ia != 0: cv2.destroyWindow('no light: %da' % ia)
            ia += 1
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度图
            cv2.imshow('no light: %da' % ia, gray_image)
            cv2.moveWindow('no light: %da' % ia, 960, 0)
            cv2.imwrite(filedir + 'oa' + str(ia) + suffix, image)  # 原图oa
            cv2.imwrite(filedir + 'a' + str(ia) + suffix, gray_image)  # 灰度a
            print('no light ok:', 'a' + str(ia) + suffix)

        elif k == ord('b'):  # 有光的
            if ib != 0: cv2.destroyWindow('light: %db' % ib)
            ib += 1
            enhanced_image = _red_enhance(image)
            enhanced_gray_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)  # 增强过的
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 灰度图
            cv2.imshow('light: %db' % ib, gray_image)
            cv2.moveWindow('light: %db' % ib, 0, 540)
            cv2.imwrite(filedir + 'ob' + str(ib) + suffix, image)
            cv2.imwrite(filedir + 'eb' + str(ib) + suffix, enhanced_gray_image)
            cv2.imwrite(filedir + 'b' + str(ib) + suffix, gray_image)
            print('light ok:', 'b' + str(ib) + suffix)


if __name__ == '__main__':
    calib30()
    cv2.destroyAllWindows()
    cap.release()
