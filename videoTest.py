import cv2
import os
import numpy as np
from threading import Thread
import time

# 该阈值根据红光照在蓝色立方体上亮度会降低，而放宽了v值，但对h和s多了限制。
hsv_lower_hsv = np.array([112, 74, 140])  # hsv过滤范围最小值
hsv_upper_hsv = np.array([180, 255, 255])  # hsv过滤范围最大值

# 该阈值主要是根据v值进行筛选(pane使用)
light_lower_hsv = np.array([0, 0, 200])  # hsv过滤范围最小值
light_upper_hsv = np.array([180, 255, 255])  # hsv过滤范围最大值

FILEPATH = os.path.dirname(os.path.abspath(__file__))  # 获取当前py文件的路径
WORKINGPATH = FILEPATH + '/resources/'  # 存放pane、test、video的目录


class Cap:
    """获取摄像头捕获与三维重构计算体积的核心类，视频测试类"""
    def __init__(self):
        self.pane = None  # 存放固定的pane文件，cv2读取的图片，相当于本来的simage
        self.pane_canny = None  # debug需要，canny运算后得到的图像
        self.pane_2d_iamge = None  # debug需要
        self.pane_2d = None  # 存放pane的2d点列阵
        self.pane_3d = None  # 存放pane的3d点列阵
        self.pane_size = None  # 点的数量(其实就是pane_2d的行数)

        self._2d_light_points_dx = None  # 横截面积积分时，三维点在世界坐标系y轴中的距离

        self._last_stamp = 0  # 上一帧的时间戳(ms)
        self._current_stamp = 0  # 当前帧的时间戳(ms)
        self.difference_stamp = 0  # 时间戳的差值(ms)
        self.volume = 0  # 累计的体积
        self.speed = 0.01  # 传送带的速度(m/s)

        self.image = None  # 当前帧获取的图像
        self.cap = None  # cv2.VideoCapture对象
        self.handle = None  # 运行时的线程句柄（包含测试与运行）
        self.run_flag = False  # 是否运行的flag

        self.linepen = [1.0, 67.3051, -10.7165, 5213.5]  # 光平面方程参数
        self.kk = np.array([[563.0877, 0, 491.7189],
                            [0, 562.3336, 255.9260],
                            [0, 0, 1]],
                           dtype=np.float32)  # MATLAB数据 相机的内外参数
        self.p1 = [185, 130]  # 光条纹范围的起点，p1为左上角，p2是右下角
        self.p2 = [800, 222]  # 使用list代替point类型，[0]:x，[1]:y

        self.distan = np.zeros((1, 1), dtype=np.float32)
        self.scale = 0.5  # 缩放倍数

    # 辅助函数，计算三维坐标之间的欧氏距离
    @staticmethod
    def _get_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """辅助函数，计算三维坐标之间的欧氏距离
        :param p1:np.ndarray
        :param p2:np.ndarray
        :return: float
        """
        re = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
        # print("=DEBUG= p1,p2:",p1,p2,"re:",re)
        re = re ** 0.5
        return re

    # 计算三维坐标
    def _get_3d_points(self, line: list, p2d: np.ndarray) -> np.ndarray:
        kk_invert = np.linalg.inv(self.kk)
        re3_dp = np.zeros((1, 3), np.float32)
        data = np.zeros((3, 1), np.float32)
        for i in range(p2d.shape[0]):
            data[0][0] = p2d[i][0]
            data[1][0] = p2d[i][1]
            data[2][0] = 1

            # data = kk_invert * data
            data = np.dot(kk_invert, data)  # c++中直接a*b，python中需要使用dot函数
            # print("=DEBUG= i={},data.shape:{} [0][0]:{}".format(i, data.shape,  data))

            m = data[0][0] * line[0] + data[1][0] * line[1] + line[2]  # m是float
            m = line[3] / (-m)
            for x in range(3):
                re3_dp[i][x] = m * data[x][0]
            # 应该还是加一行，虽然不知道为什么是resize(i+2)
            gum = np.zeros((1, 3), dtype=np.float32)
            re3_dp = np.concatenate((re3_dp, gum), axis=0)
        return re3_dp

    # 获取光条纹中心
    @staticmethod
    def _get_light_mid(image: np.ndarray, p1: list, p2: list) -> np.ndarray:
        midp2d = np.zeros((1, 2), dtype=np.float32)
        col = 0
        maxx, minx = max(p1[0], p2[0]), min(p1[0], p2[0])
        maxy, miny = max(p1[1], p2[1]), min(p1[1], p2[1])

        for i in range(int(minx), int(maxx)):
            mix = miy = 0
            for j in range(int(miny), int(maxy)):
                pix = image[j][i]
                if pix != 0 and mix == 0:
                    m = 50
                    while m != 0:
                        if image[j + m][i] != 0: break
                        m -= 1
                    mix, miy = i, j + m // 2  # 这里应该涉及除法or整除
                    midp2d[col][0] = mix
                    midp2d[col][1] = miy
                    col += 1
                    # 似乎没有类似的resize方法（无法拓展），考虑使用concatenate函数拼接。gum就像口香糖一样黏在后面，size为(1,2)和上面定义midp2d对应，直接写死
                    gum = np.zeros((1, 2), dtype=np.float32)
                    midp2d = np.concatenate((midp2d, gum), axis=0)
                    break
        return midp2d

    @staticmethod
    # 2d点坐标，生成图像方便调试
    def _2d_to_image(points: np.ndarray):
        _temp = np.zeros((540, 960), dtype=np.uint8)
        for each in points:
            _temp[int(each[1])][int(each[0])] = 255
        return _temp

    # 该函数接收彩色原图，首先进行hvs阈值提取光条纹
    def set_pane(self, image: np.ndarray):
        hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 转换为hvs模型
        mask = cv2.inRange(hsv_frame, lowerb=light_lower_hsv, upperb=light_upper_hsv)
        # cv2.imshow('mask', mask)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pane = cv2.bitwise_and(image_gray, image_gray, mask=mask)  # 过滤范围内的部分，得到灰度图
        self.set_pane_gray(pane)  # 传给其灰度图

    # 该函数接收灰度图（同原C++函数逻辑）
    def set_pane_gray(self, pane: np.ndarray) -> None:
        self.pane = pane
        _pane = cv2.medianBlur(self.pane, 5)  # 中值滤波
        self.pane_canny = cv2.Canny(_pane, 60, 180, apertureSize=3, L2gradient=False)  # Canny检测边缘

        self.pane_2d = self._get_light_mid(self.pane_canny, self.p1, self.p2)
        self.pane_3d = self._get_3d_points(self.linepen, self.pane_2d)

        print("[DEBUG][SET_PANE]2d 0: u = %f v = %f " % (self.pane_2d[0, 0], self.pane_2d[0, 1]))
        print("[DEBUG][SET_PANE]3d 0: x = %f y = %f z = %f " % (self.pane_3d[0, 0], self.pane_3d[0, 1], self.pane_3d[0, 2]))
        self.pane_2d_iamge = self._2d_to_image(self.pane_2d)
        self.pane_size = self.pane_2d.shape[0] - 1
        # dx好像是光条纹的宽度
        self._2d_light_points_dx = self._get_distance(self.pane_3d[0], self.pane_3d[self.pane_size - 1]) / self.pane_size  # ？？为什么是这样得到dx
        print("[DEBUG][SET_PANE]_2d_light_points_dx : ", self._2d_light_points_dx, '\n')

    # 设置摄像头，可为视频、网络摄像头。如果是用的摄像头，那就把摄像头设置为1920*1080
    def set_camera(self, url):
        self.cap = cv2.VideoCapture(url)
        if url == 1 and self.cap is not None:
            self.cap.set(3, 1920)  # 宽
            self.cap.set(4, 1080)  # 高

    # 设置速度函数
    def set_speed(self, _speed):
        self.speed = _speed

    # 释放摄像头
    def release_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # 开始运行，并启动计算线程
    def on(self, _type: str):
        """on，开始运行计算
        _type: 'main','loop_test'
        """
        # 判断是否有正在运行的计算线程
        if self.handle is not None:
            print('[DEBUG][ON]有正在运行的线程')
            return 1
        # 判断参数是否合法
        if _type not in ['main', 'loop_test']:
            print('[DEBUG][ON]参数不合法')
            return 2

        self.run_flag = True   # 运行标志位
        if _type == 'main':
            # ret, iamge = self.cap.read()  # 获取摄像头当前帧的图像
            # self.image = cv2.resize(iamge, (960, 540), interpolation=cv2.INTER_CUBIC)
            # self._current_stamp = self.cap.get(cv2.CAP_PROP_POS_MSEC)  # 获取当前帧的时间戳
            # self._last_stamp = self._current_stamp
            self.handle = Thread(target=self._shoot_and_calc, args=())
        elif _type == 'loop_test':
            self.handle = Thread(target=self.loop_test, args=())
        self.handle.start()

    # 结束运行loop_test
    def off(self):
        self.run_flag = False
        self.handle = None

    # 拍照并计算，另起一个线程操作（该文件不可用）
    def _shoot_and_calc(self, **kwargs):
        return 0

    # 模拟循环拍照并计算体积，实际将拍照所得图像更改为test1、test2、test3
    def loop_test(self):
        frame_index = 0  # debug变量
        while self.run_flag:
            frame_index += 1
            print('[DEBUG][LOOP_TEST]frame_index:', frame_index)
            distan = np.zeros((self.pane_size, 1), dtype=np.float32)
            # 摄像获取图像(不起作用，仅测试函数是否正常运行）
            ret, frame = self.cap.read()
            if not ret:
                self.off()
                break
            self._current_stamp = self.cap.get(cv2.CAP_PROP_POS_MSEC)  # 获取当前帧的时间戳
            self.difference_stamp = self._current_stamp - self._last_stamp   # 得到时间戳差值
            self._last_stamp = self._current_stamp  # 更新上一帧时间戳
            # print("[DEBUG][LOOP_TEST]difference_stamp:", self.difference_stamp)

            # frame_resize = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_CUBIC)
            # image = cv2.rotate(frame_resize  , cv2.ROTATE_180)  # 摄像装反了旋转一下
            image = frame

            # DEBUG
            cv2.imshow('video', image)
            cv2.imshow('pane_2d_image', self.pane_2d_iamge)

            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(image_hsv, lowerb=hsv_lower_hsv, upperb=hsv_upper_hsv)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_gray_masked = cv2.bitwise_and(image_gray, image_gray, mask=mask)  # 过滤范围内的部分

            cv2.waitKey(1)
            dimage = cv2.medianBlur(image_gray_masked, 5)  # 中值滤波
            dimage = cv2.Canny(dimage, 60, 180, apertureSize=3, L2gradient=False)  # Canny检测边缘

            midt2d = self._get_light_mid(dimage, self.p1, self.p2)
            midt3d = self._get_3d_points(self.linepen, midt2d)

            cv2.imshow('midt2d_image', self._2d_to_image(midt2d))
            # tsize = midt2d.shape[0] - 1  # tsize似乎不起作用
            # count = pcount = mcount = 0
            # 循环计算3d距离
            # while True:
            #     if midt2d[mcount][0] == self.pane_2d[pcount][0]:
            #         distan[count][0] = self._get_distance(self.pane_3d[pcount], midt3d[mcount])
            #         mcount += 1
            #         pcount += 1
            #     else:
            #         pcount += 1
            #         distan[count][0] = 0
            #     count += 1  # 因为c里，if 和 else里都写了 count++，所以直接放这里
            #     if count == self.pane_size:
            #         break
            #
            #     gum = np.zeros((1, 1), dtype=np.float32)
            #     distan = np.concatenate((distan, gum), axis=0)

            # 新写一个试试！！！！
            # 因为原始程序distan的size始终为(panez_size,2)，所以直接循环这么多次
            mcount = 0
            for i in range(self.pane_size):
                # 如果test的光条纹比pane更偏左边  (mcount是test图像的指针)
                while midt2d[mcount][0] < self.pane_2d[i][0]:  # 此处存在可能mcount下标越界的bug，暂时假设一定有对应
                    mcount += 1
                # 跳出循环则说明，midt2d[mcount][0] >= self.pane_2d[i][0]
                if midt2d[mcount][0] == self.pane_2d[i][0]:
                    distan[i][0] = self._get_distance(self.pane_3d[i], midt3d[mcount])
                # else说明midt2d[mcount][0] > self.pane_2d[i][0]
                else:
                    distan[i][0] = 0

            # distan.resize(count+1);
            # print("count : ", count)

            # 计算截面积
            area = 0
            for i in range(self.pane_size):
                if distan[i][0] < 10:
                    distan[i][0] = 0
                area = area + distan[i][0]
            area = self._2d_light_points_dx * area * 0.8916252
            print("[DEBUG][LOOP_TEST]area:", area)
            # 计算这一帧图像所用时间物体走过的路程( m/s * ms / 1000)
            _dx = self.speed * self.difference_stamp / 1000
            # print("[DEBUG][LOOP_TEST]一帧中走过的_dx:", _dx)
            self.volume = self.volume + area * _dx  # 这里到底应该乘什么？单位
            print("[DEBUG][LOOP_TEST]volume:", self.volume, '\n')

            # # 显示光条纹中心
            # lightmidimg = np.zeros(dimage.shape, dtype=np.uint8)
            # for i in range(midt2d.shape[0] - 1):
            #     lightmidimg[int(self.pane_2d[i][1]), int(self.pane_2d[i][0])] = 255
            #
            # dsize = int(self.pane.shape[1] * self.scale), int(
            #     self.pane.shape[0] * self.scale)  # 这里应该还是先row再col(我错了,先col)
            # image2 = cv2.resize(test_image, dsize)
            # cv2.imshow("Source image", image2)
            # image2 = cv2.resize(lightmidimg, dsize)
            # cv2.imshow("Dest image", image2)

            # k = cv2.waitKey(5) & 0xff  # 所以延时xms，不考虑代码运算时间？
            # if k == 27:
            #     break

        return 0


if __name__ == '__main__':
    a = Cap()  # 实例化一个对象

    ImagePath = WORKINGPATH + 'pane.jpg'
    simage = cv2.imread(ImagePath)  # 读取pane的原图彩色图
    a.set_pane(simage)  # 设置pane图像

    a.set_camera(WORKINGPATH + 'video1.mp4')
    print('a.cap', a.cap)
    a.on('loop_test')
    # time.sleep(20)  # 播放视频会自动 off
    # a.off()  # 结束测试loop
    while a.run_flag:
        pass
    time.sleep(0.5)  # 确保拍照计算线程已经结束
    a.release_camera()
    print("finish")
    print('volume:', a.volume)
    # a.shoot_and_set_pane()
