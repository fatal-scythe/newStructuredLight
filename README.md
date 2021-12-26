# README

## 文件夹索引

calib_original_images——存放用来标定的30张图片  


## 应该怎么拍那个标定

    内外参数标定需要：20张棋盘格图片（不带光条纹）
    光平面标定需要：20张棋盘格图（10组，1组一张带光条纹的一张不带光条纹的）

    在matlab的 PD = GetLightMid(  LightMid ,lx,ly,fc,cc,kc,alpha_c,H_1 ); 这个函数中，就假设是写错了， 最后应该是每幅图对应的H矩阵。

    所以所以，
    拍照应该，一共拍30张
    a1 a2 .... a10 a11 ...a20  与 b1 ... b10

    其中a1~a20这20张用来内外参标定，得到KK（Calib_Result.mat）
    a1~a10与b1~b10这20张用来光平面标定。

## 结构光命名规范

    # cap获取的一帧叫frame
    ret, frame = cap.read() 

    # 大小缩放为规定的960*540叫frame_resize
    frame_resize  = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_CUBIC)
    
    # 被旋转之后的原图，叫做image
    image = cv2.rotate(frame_resize  cv2.ROTATE_180)  

    # 灰度图
    image_gray 

    # hsv模型图
    image_hsv

    # 被提取出光条纹后的灰度图
    image_gray_masked

