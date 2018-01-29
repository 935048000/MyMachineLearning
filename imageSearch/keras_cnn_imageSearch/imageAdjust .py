# 图片调整

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from pyprind import ProgBar
import cv2

class imgchange():

    def ichange(self,imagefile1,imagefile2,h,w):
        
        # 读取图片
        image_raw_data = tf.gfile.FastGFile (imagefile1, 'rb').read ()
        # 解码
        img_data = tf.image.decode_jpeg (image_raw_data)

        # 重新调整图片大小
        with tf.Session () as sess:
            resized = tf.image.resize_images (img_data, [h,w], method=3)
        
            # TensorFlow的函数处理图片后存储的数据是float32格式的，需要转换成uint8才能正确打印图片。
            # print("Digital type: ", resized.dtype)
            
            # 转换成uint8数据格式
            cat = np.asarray (resized.eval (), dtype='uint8')
            # cat = tf.image.convert_image_dtype (resized.eval (), tf.uint8)
            # 编码
            image_jpg = tf.image.encode_jpeg (cat)
        
            with tf.gfile.GFile(imagefile2,'wb') as f:
                f.write(image_jpg.eval ())
            
            # plt.imshow (cat)
            # plt.show ()
        return 0

    def ichange2(self,imageFile1,imageFile2,h,w):
        image = cv2.imread (imageFile1)
        res = cv2.resize (image, (w, h), interpolation=cv2.INTER_AREA)
        # cv2.imshow ('iker', res)
        # cv2.imshow('image',image)
        # cv2.waitKey (0)
        cv2.imwrite(imageFile2,res)
        # cv2.destroyWindow()
        return 0

if __name__ == '__main__':
    c = imgchange ()
    # imgfile = "D:/datasets/trainset/19700102141321203.JPEG"
    

    # imgList = c.getImageList("D:/datasets/trainset1-1")
    # bar = ProgBar (len(imgList[844:]), monitor=True, title="datasets_to_mini", bar_char="=")
    # for i in imgList[844:]:
    #     c.ichange(i,i,272,480)
    #     bar.update ()
    #     if (imgList[844:].index(i) % 50) == 0:
    #         print("%d/%d" % (imgList.index(i),len(imgList)))
    # print (bar)

    pass

    from imageSearch.keras_cnn_imageSearch.base import base

    b = base ()

    # img_list1 = b.getImageList ("学二公寓西")
    # img_list2 = b.getImageList ("教三楼南广场")
    # img_List = img_list1 + img_list2
    # print (len (img_List))
    #
    # img_List2 = []
    # for j in img_List:
    #     img_List2.append(b.getImageName(j))
    # print(len(img_List2),img_List2[0])
    #
    # bar = ProgBar (len (img_List), monitor=True, title="datasets_to_mini", bar_char="=")
    # for i1,i2 in zip(img_List,img_List2):
    #     c.ichange(i1, "./temp_image1/%s" % i2, 480, 272)
    #     bar.update ()
    # print (bar)
    #
    #
    # bar = ProgBar (len (img_List), monitor=True, title="datasets_to_mini", bar_char="=")
    # for i1, i2 in zip (img_List, img_List2):
    #     c.ichange2 (i1, "./temp_image2/%s" % i2, 480, 272)
    #     bar.update ()
    # print (bar)
    
    pass
    
    # c.ichange ("D:/datasets/testingset/20150630152018514.JPEG", "./20150630152018514.JPEG", 480, 272)
    # c.ichange2 ("D:/datasets/testingset/20150630152018514.JPEG", "./20150630152018514.JPEG", 480, 272)