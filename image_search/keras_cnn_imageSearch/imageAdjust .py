# 图片调整

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from pyprind import ProgBar

class imgchange():

    def ichange(self,imagefile1,imagefile2,w,h):
        
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
            
            plt.imshow (cat)
            plt.show ()
        return 0

    # 返回目录中所有jpg图像的文件名列表。
    def getImageList(self,path):
        return [os.path.join (path, f) for f in os.listdir (path) if f.endswith ('.JPEG')]
    
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


    c.ichange ("D:/datasets/trainset/20150630183252485.JPEG", "./123.JPEG", 272, 480)