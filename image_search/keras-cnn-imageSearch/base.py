import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import os
from pyprind import ProgBar
import time

class base():
    """
    这是一个基础方法类，封装常用方法。
    """

    # 返回目录中指定后缀的文件列表。
    def getFileList(self,path,type):
        return [os.path.join (path, f) for f in os.listdir (path) if f.endswith ('.'+type)]

    # 获取图像文件名字,有文件类别后缀的，无路径。
    def getImageName(self,imagefilepath):
        _temp = imagefilepath.split ("/")[-1]
        return _temp.split ("\\")[-1]

    # 获取图像名字,无文件类别后缀。
    def getImageName2(self,imagefile):
        return imagefile.split (".")[0]

    # 获取图像信息文件名，“.txt” 为后缀。
    def getImageTxtName(self,imagefile):
        return imagefile + ".txt"