# -*- coding: utf-8 -*-
# Author: yongyuan.name
import os
import h5py
import numpy as np
import argparse
import time
from extract_cnn_vgg16_keras import extract_feat
from memory_profiler import profile

# 命令行参数功能
# ap = argparse.ArgumentParser()
# ap.add_argument("-database", required = True,
# 	help = "通往数据库的路径，其中包含要被索引的图像")
# ap.add_argument("-index", required = True,
# 	help = "索引文件的名称")
# args = vars(ap.parse_args())


# 返回目录中所有jpg图像的文件名列表。
def get_imlist(path):
    return [os.path.join (path, f) for f in os.listdir (path) if f.endswith ('.JPEG')]

# 按指定格式读取h5文件
def rH5FileData(i,filename):
    with h5py.File (filename, 'r') as h5f:
        feats = h5f["data" + str (i)][:]
        imgNames = h5f["name" + str (i)][:]
        return feats,imgNames

# 按指定格式写入h5文件
def wH5FileData(i,feats,names,filename):
    namess = []
    # 数据编码转换
    if type(names) is list:
        for j in names:
            namess.append (j.encode ())
    else:
        names.encode ()

    h5f = h5py.File (filename, 'a')
    h5f.create_dataset ("data"+str(i), data=feats)
    h5f.create_dataset ("name"+str(i), data=namess)
    h5f.close ()
    return 0


# 提取特征并写入文件
# @profile (precision=6)
def etlFeature(img_list,h5filename):

    names = []
    # 迭代方式，提取特征值写入h5文件
    for i, img_path in enumerate (img_list):
        norm_feat = extract_feat (img_path)
        img_name = os.path.split (img_path)[1]
        names.append (img_name)
        feats2 = np.array (norm_feat)
        wH5FileData (i, feats2, names,h5filename)
        print ("从图像中提取特征: %d ,图片总:%d" % ((i + 1), len (img_list)))
    return 0


if __name__ == "__main__":
    # db = "./image"
    db = "D:/datasets/003"
    img_list = get_imlist (db)
    print ("\n特征提取开始!\n")

    feats = []
    h5filename = "./model"
    h5filename = h5filename + str(len(img_list)) + ".h5"

    etlFeature (img_list, h5filename)


    # 读取
    featsList = []
    nameList = []
    imagesize = int(h5filename[7:-3])
    for i in range (imagesize):
        feats, imgNames = rH5FileData (i, h5filename)
        featsList.append (feats)
        nameList.append (imgNames[0])

    queryVec = extract_feat("./image/19700102125648863.JPEG")
    featsList = np.array(featsList)
    scores = np.dot(queryVec, featsList.T)
    rank_ID = np.argsort(scores)[::-1] # 排序,倒序，大到小
    rank_score = scores[rank_ID] # 计算评分
    print(rank_score)






