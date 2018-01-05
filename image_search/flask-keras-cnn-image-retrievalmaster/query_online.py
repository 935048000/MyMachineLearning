# -*- coding: utf-8 -*-
# Author: yongyuan.name
from extract_cnn_vgg16_keras import extract_feat

import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-query", required = True,
	help = "通往数据库的路径，其中包含要被索引的图像")
ap.add_argument("-index", required = True,
	help = "路径指数")
ap.add_argument("-result", required = True,
	help = "输出检索图像的路径")
args = vars(ap.parse_args())


# 读取索引图像的特征向量和相应的图像名称
h5f = h5py.File(args["index"],'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()

print ("--------------------------------------------------")
print ("               搜索开始")
print ("--------------------------------------------------")
    
# 读取和显示查询图像
queryDir = args["query"]
queryImg = mpimg.imread(queryDir)
plt.title("Query Image")
plt.imshow(queryImg)
plt.show()


# 提取查询图像的特征，计算simlarity评分和排序
queryVec = extract_feat(queryDir)
scores = np.dot(queryVec, feats.T)
rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]
#print rank_ID
#print rank_score


# 显示的顶部检索图像数目
maxres = 3
imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
print ("最高的 %d 图片为: " %maxres, imlist)
 

# 显示检索结果
# for i,im in enumerate(imlist):
#     image = mpimg.imread(args["result"]+"/"+im)
#     plt.title("搜索输出 %d" %(i+1))
#     plt.imshow(image)
#     plt.show()
