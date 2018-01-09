# -*- coding: utf-8 -*-
# Author: yongyuan.name
from extract_cnn_vgg16_keras import extract_feat

import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

# 命令行参数功能
# ap = argparse.ArgumentParser()
# ap.add_argument("-query", required = True,
# 	help = "通往数据库的路径，其中包含要被索引的图像")
# ap.add_argument("-index", required = True,
# 	help = "路径指数")
# ap.add_argument("-result", required = True,
# 	help = "输出检索图像的路径")
# args = vars(ap.parse_args())
# print(args)


def showimage(queryImage,imlist,result):
    # 读取和显示查询图像
    queryImg = mpimg.imread(queryImage)
    plt.title("Query Image")
    plt.imshow(queryImg)
    plt.show()
    # 显示检索结果
    for i,im in enumerate(imlist):
        image = mpimg.imread(result+"/"+im)
        plt.title("top %d" %(i+1))
        plt.imshow(image)
        plt.show()

    return 0

# result = args['result']
result = "./imagesets"
# Model = args[index'']
Model = "./youdian2CNN.h5"

# 读取索引图像的特征向量和相应的图像名称
h5f = h5py.File(Model,'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()

# queryImage = args['query']
# queryImage = "./imagesets/19700102142532557.JPEG"
queryImage = "D:/datasets/trainingset1/19700102130026909.JPEG"




# 提取查询图像的特征，计算 simlarity 评分和排序
queryVec = extract_feat(queryImage)
scores = np.dot(queryVec, feats.T) # 计算点积,计算图像得分
rank_ID = np.argsort(scores)[::-1] # 排序,倒序，大到小
rank_score = scores[rank_ID] # 计算评分
# print("scores",scores,type(scores))
# print ("rank_ID",rank_ID,type(rank_ID))
# print ("rank_score",rank_score,type(rank_score))


# 显示的顶部检索图像数目
maxres = 3 # 显示三个
imList = []
scoresList = []
for i, index in enumerate (rank_ID[0:maxres]):
    _temp = imgNames[index].decode()
    imList.append (_temp)

for j in rank_score[0:maxres]:
    _temp = float("%.2f" % (j*100))
    scoresList.append(_temp)

print ("最高的%d张图片为: " %maxres, imList)
print ("最高%d张图片的相似度评分："%maxres,scoresList)
 

showimage(queryImage,imList,result)
