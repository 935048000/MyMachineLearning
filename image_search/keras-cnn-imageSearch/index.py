# -*- coding: utf-8 -*-
# Author: yongyuan.name
import os
import h5py
import numpy as np
import argparse
import time
from extract_cnn_vgg16_keras import extract_feat

# 命令行参数功能
# ap = argparse.ArgumentParser()
# ap.add_argument("-database", required = True,
# 	help = "通往数据库的路径，其中包含要被索引的图像")
# ap.add_argument("-index", required = True,
# 	help = "索引文件的名称")
# args = vars(ap.parse_args())


'''
 返回目录中所有jpg图像的文件名列表。
'''
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.JPEG')]


'''
 提取特征和索引图像
'''
if __name__ == "__main__":
    start = time.clock ()

    # db = args["database"]
    db = "./imagesets"
    img_list = get_imlist(db)

    print ("\n特征提取开始!\n")
    
    feats = []
    names = []

    for i, img_path in enumerate(img_list):
        norm_feat = extract_feat(img_path)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        print ("从图像中提取特征: %d ,图片总:%d" %((i+1), len(img_list)))

    feats = np.array(feats)

    # 存储提取特征的目录
    output = "./youdian3CNN.h5"
    # output = args["index"]

    print ("\n写入特征提取结果......\n")

    # 数据编码转换
    namess = []
    for j in names:
        namess.append (j.encode ())

    h5f = h5py.File(output, 'w',)
    h5f.create_dataset('dataset_1', data = feats)
    h5f.create_dataset('dataset_2',  data = namess)
    h5f.close()

    elapsed = (time.clock() - start)
    print ("Time used:{}s".format(elapsed))

