import os
import h5py
import numpy as np
import argparse
import time
from extract_cnn_vgg16_keras import extract_feat
# from memory_profiler import profile
from pyprind import ProgBar


# class feature
# 返回目录中所有jpg图像的文件名列表。
def getImageList(path):
    return [os.path.join (path, f) for f in os.listdir (path) if f.endswith ('.JPEG')]

# 命令行参数功能
def comdArgs():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", required = True, help = "训练集的路径")
    ap.add_argument("-f", required = True, help = "特征索引文件的路径名称")
    args = vars(ap.parse_args())
    # datasets = args["d"]
    # h5filename = args["f"]
    return getImageList(args["d"]),args["f"]


# 按指定格式读取h5文件
def rH5FileData(Key,filename):
    try:
        with h5py.File (filename, 'r') as h5f:
            feats = h5f["data" + str (Key)][:]
            imgNames = h5f["name" + str (Key)][:]
            return feats,imgNames[0].decode("utf-8")
    except KeyError:
        print("Read HDF5 File Key Error")
        return 1


# 按指定格式写入h5文件
def wH5FileData(Key,feats,names,filename):
    namess = []
    # 数据编码转换
    if type(names) is list:
        for j in names:
            namess.append (j.encode ())
    else:
        names.encode ()
    try:
        with h5py.File (filename, 'a') as h5f:
            h5f.create_dataset ("data"+str(Key), data=feats)
            h5f.create_dataset ("name"+str(Key), data=namess)
    except RuntimeError:
        raise NameError('Unable to create link (name already exists)')
    return 0


# 提取特征并写入文件
# @profile (precision=6)
def etlFeature(post,img_list,h5filename):

    # 迭代方式，提取特征值写入h5文件
    bar = ProgBar (len(img_list), monitor=True, title="提取图片特征,Image Total:%d" % len (img_list))
    for i, img_path in enumerate (img_list):
        norm_feat = extract_feat (img_path)
        img_name = os.path.split (img_path)[1]
        names = []
        names.append (img_name)
        feats2 = np.array (norm_feat)
        try:
            wH5FileData (i+post, feats2, names,h5filename)
        except:
            print("Feats Write Error")
            return 1
        bar.update ()
        # print ("提取图片特征！进度: %d/%d" % ((i + 1), len (img_list)))
    print (bar)
    return 0


# 获取HDF5文件数据数量，便于追加和读取。
def showHDF5Len(filename):
    # 文件不存在则重写，不追加
    if not os.path.exists(filename):
        return 0
    # 存在则追加
    with h5py.File (filename, 'r') as h5f:
        return int(len(h5f)/2)


# def main():
#     feats = []
#     h5filename = "./imageCNN.h5"
#     dataset = ""
#     img_list = getImageList(dataset)
#     etlFeature (showHDF5Len (h5filename), img_list, h5filename)
#     return 0


if __name__ == "__main__":
    pass
    feats = []
    # 数据文件
    h5filename = "./imageCNN2.h5"


    # 文件条数
    # lens = showHDF5Len (h5filename)
    # print(lens)
    import keras-cnn-imageSearch
    
    b = base()
    img_list = b.getImageList("学二公寓西")
    print(img_list)
    # etlFeature (showHDF5Len (h5filename), img_list, h5filename)





    # for i in range (showHDF5Len (h5filename)):
    #     feats, imgNames = rH5FileData (i, h5filename)
    #     print(imgNames)


    # 读取数据
    # featsList = []
    # nameList = []
    # for i in range (lens):
    #     feats, imgNames = rH5FileData (i, h5filename)
    #     featsList.append (feats)
    #     nameList.append (imgNames)

    # queryVec = extract_feat("./image/19700102125648863.JPEG")
    # featsList = np.array(featsList)
    # scores = np.dot(queryVec, featsList.T)
    # rank_ID = np.argsort(scores)[::-1] # 排序,倒序，大到小
    # rank_score = scores[rank_ID] # 计算评分
    # print(rank_score)





