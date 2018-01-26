from extract_cnn_vgg16_keras import extract_feat

import numpy as np
import h5py

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import argparse
import os
import time
from memory_profiler import profile

# 命令行参数功能
def comdArgs():
    ap = argparse.ArgumentParser()
    ap.add_argument("-image", required = True, help = "图像路径")
    ap.add_argument("-f", required = True, help = "特征文件路径")
    ap.add_argument("-result", required = True, help = "检索图像的路径")
    args = vars(ap.parse_args())
    
    result = args['result']
    Model = args['f']
    queryImage = args['image']
    return queryImage,Model,result

# 返回目录中所有jpg图像的文件名列表。
def getImageList(path):
    return [os.path.join (path, f) for f in os.listdir (path) if f.endswith ('.JPEG')]

def getInfoFileList(path):
    return [os.path.join (path, f) for f in os.listdir (path) if f.endswith ('.txt')]

# 获取HDF5文件数据数量，便于追加和读取。
def showHDF5Len(filename):
    # 文件不存在则重写，不追加
    if not os.path.exists(filename):
        return 0
    # 存在则追加
    with h5py.File (filename, 'r') as h5f:
        return int(len(h5f)/2)

# 显示图片：查询图，匹配结果图
def showimage(queryImage, imlist, result):
    # 读取和显示查询图像
    _queryImg = mpimg.imread (queryImage)
    plt.title ("Query Image")
    plt.imshow (_queryImg)
    plt.show ()
    # 显示检索结果
    for i, im in enumerate (imlist):
        image = mpimg.imread (result + "/" + im)
        plt.title ("top %d" % (i + 1))
        plt.imshow (image)
        plt.show ()
    
    return 0


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

def rH5FileData2(Key1, key2, filename):
    NameList = []
    featsArrayList = []
    try:
        with h5py.File (filename, 'r') as h5f:
            feats = h5f[Key1][:]
            imgNames = h5f[key2][:]
            for i in imgNames:
                NameList.append (i.decode ("utf-8"))
            featsArrayList = np.array (feats)
            return featsArrayList, NameList
    except KeyError:
        print ("Read HDF5 File Key Error")
        return 1


# 读取索引图像的特征向量和相应的图像名称
def readFeature(h5filename):
    featsList = []
    nameList = []
    for i in range (showHDF5Len(h5filename)):
        feats, imgNames = rH5FileData (i, h5filename)
        featsList.append (feats)
        nameList.append (imgNames)
    featsArrayList = np.array (featsList)
    return featsArrayList,nameList

# 获取图像文件名字,有文件类别后缀的。
def getImageName(imagefile):
    _temp = imagefile.split("/")[-1]
    return _temp.split("\\")[-1]

# 获取图像名字,无文件类别后缀的。
def getImageName2(imagefile):
    return imagefile.split(".")[0]

# 获取图像信息文件名
def getImageTxtName(imagefile):
    return imagefile+".txt"

# 获取图像信息
def getImageInfo(image,imagePath):
    _imgInfoList = []
    _imgList = []
    def _getImageInfoList(imgList):
        for i in imgList:
            _image = getImageName (i)
            _imageName = getImageName2 (_image)
            _imageInfoFile = getImageTxtName (_imageName)
            _imageInfoFile = imagePath + "/" + _imageInfoFile
            with open (_imageInfoFile, 'r', encoding='utf-8') as f:
                imageInfo = f.readline ().strip ("\n")
            _imgInfoList.append(imageInfo)
        return _imgInfoList
    
    
    if type(image) != list:
        _imgList.append(image)
        _imgInfoList = _getImageInfoList(_imgList)
        return _imgInfoList[0]
    else:
        _imgInfoList = _getImageInfoList (image)
        return _imgInfoList
    

# 图像特征检索，计算匹配得分
def featureSearch(queryImage,feats):
    # 提取查询图像的特征，计算 simlarity 评分和排序
    queryVec = extract_feat (queryImage)
    scores = np.dot (queryVec, feats.T)  # 计算点积（内积）,计算图像得分
    # 矩阵乘法并把（纵列）向量当作n×1 矩阵，点积还可以写为：a·b=a^T*b。
    # 点积越大，说明向量夹角越小。点积等于1，则向量为同向，向量夹角0度。
    rank_ID = np.argsort (scores)[::-1]  # 排序,倒序，大到小
    rank_score = scores[rank_ID]  # 计算评分
    # print("scores",scores,type(scores))
    # print ("rank_ID",rank_ID,type(rank_ID))
    # print ("rank_score",rank_score,type(rank_score))
    return rank_ID,rank_score


# 检索显示的相似度最高的图像数目
def getSearchResult(maxres, imgNames, rank_ID, rank_score):
    _imList = []
    _scoresList = []
    
    # 生成图片列表
    for i, index in enumerate (rank_ID[0:maxres]):
        _temp = imgNames[index]
        _imList.append (_temp)
    
    # 生成搜索得分列表
    for j in rank_score[0:maxres]:
        _temp = float ("%.2f" % (j * 100))
        _scoresList.append (_temp)
    
    return _imList, _scoresList

# 测试测试集
def testSetTest(testSet,imageinfopath,feats,imgNames):
    resultnum = 2
    imageList = getImageList (testSet)
    probability = 00.00
    num = len (imageList)
    errorNum = 0
    
    # testErrorList = []
    # trainErrorList = []
    # testImgErrorList = []
    # trainImgErrorList = []
    
    for i in imageList:
        rank_ID, rank_score = featureSearch (i, feats)  # 获取图像得分信息
        imList, scoresList = getSearchResult (resultnum, imgNames, rank_ID, rank_score)
        _imageInfo1 = getImageInfo (i, imageinfopath)  # 获取图片信息
        _imageInfo2 = getImageInfo (imList[0], imageinfopath)
        if _imageInfo1 != _imageInfo2:
            errorNum += 1
            # testErrorList.append (_imageInfo1)
            # trainErrorList.append (_imageInfo2)
            # testImgErrorList.append (i)
            # trainImgErrorList.append (imList[0])
            print ("测试集错误的图片信息： ", _imageInfo1)
            print ("错误的图片最匹配的图片信息：", _imageInfo2)
            print ("测试集错误的图片： ", i)
            print ("错误的图片最匹配的图片：", imList[0])
            print ("相似度评分：", scoresList)
    
    probability = errorNum / num
    print ("错误率: %.2f%%" % (probability*100))
    # print ("测试集错误的图片信息： ", testErrorList)
    # print ("错误的图片最匹配的图片信息：", trainErrorList)
    # print ("测试集错误的图片： ", testImgErrorList)
    # print ("错误的图片最匹配的图片：", trainImgErrorList)
    return 0

# 显示搜索结果
def showSearchResult(resultnum,queryImage,ModelFile,imageinfopath):
    start2 = time.time ()
    outputInfo = ""
    feats, imgNames = rH5FileData2 ("feature", "imagename", ModelFile)
    rank_ID, rank_score = featureSearch(queryImage,feats)
    imList, scoresList = getSearchResult(resultnum,imgNames,rank_ID,rank_score)
    imgInfoList = getImageInfo(imList,imageinfopath)
    _imageInfo = getImageInfo (queryImage, imageinfopath)  # 获取图片信息
    # if _imageInfo == imgInfoList[0]:
    #     outputInfo = "完美匹配！"
    # elif _imageInfo == imgInfoList[1]:
    #     outputInfo = "次要匹配！"
    # else:
    #     outputInfo = "匹配失败！"
        
    
    print ("原图为  :", queryImage)
    print ("原图信息:",_imageInfo)
    print ("最高相似度的%d张图片为: " % resultnum, imList)
    print ("最高%d张图片的评分：" % resultnum, scoresList)
    print ("图片信息为: ", imgInfoList)
    # print ("搜索结果 ：",outputInfo)
    print ("本次查询搜索总耗时(秒)：%.2f s" % (time.time () - start2))
    
    # showimage(queryImage,imList,result)
    return 0

# @profile (precision=6)
def main():

    # feats, imgNames = readFeature (Model)
    

    showSearchResult (3, queryImage, Model, imageinfopath)
    
    
    # testSetTest (testSet, imageinfopath, feats, imgNames)
    
    

if __name__ == '__main__':
    
    # 相关参数
    result = "D:/datasets/trainset"
    imageinfopath = "D:/datasets/imageinfo"
    Model = "./model/imageCNNAll.h5"
    # Model = "./model/imageCNN6442.h5"
    # queryImage = "D:/datasets/trainingset1/19700102130430799.JPEG"
    # queryImage = "D:/datasets/001/19700102130428480.JPEG"
    testSet = "D:/datasets/testingset"
    trainSet = "D:/datasets/trainset"

    # imgList = getImageList(testSet)
    #
    # imgInfoList = getImageInfo(imgList,imageinfopath)
    # print(imgInfoList[0])

    # 统计图片信息类
    # infoFileList = getInfoFileList(imageinfopath)
    # infoFileInfo = []
    # for i in infoFileList:
    #     with open (i, 'r', encoding='utf-8') as f:
    #         imageInfo = f.readline ().strip ("\n")
    #         infoFileInfo.append(imageInfo)
    # infoFileInfo = list (set (infoFileInfo))
    #
    # for j in range(len(infoFileInfo)):
    #     with open("./imageinfoclass.txt",'a',encoding="utf-8") as ff:
    #         ff.write(infoFileInfo[j]+"\n")

    # 错误的图片
    # imgList = []
    # with open ("./image2info.txt", 'r', encoding='utf-8') as f:
    #     image = f.readlines ()
    #
    # for i in image[::2]:
    #     with open ("./image_search_error.txt", 'a', encoding='utf-8') as ff:
    #         ff.write(i)

    # 搜索错误列表中的图片
    # with open ("./image_search_error.txt", 'r', encoding='utf-8') as f:
    #     errorImage = f.readlines ()
    #
    # for i in errorImage[:10]:
    #     queryImage = testSet + "/" + i.strip("\n")
    #     main()
    
    pass
    # from image_search.keras_cnn_imageSearch.base import base
    # b = base ()
    

    queryImage = "D:/datasets/testingset/20150630163114680.JPEG"
    # queryImage = "./19700102125648863.JPEG"

    main()
    
    
    
    
    
    


