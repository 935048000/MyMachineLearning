import cv2
import scipy as sp
from matplotlib import pyplot as plt
import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import pdist
from time import time

# 应用比值判别法
def imageSearch(matches):
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append (m)
    # print('good', len (good))
    return (len (good)/len (matches))

# 应用比值判别法
def imageSearch2(avg1,avg2):
    good = []
    for m, n in zip(avg1,avg2):
        if m < 0.75 * n:
            good.append (m)
    # print('good', len (good))
    return (len (good)/len (avg1))

# 归一化
def normalization(vec):
    return vec / LA.norm (vec)

# 计算余弦距离
def Cosine_distance(hash1,hash2):
    dist = pdist (np.vstack ([hash1, hash2]), 'cosine')
    return dist

# KNN匹配
def knn(des1,des2):
    # FLANN 特征
    FLANN_INDEX_KDTREE = 0
    index_params = dict (algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict (checks=10)
    flann = cv2.FlannBasedMatcher (index_params, search_params)

    bf = cv2.BFMatcher (cv2.NORM_HAMMING)

    # K-近邻算法 匹配
    matches = flann.knnMatch (des1, des2, k=2)
    # print ("匹配度1： ",imageSearch (matches))
    return imageSearch (matches)

def imagess(image1,image2):
    img1 = cv2.imread (image1,0)
    img2 = cv2.imread (image2, 0)

    t = time.time ()
    # 启动 SIFT
    sift = cv2.xfeatures2d.SIFT_create ()

    
    DES = []
    # 使用SIFT找到关键点和描述符
    kp1, des1 = sift.detectAndCompute (img1, None)
    print ("特征提取耗时：", time.time () - t)
    
    kp2, des2 = sift.detectAndCompute (img2, None)

    vec = []
    vec.append(des1)
    vec.append(des2)
    
    d1 = []
    d2 = []
    d1.append(des1)
    d2.append(des2)

    # np.save ('arr1.npy', vec)
    # vec2 = np.load ('arr1.npy')

    t = time.time ()
    cd = Cosine_distance(des2[0],des1[0])
    print ("余弦耗时：", time.time () - t)
    print("余弦:",cd)
    
    # t2 = time.time ()
    # print("比值判别：",imageSearch2(des2[0],des1[0]))
    # print ("比值判别时间：", time.time () - t2)


    return 0


    # 可视化

# 显示图像
def showImage(img1,img2,kp1,kp2,good):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    view = sp.zeros ((max (h1, h2), w1 + w2, 3), sp.uint8)
    view[:h1, :w1, 0] = img1
    view[:h2, w1:, 0] = img2
    view[:, :, 1] = view[:, :, 0]
    view[:, :, 2] = view[:, :, 0]

    for m in good:
        # 画出要点
        # print m.queryIdx, m.trainIdx, m.distance
        color = tuple ([sp.random.randint (0, 255) for _ in range (3)])
        # print 'kp1,kp2',kp1,kp2
        cv2.line (view, (int (kp1[m.queryIdx].pt[0]), int (kp1[m.queryIdx].pt[1])),
                  (int (kp2[m.trainIdx].pt[0] + w1), int (kp2[m.trainIdx].pt[1])), color)

    cv2.imshow ("view", view)
    cv2.waitKey ()

# 获取特征值
def getFeatus(img):
    _img = cv2.imread (img, 0)
    sift = cv2.xfeatures2d.SIFT_create ()
    kp, des = sift.detectAndCompute (_img, None)
    return des
    
    

if __name__ == '__main__':
    # image1 = './test01/5.JPEG'
    # image2 = './test01/6.JPEG'
    # image3 = './test01/1.JPG'
    testingset = "H:/datasets/testingset/"
    trainingset = "H:/datasets/trainset/"
    #
    #
    # t = time.time()
    # imagess ("H:/datasets/testingset/20150710161158727.JPEG", "H:/datasets/testingset/20150710161156773.JPEG")
    # print("总时间：",time.time() - t)
    # imagess ("H:/datasets/testingset/20150710161156773.JPEG", "H:/datasets/testingset/20150710150006199.JPEG")
    # imagess ("H:/datasets/testingset/20150710161158727.JPEG", "H:/datasets/testingset/20150710150006199.JPEG")

    