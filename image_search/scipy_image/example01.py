import cv2
import scipy as sp
from matplotlib import pyplot as plt
import numpy as np

# 应用比值判别法
def imageSearch(matches):
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append (m)
    # print('good', len (good))
    return (len (good))

def imagess(image1,image2,image3):
    img1 = cv2.imread (image1,0)
    img2 = cv2.imread (image2, 0)
    img3 = cv2.imread (image3, 0)

    # 启动 SIFT
    sift = cv2.xfeatures2d.SIFT_create ()

    DES = []
    # 使用SIFT找到关键点和描述符
    kp1, des1 = sift.detectAndCompute (img1, None)
    # kp2, des2 = sift.detectAndCompute (img2, None)
    # DES.append(des2)
    # kp3, des3 = sift.detectAndCompute (img3, None)
    # DES.append(des3)


    # FLANN 特征
    FLANN_INDEX_KDTREE = 0
    index_params = dict (algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict (checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher (index_params, search_params)



    # for i in DES:
        # matches = flann.knnMatch (des1, i, k=2)
        # print(imageSearch(matches))
        # np.savetxt ("./test.txt", i)
        # print(np.loadtxt ("./test.txt"))

    # 矩阵读写文件
    # np.savetxt ("./test.txt", des2)
    des4 = np.loadtxt ("./test.txt", dtype=float)

    # float64 --》 float32
    des4 = np.float32(des4)

    # 矩阵比较
    # if (des4 == des2).all():
    #     print("des4 == des2")
    #     print (des2.dtype, des4.dtype)
    # else:
    #     print("des4 ！= des2")
    #     print (des2.dtype, des4.dtype)
    #


    matches = flann.knnMatch (des1, des4, k=2)
    print ("匹配度： ",imageSearch (matches))


    # 特征点总数
    # print('特征点总数:', len (matches))
    return 0


    # 可视化
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

if __name__ == '__main__':
    image1 = './test01/5.JPEG'
    image2 = './test01/6.JPEG'
    image3 = './test01/1.JPG'

    imagess(image1,image2,image3)

