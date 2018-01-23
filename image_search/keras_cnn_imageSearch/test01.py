# from memory_profiler import profile
#
#
# @profile (precision=6)
# def primes(n):
#     I = 0
#     J = []
#     for i in range(n):
#         I += i
#         J.append(i)
#     return J
#
#
# primes (100000)

# from psutil import virtual_memory
# import time
#
# mem = virtual_memory ()
# start = time.time()
# print ("任务01 Used Time:%s s，Mem Used：%dM  %.2f%% " % ((time.time () - start), mem.used / 1024 / 1024, mem.percent))

# import sys, time
# import pyprind
#
# from pyprind import ProgBar
# import pyprind
# import time
# bar = pyprind.ProgBar(30,monitor=True,title="job01",bar_char="=")
# for i in range(30):
#     time.sleep (1)
#     bar.update()
# print(bar)


# import pyprind
# import time
# for progress in pyprind.prog_bar(range(20)):
#     time.sleep (1)

# import pyprind
# import time
# bar = pyprind.prog_percent(30,monitor=True,title="job01")
# for progress in pyprind.prog_percent (range (20)):
# for i in range(30):
#     time.sleep (1)
#     bar.update()
# print(bar)


queryImage = "D:/datasets/002/19700102132424765.JPEG"
queryImage2 = "19700102132424765.JPEG"
# print(queryImage.split("/")[-1])

def getImageName(imagefile):
    return imagefile.split("/")[-1]

# print(getImageName(queryImage2))
# print(queryImage2.split(".")[0])

# a = [[2,"啊"],[1,"吧"],[6,"才"],[3,"的"]]
a = ["a","b"]
import operator
# a.sort(key=operator.itemgetter(1))
# print(a.index("a"))

# print(100%50)

import cv2
dirfile = 'D:/datasets/trainset/20150630183252485.JPEG'
im = "D:/datasets/trainset/20150716154609131.JPEG"
# img = cv2.imread(im)
# size = img.shape
# print (size)
# cv2.imshow(img)
# # cv2.imshow()


# image=cv2.imread(dirfile)
# res=cv2.resize(image,(272,480),interpolation=cv2.INTER_AREA)
# cv2.imshow('iker',res)
# cv2.imshow('image',image)
# cv2.waitKey(0)
# cv2.imwrite('12345.JPEG',res)
# cv2.destroyWindow()


