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
# # from pyprind import ProgBar
# import pyprind
# import time
# bar = pyprind.ProgBar(30,monitor=True,title="job01",bar_char="-")
# for i in range(30):
#     time.sleep (0.5)
#     bar.update()
# print(bar)


# import pyprind
# import time
# for progress in pyprind.prog_bar(range(20)):
#     time.sleep (1)

# import pyprind
# import time
# for progress in pyprind.prog_percent (range (20)):
#     time.sleep (1)


queryImage = "D:/datasets/002/19700102132424765.JPEG"
print(queryImage.split("/")[-1])