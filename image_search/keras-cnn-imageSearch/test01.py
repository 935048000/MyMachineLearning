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

import sys, time
import pyprind
from pyprind import ProgBar

bar = ProgBar(30,monitor=True,title="job01")
# for progress in pyprind.prog_bar(range(100)):
# for progress in pyprind.prog_percent (range (100)):
for i in range(30):

    time.sleep (0.5)
    bar.update()

print(bar)