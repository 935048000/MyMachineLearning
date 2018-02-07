# Image Retrieval Engine Based on Keras

## 环境

```python
In [1]: import keras
Using Theano backend.
```
推荐Python 3.x

此外需要numpy, matplotlib, os, h5py, argparse. 推荐使用anaconda安装

https://github.com/willard-yuan/flask-keras-cnn-image-retrieval


### 使用个人数据集测试的模型性能
模型性能：
batch1 错误率: 5.00%  10/200
batch2 错误率: 1.00%  2/200
batch3 错误率: 4.00%  8/200
batch4 错误率: 4.00%  8/200
batch5 错误率: 6.50%  13/200
batch6 错误率: 6.50%  13/200
batch7 错误率: 5.00%  10/200
batch8 错误率: 9.09%  20/220

错误识别数：76
识别总数：1620
错误率：4.69%
识别速度：3s


### 使用

- 步骤一

`python index.py -database <path-to-dataset> -index <name-for-output-index>`

- 步骤二

`python query_online.py -query <path-to-query-image> -index <path-to-index-flie> -result <path-to-images-for-retrieval>`

```sh
├── database 图像数据集
├── extract_cnn_vgg16_keras.py 使用预训练vgg16模型提取图像特征
|── index.py 对图像集提取特征，建立索引
├── query_online.py 库内搜索
└── README.md
```

#### 示例

```sh
# 对database文件夹内图片进行特征提取，建立索引文件 imageCNN.h5
C:\Users\pop\Desktop\python\MyMachineLearning\image_search\keras_cnn_imageSearch

C:\Python\Python36\python.exe index.py -database imagesets -index imageCNN.h5

# 使用database文件夹内001_accordion_image_0001.jpg作为测试图片，在database内以 imageCNN.h5 进行近似图片查找，并显示最近似的3张图片
python query_online.py -query database/001_accordion_image_0001.jpg -index imageCNN.h5 -result database
C:\Python\Python36\python.exe imageQyery.py -query imagesets/19700102130428480.JPEG -index youdianCNN.h5 -result imagesets

```
