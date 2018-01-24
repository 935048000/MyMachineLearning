# Image Retrieval Engine Based on Keras

## 环境

```python
In [1]: import keras
Using Theano backend.
```

keras 2.0.1 及 2.0.5 版本均经过测试可用。
推荐Python 3.x

此外需要numpy, matplotlib, os, h5py, argparse. 推荐使用anaconda安装

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
