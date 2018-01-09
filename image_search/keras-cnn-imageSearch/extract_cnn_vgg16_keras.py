# -*- coding: utf-8 -*-
# Author: yongyuan.name

import numpy as np
from numpy import linalg as LA

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


'''
VGG16模型,权重由ImageNet训练而来
使用vgg16模型提取特征
输出归一化特征向量
'''
def extract_feat(img_path):
    # weights: None代表随机初始化，即不加载预训练权重。'imagenet'代表加载预训练权重
    # pooling: pooling：当include_top=False时，该参数指定了池化方式。None代表不池化，最后一个卷积层的输出为4D张量。
    #                   ‘avg’代表全局平均池化，‘max’代表全局最大值池化。
    # input_shape: (width, height, 3)
    #               仅当include_top = False有效，应为长为3的tuple，指明输入图片的shape，
    #               图片的宽高必须大于48，如 (200, 200, 3)
    # include_top：是否保留顶层的3个全连接网络
    
    input_shape = (272, 480, 3)
    model = VGG16(weights = 'imagenet', input_shape = (input_shape[0],input_shape[1],input_shape[2]), pooling = 'max', include_top = False)
        
    img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    feat = model.predict(img)
    norm_feat = feat[0]/LA.norm(feat[0])
    return norm_feat


if __name__ == '__main__':
    print("local run .....")


    # model = VGG16 (weights='imagenet', pooling = 'max', include_top=False)
    # img_path = './database/001_accordion_image_0001.jpg'
    # img = image.load_img (img_path, target_size=(224, 224))
    # x = image.img_to_array (img)
    # x = np.expand_dims (x, axis=0)
    # x = preprocess_input (x)
    # features = model.predict (x)
    # norm_feat = features[0]/LA.norm(features[0])
    # feats = np.array(norm_feat)
    # print(norm_feat.shape)
    # print(feats.shape)

    # norm_feat = extract_feat(img_path)
    # print(norm_feat.shape)

