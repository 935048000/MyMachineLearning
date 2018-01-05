from numpy import *
import matplotlib.pyplot as plt
import time


# 预测 Sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + exp (-inX))


# 使用一些可选的优化算法来训练逻辑回归模型
# 输入: train_x是一个mat数据类型，每一行代表一个样本
#       train_y也是mat数据类型，每行都是对应的标签
#       opts是优化选项，包括步骤和最大迭代次数
def trainLogRegres(train_x, train_y, opts):
    # 计算训练时间
    startTime = time.time ()

    numSamples, numFeatures = shape (train_x)
    alpha = opts['alpha']
    maxIter = opts['maxIter']
    weights = ones ((numFeatures, 1))

    # 通过梯度下降algorilthm优化
    for k in range (maxIter):
        if opts['optimizeType'] == 'gradDescent':  # 梯度下降法algorilthm
            output = sigmoid (train_x * weights)
            error = train_y - output
            weights = weights + alpha * train_x.transpose () * error
        elif opts['optimizeType'] == 'stocGradDescent':  # 随机梯度下降
            for i in range (numSamples):
                output = sigmoid (train_x[i, :] * weights)
                error = train_y[i, 0] - output
                weights = weights + alpha * train_x[i, :].transpose () * error
        elif opts['optimizeType'] == 'smoothStocGradDescent':  # 平稳随机梯度下降
            # 随机选取样本进行优化以减少周期波动
            dataIndex = list(range (numSamples))
            for i in range (numSamples):
                alpha = 4.0 / (1.0 + k + i) + 0.01
                randIndex = int (random.uniform (0, len (dataIndex)))
                output = sigmoid (train_x[randIndex, :] * weights)
                error = train_y[randIndex, 0] - output
                weights = weights + alpha * train_x[randIndex, :].transpose () * error
                del (dataIndex[randIndex])  # 在一次交互中，删除优化的示例
        else:
            raise NameError ('不支持优化方法类型!')

    print('祝贺你,培训完成了!花了 %fs!' % (time.time () - startTime))
    return weights


# 测试您训练过的逻辑回归模型
def testLogRegres(weights, test_x, test_y):
    numSamples, numFeatures = shape (test_x)
    matchCount = 0
    for i in range (numSamples):
        predict = sigmoid (test_x[i, :] * weights)[0, 0] > 0.5
        if predict == bool (test_y[i, 0]):
            matchCount += 1
    accuracy = float (matchCount) / numSamples
    return accuracy


# 展示你训练过的逻辑回归模型，只提供二维数据
def showLogRegres(weights, train_x, train_y):
    # 注意:train_x和train_y是mat数据类型
    numSamples, numFeatures = shape (train_x)
    if numFeatures != 3:
        print("对不起!我不能画因为你的数据的尺寸不是2 !")
        return 1

        # 画出所有样品
    for i in range (numSamples):
        if int (train_y[i, 0]) == 0:
            plt.plot (train_x[i, 1], train_x[i, 2], 'or')
        elif int (train_y[i, 0]) == 1:
            plt.plot (train_x[i, 1], train_x[i, 2], 'ob')

            # 画线分类
    min_x = min (train_x[:, 1])[0, 0]
    max_x = max (train_x[:, 1])[0, 0]
    # weights = weights.getA ()  # 转换为数组
    y_min_x = float (-weights[0] - weights[1] * min_x) / weights[2]
    y_max_x = float (-weights[0] - weights[1] * max_x) / weights[2]
    plt.plot ([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel ('X1')
    plt.ylabel ('X2')
    plt.show ()


# from numpy import *
# import matplotlib.pyplot as plt
# import time


def loadData():
    train_x = []
    train_y = []
    fileIn = open ('./testSet.txt')
    for line in fileIn.readlines ():
        lineArr = line.strip ().split ()
        train_x.append ([1.0, float (lineArr[0]), float (lineArr[1])])
        train_y.append (float (lineArr[2]))
    return mat (train_x), mat (train_y).transpose ()


## 步骤1:加载数据
print("步骤1:加载数据…")

train_x, train_y = loadData ()
test_x = train_x
test_y = train_y

## 步骤2:培训
print("步骤2:培训……")
opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'smoothStocGradDescent'}
optimalWeights = trainLogRegres (train_x, train_y, opts)

## 步骤3:测试
print("步骤3:测试……")
accuracy = testLogRegres (optimalWeights, test_x, test_y)

## 步骤4:显示结果
print("第四步:展示结果…")

print('分类精度是: %.3f%%' % (accuracy * 100))
showLogRegres (optimalWeights, train_x, train_y)







