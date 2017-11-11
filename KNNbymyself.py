# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 23:39:29 2017
python 3.5
@author: zyy
"""

"""
KNN算法：
原理：李航.统计学习方法:37-45
最近邻搜索方法（查找最近的k个点）：
（1）线性扫描。（也就是依次求全部点的距离取最小值，效率低，一般不采用这种方法）
     这里有一个python实现，可以帮助理解原理：http://blog.csdn.net/mrlevo520/article/details/52425121
（2）KDtree。
     kdtree相关内容同可参考：李航.统计学习方法:37-45
     kdtree的python实现:http://blog.csdn.net/lipengcn/article/details/50938184
     kdtree进阶：kdtree with BBF：http://blog.sina.com.cn/s/blog_6f611c300101bysf.html                 
     kdtree搜索的平均计算复杂度O(logN)：李航.统计学习方法:37-45
     适用范围：训练实例数远大于空间维数时的kNN算法
（3）。。。

算法实现：
（1）根据原理自编写（参考了 kdtree的python实现）：见下文
（2）使用库函数：http://www.cnblogs.com/90zeng/p/python_knn.html
 (3) knn算法的时间空间复杂度：http://blog.csdn.net/lsldd/article/details/41357931

下面算法没有用到迭代器、字典等python风格的东西，有空再更新，改写一下
"""

import numpy as np
def testData():
    testData = np.array([[2,3],[7,4],[9,6],[2,7],[8,1],[7,2]])
    labels = np.array(["A","A","B","B","C","C"])
    return testData,labels

    
class KD_node(object):
    def _init_ (self,point,split,left,right):
        '''
        point:节点，或者说数据点
        split:分裂维度
        left，right:节点的左子空间和右子空间
        left，right均为kd_node类
        '''
        self.point = point
        self.split = split
        self.left = left
        self.right = right
        
#如何选择分裂维度？    
#选择哪一个维度来进行划分，可以计算所有数据点在每个维度上的方差，方差越大，包含的信息越多
def createKdtree(testData,root=None):
    """
    root:当前树的根节点
    testData:测试集
    return：构造的kdtree树根    
    """
    LEN = len(testData) 
    if LEN == 0:
        return
    #所有数据点在每个维度上的方差
    ColVar = np.var(testData,axis=0)
    #确定分裂维度:选择方差最大的维度     
    split = np.where(ColVar == max(ColVar))[0][0]
    #选取切分点point:对裂维度上坐标数据进行排序,选取中位数作为切分点
    sortedData = sorted(testData, key=lambda t: t[split]) 
    point = sortedData[LEN//2] 
    root = KD_node()
    root.point = point
    root.split = split
    left = createKdtree(sortedData[0:(LEN//2)])
    right = createKdtree(sortedData[((LEN//2)+1):LEN])   
    root.left = left
    root.right = right
    return root


def draw_kdtree(root): 
    tree = [root]
    while root:
        root_left = root.left
        root_right = root.right
        
    tree=[]
    tree.append(root)
    root_left = root.left
    root_right = root.right
    tree.append([root_left,root_right])    
    print(root.point)
    return tree
    
def FindNN(root,query):
    """    
    root:kdtree的树根
    query：查询点
    return：返回距离data最近的点NN，同时返回最短距离min_distance
    """
    #初始化为root的节点
    NN = root.point
    min_dist = computeDist(query,NN)
    nodeList = [] #nodeList存放的是query点所在的枝
    temp_root = root #temp_root：查找离query最近的叶节点
    ##二分查找建立路径,在kd树找出包含目标点x的叶节点
    while temp_root:
        nodeList.append(temp_root)
        dd = computeDist(query,temp_root.point)
        if min_dist > dd: #统计学习方法 算法3.3（3）a
            NN = temp_root.point
            min_dist = dd
        #当前节点的划分域
        sp = temp_root.split
        if query[sp] <= temp_root.point[sp]:
            temp_root = temp_root.left
        else:
            temp_root = temp_root.right
    #print(nodeList)
    #回溯查找
    while nodeList:
        #使用list模拟栈，后进先出
        back_point = nodeList.pop()
        sp = back_point.split
        #print("back.point = ",back_point.point)
        ##判断是否需要进入父亲节点的子空间进行搜索
        if abs(query[sp] - back_point.point[sp]) < min_dist:
            if query[sp] <= back_point.point[sp]:
                temp_root = back_point.right
            else:
                temp_root = back_point.left
                
            if temp_root != None:
                nodeList.append(temp_root)
                curDist = computeDist(query,temp_root.point)
                if min_dist > curDist:
                    min_dist = curDist
                    NN = temp_root.point
    return NN,min_dist


def computeDist(pt1,pt2,p=2):
    """
    pt1,pt2为两个点
    p:Lp距离中的参数p，p=2时为欧式距离
    p=n(表示无穷)时则为曼哈顿距离
    return:pt1和pt2之间的距离
    """
    #下面计算Lp距离
    d = abs(pt1-pt2)
    if p != 'n':        
        Lp = sum(d ** p) ** (1/p)
    else:
        Lp = max(d)        
    return Lp

def Xlabel(point,testData,labels,k=1,weight = False):
    #加权的投票法还未完成
    root = createKdtree(testData)
    kpoint = [] #存放k个最近邻
    dist = [] #这k个点的距离
    klabels = []
    for i in range(k):
        root = createKdtree(testData)
        NN,min_dist= FindNN(root,point)
        kpoint.append(NN.tolist())
        testData = testData.tolist()
        index = testData.index(NN.tolist())
        klabels.append(labels[index])
        dist.append(min_dist)
        testData.pop(index)
        testData = np.array(testData)
    from collections import Counter
    if weight == False:
        Count = Counter(klabels).most_common(1)
        point_label = Count[0][0]
    else:
        w = 1/np.array(dist)
        w = w.tolist()
    return point_label
    
def KNN(dataset,pointlabels,ratio=0.3,k=1,weight = False):
    """
    dataset:所有的数据
    labels:为训练集
    label:为训练集的类别
    k:选择相邻的k个点，默认为1，为最近邻
    weight:是否适用距离进行加权，默认False
    """    
    rows = len(dataset)
    l = range(0,rows)
    import random
    samplenum = random.sample(l,round(rows * ratio)) #随机抽样选出供预测的值
    Points = dataset[samplenum,]
    pointLabels = pointlabels[samplenum,]
    testnum = list(set(l).difference(set(samplenum)))
    testData = dataset[testnum,]
    labels = pointlabels[testnum,]

    predictLabels = []
    for i in range(len(Points)):
        label = Xlabel(Points[i],testData,labels,k=1,weight = False)
        predictLabels.append(label)
    predictLabels = np.array(predictLabels)
    errNum = np.where(predictLabels != pointLabels)[0]
    errRatio = len(errNum)/rows
    errPoint = Points[errNum,]
    return predictLabels,pointLabels,errRatio,errPoint


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    iris = load_iris()
    irisdata = iris.data
    labels = iris.target
    dataset = irisdata               
    pointlabels = labels
    predictLabels,pointLabels,errRatio,errPoint = KNN(dataset,pointlabels,ratio=0.3,k=3,weight = False)
    print(errRatio,'\n',errPoint)
   

