# coding: utf-8
# xionglidong
# ubuntu16
# jupyter
# python3.5
# 2017-6-28


import numpy as np
from numpy import genfromtxt


def getData(dataset): #取得数据
    m,n=np.shape(dataset) #m,n分别是导入数据行和列
    traindata=np.ones((m,n-1)) #建立一个m行，n-1列全为1的矩阵
    traindata[:,:]=dataset[:,:-1] #切片，去掉原来数据最后一行，赋值给traindata 
    trainlabel=dataset[:,-1] #dataset最后一列赋值给tainlabel
    return traindata,trainlabel #返回所需数据


def batchGradientDescent(x,y,theta,alpha,m,maxIteration): #批处理梯度下降算法
    xTrains=x.transpose() #转置
    for i in range(0,maxIteration): #循环迭代maxIteration
        hypothesis=np.dot(x,theta) #假设函数
        loss=hypothesis-y #损失函数
        gradient=np.dot(xTrains,loss)/m #对损失函数求梯度
        theta=theta-alpha*gradient #得到theta
        
    print("theta={}".format(theta))        
    return theta


def predict(x,theta): #对得到的theta值进行测验
    m,n=np.shape(x)
    xtest=np.ones((m,n))
    xtest[:,:]=x
    ypre=np.dot(xtest,theta)
    return ypre


datapath=r"/home/xiong/gradientdesent.csv" #数据路径
dataset=genfromtxt(datapath,delimiter=',') #导入
traindata,trainlabel=getData(dataset) 
m,n=np.shape(dataset)
alpha=0.005 #更新速率
maxIteration=300 #迭代次数
theta=np.array([2,5]).astype(np.float64) #权重theta值
theta=batchGradientDescent(traindata,trainlabel,theta,alpha,m,maxIteration)
x=np.array([[1.05,4],[2.1,5],[5,1],[4,2]]).astype(np.float64) #测验数据集
print(predict(x,theta)) #得出结果



