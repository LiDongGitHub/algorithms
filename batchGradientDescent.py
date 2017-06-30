# coding: utf-8
# xionglidon
# ubuntu16
# jupyter
# python=3.5
# 2017-6-28

import numpy as np
from numpy import genfromtxt

datapath=r"/home/xiong/gradientdesent.csv" #数据地址
dataset=genfromtxt(datapath,delimiter=',') #导入数据

def getData(dataset): #得到数据
    m,n=np.shape(dataset) #m,n分别是导入数据行和列
    traindata=np.ones((m,n)) #建立一个m行，n列全为1的矩阵
    traindata[:,:-1]=dataset[:,:-1] #切片，去掉原来数据最后一行，赋值给traindata
    trainlabel=dataset[:,-1] #dataset最后一列赋值给tainlabel
    return traindata,trainlabel

def batchGradientDescent(x,y,theta,alpha,m,maxIterations): #批处理梯度算法
    xTrains=x.transpose() #矩阵转置 
    for i in range(0,maxIterations):
        hypothesis=np.dot(x,theta)
        loss=hypothesis-y
        gradient=np.dot(xTrains,loss)/m
        theta=theta-alpha*gradient 
    return theta

def predict(x,theta):
    m,n=np.shape(x)
    xtest=np.ones((m,n+1))
    xtest[:,:-1]=x
    ypre=np.dot(xtest,theta)
    return ypre

traindata,trainlabel=getData(dataset)
m,n=np.shape(traindata)
theta=np.ones(n)
alpha=0.01
maxIteration=500000
theta=batchGradientDescent(traindata,trainlabel,theta,alpha,m,maxIteration)
x=np.array([[3.1,5.5],[3.3,5.9],[3.5,6.3],[3.7,6.7],[3.9,7.1]])
print(predict(x,theta))
    


