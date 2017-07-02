# coding: utf-8
# xionglidong
# ubuntu16
# jupyter
# python3.5
# 2017-7-2


import random
import numpy as np
from numpy import genfromtxt


def getData(dataset): 
    m,n=np.shape(dataset)
    traindata=np.ones((m,n-1))
    traindata[:,:]=dataset[:,:-1]
    trainlabel=dataset[:,-1]
    return traindata,trainlabel


def stachasticGradientDescent(traindata,trainlabel,alph,theta,maxiteration):
    m,n=np.shape(traindata)
    for index in range(0,maxiteration):
        i=random.randint(0,m-1)
        hypothesis=theta[0]*traindata[i][0]+theta[1]*traindata[i][1]
        loss_fuction=trainlabel[i]-hypothesis
        gradient_1=loss_fuction*traindata[i][0]
        gradient_2=loss_fuction*traindata[i][1]
        theta[0]=theta[0]+alph*gradient_1
        theta[1]=theta[1]+alph*gradient_2
      
    print("theta={}".format(theta))
    return theta


def predict(x,theta):
    m,n=np.shape(x)
    xtest=np.ones((m,n))
    xtest[:,:]=x
    ypre=np.dot(xtest,theta)
    return ypre       
       
    
datapath=r"/home/xiong/gradientdesent.csv" 
dataset=genfromtxt(datapath,delimiter=',')
traindata,trainlabel=getData(dataset)
alph=0.005
maxiteration=300
theta=np.array([2,5]).astype(np.float64)
theta=stachasticGradientDescent(traindata,trainlabel,alph,theta,maxiteration)
x=np.array([[1.05,4],[2.1,5],[5,1],[4,2]])
print(predict(x,theta))


