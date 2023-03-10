#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt
import torch
import time
from torch.utils import data


# In[2]:


# 使用svg格式在jupyter中显示绘图
def use_svg_display():
    backend_inline.set_matplotlib_formats('svg')


# In[3]:


# 设置图表大小
def set_figsize(figsize=(3.5,2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


# In[4]:


# 生成图表的轴属性 
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


# In[5]:


# 绘图
"""
可针对一个x，多个y同时画图
使用格式:plot(x,[y1,y2],[xlabel=,ylabel=,legend=[,……,],xlim=,ylim=,……])
"""
def plot(X,Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,
        xscale='linear', yscale='linear', 
        fmts=('-','m--','g-.','r:'),figsize=(3.5,2.5),axes=None):
    if legend is None:
        legend=[]
    
    set_figsize(figsize)
    axes = axes if axes else plt.gca()
    # 若X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X,"ndim") and X.ndim == 1 or isinstance(X,list)
               and not hasattr(X[0],"__len__"))
    
    if has_one_axis(X):
        X = [X]
    if Y is None:
        X,Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes,xlabel,ylabel,xlim,ylim,xscale,yscale,legend)


# In[6]:


# 定义计时器 
class Timer:
    # 记录多次运行时间
    def __init__(self):
        self.times=[]
        self.start()
    
    def start(self):
        # 启动计时器
        self.tik = time.time()
    
    def stop(self):
        # 停止计时器并将时间记录在列表
        self.times.append(time.time()-self.tik)
        return self.times[-1]
    
    def avg(self):
        # 返回平均时间
        return sum(self.times) / len(self.times)
    
    def sum(self):
        # 返回时间总和 
        return sum(self.times)
    
    def cumsum(self):
        # 返回累计时间
        return np.array(self.times).cumsum().tolist()


# In[7]:


# 生成具有噪声的数据集 (x~(0,1), y增加的噪声e~(0,0.01))
def synthetic_data(w,b,num_examples):
    x = torch.normal(0,1,(num_examples,len(w)))
    y = torch.matmul(x,w) + b
    # 增加噪声
    y += torch.normal(0,0.01,y.shape)
    return x, y.reshape((-1,1))


# In[8]:


# 线性回归模型
def linreg(x,w,b):
   return torch.matmul(x,w)+b


# In[9]:


# 均方误差
def squared_loss(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2


# In[10]:


# 小批量随机梯度下降
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size # 本地修改，无需return
            param.grad.zero_()


# In[11]:


# Huber_loss
def Huber(y_hat,y,delta):
    loss = torch.where(abs(y_hat - y) > delta, abs(y_hat - y) - delta / 2, (y_hat - y)**2 / (2 * delta)) 
    return torch.sum(loss)


# In[12]:


# 读取数据集 (is_train ——> 是否打乱顺序)
"""
传入（x,y）和batch_size,封装成Pytorch数据迭代器
"""
def load_array(data_arrays, batch_size, is_train=True):
    # 构造Pytorch数据迭代器
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# In[ ]:




