#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import time


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


# In[13]:


# 将数字索引与文本名称进行转换(fashion_mnist)
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# In[14]:


# 可视化样本
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    # 绘制图像列表
    figsize = (num_cols * scale, num_rows * scale)
    _,axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes,imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        
        else:
            ax.imshow(img)
        # 隐藏坐标轴
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


# In[15]:


# 使用4个线程读取数据
def get_dataloader_workers():
    return 4


# In[16]:


# 整合所有组件(获取和读取Fashion-MNIST数据集)
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
        # 相当于 [transforms.Resize(resize),transforms.ToTensor()]
    # 将变换组合在一起
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root = "../data", train = True, transform = trans, download = True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root = "../data" , train = False, transform = trans, download = True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, 
                             num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False, 
                             num_workers=get_dataloader_workers()))


# In[ ]:




