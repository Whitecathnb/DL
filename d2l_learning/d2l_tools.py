#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt
from IPython import display
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


# In[17]:


# Softmax
def softmax(X):
    X_exp = torch.exp(X)
    # 考虑到 X 为 batch_size 个 向量组成
    partition = X_exp.sum(1,keepdim=True)
    return X_exp / partition # 广播机制


# In[18]:


# 交叉熵
def cross_entropy(y_hat, y): 
    return -torch.log(y_hat[range(len(y_hat)), y])


# In[27]:


def softmax_improve(x):
    x_exp = torch.exp(x - x.max(1,keepdim=True)[0])
    if len(x.shape) > 1:
        partition = x_exp.sum(1,keepdim=True)
    else:
        partition = x_exp.sum()
    return x_exp / partition


# In[28]:


def CrossEntropyLoss(y_hat, y):
    soft_tmp = softmax_improve(y_hat)
    LSE = y_hat - y_hat.max(1,keepdim=True)[0] - torch.log(torch.exp(y_hat - y_hat.max(1,keepdim=True)[0]).sum(1,keepdim=True))
    return -LSE[range(len(LSE)), y]


# In[19]:


#  计算准确数
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 每个样本中选出概率最大的类别,argmax返回最大值的索引
        y_hat = y_hat.argmax(axis=1)
    
    cmp = y_hat.type(y.dtype) == y
    # 返回正确的总数
    return float(cmp.type(y.dtype).sum())


# In[20]:


# 计算指定数据集上的模型精度
def evaluate_accuracy(net, data_iter):    
    if isinstance(net, torch.nn.Module):
        net.eval() # 将模型设置为评估模式
    metric = Accumulator(2) # 正确预测数，预测总数
    with torch.no_grad():
        for x, y in data_iter:
            metric.add(accuracy(net(x), y), y.numel())
    return metric[0] / metric[1]


# In[21]:


# 在多个变量进行累加

"""
使用方法: 

metrics = Accumulator(number)  number --- 为你的变量个数  
metrics.add(……)   
metrics[index] --- 获取该索引下的值

"""
class Accumulator: 
    def __init__(self,n):
        self.data = [0.0] * n
    
    def add(self,*args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)
    
    # 根据索引取值
    def __getitem__(self,idx):
        return self.data[idx]


# In[22]:


"""
updater example:

lr = 0.1

def updater(batch_size):
    return d2l.sgd([w, b], lr, batch_size)

"""
def train_epoch_ch3(net, train_iter, loss, updater):
    # 训练模型
    # 若为torch.nn.Module内置网络，则把模式设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和，训练准确度总和，样本数
    metric = Accumulator(3)
    for x, y in train_iter:
        y_hat = net(x)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用自定义的优化器和损失函数
            l.sum().backward()
            updater(x.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


# In[23]:


"""
使用案例:

参加train_ch3

"""
# 动画中绘制数据
class Animator: 
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                ylim=None, xscale='linear', yscale='linear',
                fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                figsize=(3.5,2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend
        )
        self.X, self.Y, self.fmts = None, None, fmts
        
    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        # 清除单元格的输出
        display.clear_output(wait=True)


# In[24]:


"""
使用案例:
multi_animator = Multi_Animator(xlabels=['x','x'],ylabels=['sin(x)','cos(x)'],nrows=2, ncols=1)
for x in range(100):
    multi_animator(x,[np.sin(x),np.cos(x)])

"""

# 动画中绘制数据 (同时绘多个图)
class Multi_Animator: 
    def __init__(self, xlabels='x', ylabels='y', legends=None, xlims=None,
                ylims=None, xscale='linear', yscale='linear',
                fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                figsize=(3.5,2.5)):
        # 增量地绘制多条线
        if legends is None:
            legends = []
        
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda i: set_axes(
            self.axes[i], xlabels[i], ylabels[i], xlims, ylims, xscale, yscale, legends
        )
        self.X, self.Y, self.fmts = None, None, fmts
        
    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
            self.axes[i].cla()
        for i, (x, y, fmt) in enumerate(zip(self.X, self.Y, self.fmts)):
            self.axes[i].plot(x, y, fmt)
            self.config_axes(i)
        display.display(self.fig)
        # 清除单元格的输出
        display.clear_output(wait=True)


# In[25]:


# 可视化训练进度(动画版)
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater): 
    animator = Animator(xlabel='epoch', xlim=[1,num_epochs],ylim=[0.3,0.9],
                       legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net,test_iter)
        # 向元组中增加一列(test_acc)
        animator.add(epoch+1, train_metrics + (test_acc, ))
    
    train_loss, train_acc = train_metrics
    
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <=1 and test_acc > 0.7, test_acc


# In[26]:


# 预测并展示部分结果
def predict_ch3(net, test_iter, n=6):
    # 只查看一组
    for x, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(x).argmax(axis=1))
    titles = ["True:" + true + "\n" + "pred:" + pred for true, pred in zip(trues, preds)]
    show_images(
        x[0:n].reshape((n, 28, 28)), 1, n, titles=titles)
   


# In[ ]:




