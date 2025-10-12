import numpy as np
import tensorflow as tf
from keras import datasets, layers, models
import tensorflow as tf

def load_mnist_data(flatten=False, normalize=True, reshape_for_cnn=True):
    """
    加载MNIST数据集
    
    para:
        flatten: 是否将图像展平为向量
        normalize: 是否将像素值归一化到[0,1]
        reshape_for_cnn: 是否重塑为CNN输入格式
        
    return:
        (x_train, y_train), (x_test, y_test): 训练集和测试集
    """
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    if normalize:
        x_train /= 255.0
        x_test /= 255.0
    
    if reshape_for_cnn and not flatten:
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
    
    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    
    print(f"MNIST数据加载完成:")
    print(f"  训练集: {x_train.shape}, {y_train.shape}")
    print(f"  测试集: {x_test.shape}, {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)

def get_mnist_subset(x, y, classes=None, samples_per_class=None, shuffle=True):
    """
    获取MNIST数据子集
    
    pa ra:
        x: 输入数据
        y: 标签
        classes: 要选择的类别列表，默认为所有类别
        samples_per_class: 每个类别选择的样本数，默认为全部
        shuffle: 是否打乱数据
        
    return:
        x_subset, y_subset: 数据子集
    """
    if classes is None:
        classes = np.unique(y)
    
    x_subset = []
    y_subset = []
    
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        
        if samples_per_class is not None:
            if shuffle:
                np.random.shuffle(cls_indices)
            cls_indices = cls_indices[:samples_per_class]
        
        x_subset.append(x[cls_indices])
        y_subset.append(y[cls_indices])
    
    x_subset = np.vstack(x_subset)
    y_subset = np.hstack(y_subset)
    
    if shuffle:
        indices = np.random.permutation(len(x_subset))
        x_subset = x_subset[indices]
        y_subset = y_subset[indices]
    
    return x_subset, y_subset

def create_dynamic_dataset(x_train, y_train, x_test, y_test, stages=3):
    """
    创建一个动态变化的数据集，for concept drift simulation
    
    para:
        x_train, y_train: 原始训练数据
        x_test, y_test: 原始测试数据
        stages: 数据变化的阶段数
        
    return:
        train_stages, test_stages: 每个阶段的训练集和测试集列表
    """
    train_stages = []
    test_stages = []
    
    for stage in range(stages):
        noise_level = (stage + 1) * 0.05
        
        x_train_mod = x_train + np.random.normal(0, noise_level, x_train.shape)
        x_train_mod = np.clip(x_train_mod, 0, 1) 
        
        x_test_mod = x_test + np.random.normal(0, noise_level, x_test.shape)
        x_test_mod = np.clip(x_test_mod, 0, 1)
        
        train_stages.append((x_train_mod, y_train))
        test_stages.append((x_test_mod, y_test))
    
    return train_stages, test_stages