import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_data(x, flatten=False, normalize=True):
    """
    para:
        x: 输入数据
        flatten: 是否将图像展平为向量
        normalize: 是否将像素值归一化到[0,1]
        
    return:
        x_processed: 处理后的数据
    """
    x_processed = x.copy()
    
    if x_processed.dtype != np.float32:
        x_processed = x_processed.astype('float32')
    
    if normalize and np.max(x_processed) > 1.0:
        x_processed /= 255.0
    
    if flatten:
        original_shape = x_processed.shape
        if len(original_shape) > 2:  
            x_processed = x_processed.reshape(original_shape[0], -1)
    
    return x_processed

def split_data(x, y, test_size=0.2, validation_size=0.1, random_state=42):
    """
    para:
        x: 输入数据
        y: 标签
        test_size: 测试集比例
        validation_size: 验证集比例
        random_state: 随机种子
        
    return:
        (x_train, y_train), (x_val, y_val), (x_test, y_test): 训练集、验证集和测试集
    """
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    if validation_size > 0:
        val_size_adjusted = validation_size / (1 - test_size)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val, 
            test_size=val_size_adjusted, 
            random_state=random_state,
            stratify=y_train_val
        )
        
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    
    return (x_train_val, y_train_val), None, (x_test, y_test)

def augment_data(x, y, augmentation_factor=2):

    if len(x.shape) == 3:
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    
    x_augmented = []
    y_augmented = []
    
    x_augmented.append(x)
    y_augmented.append(y)
    
    for i in range(augmentation_factor - 1):
        shifted_x = np.zeros_like(x)
        for j in range(len(x)):
            h_shift = np.random.randint(-2, 3)
            v_shift = np.random.randint(-2, 3)
            
            if h_shift > 0:
                shifted_x[j, :, h_shift:, :] = x[j, :, :-h_shift, :]
            elif h_shift < 0:
                shifted_x[j, :, :h_shift, :] = x[j, :, -h_shift:, :]
            else:
                shifted_x[j] = x[j]
                
            if v_shift > 0:
                shifted_x[j, v_shift:, :, :] = shifted_x[j, :-v_shift, :, :]
            elif v_shift < 0:
                shifted_x[j, :v_shift, :, :] = shifted_x[j, -v_shift:, :, :]
        
        x_augmented.append(shifted_x)
        y_augmented.append(y)
    
    x_augmented = np.vstack(x_augmented)
    y_augmented = np.hstack(y_augmented)
    
    return x_augmented, y_augmented