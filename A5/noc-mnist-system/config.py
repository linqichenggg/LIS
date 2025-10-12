"""
配置参数模块
"""

# 系统参数
SYSTEM_CONFIG = {
    'num_experts': 0,  # 初始专家智能体数量，0表示自动创建
}

# SOM参数
SOM_CONFIG = {
    'main_map_size': (10, 10),  # 主智能体SOM大小
    'expert_map_size': (5, 5),  # 专家智能体SOM大小
    'iterations': 1000,  # SOM训练迭代次数
    'learning_rate': 0.1,  # SOM学习率
}

# 分类器参数
CLASSIFIER_CONFIG = {
    'type': 'cnn',  # 分类器类型：'mlp'或'cnn'
    'epochs': 2,  # 训练轮数
    'batch_size': 64,  # 批次大小
}

# 训练参数
TRAINING_CONFIG = {
    'val_ratio': 0.1,  # 验证集比例
    'samples_per_class': 100,  # 每个类别的最大样本数
    'min_samples_for_classifier': 10,  # 训练分类器的最小样本数
}

# 新颖性检测参数
NOVELTY_CONFIG = {
    'contamination': 0.05,  # 预期异常比例
    'buffer_size': 100,  # 新颖样本缓冲区大小
}

# 适应参数
ADAPTATION_CONFIG = {
    'min_novel_for_expert': 50,  # 创建新专家的最小新颖样本数
    'adaptation_epochs': 3,  # 适应训练轮数
}

# 实验参数
EXPERIMENT_CONFIG = {
    'num_experiments': 1,  # 实验次数
    'dynamic_stages': 2,  # 动态环境阶段数
    'noise_levels': [0.0, 0.1],  # 噪声级别
}