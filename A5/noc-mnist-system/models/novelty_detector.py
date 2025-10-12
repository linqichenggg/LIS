import numpy as np
from sklearn.neighbors import LocalOutlierFactor

class NoveltyDetector:
    
    def __init__(self, som, threshold=None):
        """
        初始化新颖性检测器
        
        para:
            som: 训练好的SOM对象
            threshold: 新颖性阈值，默认为None(自动确定)
        """
        self.som = som
        self.threshold = threshold
        self.distances = [] 
        
        self.lof = None
        
    def compute_novelty_score(self, x):
        """
        计算样本的新颖性分数(与BMU的距离)
        
        para:
            x: 输入样本
            
        return:
            score: 新颖性分数
        """
        if len(x.shape) > 1:
            x = x.flatten()
            
        bmu_idx = self.som.find_bmu(x)
        
        distance = np.sqrt(np.sum((self.som.weights[bmu_idx] - x) ** 2))
        
        return distance
    
    def fit(self, data, auto_threshold=True, contamination=0.05):
        """
        训练新颖性检测器，计算阈值
        
        para:
            data: 训练数据
            auto_threshold: 是否自动设置阈值
            contamination: 预期异常比例
        """
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
            
        distances = []
        for x in data:
            distances.append(self.compute_novelty_score(x))
        
        self.distances = np.array(distances)
        
        if auto_threshold and self.threshold is None:
            self.threshold = np.percentile(self.distances, (1 - contamination) * 100)
        
        self.lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        self.lof.fit(data)
    
    def is_novel(self, x):
        """
        判断样本是否为新颖样本
        
        para:
            x: 输入样本
            
        return:
            is_novel: 布尔值，是否为新颖样本
            score: 新颖性分数
        """
        if self.threshold is None:
            raise ValueError("Detector must be fitted before use")
            
        score = self.compute_novelty_score(x)
        
        return score > self.threshold, score