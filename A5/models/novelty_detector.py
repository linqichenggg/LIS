import numpy as np
from sklearn.neighbors import LocalOutlierFactor

class NoveltyDetector:
    
    def __init__(self, som, threshold=None):
        self.som = som
        self.threshold = threshold
        self.distances = [] 
        
        self.lof = None
        
    def compute_novelty_score(self, x):
        if len(x.shape) > 1:
            x = x.flatten()
            
        bmu_idx = self.som.find_bmu(x)
        
        distance = np.sqrt(np.sum((self.som.weights[bmu_idx] - x) ** 2))
        
        return distance
    
    def fit(self, data, auto_threshold=True, contamination=0.05):
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
        if self.threshold is None:
            raise ValueError("Detector must be fitted before use")
            
        score = self.compute_novelty_score(x)
        
        return score > self.threshold, score