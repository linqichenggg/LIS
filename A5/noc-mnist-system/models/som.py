import numpy as np
import matplotlib.pyplot as plt

class SelfOrganizingMap:
    
    def __init__(self, input_dim, map_size=(10, 10), learning_rate=0.1, sigma=None):
        """
        初始化自组织映射
        
        para:
            input_dim: 输入向量维度
            map_size: SOM地图大小，元组(width, height)
            learning_rate: 初始学习率
            sigma: 初始邻域半径，默认为max(map_size)/2
        """
        self.input_dim = input_dim
        self.map_size = map_size
        self.initial_learning_rate = learning_rate
        self.initial_sigma = sigma if sigma is not None else max(map_size) / 2
        
        self.weights = np.random.random((map_size[0], map_size[1], input_dim))
        
        self.iterations = 0
        self.trained = False
        
        self.activation_count = np.zeros(map_size)
    
    def find_bmu(self, x):
        """
        找到最佳匹配单元(Best Matching Unit)
        
        pa ra:
            x: 输入向量
            
        return:
            bmu_idx: BMU索引，元组(i, j)
        """
        distances = np.sum((self.weights - x) ** 2, axis=2)
        
        bmu_idx = np.unravel_index(np.argmin(distances), self.map_size)
        return bmu_idx
    
    def update_weights(self, x, bmu_idx, iteration):
        """
        更新权重向量
        
        para:
            x: 输入向量
            bmu_idx: BMU索引
            iteration: 当前迭代次数
        """
        learning_rate = self.initial_learning_rate * np.exp(-iteration / 1000.0)
        sigma = self.initial_sigma * np.exp(-iteration / 1000.0)
        
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                dist = np.sqrt((i - bmu_idx[0]) ** 2 + (j - bmu_idx[1]) ** 2)
                
                influence = np.exp(-(dist ** 2) / (2 * (sigma ** 2)))
                
                self.weights[i, j] += learning_rate * influence * (x - self.weights[i, j])
    
    def train(self, data, iterations=10000, verbose=True):
        """
        训练自组织映射
        
        para:
            data: 训练数据，形状为(n_samples, input_dim)
            iterations: 训练迭代次数
            verbose: 是否打印进度
        """
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        
        if verbose:
            print(f"Training SOM with {len(data)} samples for {iterations} iterations...")
        
        for i in range(iterations):
            sample_idx = np.random.randint(0, len(data))
            x = data[sample_idx]
            
            bmu_idx = self.find_bmu(x)
            
            self.update_weights(x, bmu_idx, i)
            
            self.activation_count[bmu_idx] += 1
            
            if verbose and (i+1) % (iterations // 10) == 0:
                print(f"  Iteration {i+1}/{iterations} completed")
        
        self.iterations += iterations
        self.trained = True
        
        if verbose:
            print("SOM training completed")
    
    def transform(self, x):
        """
        将输入向量映射到SOM上，returnBMU索引
        
        para:
            x: 输入向量或向量集合
            
        return:
            bmu_indices: BMU索引或索引集合
        """
        if not self.trained:
            raise ValueError("SOM must be trained before transform")
        
        if len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[0] == 1):
            if len(x.shape) > 1:
                x = x.flatten()
            return self.find_bmu(x)
        
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
            
        bmu_indices = []
        for sample in x:
            bmu_indices.append(self.find_bmu(sample))
            
        return np.array(bmu_indices)
    
    def get_distance_map(self):
        """
        计算U-Matrix (统一距离矩阵)，显示节点间距离
        
        return:
            u_matrix: U-Matrix，形状与SOM相同
        """
        u_matrix = np.zeros(self.map_size)
        
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.map_size[0] and 0 <= nj < self.map_size[1]:
                            neighbors.append((ni, nj))
                
                if neighbors:
                    dist_sum = 0
                    for ni, nj in neighbors:
                        dist_sum += np.sqrt(np.sum((self.weights[i, j] - self.weights[ni, nj]) ** 2))
                    u_matrix[i, j] = dist_sum / len(neighbors)
        
        return u_matrix
    
    def visualize(self, data=None, labels=None, show_activation=True):
        """
        可视化SOM
        
        para:
            data: 映射到SOM的数据，可选
            labels: 数据标签，可选
            show_activation: 是否显示激活热图
        """
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        u_matrix = self.get_distance_map()
        plt.imshow(u_matrix, cmap='viridis')
        plt.colorbar()
        plt.title('U-Matrix (Node Distances)')
        
        if show_activation:
            plt.subplot(1, 3, 2)
            plt.imshow(self.activation_count, cmap='hot')
            plt.colorbar()
            plt.title('Node Activation Count')
        
        if data is not None and labels is not None:
            plt.subplot(1, 3, 3)
            if len(data.shape) > 2:
                data = data.reshape(data.shape[0], -1)
            
            cmap = plt.cm.get_cmap('tab10', len(np.unique(labels)))
            
            class_map = np.zeros(self.map_size + (3,))
            counts = np.zeros(self.map_size)
            
            for i, x in enumerate(data):
                bmu = self.find_bmu(x)
                color = cmap(labels[i])[:3]  
                if counts[bmu] == 0:
                    class_map[bmu] = color
                else:
                    class_map[bmu] = (class_map[bmu] * counts[bmu] + color) / (counts[bmu] + 1)
                counts[bmu] += 1
            
            plt.imshow(class_map)
            plt.title('Class Distribution')
            
            unique_labels = np.unique(labels)
            for i, label in enumerate(unique_labels):
                plt.plot([], [], 'o', color=cmap(i), label=f'Class {label}')
            plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()