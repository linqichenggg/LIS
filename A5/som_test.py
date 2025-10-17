import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# ======== 这里复制我们定义好的 SOM 类 ========

class SelfOrganizingMap:
    
    def __init__(self, input_dim, map_size=(10, 10), learning_rate=0.1, sigma=None):
        self.input_dim = input_dim
        self.map_size = map_size
        self.initial_learning_rate = learning_rate
        self.initial_sigma = sigma if sigma is not None else max(map_size) / 2
        self.weights = np.random.random((map_size[0], map_size[1], input_dim))
        self.iterations = 0
        self.trained = False
        self.activation_count = np.zeros(map_size)
    
    def find_bmu(self, x):
        distances = np.sum((self.weights - x) ** 2, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), self.map_size)
        return bmu_idx
    
    def update_weights(self, x, bmu_idx, iteration):
        learning_rate = self.initial_learning_rate * np.exp(-iteration / 1000.0)
        sigma = self.initial_sigma * np.exp(-iteration / 1000.0)
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                dist = np.sqrt((i - bmu_idx[0]) ** 2 + (j - bmu_idx[1]) ** 2)
                influence = np.exp(-(dist ** 2) / (2 * (sigma ** 2)))
                self.weights[i, j] += learning_rate * influence * (x - self.weights[i, j])
    
    def train(self, data, iterations=5000, verbose=True):
        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
        if verbose:
            print(f"Training SOM with {len(data)} samples for {iterations} iterations...")
        for i in range(iterations):
            x = data[np.random.randint(0, len(data))]
            bmu_idx = self.find_bmu(x)
            self.update_weights(x, bmu_idx, i)
            self.activation_count[bmu_idx] += 1
            if verbose and (i + 1) % (iterations // 10) == 0:
                print(f"Iteration {i+1}/{iterations} done")
        self.trained = True
        print("Training complete!")

    def get_distance_map(self):
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

    def visualize(self, data=None, labels=None):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(self.get_distance_map(), cmap='viridis')
        plt.colorbar()
        plt.title('U-Matrix (Node Distances)')
        plt.subplot(1, 2, 2)
        plt.imshow(self.activation_count, cmap='hot')
        plt.colorbar()
        plt.title('Activation Count')
        plt.show()

# ======== 下面是最小实验部分 ========

# 加载 sklearn 自带的 digits 数据集（8x8 手写数字）
data = load_digits()
x = data.data / 16.0 
y = data.target

# 初始化 SOM
som = SelfOrganizingMap(input_dim=64, map_size=(10, 10), learning_rate=0.5)

# 训练
som.train(x, iterations=3000, verbose=True)

# 可视化
som.visualize(data=x, labels=y)
plt.figure(figsize=(6,6))
class_map = np.zeros(som.map_size)
for i, x in enumerate(x):
    bmu = som.find_bmu(x)
    class_map[bmu] = y[i]
plt.imshow(class_map, cmap='tab10')
plt.title("Dominant Class per Node")
plt.colorbar()
plt.show()
