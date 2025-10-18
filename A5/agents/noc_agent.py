import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from models.som import SelfOrganizingMap
from models.classifiers import LocalClassifier
from models.novelty_detector import NoveltyDetector

class NOCAgent:

    def __init__(self, input_shape, num_classes, map_size=(10, 10), 
                 classifier_type='mlp', agent_id=None, name=None):

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.map_size = map_size
        self.classifier_type = classifier_type
        self.agent_id = agent_id if agent_id is not None else id(self)
        self.name = name if name is not None else f"NOCAgent-{self.agent_id}"
        
        self.input_dim = np.prod(input_shape)
        
        self.som = SelfOrganizingMap(input_dim=self.input_dim, map_size=map_size)
        
        self.classifiers = {}
        
        self.novelty_detector = None
        
        self.is_trained = False
        self.samples_seen = 0
        self.samples_by_class = np.zeros(num_classes)
        self.performance_history = []
        
        self.expertise = np.zeros(num_classes)
        
        self.active_nodes = set()
    
    def train_som(self, data, iterations=10000, verbose=True):

        if len(data.shape) > 2:
            data = data.reshape(data.shape[0], -1)
            
        self.som.train(data, iterations=iterations, verbose=verbose)
        
        if verbose:
            print(f"智能体 {self.name} SOM训练完成，活动节点: {len(np.where(self.som.activation_count > 0)[0])}")
    
    def _create_classifier(self, bmu_idx):

        classifier = LocalClassifier(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            model_type=self.classifier_type
        )
        
        self.classifiers[bmu_idx] = classifier
        return classifier
    
    def _get_classifier(self, bmu_idx):

        bmu_key = tuple(bmu_idx)
        if bmu_key not in self.classifiers:
            return self._create_classifier(bmu_key)
        return self.classifiers[bmu_key]
    
    def map_data(self, data):

        if len(data.shape) > 2:
            flat_data = data.reshape(data.shape[0], -1)
        else:
            flat_data = data
            
        bmu_indices = []
        for x in flat_data:
            bmu_idx = self.som.find_bmu(x)
            bmu_indices.append(bmu_idx)
            
        return np.array(bmu_indices)
    
    def train_classifiers(self, data, labels, epochs=5, min_samples=10, verbose=False):

        if not self.som.trained:
            raise ValueError("必须先训练SOM才能训练分类器")
            
        if verbose:
            print(f"映射{len(data)}个样本到SOM...")
            
        bmu_samples = {}
        
        for i in range(len(data)):
            x = data[i]
            y = labels[i]
            
            self.samples_seen += 1
            self.samples_by_class[y] += 1
            
            if len(x.shape) > 1 and np.prod(x.shape) == self.input_dim:
                flat_x = x.flatten()
            else:
                flat_x = x
                
            bmu_idx = self.som.find_bmu(flat_x)
            bmu_key = tuple(bmu_idx)
            
            self.active_nodes.add(bmu_key)
            
            if bmu_key not in bmu_samples:
                bmu_samples[bmu_key] = {'x': [], 'y': []}
            
            bmu_samples[bmu_key]['x'].append(x)
            bmu_samples[bmu_key]['y'].append(y)
        
        trained_classifiers = 0
        for bmu_key, samples in bmu_samples.items():
            if len(samples['y']) >= min_samples:
                node_x = np.array(samples['x'])
                node_y = np.array(samples['y'])
                
                classifier = self._get_classifier(bmu_key)
                
                if verbose:
                    print(f"  训练节点{bmu_key}的分类器，样本数：{len(node_y)}")
                
                classifier.fit(node_x, node_y, epochs=epochs, batch_size=16, verbose=0)
                trained_classifiers += 1
                
                class_dist = classifier.get_class_distribution()
                for cls in range(self.num_classes):
                    if class_dist[cls] > 0.1:  
                        self.expertise[cls] += class_dist[cls] * len(node_y)
        
        if np.sum(self.expertise) > 0:
            self.expertise = self.expertise / np.sum(self.expertise)
            
        if verbose:
            print(f"训练了{trained_classifiers}个分类器，覆盖{len(self.active_nodes)}个活动节点")
            print(f"专业化分布: {self.expertise}")

        if self.novelty_detector is None:
            self.novelty_detector = NoveltyDetector(self.som)
            
        if len(data.shape) > 2:
            flat_data = data.reshape(data.shape[0], -1)
        else:
            flat_data = data
            
        self.novelty_detector.fit(flat_data)
        
        self.is_trained = True
    
    def predict(self, x):

        if not self.is_trained:
            raise ValueError("智能体必须先训练才能进行预测")
            
        single_sample = len(x.shape) == len(self.input_shape)
        if single_sample:
            x = np.expand_dims(x, axis=0)
            
        predictions = []
        confidences = []
        
        for i in range(len(x)):
            sample = x[i]
            
            if len(sample.shape) > 1:
                flat_sample = sample.flatten()
            else:
                flat_sample = sample
                
            bmu_idx = self.som.find_bmu(flat_sample)
            bmu_key = tuple(bmu_idx)
            
            if bmu_key in self.classifiers:
                classifier = self.classifiers[bmu_key]
                pred, conf = classifier.predict(np.expand_dims(sample, axis=0))
                class_idx = pred[0]
                confidence = conf[0]
            else:
                class_idx, confidence = self._find_nearest_classifier_prediction(sample, bmu_idx)
            
            predictions.append(class_idx)
            confidences.append(confidence)
        
        if single_sample:
            return predictions[0], confidences[0]
        return np.array(predictions), np.array(confidences)
    
    def _find_nearest_classifier_prediction(self, sample, bmu_idx):

        if not self.classifiers:
            return np.random.randint(0, self.num_classes), 0.1
        
        distances = {}
        for node_idx in self.classifiers.keys():
            dist = np.sqrt((bmu_idx[0] - node_idx[0])**2 + (bmu_idx[1] - node_idx[1])**2)
            distances[node_idx] = dist
        
        nearest_node = min(distances.items(), key=lambda x: x[1])[0]
        
        classifier = self.classifiers[nearest_node]
        if len(sample.shape) > 1:
            pred, conf = classifier.predict(np.expand_dims(sample, axis=0))
        else:
            pred, conf = classifier.predict(np.expand_dims(sample.reshape(self.input_shape), axis=0))
            
        return pred[0], conf[0]
    
    def detect_novelty(self, x):

        if self.novelty_detector is None:
            return False, 0.0
            
        if len(x.shape) == len(self.input_shape):
            if len(x.shape) > 1:
                flat_x = x.flatten()
            else:
                flat_x = x
            return self.novelty_detector.is_novel(flat_x)
        
        results = []
        scores = []
        
        for i in range(len(x)):
            sample = x[i]
            if len(sample.shape) > 1:
                flat_sample = sample.flatten()
            else:
                flat_sample = sample
                
            is_novel, score = self.novelty_detector.is_novel(flat_sample)
            results.append(is_novel)
            scores.append(score)
            
        return np.array(results), np.array(scores)
    
    def get_expertise_level(self, class_id):

        if np.sum(self.expertise) == 0:
            return 0.0
        
        return self.expertise[class_id]
    
    def update_expertise(self, x_val, y_val):

        if not self.is_trained:
            return
            
        class_accuracies = np.zeros(self.num_classes)
        class_counts = np.zeros(self.num_classes)
        
        for cls in range(self.num_classes):
            cls_mask = (y_val == cls)
            if np.sum(cls_mask) == 0:
                continue
                
            x_cls = x_val[cls_mask]
            y_cls = y_val[cls_mask]
            
            preds, _ = self.predict(x_cls)
            
            accuracy = np.mean(preds == y_cls)
            
            class_accuracies[cls] = accuracy
            class_counts[cls] = np.sum(cls_mask)
        
        for cls in range(self.num_classes):
            if class_counts[cls] > 0:
                self.expertise[cls] = class_accuracies[cls] * np.sqrt(class_counts[cls])
                
        if np.sum(self.expertise) > 0:
            self.expertise = self.expertise / np.sum(self.expertise)
    
    def visualize_som_state(self):
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.imshow(self.som.activation_count, cmap='hot')
        plt.colorbar()
        plt.title('node activation count')
        
        plt.subplot(2, 2, 2)
        coverage_map = np.zeros(self.map_size)
        for node_idx in self.classifiers.keys():
            coverage_map[node_idx] = 1
        plt.imshow(coverage_map, cmap='Blues')
        plt.title('classifier coverage')
        
        plt.subplot(2, 2, 3)
        plt.bar(range(self.num_classes), self.expertise)
        plt.xlabel('class')
        plt.ylabel('specialization level')
        plt.title('class specialization distribution')
        plt.xticks(range(self.num_classes))
        
        plt.subplot(2, 2, 4)
        u_matrix = self.som.get_distance_map()
        plt.imshow(u_matrix, cmap='viridis')
        plt.colorbar()
        plt.title('U-Matrix (node distance)')
        
        plt.suptitle(f"agent {self.name} SOM state")
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()