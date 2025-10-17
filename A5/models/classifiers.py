import numpy as np
import tensorflow as tf
from keras import layers, models

class LocalClassifier:
    """
    与SOM节点关联的局部分类器
    """
    def __init__(self, input_shape, num_classes, model_type='mlp'):
        """
        初始化局部分类器
        
        para:
            input_shape: 输入数据形状，可以是展平的向量或原始形状(如MNIST为(28,28,1))
            num_classes: 类别数量
            model_type: 模型类型，'mlp'或'cnn'
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_type = model_type
        
        self.model = self._build_model()
        
        self.train_history = []
        self.samples_seen = 0
        
        self.class_distribution = np.zeros(num_classes)
        
    def _build_model(self):
        if self.model_type == 'mlp':
            flat_dim = np.prod(self.input_shape)
            
            model = models.Sequential([
                layers.Input(shape=(flat_dim,)),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(64, activation='relu'),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
        elif self.model_type == 'cnn':
            model = models.Sequential([
                layers.Input(shape=self.input_shape),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess(self, x):
        if self.model_type == 'mlp':
            if len(x.shape) > 2:
                return x.reshape(x.shape[0], -1)
            return x
        else:
            if len(x.shape) == 2: 
                return x.reshape((-1,) + self.input_shape)
            return x
    
    def fit(self, x, y, epochs=10, batch_size=32, verbose=0):
        """
        训练分类器
        
        para:
            x: 输入数据
            y: 标签
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 详细程度
        """
        x = self.preprocess(x)
        
        for label in y:
            self.class_distribution[label] += 1
        
        history = self.model.fit(
            x, y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        self.train_history.append(history.history)
        self.samples_seen += len(x)
        
        return history
    
    def predict(self, x):
        """
        预测类别和置信度
        
        para:
            x: 输入数据
            
        return:
            predictions: 预测类别
            confidences: 预测置信度
        """
        x = self.preprocess(x)
        
        probs = self.model.predict(x)
        
        predictions = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        
        return predictions, confidences
    
    def get_class_distribution(self):
        if np.sum(self.class_distribution) > 0:
            return self.class_distribution / np.sum(self.class_distribution)
        return self.class_distribution