import numpy as np
from .noc_agent import NOCAgent

class ExpertAgent(NOCAgent):
    
    def __init__(self, input_shape, num_classes, map_size=(8, 8), 
                 classifier_type='mlp', agent_id=None, name=None,
                 specialty_classes=None, specialty_region=None):
        """
        初始化
            input_shape: 输入数据形状
            num_classes: 类别数量
            map_size: SOM映射尺寸
            classifier_type: 分类器类型
            agent_id
            name
            specialty_classes: 专注的类别列表
            specialty_region: 专注的输入空间区域
        """
        super().__init__(
            input_shape=input_shape,
            num_classes=num_classes,
            map_size=map_size,
            classifier_type=classifier_type,
            agent_id=agent_id,
            name=name if name else "ExpertAgent"
        )
        
        self.specialty_classes = specialty_classes
        self.specialty_region = specialty_region
        
        self.trust_score = 1.0
        self.predictions_made = 0
        self.correct_predictions = 0
    
    def is_responsible_for(self, x, y=None):
        """
        判断智能体是否负责处理给定样本
        
        para:
            x: 输入样本
            y: 样本标签(可选)
            
        return:
            is_responsible: 是否负责处理
        """
        if not self.is_trained:
            return False
            
        if self.specialty_classes is not None and y is not None:
            return y in self.specialty_classes
            
        if self.specialty_region is not None:
            if len(x.shape) > 1 and x.shape[0] > 1:
                bmu_indices = self.map_data(x)
                responsible_count = 0
                for bmu_idx in bmu_indices:
                    bmu_key = tuple(bmu_idx)
                    if bmu_key in self.active_nodes:
                        responsible_count += 1
                return responsible_count > len(bmu_indices) / 2
            else:
                if len(x.shape) == len(self.input_shape):
                    sample = x
                else:
                    sample = x[0]
                    
                flat_sample = sample.flatten() if len(sample.shape) > 1 else sample
                bmu_idx = self.som.find_bmu(flat_sample)
                bmu_key = tuple(bmu_idx)
                return bmu_key in self.active_nodes
                
        if self.novelty_detector is not None:
            is_novel, _ = self.detect_novelty(x)
            return not is_novel
            
        return False
    
    def update_trust_score(self, correct, total=1):
        """
        更新智能体的信任度分数
        
        para:
            correct: 正确预测的数量
            total: 预测总数
        """
        self.predictions_made += total
        self.correct_predictions += correct
        
        if self.predictions_made > 0:
            self.trust_score = self.correct_predictions / self.predictions_made
            
    def get_confidence(self, x):
        """
        获取对样本的预测置信度
        
        para:
            x: 输入样本
            
        return:
            confidence: 置信度
        """
        if not self.is_responsible_for(x):
            return 0.2
            
        _, confidence = self.predict(x)
        
        adjusted_confidence = confidence * self.trust_score
        
        return adjusted_confidence
    
    def specialize(self, x_train, y_train, target_classes=None, min_samples_per_class=100):
        """
        使智能体专注于特定类别
        
        para:
            x_train: 训练数据
            y_train: 训练标签
            target_classes: 目标类别列表，如果为None则自动选择
            min_samples_per_class: 每个类别的最小样本数
        """
        if target_classes is None:
            if np.sum(self.expertise) > 0:
                target_classes = [np.argmax(self.expertise)]
            else:
                target_classes = [np.random.randint(0, self.num_classes)]
        
        self.specialty_classes = target_classes
        
        selected_indices = []
        for cls in target_classes:
            cls_indices = np.where(y_train == cls)[0]
            if len(cls_indices) > min_samples_per_class:
                selected = np.random.choice(cls_indices, min_samples_per_class, replace=False)
            else:
                selected = cls_indices
            selected_indices.extend(selected)
        
        x_selected = x_train[selected_indices]
        y_selected = y_train[selected_indices]
        
        self.train_som(x_selected)
        self.train_classifiers(x_selected, y_selected)
        
        active_bmuis = []
        for x in x_selected:
            if len(x.shape) > 1:
                flat_x = x.flatten()
            else:
                flat_x = x
            bmu_idx = self.som.find_bmu(flat_x)
            active_bmuis.append(bmu_idx)
        
        self.specialty_region = np.array(active_bmuis)
        
        print(f"智能体 {self.name} 专注于类别 {target_classes}，"
              f"使用 {len(x_selected)} 个样本，覆盖 {len(self.active_nodes)} 个SOM节点")