import numpy as np

class AgentCoordinator:
    """
    智能体协调器
    负责分配任务给适当的智能体并整合结果
    """
    
    def __init__(self, agents=None):
        """
        初始化协调器
        
        参数:
            agents: 初始智能体列表
        """
        self.agents = agents if agents is not None else []
        
        # 各智能体的信任度
        self.trust_scores = {agent.agent_id: 1.0 for agent in self.agents}
        
        # 任务分配历史
        self.assignment_history = []
        
        # 类别专业化映射：记录哪些智能体专注于哪些类别
        self.class_specialists = {}
    
    def add_agent(self, agent):
        """
        添加智能体
        
        参数:
            agent: 要添加的智能体
        """
        self.agents.append(agent)
        self.trust_scores[agent.agent_id] = 1.0
    
    def remove_agent(self, agent_id):
        """
        移除智能体
        
        参数:
            agent_id: 要移除的智能体ID
        """
        self.agents = [agent for agent in self.agents if agent.agent_id != agent_id]
        if agent_id in self.trust_scores:
            del self.trust_scores[agent_id]
    
    def assign_sample(self, x, return_all_confidences=False):
        """
        将样本分配给最合适的智能体
        
        参数:
            x: 输入样本
            return_all_confidences: 是否返回所有智能体的置信度
            
        返回:
            best_agent: 最佳智能体
            confidence: 最佳智能体的置信度
        """
        if not self.agents:
            raise ValueError("没有可用的智能体")
            
        # 收集所有智能体的置信度
        agent_confidences = []
        
        for agent in self.agents:
            # 只考虑已训练的智能体
            if not agent.is_trained:
                continue
                
            # 获取智能体的置信度
            confidence = agent.get_confidence(x) if hasattr(agent, 'get_confidence') else 0.5
            
            # 考虑智能体的信任度
            adjusted_confidence = confidence * self.trust_scores[agent.agent_id]
            
            agent_confidences.append((agent, adjusted_confidence))
        
        if not agent_confidences:
            # 如果没有智能体可用
            if return_all_confidences:
                return None, 0.0, {}
            return None, 0.0
        
        # 选择置信度最高的智能体
        agent_confidences.sort(key=lambda x: x[1], reverse=True)
        best_agent, best_confidence = agent_confidences[0]
        
        # 记录分配
        self.assignment_history.append(best_agent.agent_id)
        
        if return_all_confidences:
            # 返回所有智能体的置信度
            confidence_dict = {agent.agent_id: conf for agent, conf in agent_confidences}
            return best_agent, best_confidence, confidence_dict
        
        return best_agent, best_confidence
    
    def aggregate_predictions(self, predictions_with_confidences):
        """
        整合多个智能体的预测结果
        
        参数:
            predictions_with_confidences: [(agent_id, prediction, confidence), ...]
            
        返回:
            final_prediction: 最终预测
        """
        if not predictions_with_confidences:
            return None
            
        # 如果只有一个预测，直接返回
        if len(predictions_with_confidences) == 1:
            return predictions_with_confidences[0][1]
            
        # 加权投票
        class_votes = np.zeros(10)  # 假设有10个类别
        
        for agent_id, pred, conf in predictions_with_confidences:
            # 获取智能体的信任度
            trust = self.trust_scores.get(agent_id, 0.5)
            
            # 计算投票权重
            weight = trust * conf
            
            # 累加投票
            class_votes[pred] += weight
        
        # 返回得票最高的类别
        return np.argmax(class_votes)
    
    def update_trust(self, agent_id, performance):
        """
        更新对智能体的信任度
        
        参数:
            agent_id: 智能体ID
            performance: 性能分数(0-1)
        """
        if agent_id not in self.trust_scores:
            return
            
        # 平滑更新信任度
        alpha = 0.1  # 学习率
        self.trust_scores[agent_id] = (1 - alpha) * self.trust_scores[agent_id] + alpha * performance
    
    def get_best_agent_for_class(self, class_id):
        """
        获取对特定类别最擅长的智能体
        
        参数:
            class_id: 类别ID
            
        返回:
            best_agent: 最佳智能体
        """
        best_agent = None
        best_expertise = -1
        
        for agent in self.agents:
            if not agent.is_trained:
                continue
                
            # 获取智能体对该类别的专业化程度
            expertise = agent.get_expertise_level(class_id) if hasattr(agent, 'get_expertise_level') else 0
            
            # 考虑信任度
            adjusted_expertise = expertise * self.trust_scores[agent.agent_id]
            
            if adjusted_expertise > best_expertise:
                best_expertise = adjusted_expertise
                best_agent = agent
                
        return best_agent
    
    def update_specialists(self):
        """更新类别专家映射"""
        self.class_specialists = {}
        
        for cls in range(10):  # 假设有10个类别
            best_agent = self.get_best_agent_for_class(cls)
            if best_agent is not None:
                self.class_specialists[cls] = best_agent.agent_id
    
    def get_agent_by_id(self, agent_id):
        """
        根据ID获取智能体
        
        参数:
            agent_id: 智能体ID
            
        返回:
            agent: 智能体对象，如果不存在则返回None
        """
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None