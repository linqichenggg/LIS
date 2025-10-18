import numpy as np

class AgentCoordinator:
    def __init__(self, agents=None):
        self.agents = agents if agents is not None else []
        
        self.trust_scores = {agent.agent_id: 0.9 for agent in self.agents}
        
        self.assignment_history = []
        
        self.class_specialists = {}
    
    def add_agent(self, agent):
        self.agents.append(agent)
        self.trust_scores[agent.agent_id] = 0.9
    
    def remove_agent(self, agent_id):
        self.agents = [agent for agent in self.agents if agent.agent_id != agent_id]
        if agent_id in self.trust_scores:
            del self.trust_scores[agent_id]
    
    def assign_sample(self, x, return_all_confidences=False):
        if not self.agents:
            raise ValueError("没有可用的智能体")
            
        agent_confidences = []
        
        for agent in self.agents:
            if not agent.is_trained:
                continue
                
            confidence = agent.get_confidence(x) if hasattr(agent, 'get_confidence') else 0.5
            
            adjusted_confidence = confidence * self.trust_scores[agent.agent_id]
            
            agent_confidences.append((agent, adjusted_confidence))
        
        if not agent_confidences:
            if return_all_confidences:
                return None, 0.0, {}
            return None, 0.0
        
        agent_confidences.sort(key=lambda x: x[1], reverse=True)
        best_agent, best_confidence = agent_confidences[0]
        
        self.assignment_history.append(best_agent.agent_id)
        
        if return_all_confidences:
            confidence_dict = {agent.agent_id: conf for agent, conf in agent_confidences}
            return best_agent, best_confidence, confidence_dict
        
        return best_agent, best_confidence
    
    def aggregate_predictions(self, predictions_with_confidences):
        if not predictions_with_confidences:
            return None
            
        if len(predictions_with_confidences) == 1:
            return predictions_with_confidences[0][1]
            
        class_votes = np.zeros(10) 
        
        for agent_id, pred, conf in predictions_with_confidences:
            trust = self.trust_scores.get(agent_id, 0.5)
            
            weight = trust * conf
            
            class_votes[pred] += weight
        
        return np.argmax(class_votes)
    
    def update_trust(self, agent_id, performance):
        if agent_id not in self.trust_scores:
            return
            
        alpha = 0.1  # 学习率
        self.trust_scores[agent_id] = (1 - alpha) * self.trust_scores[agent_id] + alpha * performance
    
    def get_best_agent_for_class(self, class_id):
        best_agent = None
        best_expertise = -1
        
        for agent in self.agents:
            if not agent.is_trained:
                continue
                
            expertise = agent.get_expertise_level(class_id) if hasattr(agent, 'get_expertise_level') else 0
            
            adjusted_expertise = expertise * self.trust_scores[agent.agent_id]
            
            if adjusted_expertise > best_expertise:
                best_expertise = adjusted_expertise
                best_agent = agent
                
        return best_agent
    
    def update_specialists(self):
        self.class_specialists = {}
        
        for cls in range(10):  
            best_agent = self.get_best_agent_for_class(cls)
            if best_agent is not None:
                self.class_specialists[cls] = best_agent.agent_id
    
    def get_agent_by_id(self, agent_id):
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None