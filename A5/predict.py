import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras import datasets
from system.multi_agent_system import MultiAgentSystem
from agents.expert_agent import ExpertAgent

input_shape = (28, 28, 1)
num_classes = 10
system = MultiAgentSystem(input_shape=input_shape, num_classes=num_classes)

with open('/Users/lqcmacmini/cursor/trained_noc_system.pkl', 'rb') as f:
    saved_system = pickle.load(f)
    
print("加载主智能体...")
system.main_agent.som.weights = saved_system['main_agent']['som_weights']
for node_str, weights in saved_system['main_agent']['classifiers'].items():
    node = eval(node_str)
    if node not in system.main_agent.classifiers:
        system.main_agent._create_classifier(node)
    system.main_agent.classifiers[node].model.set_weights(weights)

print("加载专家智能体...")
if 'expert_agents' in saved_system and saved_system['expert_agents']:
    for expert_data in saved_system['expert_agents']:
        original_agent_id = None
        if 'agent_id' in expert_data:
            original_agent_id = expert_data['agent_id']
        
        expert = ExpertAgent(
            input_shape=input_shape,
            num_classes=num_classes,
            map_size=(8, 8),
            name=expert_data['name'],
            specialty_classes=expert_data['specialty_classes'],
            agent_id=original_agent_id
        )
        expert.som.weights = expert_data['som_weights']
        for node_str, weights in expert_data['classifiers'].items():
            node = eval(node_str)
            if node not in expert.classifiers:
                expert._create_classifier(node)
            expert.classifiers[node].model.set_weights(weights)
        
        expert.is_trained = True
        expert.som.trained = True
        system.expert_agents.append(expert)
        system.all_agents.append(expert)
    print(f"已加载 {len(saved_system['expert_agents'])} 个专家智能体")

print("恢复协调器状态...")
if 'coordinator' in saved_system:
    system.coordinator.trust_scores = saved_system['coordinator']['trust_scores'].copy()
    
    for agent in system.all_agents:
        if agent.agent_id not in system.coordinator.trust_scores:
            print(f"警告: 智能体 {agent.name} (ID: {agent.agent_id}) 不在信任分数字典中，添加默认值")
            system.coordinator.trust_scores[agent.agent_id] = 1.0
            
    try:
        system.coordinator.update_specialists()
    except Exception as e:
        print(f"更新专家映射出错: {str(e)}")
        print("继续执行，但协调功能可能受限")

system.is_trained = True
system.main_agent.som.trained = True
system.main_agent.is_trained = True

(_, _), (x_test, y_test) = datasets.mnist.load_data()
x_test = x_test.astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

sample_idx = 43
sample = x_test[sample_idx:sample_idx+1]
true_label = y_test[sample_idx]

prediction = system.predict(sample)

plt.figure(figsize=(6, 3))
plt.imshow(x_test[sample_idx, :, :, 0], cmap='gray')
plt.title(f'Prediction: {prediction[0]}, True Label: {true_label}')
plt.axis('off')
plt.show()

quick_accuracy = system.evaluate(x_test[:500], y_test[:500])
print(f"系统在500个样本上的准确率: {quick_accuracy:.4f}")

# full_accuracy = system.evaluate(x_test, y_test)
# print(f"系统在全部测试集上的准确率: {full_accuracy:.4f}")

print(f"系统共有 {len(system.all_agents)} 个智能体:")
print(f"  - 1个主智能体")
print(f"  - {len(system.expert_agents)} 个专家智能体")