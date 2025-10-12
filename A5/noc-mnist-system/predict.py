import numpy as np
import pickle
import matplotlib.pyplot as plt
# 使用与mnist_loader.py相同的导入方式
from keras import datasets
from system.multi_agent_system import MultiAgentSystem

input_shape = (28, 28, 1)
num_classes = 10
system = MultiAgentSystem(input_shape=input_shape, num_classes=num_classes)

with open('/Users/lqcmacmini/cursor/LIS/trained_noc_system2.pkl', 'rb') as f:
    saved_system = pickle.load(f)
    
# 3. 将保存的状态应用到系统
# 注意：这部分取决于您的save_system方法的实现
system.main_agent.som.weights = saved_system['main_agent']['som_weights']
for node_str, weights in saved_system['main_agent']['classifiers'].items():
    # 转换字符串键回元组
    node = eval(node_str)
    if node not in system.main_agent.classifiers:
        system.main_agent._create_classifier(node)
    system.main_agent.classifiers[node].model.set_weights(weights)

system.is_trained = True
# 也设置SOM为已训练状态
system.main_agent.som.trained = True
system.main_agent.is_trained = True

# 4. 加载一些测试数据 - 使用keras.datasets方式
(_, _), (x_test, y_test) = datasets.mnist.load_data()
x_test = x_test.astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

# 5. 选择一个样本并预测
sample_idx = 43
sample = x_test[sample_idx:sample_idx+1]
true_label = y_test[sample_idx]

# 进行预测
prediction = system.predict(sample)

# 6. 显示结果
plt.figure(figsize=(6, 3))
plt.imshow(x_test[sample_idx, :, :, 0], cmap='gray')
plt.title(f'Prediction: {prediction[0]}, True Label: {true_label}')
plt.axis('off')
plt.show()

# 7. 评估一部分测试集
accuracy = system.evaluate(x_test[:500], y_test[:500])
print(f"System accuracy on test set: {accuracy}")