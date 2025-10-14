import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras import datasets
from system.multi_agent_system import MultiAgentSystem

input_shape = (28, 28, 1)
num_classes = 10
system = MultiAgentSystem(input_shape=input_shape, num_classes=num_classes)

with open('/Users/lqcmacmini/cursor/LIS/trained_noc_system2.pkl', 'rb') as f:
    saved_system = pickle.load(f)
    
system.main_agent.som.weights = saved_system['main_agent']['som_weights']
for node_str, weights in saved_system['main_agent']['classifiers'].items():
    node = eval(node_str)
    if node not in system.main_agent.classifiers:
        system.main_agent._create_classifier(node)
    system.main_agent.classifiers[node].model.set_weights(weights)

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

accuracy = system.evaluate(x_test[:500], y_test[:500])
print(f"System accuracy on test set: {accuracy}")