import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import time
from datetime import datetime
import os

from data.mnist_loader import load_mnist_data, get_mnist_subset, create_dynamic_dataset
from system.multi_agent_system import MultiAgentSystem
from utils.visualization import visualize_som_results, plot_confusion_matrix
from utils.evaluation import evaluate_system, evaluate_on_dynamic_data
import config

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='NOC多智能体MNIST分类系统')
    
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'test', 'adapt', 'experiment'],
                        help='运行模式')
    parser.add_argument('--experts', type=int, default=0,
                        help='初始专家智能体数量(0表示自动创建)')
    parser.add_argument('--som_iter', type=int, default=5000,
                        help='SOM训练迭代次数')
    parser.add_argument('--epochs', type=int, default=10,
                        help='分类器训练轮数')
    parser.add_argument('--samples', type=int, default=None,
                        help='每个类别的训练样本数(None表示使用全部)')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化结果')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print("多智能体MNIST分类系统")
    
    (x_train_full, y_train_full), (x_test_full, y_test_full) = load_mnist_data()
    
    '''
    max_samples = 1000 
    indices = np.random.choice(len(x_train_full), max_samples, replace=False)
    x_train = x_train_full[indices]
    y_train = y_train_full[indices]
    
    test_indices = np.random.choice(len(x_test_full), 500, replace=False)
    x_test = x_test_full[test_indices]
    y_test = y_test_full[test_indices]
    
    print(f"训练集减少到: {len(x_train)}个样本")
    print(f"测试集减少到: {len(x_test)}个样本")
    '''

    if args.samples is not None:
        samples_per_class = 500
        print(f"为每个类别选择{args.samples}个样本...")
        x_train, y_train = get_mnist_subset(
            x_train_full, y_train_full, samples_per_class=args.samples
        )
        x_test, y_test = get_mnist_subset(
            x_test_full, y_test_full, samples_per_class=min(100, args.samples)
        )
    else:
        x_train, y_train = x_train_full, y_train_full
        x_test, y_test = x_test_full, y_test_full

    input_shape = (28, 28, 1)
    num_classes = 10
    
    system = MultiAgentSystem(
        input_shape=input_shape,
        num_classes=num_classes,
        num_experts=args.experts
    )
    
    if args.mode == 'train':
        system.train(
            x_train, y_train,
            som_iterations=args.som_iter,
            classifier_epochs=args.epochs,
            verbose=True
        )
        
        print("\n评估系统...")
        system.evaluate(x_test, y_test, verbose=True)

        save_path = 'trained_noc_system2.pkl'
        system.save_system(save_path)
        print(f"系统已保存到: {save_path}")
        
        if args.visualize:
            print("\n可视化系统...")
            system.visualize_system()
            
            y_pred = system.predict(x_test)
            plot_confusion_matrix(y_test, y_pred, 
                                 class_names=[str(i) for i in range(10)],
                                 title="MNIST分类混淆矩阵")
        
        # print("\n实验结果:")
        # for i, acc in enumerate(stage_accuracies):
        #     print(f"阶段{i+1}准确率: {acc:.4f}")
        
        # if args.visualize:
        #     print("\n可视化系统...")
        #     system.visualize_system()
            
        #     plt.figure(figsize=(10, 6))
        #     plt.plot(range(1, len(stage_accuracies) + 1), stage_accuracies, marker='o')
        #     plt.title('system performance in dynamic environment')
        #     plt.xlabel('environment stage')
        #     plt.ylabel('accuracy')
        #     plt.grid(True, alpha=0.3)
        #     plt.xticks(range(1, len(stage_accuracies) + 1))
        #     plt.show()
    
    print("\n程序执行完毕")

if __name__ == "__main__":
    main()