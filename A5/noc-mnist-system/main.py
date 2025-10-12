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
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    print("=" * 50)
    print("NOC多智能体MNIST分类系统")
    print("=" * 50)
    
    # 加载MNIST数据
    print("\n加载MNIST数据...")
    (x_train_full, y_train_full), (x_test_full, y_test_full) = load_mnist_data()
    
    '''
    # 添加这段代码 - 不管命令行参数，强制限制样本数
    print("使用快速训练设置...")
    max_samples = 1000  # 总共只使用1000个样本
    indices = np.random.choice(len(x_train_full), max_samples, replace=False)
    x_train = x_train_full[indices]
    y_train = y_train_full[indices]
    
    # 测试集也减少样本
    test_indices = np.random.choice(len(x_test_full), 500, replace=False)
    x_test = x_test_full[test_indices]
    y_test = y_test_full[test_indices]
    
    print(f"训练集减少到: {len(x_train)}个样本")
    print(f"测试集减少到: {len(x_test)}个样本")
    '''

    # 如果指定了每类样本数，创建子集
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

    # 创建系统
    input_shape = (28, 28, 1)
    num_classes = 10
    
    system = MultiAgentSystem(
        input_shape=input_shape,
        num_classes=num_classes,
        num_experts=args.experts
    )
    
    # 根据模式执行不同操作
    if args.mode == 'train':
        # 训练模式
        print("\n训练多智能体系统...")
        system.train(
            x_train, y_train,
            som_iterations=args.som_iter,
            classifier_epochs=args.epochs,
            verbose=True
        )
        
        # 评估系统
        print("\n评估系统...")
        system.evaluate(x_test, y_test, verbose=True)

        # 添加保存功能
        save_path = 'trained_noc_system2.pkl'
        system.save_system(save_path)
        print(f"系统已保存到: {save_path}")
        
        # 可视化
        if args.visualize:
            print("\n可视化系统...")
            system.visualize_system()
            
            # 绘制混淆矩阵
            y_pred = system.predict(x_test)
            plot_confusion_matrix(y_test, y_pred, 
                                 class_names=[str(i) for i in range(10)],
                                 title="MNIST分类混淆矩阵")
    
    elif args.mode == 'test':
        print("测试模式尚未实现，请先训练系统")
        
    elif args.mode == 'adapt':
        print("适应模式尚未实现，请先训练系统")
        
    elif args.mode == 'experiment':
        # 实验模式：测试系统在动态环境中的表现
        print("\n执行动态环境实验...")
        
        # 创建动态数据集
        train_stages, test_stages = create_dynamic_dataset(
            x_train, y_train, x_test, y_test,
            stages=config.EXPERIMENT_CONFIG['dynamic_stages']
        )
        
        # 训练系统
        print("\n阶段1：初始训练...")
        system.train(
            train_stages[0][0], train_stages[0][1],
            som_iterations=args.som_iter,
            classifier_epochs=args.epochs,
            verbose=True
        )
        
        # 初始评估
        initial_accuracy = system.evaluate(
            test_stages[0][0], test_stages[0][1], verbose=True
        )
        
        # 执行后续阶段
        stage_accuracies = [initial_accuracy]
        
        for i in range(1, len(test_stages)):
            print(f"\n阶段{i+1}：适应新环境...")
            
            # 适应新数据
            system.adapt(
                train_stages[i][0], train_stages[i][1],
                epochs=config.ADAPTATION_CONFIG['adaptation_epochs'],
                verbose=True
            )
            
            # 评估新环境
            accuracy = system.evaluate(
                test_stages[i][0], test_stages[i][1], verbose=True
            )
            stage_accuracies.append(accuracy)
        
        # 显示结果
        print("\n实验结果:")
        for i, acc in enumerate(stage_accuracies):
            print(f"阶段{i+1}准确率: {acc:.4f}")
        
        # 可视化
        if args.visualize:
            print("\n可视化系统...")
            system.visualize_system()
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(stage_accuracies) + 1), stage_accuracies, marker='o')
            plt.title('system performance in dynamic environment')
            plt.xlabel('environment stage')
            plt.ylabel('accuracy')
            plt.grid(True, alpha=0.3)
            plt.xticks(range(1, len(stage_accuracies) + 1))
            plt.show()
    
    print("\n程序执行完毕")

if __name__ == "__main__":
    main()