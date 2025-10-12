import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_system(system, x_test, y_test, verbose=True):
    """
    评估系统性能
    
    参数:
        system: 多智能体系统对象
        x_test: 测试数据
        y_test: 测试标签
        verbose: 是否打印详细信息
        
    返回:
        results: 包含各种性能指标的字典
    """
    # 记录开始时间
    start_time = time.time()
    
    # 生成预测
    y_pred = system.predict(x_test)
    
    # 计算推理时间
    inference_time = time.time() - start_time
    
    # 计算性能指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # 计算每个类别的准确率
    class_accuracy = {}
    for cls in np.unique(y_test):
        cls_idx = (y_test == cls)
        cls_acc = accuracy_score(y_test[cls_idx], y_pred[cls_idx])
        class_accuracy[cls] = cls_acc
    
    # 存储结果
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_accuracy': class_accuracy,
        'inference_time': inference_time,
        'samples': len(y_test)
    }
    
    # 打印结果
    if verbose:
        print(f"系统评估结果 (样本数: {len(y_test)}):")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  推理时间: {inference_time:.4f}秒 ({inference_time/len(y_test)*1000:.2f}毫秒/样本)")
        print(f"  类别准确率:")
        for cls, acc in class_accuracy.items():
            print(f"    类别 {cls}: {acc:.4f}")
    
    return results

def compare_models(models, model_names, x_test, y_test):
    """
    比较多个模型的性能
    
    参数:
        models: 模型列表
        model_names: 模型名称列表
        x_test: 测试数据
        y_test: 测试标签
        
    返回:
        results: 包含各个模型性能的字典
    """
    results = {}
    
    for model, name in zip(models, model_names):
        print(f"\n评估模型: {name}")
        model_results = evaluate_system(model, x_test, y_test, verbose=False)
        results[name] = model_results
        
        # 打印摘要
        print(f"  准确率: {model_results['accuracy']:.4f}")
        print(f"  F1分数: {model_results['f1']:.4f}")
        print(f"  推理时间: {model_results['inference_time']:.4f}秒")
    
    return results

def evaluate_on_dynamic_data(system, data_stages, stage_names=None):
    """
    在动态变化的数据上评估系统
    
    参数:
        system: 多智能体系统对象
        data_stages: 数据阶段列表，每个元素是(x_test, y_test)
        stage_names: 阶段名称列表
        
    返回:
        stage_results: 每个阶段的评估结果
    """
    if stage_names is None:
        stage_names = [f"阶段{i+1}" for i in range(len(data_stages))]
    
    stage_results = {}
    
    for (x_test, y_test), name in zip(data_stages, stage_names):
        print(f"\n评估阶段: {name}")
        results = evaluate_system(system, x_test, y_test, verbose=True)
        stage_results[name] = results
    
    return stage_results