import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def visualize_som_results(som, data, labels, title="SOM Visualization"):
    """
    Visualize SOM results
    
    Parameters:
        som: Trained SOM object
        data: Data mapped to SOM
        labels: Data labels
        title: Chart title
    """
    # Ensure data is 2D
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)
    
    plt.figure(figsize=(15, 10))
    
    # 1. Draw U-Matrix
    plt.subplot(2, 2, 1)
    u_matrix = som.get_distance_map()
    plt.imshow(u_matrix, cmap='viridis')
    plt.colorbar()
    plt.title('U-Matrix (Node Distances)')
    
    # 2. Draw activation heatmap
    plt.subplot(2, 2, 2)
    plt.imshow(som.activation_count, cmap='hot')
    plt.colorbar()
    plt.title('Node Activation Count')
    
    # 3. Draw class distribution
    plt.subplot(2, 2, 3)
    
    # Get BMU for all samples
    bmu_indices = np.zeros((som.map_size[0], som.map_size[1]))
    class_count = np.zeros((som.map_size[0], som.map_size[1], 10))
    
    for i, x in enumerate(data):
        bmu = som.find_bmu(x)
        class_count[bmu[0], bmu[1], labels[i]] += 1
    
    # Calculate dominant class for each node
    dominant_class = np.argmax(class_count, axis=2)
    
    # Create class distribution map
    cmap = plt.cm.get_cmap('tab10', 10)
    plt.imshow(dominant_class, cmap=cmap)
    plt.colorbar(ticks=range(10))
    plt.title('Dominant Class per Node')
    
    # 4. Draw sample distribution
    plt.subplot(2, 2, 4)
    
    # Create scatter plot showing sample distribution on SOM
    x_coords = []
    y_coords = []
    colors = []
    
    for i, sample in enumerate(data):
        bmu = som.find_bmu(sample)
        x_coords.append(bmu[1])  # Column (j)
        y_coords.append(bmu[0])  # Row (i)
        colors.append(labels[i])
    
    plt.scatter(x_coords, y_coords, c=colors, cmap='tab10', alpha=0.5, s=20)
    plt.colorbar(ticks=range(10))
    plt.title('Sample Distribution on SOM')
    plt.xlim(0, som.map_size[1]-1)
    plt.ylim(0, som.map_size[0]-1)
    plt.gca().invert_yaxis()  # Match SOM coordinate system
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def plot_training_history(history, title="Model Training History"):
    """
    Plot model training history
    
    Parameters:
        history: Training history object or history dictionary
        title: Chart title
    """
    # Handle different history object types
    if hasattr(history, 'history'):
        history_dict = history.history
    else:
        history_dict = history
    
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    if 'loss' in history_dict:
        plt.plot(history_dict['loss'], label='Training Loss')
    if 'val_loss' in history_dict:
        plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    if 'accuracy' in history_dict:
        plt.plot(history_dict['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history_dict:
        plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    """
    Plot confusion matrix
    
    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        title: Chart title
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    
    # Use seaborn to draw heatmap
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.show()
    
    # Print classification report
    print(f"Confusion Matrix:")
    print(cm)
    print(f"Overall Accuracy: {np.trace(cm) / np.sum(cm):.4f}")

def plot_novelty_detection(novelty_detector, data, labels, novel_data=None, novel_labels=None):
    """
    Visualize novelty detection results
    
    Parameters:
        novelty_detector: Novelty detector object
        data: Normal data
        labels: Normal data labels
        novel_data: Novel data
        novel_labels: Novel data labels
    """
    plt.figure(figsize=(10, 8))
    
    # Calculate novelty scores for normal samples
    normal_scores = []
    for x in data:
        _, score = novelty_detector.is_novel(x)
        normal_scores.append(score)
    
    # Plot histogram of normal sample scores
    plt.hist(normal_scores, bins=30, alpha=0.7, label='Normal Samples')
    
    # If novel samples exist, calculate and plot their scores
    if novel_data is not None:
        novel_scores = []
        for x in novel_data:
            _, score = novelty_detector.is_novel(x)
            novel_scores.append(score)
        
        plt.hist(novel_scores, bins=30, alpha=0.7, label='Novel Samples')
    
    # Draw threshold line
    if hasattr(novelty_detector, 'threshold') and novelty_detector.threshold is not None:
        plt.axvline(x=novelty_detector.threshold, color='r', linestyle='--', label='Novelty Threshold')
    
    plt.title('Novelty Score Distribution')
    plt.xlabel('Novelty Score')
    plt.ylabel('Sample Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()