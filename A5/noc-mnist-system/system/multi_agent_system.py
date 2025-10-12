import numpy as np
import time
import matplotlib.pyplot as plt

from agents.noc_agent import NOCAgent
from agents.expert_agent import ExpertAgent
from .coordinator import AgentCoordinator

class MultiAgentSystem:
    """
    NOC Multi-Agent System
    Combines multiple NOC agents into a complete classification system
    """
    
    def __init__(self, input_shape, num_classes, num_experts=0):
        """
        Initialize multi-agent system
        
        Parameters:
            input_shape: Input data shape
            num_classes: Number of classes
            num_experts: Initial number of expert agents
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Create main agent
        self.main_agent = NOCAgent(
            input_shape=input_shape,
            num_classes=num_classes,
            map_size=(15, 15),
            name="MainAgent"
        )
        
        # Create expert agents
        self.expert_agents = []
        for i in range(num_experts):
            expert = ExpertAgent(
                input_shape=input_shape,
                num_classes=num_classes,
                map_size=(8, 8),
                name=f"ExpertAgent-{i}"
            )
            self.expert_agents.append(expert)
            
        # List of all agents
        self.all_agents = [self.main_agent] + self.expert_agents
        
        # Create coordinator
        self.coordinator = AgentCoordinator(self.all_agents)
        
        # System status
        self.is_trained = False
        self.training_time = 0
        self.performance_history = []
        
        # Novel sample record
        self.novel_samples_buffer = []
        self.novel_labels_buffer = []
        self.max_novel_buffer_size = 1000
    
    def train(self, x_train, y_train, som_iterations=1000, classifier_epochs=5, val_ratio=0.1, verbose=True):
        """
        Train the multi-agent system
        
        Parameters:
            x_train: Training data
            y_train: Training labels
            som_iterations: SOM training iterations
            classifier_epochs: Classifier training epochs
            val_ratio: Validation set ratio
            verbose: Whether to print progress
        """
        start_time = time.time()
        
        if verbose:
            print("Starting multi-agent system training...")
            
        # Split training and validation sets
        if val_ratio > 0:
            val_size = int(len(x_train) * val_ratio)
            val_indices = np.random.choice(len(x_train), val_size, replace=False)
            train_indices = np.array([i for i in range(len(x_train)) if i not in val_indices])
            
            x_val = x_train[val_indices]
            y_val = y_train[val_indices]
            x_train_split = x_train[train_indices]
            y_train_split = y_train[train_indices]
        else:
            x_train_split = x_train
            y_train_split = y_train
            x_val = None
            y_val = None
            
        # Step 1: Train main agent
        if verbose:
            print("Training main agent...")
            
        # Train the main agent's SOM
        self.main_agent.train_som(x_train_split, iterations=som_iterations, verbose=verbose)
        
        # Train the main agent's classifiers
        self.main_agent.train_classifiers(x_train_split, y_train_split, epochs=classifier_epochs, verbose=verbose)
        
        # If validation set exists, update the main agent's expertise
        if x_val is not None:
            self.main_agent.update_expertise(x_val, y_val)
        
        # Step 2: Create expert agents (if needed)
        if len(self.expert_agents) == 0 and self.num_classes > 1:
            if verbose:
                print("Creating expert agents...")
                
            # Create expert agents for each class
            for cls in range(self.num_classes):
                expert = ExpertAgent(
                    input_shape=self.input_shape,
                    num_classes=self.num_classes,
                    map_size=(8, 8),
                    name=f"ExpertAgent-Class{cls}",
                    specialty_classes=[cls]
                )
                self.expert_agents.append(expert)
                self.all_agents.append(expert)
                
                # Add agent to coordinator
                self.coordinator.add_agent(expert)
        
        # Step 3: Train expert agents
        if self.expert_agents and verbose:
            print("Training expert agents...")
            
        for i, expert in enumerate(self.expert_agents):
            if expert.specialty_classes:
                # Get classes this expert is responsible for
                target_classes = expert.specialty_classes
                
                # Prepare specialized training data for each expert
                expert_samples_indices = []
                for cls in target_classes:
                    cls_indices = np.where(y_train_split == cls)[0]
                    # Use at most 1000 samples to avoid overtraining
                    if len(cls_indices) > 1000:
                        cls_indices = np.random.choice(cls_indices, 1000, replace=False)
                    expert_samples_indices.extend(cls_indices)
                
                if expert_samples_indices:
                    # Prepare training data
                    x_expert = x_train_split[expert_samples_indices]
                    y_expert = y_train_split[expert_samples_indices]
                    
                    if verbose:
                        print(f"  Training expert agent {expert.name}, samples: {len(x_expert)}")
                    
                    # Train expert's SOM
                    expert.train_som(x_expert, iterations=som_iterations//2, verbose=False)
                    
                    # Train expert's classifiers
                    expert.train_classifiers(x_expert, y_expert, epochs=classifier_epochs, verbose=False)
                    
                    # Update expert's expertise
                    if x_val is not None:
                        # Filter validation samples for this expert's classes
                        val_indices = []
                        for cls in target_classes:
                            val_indices.extend(np.where(y_val == cls)[0])
                        
                        if val_indices:
                            x_val_expert = x_val[val_indices]
                            y_val_expert = y_val[val_indices]
                            expert.update_expertise(x_val_expert, y_val_expert)
                            
        # Step 4: Update coordinator's trust scores and specialist mapping
        self.coordinator.update_specialists()
        
        self.is_trained = True
        self.training_time = time.time() - start_time
        
        if verbose:
            print(f"Training completed, time: {self.training_time:.2f} seconds")
            print(f"System contains {len(self.all_agents)} agents")
    
    def predict(self, x):
        """
        Predict class for input data
        
        Parameters:
            x: Input data
            
        Returns:
            predictions: Predicted classes
        """
        if not self.is_trained:
            raise ValueError("System must be trained before prediction")
            
        # Handle single sample
        single_sample = len(x.shape) == len(self.input_shape)
        if single_sample:
            x = np.expand_dims(x, axis=0)
            
        predictions = []
        
        for i in range(len(x)):
            sample = x[i]
            
            # Check if sample is novel
            is_novel, _ = self.main_agent.detect_novelty(sample)
            
            if is_novel:
                # If novel, collect predictions from all agents
                all_predictions = []
                
                for agent in self.all_agents:
                    if agent.is_trained:
                        pred, conf = agent.predict(sample)
                        all_predictions.append((agent.agent_id, pred, conf))
                
                # Use coordinator to aggregate predictions
                if all_predictions:
                    final_pred = self.coordinator.aggregate_predictions(all_predictions)
                    predictions.append(final_pred)
                else:
                    # If no predictions available, use main agent
                    pred, _ = self.main_agent.predict(sample)
                    predictions.append(pred)
                    
                # Add novel sample to buffer
                if len(self.novel_samples_buffer) < self.max_novel_buffer_size:
                    self.novel_samples_buffer.append(sample)
                    # Use predicted label as temporary label
                    self.novel_labels_buffer.append(predictions[-1])
            else:
                # For non-novel samples, let coordinator assign the most suitable agent
                best_agent, _ = self.coordinator.assign_sample(sample)
                
                if best_agent is not None:
                    # Use best agent for prediction
                    pred, _ = best_agent.predict(sample)
                else:
                    # If no suitable agent, use main agent
                    pred, _ = self.main_agent.predict(sample)
                    
                predictions.append(pred)
        
        # Return results
        if single_sample:
            return predictions[0]
        return np.array(predictions)
    
    def evaluate(self, x_test, y_test, verbose=True):
        """
        Evaluate system performance
        
        Parameters:
            x_test: Test data
            y_test: Test labels
            verbose: Whether to print progress
            
        Returns:
            accuracy: Accuracy
        """
        # Generate predictions
        predictions = self.predict(x_test)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == y_test)
        self.performance_history.append(accuracy)
        
        # Update coordinator's trust in each agent
        for agent in self.all_agents:
            if not agent.is_trained:
                continue
                
            # Evaluate agent's performance on its specialty classes
            if hasattr(agent, 'specialty_classes') and agent.specialty_classes:
                # Filter test samples for this agent's classes
                test_indices = []
                for cls in agent.specialty_classes:
                    test_indices.extend(np.where(y_test == cls)[0])
                    
                if test_indices:
                    x_test_agent = x_test[test_indices]
                    y_test_agent = y_test[test_indices]
                    
                    agent_predictions = agent.predict(x_test_agent)[0]
                    agent_accuracy = np.mean(agent_predictions == y_test_agent)
                    
                    # Update coordinator's trust
                    self.coordinator.update_trust(agent.agent_id, agent_accuracy)
            else:
                # Evaluate main agent
                agent_predictions = agent.predict(x_test)[0]
                agent_accuracy = np.mean(agent_predictions == y_test)
                self.coordinator.update_trust(agent.agent_id, agent_accuracy)
        
        if verbose:
            print(f"System accuracy: {accuracy:.4f}")
            print("Agent trust scores:")
            # Use dictionary to filter duplicate agents
            shown_agents = {}
            for agent in self.all_agents:
                if agent.name not in shown_agents:
                    shown_agents[agent.name] = agent
            
            # Only show unique agents
            for name, agent in shown_agents.items():
                trust = self.coordinator.trust_scores.get(agent.agent_id, 0)
                print(f"  {name}: {trust:.4f}")
        
        return accuracy
    
    def adapt(self, x_new, y_new=None, epochs=3, verbose=True):
        """
        Adapt to new data
        
        Parameters:
            x_new: New data
            y_new: New data labels (optional)
            epochs: Training epochs
            verbose: Whether to print progress
        """
        if not self.is_trained:
            raise ValueError("System must be trained before adaptation")
            
        if verbose:
            print(f"Adapting to new data ({len(x_new)} samples)...")
            
        # Handle unlabeled data
        if y_new is None:
            # Use current system to predict labels
            y_new = self.predict(x_new)
            
        # Check for novel samples
        novel_flags, _ = self.main_agent.detect_novelty(x_new)
        novel_indices = np.where(novel_flags)[0]
        
        if len(novel_indices) > 0:
            x_novel = x_new[novel_indices]
            y_novel = y_new[novel_indices]
            
            if verbose:
                print(f"Detected {len(novel_indices)} novel samples")
                
            # Add novel samples to buffer
            buffer_space = self.max_novel_buffer_size - len(self.novel_samples_buffer)
            if buffer_space > 0:
                samples_to_add = min(buffer_space, len(x_novel))
                self.novel_samples_buffer.extend(x_novel[:samples_to_add])
                self.novel_labels_buffer.extend(y_novel[:samples_to_add])
                
            # Check if new expert agents are needed
            if len(novel_indices) > 100:  # If enough novel samples
                # Analyze novel sample class distribution
                novel_class_counts = np.zeros(self.num_classes)
                for cls in range(self.num_classes):
                    novel_class_counts[cls] = np.sum(y_novel == cls)
                    
                # Find main novel classes
                main_novel_classes = []
                for cls in range(self.num_classes):
                    if novel_class_counts[cls] > 20:  # At least 20 samples
                        main_novel_classes.append(cls)
                        
                # Create new expert agents for main novel classes
                for cls in main_novel_classes:
                    # Check if expert for this class already exists
                    has_expert = False
                    for agent in self.expert_agents:
                        if hasattr(agent, 'specialty_classes') and cls in agent.specialty_classes:
                            has_expert = True
                            break
                            
                    if not has_expert:
                        if verbose:
                            print(f"Creating new expert agent for class {cls}")
                            
                        # Create new expert
                        new_expert = ExpertAgent(
                            input_shape=self.input_shape,
                            num_classes=self.num_classes,
                            map_size=(8, 8),
                            name=f"ExpertAgent-Class{cls}-New",
                            specialty_classes=[cls]
                        )
                        
                        # Get samples for this class
                        cls_indices = np.where(y_novel == cls)[0]
                        x_cls = x_novel[cls_indices]
                        y_cls = y_novel[cls_indices]
                        
                        # Train new expert
                        new_expert.train_som(x_cls, iterations=5000, verbose=False)
                        new_expert.train_classifiers(x_cls, y_cls, epochs=epochs, verbose=False)
                        
                        # Add to system
                        self.expert_agents.append(new_expert)
                        self.all_agents.append(new_expert)
                        self.coordinator.add_agent(new_expert)
        
        # Update main agent
        if len(x_new) > 0:
            # Update SOM (incremental training)
            self.main_agent.som.train(x_new, iterations=min(1000, len(x_new)*10), verbose=False)
            
            # Update classifiers
            self.main_agent.train_classifiers(x_new, y_new, epochs=epochs, verbose=False)
            
        # Update coordinator
        self.coordinator.update_specialists()
        
        if verbose:
            print(f"Adaptation complete, system now has {len(self.all_agents)} agents")
    
    def visualize_system(self):
        """Visualize system status"""
        plt.figure(figsize=(15, 10))
        
        # Plot performance history
        if self.performance_history:
            plt.subplot(2, 2, 1)
            plt.plot(self.performance_history)
            plt.title('System Accuracy History')
            plt.xlabel('Evaluation Number')
            plt.ylabel('Accuracy')
            plt.grid(True, alpha=0.3)
        
        # Plot main agent SOM
        plt.subplot(2, 2, 2)
        plt.imshow(self.main_agent.som.activation_count, cmap='hot')
        plt.colorbar()
        plt.title('Main Agent SOM Activation Heatmap')
        
        # Plot agent trust scores
        plt.subplot(2, 2, 3)
        agent_names = [agent.name for agent in self.all_agents]
        trust_values = [self.coordinator.trust_scores.get(agent.agent_id, 0) for agent in self.all_agents]
        
        # Horizontal bar chart
        y_pos = np.arange(len(agent_names))
        plt.barh(y_pos, trust_values)
        plt.yticks(y_pos, agent_names)
        plt.xlabel('Trust Score')
        plt.title('Agent Trust Scores')
        
        # Plot class specialization distribution
        plt.subplot(2, 2, 4)
        # Create class-agent matrix
        expertise_matrix = np.zeros((self.num_classes, len(self.all_agents)))
        
        for i, agent in enumerate(self.all_agents):
            if hasattr(agent, 'expertise'):
                expertise_matrix[:, i] = agent.expertise
        
        plt.imshow(expertise_matrix, cmap='Blues')
        plt.colorbar()
        plt.xlabel('Agents')
        plt.ylabel('Classes')
        plt.title('Class Specialization Distribution')
        plt.xticks(range(len(self.all_agents)), [f"A{i}" for i in range(len(self.all_agents))], rotation=90)
        plt.yticks(range(self.num_classes))
        
        plt.suptitle('Multi-Agent System Status')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

        # 添加到multi_agent_system.py中
    def save_system(self, save_path):
        """保存整个系统状态"""
        system_data = {
            'main_agent': {
                'som_weights': self.main_agent.som.weights,
                'classifiers': {}
            },
            'expert_agents': [],
            'coordinator': {
                'trust_scores': self.coordinator.trust_scores
            }
        }
        
        # 保存主智能体分类器
        for node, classifier in self.main_agent.classifiers.items():
            system_data['main_agent']['classifiers'][str(node)] = classifier.model.get_weights()
        
        # 保存专家智能体
        for expert in self.expert_agents:
            expert_data = {
                'name': expert.name,
                'specialty_classes': expert.specialty_classes,
                'som_weights': expert.som.weights,
                'classifiers': {}
            }
            
            for node, classifier in expert.classifiers.items():
                expert_data['classifiers'][str(node)] = classifier.model.get_weights()
                
            system_data['expert_agents'].append(expert_data)
        
        # 保存系统数据
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(system_data, f)
        
        print(f"系统已保存到: {save_path}")

    def load_system(self, load_path):
        """加载系统状态"""
        import pickle
        with open(load_path, 'rb') as f:
            system_data = pickle.load(f)
        
        # 加载主智能体
        self.main_agent.som.weights = system_data['main_agent']['som_weights']