"""
Deep Q-Network (DQN) Agent for RPSLS using PyTorch
Neural network that learns to predict player patterns
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import json
from pathlib import Path

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHOICES = ['rock', 'paper', 'scissors', 'lizard', 'spock']
CHOICE_TO_IDX = {choice: idx for idx, choice in enumerate(CHOICES)}
IDX_TO_CHOICE = {idx: choice for idx, choice in enumerate(CHOICES)}

class DQNetwork(nn.Module):
    """
    Deep Q-Network for pattern recognition
    Input: One-hot encoded sequence of player moves (pattern_length * 5)
    Output: Q-values for each possible next move (5)
    """
    def __init__(self, input_size, hidden_size=128):
        super(DQNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # 5 outputs for 5 possible moves
        )
        
    def forward(self, x):
        return self.network(x)


class DeepLearningAgent:
    """
    DQN Agent that learns player patterns using neural networks
    """
    def __init__(self, 
                 pattern_length=10,
                 learning_rate=0.001,
                 epsilon=0.4,
                 epsilon_decay=0.9995,
                 epsilon_min=0.1,
                 gamma=0.95,
                 memory_size=10000,
                 batch_size=32):
        
        self.pattern_length = pattern_length
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Input size: pattern_length * 5 (one-hot encoding)
        input_size = pattern_length * 5
        
        # Neural networks
        self.policy_net = DQNetwork(input_size).to(device)
        self.target_net = DQNetwork(input_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Player history
        self.player_history = deque(maxlen=pattern_length)
        
        # Stats
        self.games_played = 0
        self.correct_predictions = 0
        self.total_predictions = 0
        self.update_counter = 0
        
    def encode_history(self, history):
        """
        One-hot encode the player history
        Returns: tensor of shape (pattern_length * 5,)
        """
        encoded = np.zeros(self.pattern_length * 5)
        
        for i, move in enumerate(history):
            idx = CHOICE_TO_IDX[move]
            encoded[i * 5 + idx] = 1.0
        
        return torch.FloatTensor(encoded).unsqueeze(0).to(device)
    
    def get_state(self):
        """Get current state as encoded tensor"""
        # Pad with random moves if not enough history
        history = list(self.player_history)
        while len(history) < self.pattern_length:
            history.insert(0, random.choice(CHOICES))
        
        return self.encode_history(history[-self.pattern_length:])
    
    def predict_player_move(self, state):
        """
        Predict next player move using epsilon-greedy policy
        Returns: predicted choice (string)
        """
        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(CHOICES)
        
        # Get Q-values from network
        with torch.no_grad():
            q_values = self.policy_net(state)
            predicted_idx = q_values.argmax().item()
        
        return IDX_TO_CHOICE[predicted_idx]
    
    def choose_action(self):
        """
        AI chooses action: predict player move and counter it
        Returns: (ai_choice, prediction)
        """
        state = self.get_state()
        prediction = self.predict_player_move(state)
        
        # Counter the prediction
        counter = self.get_counter_move(prediction)
        
        return counter, prediction
    
    def get_counter_move(self, opponent_move):
        """Find a move that beats the opponent"""
        rules = {
            'rock': ['paper', 'spock'],
            'paper': ['scissors', 'lizard'],
            'scissors': ['rock', 'spock'],
            'lizard': ['rock', 'scissors'],
            'spock': ['paper', 'lizard']
        }
        return random.choice(rules[opponent_move])
    
    def store_transition(self, state, action, reward, next_state):
        """Store transition in replay memory"""
        self.memory.append((state, action, reward, next_state))
    
    def learn_from_game(self, actual_player_choice, ai_choice, prediction):
        """
        Learn from game result
        """
        self.games_played += 1
        self.total_predictions += 1
        
        # Calculate reward based on prediction accuracy
        if prediction == actual_player_choice:
            reward = 1.0
            self.correct_predictions += 1
        else:
            reward = -0.5
        
        # Get states
        if len(self.player_history) >= self.pattern_length:
            current_state = self.get_state()
            
            # Add actual player choice to history
            self.player_history.append(actual_player_choice)
            
            next_state = self.get_state()
            
            # Store transition
            action_idx = CHOICE_TO_IDX[prediction]
            self.store_transition(current_state, action_idx, reward, next_state)
            
            # Train if enough samples
            if len(self.memory) >= self.batch_size:
                self.train_step()
        else:
            # Just add to history if not enough data yet
            self.player_history.append(actual_player_choice)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % 50 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def train_step(self):
        """
        Perform one training step using experience replay
        """
        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.cat([t[0] for t in batch])
        actions = torch.LongTensor([t[1] for t in batch]).to(device)
        rewards = torch.FloatTensor([t[2] for t in batch]).to(device)
        next_states = torch.cat([t[3] for t in batch])
        
        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
    
    def get_stats(self):
        """Return agent statistics"""
        prediction_accuracy = (self.correct_predictions / self.total_predictions * 100) if self.total_predictions > 0 else 0
        
        return {
            'games_played': self.games_played,
            'correct_predictions': self.correct_predictions,
            'total_predictions': self.total_predictions,
            'prediction_accuracy': round(prediction_accuracy, 1),
            'epsilon': round(self.epsilon, 3),
            'memory_size': len(self.memory),
            'pattern_length': self.pattern_length
        }
    
    def save_model(self, filepath):
        """Save model to file"""
        save_dict = {
            'policy_net_state': self.policy_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'games_played': self.games_played,
            'correct_predictions': self.correct_predictions,
            'total_predictions': self.total_predictions,
            'player_history': list(self.player_history)
        }
        torch.save(save_dict, filepath)
    
    def load_model(self, filepath):
        """Load model from file"""
        if not Path(filepath).exists():
            return False
        
        try:
            checkpoint = torch.load(filepath, map_location=device)
            
            self.policy_net.load_state_dict(checkpoint['policy_net_state'])
            self.target_net.load_state_dict(checkpoint['target_net_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.epsilon = checkpoint['epsilon']
            self.games_played = checkpoint['games_played']
            self.correct_predictions = checkpoint['correct_predictions']
            self.total_predictions = checkpoint['total_predictions']
            self.player_history = deque(checkpoint['player_history'], maxlen=self.pattern_length)
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


if __name__ == "__main__":
    print("🧠 Testing Deep Learning Agent...")
    
    agent = DeepLearningAgent(pattern_length=10)
    
    # Simulate games with a pattern
    print("Training on 500 games with a simple pattern...")
    
    def pattern_player(history):
        """Player that follows a pattern"""
        if len(history) >= 2:
            if history[-1] == 'rock' and history[-2] == 'rock':
                return 'paper'
            elif history[-1] == 'paper':
                return 'scissors'
        return random.choice(CHOICES)
    
    player_hist = []
    wins = 0
    
    for i in range(500):
        ai_choice, prediction = agent.choose_action()
        
        player_choice = pattern_player(player_hist)
        player_hist.append(player_choice)
        
        # Determine winner
        rules = {
            'rock': ['scissors', 'lizard'],
            'paper': ['rock', 'spock'],
            'scissors': ['paper', 'lizard'],
            'lizard': ['paper', 'spock'],
            'spock': ['rock', 'scissors']
        }
        
        if player_choice in rules[ai_choice]:
            wins += 1
        
        agent.learn_from_game(player_choice, ai_choice, prediction)
        
        if (i + 1) % 100 == 0:
            stats = agent.get_stats()
            winrate = wins / (i + 1) * 100
            print(f"\nGame {i+1}:")
            print(f"  AI Winrate: {winrate:.1f}%")
            print(f"  Prediction Accuracy: {stats['prediction_accuracy']}%")
            print(f"  Epsilon: {stats['epsilon']}")
    
    print("\n✅ Training complete!")
    print(f"Final stats: {agent.get_stats()}")