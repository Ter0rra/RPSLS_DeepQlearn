# 🖖 Rock-Paper-Scissors-Lizard-Spock - Deep RL Edition

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An AI-powered web game where a Deep Q-Network learns to predict and beat human players in the classic Rock-Paper-Scissors-Lizard-Spock game from *The Big Bang Theory*.

![Game Screenshot](.\data\rpsls_demo.jpeg)

## 🎮 Overview

This project implements a **self-learning AI** that plays Rock-Paper-Scissors-Lizard-Spock (RPSLS) against human players. The AI uses **Deep Reinforcement Learning** (specifically a Deep Q-Network) to:

- 📊 Observe patterns in the player's last 10 moves
- 🎯 Predict the player's next move with increasing accuracy
- 🛡️ Counter the prediction to maximize win rate
- 📈 Continuously improve through experience replay

**After 200-500 games, the AI typically achieves 50-60% prediction accuracy, making it extremely difficult to beat!**

**Let's try it if you want on [render.com](https://rpsls-deepqlearn.onrender.com)**

## ✨ Features

### 🧠 Advanced AI
- **Deep Q-Network (DQN)** with PyTorch neural network
- **Experience replay** for stable learning
- **Epsilon-greedy exploration** that decays over time
- **Pattern recognition** on sequences of 10 moves
- **Persistent memory** - AI remembers everything between sessions

### 🎨 Beautiful UI
- **Big Bang Theory themed** with animated atoms
- **Responsive design** - works on mobile and desktop
- **Real-time statistics** showing AI performance
- **Smooth animations** and effects
- **Dark theme** with neon accents

### 🚀 Production Ready
- **Flask backend** for easy deployment
- **JSON file storage** for simplicity (can scale to database)
- **Multi-user support** - all players train the same AI
- **Auto-save** every 10 games

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Flask 3.0+
- NumPy

## 🔧 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/rpsls-deep-rl.git
cd rpsls-deep-rl
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 2.1. optionnal *check env/files/folder/app* 
```bash
python test-script.py
```

### 3. Run the application
```bash
python app.py
```

### 4. Open your browser
Navigate to: `http://localhost:5000`

## 📁 Project Structure

```
rpsls-deep-rl/
│
├── app.py                      # Flask game
├── deep_learning_agent.py      # DQN Agent implementation
├── requirements.txt            # Python dependencies
├── test-script.py              # to check all env & files
│
├── templates/
│   └── index.html              # Frontend UI
│
├── data/                       # Auto-created on first run
│   ├── dqn_model.pth           # Saved neural network weights
│   └── game_stats.json         # Game statistics
│
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## 🎯 How It Works

### Game Rules (RPSLS)
```
Rock crushes Scissors
Rock crushes Lizard
Paper covers Rock
Paper disproves Spock
Scissors cuts Paper
Scissors decapitates Lizard
Lizard eats Paper
Lizard poisons Spock
Spock vaporizes Rock
Spock smashes Scissors
```

### AI Architecture

#### 1. **Neural Network**
```python
Input Layer:  50 neurons (10 moves × 5 choices, one-hot encoded)
Hidden Layer: 128 neurons + ReLU + Dropout(0.2)
Hidden Layer: 128 neurons + ReLU + Dropout(0.2)
Hidden Layer: 64 neurons + ReLU
Output Layer: 5 neurons (Q-values for each choice)
```

#### 2. **Learning Process**
1. **Observe**: AI receives the last 10 player moves as input
2. **Predict**: Neural network outputs Q-values for each possible next move
3. **Act**: AI chooses the move that counters the highest Q-value prediction
4. **Learn**: After seeing the actual player move:
   - Reward = +1.0 if prediction was correct
   - Reward = -0.5 if prediction was wrong
   - Update Q-values using Bellman equation
   - Store experience in replay memory
5. **Train**: Sample random batch from memory and train neural network

#### 3. **Key Features**
- **Experience Replay**: Stores last 10,000 games to break correlation
- **Target Network**: Separate network updated every 50 steps for stability
- **Epsilon Decay**: Starts at 40% exploration, decays to 10%
- **Adam Optimizer**: Learning rate = 0.001
- **Gradient Clipping**: Prevents exploding gradients

## 📊 Performance

| Games Played | Prediction Accuracy | AI Win Rate | Description |
|--------------|-------------------|-------------|-------------|
| 0-100         | ~20-30%           | ~20-30%      | Random play, exploring |
| 100-500       | ~30-40%           | ~30-40%      | Starting to learn patterns |
| 500-1000      | ~40-50%           | ~40-50%      | Good pattern recognition |
| 1000+         | ~50-60%           | ~50-60%      | Very strong, hard to beat |

## 🚀 Deployment

### Option 1: Render.com (Recommended)
1. Create account on [render.com](https://render.com)
2. Create new Web Service
3. Connect your GitHub repository
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `gunicorn app:app`
6. Deploy!

### Option 2: Railway.app
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### Option 3: Heroku
```bash
# Create Procfile
echo "web: gunicorn app:app" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

## 🔬 Technical Details

### Q-Learning Formula
```
Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]

Where:
- Q(s,a) = Q-value for state s and action a
- α = learning rate (0.001)
- r = reward (+1 for correct prediction, -0.5 for wrong)
- γ = discount factor (0.95)
- s' = next state
```

### Neural Network Training
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with learning rate 0.001
- **Batch Size**: 32 experiences per training step
- **Gradient Clipping**: Max norm = 1.0
- **Target Network Update**: Every 50 games

## 🎨 Customization

### Change Pattern Length
In `app.py`:
```python
agent = DeepLearningAgent(
    pattern_length=15,  # Change from 10 to 15
    ...
)
```

### Adjust Learning Parameters
In `deep_learning_agent.py`:
```python
agent = DeepLearningAgent(
    learning_rate=0.002,    # Faster learning
    epsilon=0.5,            # More exploration
    epsilon_decay=0.999,    # Slower decay
    gamma=0.98,             # More future-focused
    ...
)
```

### Change UI Colors
In `templates/index.html`, modify CSS variables:
```css
/* Primary color: cyan */
#00d9ff → your color

/* Accent color: red */
#ff6b6b → your color
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **The Big Bang Theory** for popularizing RPSLS
- **Sam Kass** for inventing the original game
- **DeepMind** for pioneering Deep Q-Networks
- **All frontend dev's friends** for help me with the template

## 📧 Contact

Ter0rra - [Mail](mailto:terorra.ia.data@gmail.com)

Project Link: [https://github.com/yourusername/rpsls-deep-rl](https://github.com/yourusername/rpsls-deep-rl)

---

Made with 🥳 and 🧠 Deep Learning

**BAZINGA!** 🎭