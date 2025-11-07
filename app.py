"""
Rock-Paper-Scissors-Lizard-Spock with Deep Reinforcement Learning
Flask web application where AI learns to beat human players
"""

from flask import Flask, render_template, jsonify, request
import json
import random
import os
from pathlib import Path
from deep_learning_agent import DeepLearningAgent

app = Flask(__name__)

# Configuration
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
MODEL_FILE = DATA_DIR / "dqn_model.pth"
STATS_FILE = DATA_DIR / "game_stats.json"

CHOICES = ['rock', 'paper', 'scissors', 'lizard', 'spock']

RULES = {
    'rock': ['scissors', 'lizard'],
    'paper': ['rock', 'spock'],
    'scissors': ['paper', 'lizard'],
    'lizard': ['paper', 'spock'],
    'spock': ['rock', 'scissors']
}

EMOJIS = {
    'rock': '🗿',
    'paper': '📄',
    'scissors': '✂️',
    'lizard': '🦎',
    'spock': '🖖'
}

# Initialize AI agent (pattern length = 10)
agent = DeepLearningAgent(
    pattern_length=10,
    learning_rate=0.001,
    epsilon=0.4,
    epsilon_decay=0.9995,
    epsilon_min=0.1
)

# Try to load existing model
if MODEL_FILE.exists():
    print("📚 Loading existing AI model...")
    agent.load_model(MODEL_FILE)
    print("✅ Model loaded successfully!")
else:
    print("🆕 Starting with fresh AI model")

# ==================== GAME STATS ====================

def load_game_stats():
    """Load game statistics"""
    if STATS_FILE.exists():
        with open(STATS_FILE, 'r') as f:
            return json.load(f)
    return {
        'total_games': 0,
        'player_wins': 0,
        'ai_wins': 0,
        'ties': 0
    }

def save_game_stats(stats):
    """Save game statistics"""
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)

# ==================== GAME LOGIC ====================

def determine_winner(player, ai):
    """Determine the winner"""
    if player == ai:
        return 'tie'
    if ai in RULES[player]:
        return 'player'
    return 'ai'

def get_explanation(player, ai, winner):
    """Get game result explanation"""
    if winner == 'tie':
        return f"It's a tie! Both played {EMOJIS[player]} {player.capitalize()}"
    elif winner == 'player':
        return f"You win! {EMOJIS[player]} {player.capitalize()} beats {EMOJIS[ai]} {ai.capitalize()}"
    else:
        return f"AI wins! {EMOJIS[ai]} {ai.capitalize()} beats {EMOJIS[player]} {player.capitalize()}"

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    """Get current statistics"""
    game_stats = load_game_stats()
    agent_stats = agent.get_stats()
    
    # Calculate win rates
    total = game_stats['total_games']
    ai_winrate = (game_stats['ai_wins'] / total * 100) if total > 0 else 0
    player_winrate = (game_stats['player_wins'] / total * 100) if total > 0 else 0
    
    return jsonify({
        'total_games': game_stats['total_games'],
        'player_wins': game_stats['player_wins'],
        'ai_wins': game_stats['ai_wins'],
        'ties': game_stats['ties'],
        'ai_winrate': round(ai_winrate, 1),
        'player_winrate': round(player_winrate, 1),
        'prediction_accuracy': agent_stats['prediction_accuracy'],
        'epsilon': agent_stats['epsilon'],
        'memory_size': agent_stats['memory_size'],
        'pattern_length': agent_stats['pattern_length'],
        'correct_predictions': agent_stats['correct_predictions'],
        'total_predictions': agent_stats['total_predictions']
    })

@app.route('/api/play', methods=['POST'])
def play():
    """Play a round"""
    data = request.json
    player_choice = data.get('choice')
    
    if player_choice not in CHOICES:
        return jsonify({'error': 'Invalid choice'}), 400
    
    # AI makes decision
    ai_choice, prediction = agent.choose_action()
    
    # Determine winner
    winner = determine_winner(player_choice, ai_choice)
    explanation = get_explanation(player_choice, ai_choice, winner)
    
    # AI learns from this game
    agent.learn_from_game(player_choice, ai_choice, prediction)
    
    # Update game stats
    game_stats = load_game_stats()
    game_stats['total_games'] += 1
    
    if winner == 'player':
        game_stats['player_wins'] += 1
    elif winner == 'ai':
        game_stats['ai_wins'] += 1
    else:
        game_stats['ties'] += 1
    
    save_game_stats(game_stats)
    
    # Save AI model every 10 games
    if game_stats['total_games'] % 10 == 0:
        agent.save_model(MODEL_FILE)
        print(f"💾 Model saved at game {game_stats['total_games']}")
    
    return jsonify({
        'player_choice': player_choice,
        'ai_choice': ai_choice,
        'winner': winner,
        'explanation': explanation,
        'prediction': prediction,
        'prediction_correct': prediction == player_choice,
        'player_emoji': EMOJIS[player_choice],
        'ai_emoji': EMOJIS[ai_choice],
        'prediction_emoji': EMOJIS[prediction]
    })

@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset the game and AI"""
    global agent
    
    # Delete saved files
    if MODEL_FILE.exists():
        MODEL_FILE.unlink()
    if STATS_FILE.exists():
        STATS_FILE.unlink()
    
    # Create new agent
    agent = DeepLearningAgent(
        pattern_length=10,
        learning_rate=0.001,
        epsilon=0.4,
        epsilon_decay=0.9995,
        epsilon_min=0.1
    )
    
    print("🔄 Game and AI reset!")
    
    return jsonify({'success': True, 'message': 'Game reset successfully'})

if __name__ == '__main__':
    print("=" * 60)
    print("🎮 Rock-Paper-Scissors-Lizard-Spock - DQN Edition")
    print("=" * 60)
    print(f"🧠 AI Pattern Length: {agent.pattern_length} moves")
    print(f"📊 Total Games Played: {agent.games_played}")
    print(f"🎯 Prediction Accuracy: {agent.get_stats()['prediction_accuracy']}%")
    print("=" * 60)
    print("🚀 Starting server...")
    print("📍 Open http://localhost:5000 in your browser")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)