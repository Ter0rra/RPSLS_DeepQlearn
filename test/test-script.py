#!/usr/bin/env python3
"""
Automatic installation test script for RPSLS Deep RL
Checks that everything is installed and working correctly
"""

import sys
import os
from pathlib import Path

def print_header(text):
    """Print a nice header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_success(text):
    """Print success message"""
    print(f"✅ {text}")

def print_error(text):
    """Print error message"""
    print(f"❌ {text}")

def print_info(text):
    """Print info message"""
    print(f"ℹ️  {text}")

def check_python_version():
    """Check Python version >= 3.8"""
    print_header("Checking Python Version")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print_info(f"Python version: {version_str}")
    
    if version.major >= 3 and version.minor >= 8:
        print_success("Python version is compatible (3.8+)")
        return True
    else:
        print_error(f"Python 3.8+ required, you have {version_str}")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    print_header("Checking Dependencies")
    
    required_packages = {
        'flask': 'Flask',
        'torch': 'PyTorch',
        'numpy': 'NumPy',
    }
    
    all_installed = True
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print_success(f"{name} is installed")
        except ImportError:
            print_error(f"{name} is NOT installed")
            all_installed = False
    
    if not all_installed:
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
    
    return all_installed

def check_files():
    """Check if all required files exist"""
    print_header("Checking Project Files")
    
    required_files = [
        'app.py',
        'deep_learning_agent.py',
        'requirements.txt',
        'templates/index.html',
        'README.md'
    ]
    
    all_exist = True
    
    for file in required_files:
        if Path(file).exists():
            print_success(f"{file} exists")
        else:
            print_error(f"{file} is MISSING")
            all_exist = False
    
    return all_exist

def check_data_folder():
    """Check if data folder exists or can be created"""
    print_header("Checking Data Folder")
    
    data_dir = Path("data")
    
    if data_dir.exists():
        print_success("data/ folder exists")
    else:
        try:
            data_dir.mkdir(exist_ok=True)
            print_success("data/ folder created")
        except Exception as e:
            print_error(f"Cannot create data/ folder: {e}")
            return False
    
    # Check write permissions
    test_file = data_dir / "test.txt"
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        test_file.unlink()
        print_success("data/ folder is writable")
        return True
    except Exception as e:
        print_error(f"data/ folder is not writable: {e}")
        return False

def test_agent_import():
    """Test if agent can be imported"""
    print_header("Testing AI Agent Import")
    
    try:
        from deep_learning_agent import DeepLearningAgent
        print_success("DeepLearningAgent imported successfully")
        
        # Try creating an agent
        agent = DeepLearningAgent(pattern_length=2)
        print_success("Agent instance created successfully")
        
        # Test basic functionality
        state = agent.get_state()
        print_success(f"Agent state: {state}")
        
        return True
    except Exception as e:
        print_error(f"Failed to import/create agent: {e}")
        return False

def test_flask_import():
    """Test if Flask app can be imported"""
    print_header("Testing Flask App Import")
    
    try:
        from app import app
        print_success("Flask app imported successfully")
        
        # Check routes
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        print_success(f"Found {len(routes)} routes:")
        for route in routes:
            if not route.startswith('/static'):
                print(f"   - {route}")
        
        return True
    except Exception as e:
        print_error(f"Failed to import Flask app: {e}")
        return False

def test_neural_network():
    """Test if neural network can be created"""
    print_header("Testing Neural Network")
    
    try:
        import torch
        import torch.nn as nn
        from deep_learning_agent import DQNetwork
        
        # Create network
        input_size = 10 * 5  # 10 moves, 5 choices
        network = DQNetwork(input_size, hidden_size=64)
        print_success("Neural network created successfully")
        
        # Test forward pass
        dummy_input = torch.randn(1, input_size)
        output = network(dummy_input)
        print_success(f"Forward pass successful, output shape: {output.shape}")
        
        # Check output size
        if output.shape[1] == 5:
            print_success("Output has correct size (5 Q-values)")
        else:
            print_error(f"Output size incorrect: {output.shape[1]} (expected 5)")
            return False
        
        return True
    except Exception as e:
        print_error(f"Neural network test failed: {e}")
        return False

def test_game_simulation():
    """Test a complete game simulation"""
    print_header("Testing Game Simulation")
    
    try:
        from deep_learning_agent import DeepLearningAgent
        
        agent = DeepLearningAgent(pattern_length=2)
        print_success("Agent created")
        
        # Simulate 5 games
        choices = ['rock', 'paper', 'scissors', 'lizard', 'spock']
        
        for i in range(5):
            # AI makes decision
            ai_choice, prediction = agent.choose_action()
            print_info(f"Game {i+1}: AI played {ai_choice}, predicted {prediction}")
            
            # Simulate player choice
            import random
            player_choice = random.choice(choices)
            
            # AI learns
            agent.learn_from_game(player_choice, ai_choice, prediction)
        
        stats = agent.get_stats()
        print_success(f"Simulated 5 games successfully")
        print_info(f"Final stats: {stats['games_played']} games, {stats['prediction_accuracy']:.1f}% accuracy")
        
        return True
    except Exception as e:
        print_error(f"Game simulation failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n")
    print("🧪" * 30)
    print("  RPSLS Deep RL - Installation Test Suite")
    print("🧪" * 30)
    
    results = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'Project Files': check_files(),
        'Data Folder': check_data_folder(),
        'Agent Import': test_agent_import(),
        'Flask Import': test_flask_import(),
        'Neural Network': test_neural_network(),
        'Game Simulation': test_game_simulation()
    }
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All tests passed! Your installation is ready to go!")
        print("\n👉 Next steps:")
        print("   1. Run: python app.py")
        print("   2. Open: http://localhost:5000")
        print("   3. Play and watch the AI learn!")
        print("\n📚 Need help? Check QUICK_START.md")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\n💡 Common solutions:")
        print("   - Install dependencies: pip install -r requirements.txt")
        print("   - Check file structure: ls -la")
        print("   - Read QUICK_START.md for detailed instructions")
        return 1

if __name__ == '__main__':
    sys.exit(run_all_tests())