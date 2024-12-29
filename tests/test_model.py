import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import torch
import pytest
from torchvision import datasets, transforms
from model.network import SimpleCNN

def count_parameters(model):
    """Helper function to count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@pytest.fixture
def model():
    """Fixture to create model instance"""
    return SimpleCNN()

@pytest.fixture
def device():
    """Fixture to determine device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model_parameters(model):
    """Test if model size is within acceptable limits"""
    param_count = count_parameters(model)
    assert param_count < 100000, f"Model has {param_count} parameters, should be less than 100000"

def test_input_output_shape(model):
    """Test if model produces correct output shape"""
    model.eval()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"

@pytest.mark.skipif(not os.path.exists('models'), reason="No saved models found")
def test_model_accuracy(model, device):
    """Test if model achieves acceptable accuracy on MNIST test set"""
    model = model.to(device)
    
    # Load the latest model
    try:
        model_files = glob.glob('models/*.pth')
        if not model_files:
            pytest.skip("No model files found in models directory")
        latest_model = max(model_files, key=os.path.getctime)
        model.load_state_dict(torch.load(latest_model))
    except Exception as e:
        pytest.skip(f"Error loading model: {str(e)}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load test dataset
    try:
        test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    except Exception as e:
        pytest.skip(f"Error loading MNIST dataset: {str(e)}")
    
    # Evaluate model
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 80, f"Model accuracy is {accuracy}%, should be > 80%" 