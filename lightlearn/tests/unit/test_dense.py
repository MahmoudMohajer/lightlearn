import pytest
import numpy as np

# Define fixtures for common setup
@pytest.fixture
def dense_layer():
    return Dense(input_size=3, output_size=2)

@pytest.fixture
def input_data():
    return np.random.randn(3, 1)

def test_forward(dense_layer, input_data):
    result = dense_layer.forward(input_data)
    assert result.shape == (2, 1)

def test_backward(dense_layer, input_data):
    output_gradient = np.random.randn(2, 1)
    learning_rate = 0.01

    # Store initial weights and biases for later comparison
    initial_weights = dense_layer.weights.copy()
    initial_bias = dense_layer.bias.copy()

    # Perform backward pass
    dense_layer.backward(output_gradient, learning_rate)

    # Check if weights and biases are updated
    assert not np.array_equal(dense_layer.weights, initial_weights)
    assert not np.array_equal(dense_layer.bias, initial_bias)

def test_backward_output_shape(dense_layer, input_data):
    output_gradient = np.random.randn(2, 1)
    learning_rate = 0.01

    result = dense_layer.backward(output_gradient, learning_rate)

    # Check if the output shape matches the input shape
    assert result.shape == input_data.shape

# You can add more tests as needed
