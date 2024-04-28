import numpy as np


class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases with small random values
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias2 = np.zeros((1, output_size))

    def print_internal_states(self, description="Internal state"):
        print(f"{description}:")
        print("Weights 1:\n", self.weights1)
        print("Bias 1:\n", self.bias1)
        print("Weights 2:\n", self.weights2)
        print("Bias 2:\n", self.bias2)

    def forward_pass(self, input_data):
        self.hidden = sigmoid(np.dot(input_data, self.weights1) + self.bias1)
        return sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)

    def backpropagation(self, input_data, actual_output, predicted_output):
        output_error = actual_output - predicted_output
        output_delta = output_error * sigmoid_derivative(predicted_output)

        hidden_error = output_delta.dot(self.weights2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden)

        self.weights1 += input_data.T.dot(hidden_delta) * 0.1
        self.weights2 += self.hidden.T.dot(output_delta) * 0.1
        self.bias1 += np.sum(hidden_delta, axis=0, keepdims=True) * 0.1
        self.bias2 += np.sum(output_delta, axis=0, keepdims=True) * 0.1

    def train(self, input_data, actual_output, iterations):
        for i in range(iterations):
            predicted_output = self.forward_pass(input_data)
            self.backpropagation(input_data, actual_output, predicted_output)
            loss = np.mean((actual_output - predicted_output) ** 2)
            if i % 10 == 0:  # Debug: Print detailed info every 10 iterations
                print(f"Iteration {i}: Loss = {loss:.5f}")
                self.print_internal_states(f"After iteration {i}")


# Example usage
if __name__ == "__main__":
    input_size = 3
    hidden_size = 5
    output_size = 1

    nn = SimpleNeuralNetwork(input_size, hidden_size, output_size)

    input_data_main = np.random.rand(4, input_size)
    actual_output_main = np.array([[0], [1], [0], [1]])

  #  nn.train(input_data_main, actual_output_main, iterations=100)

    sample_input = np.random.rand(1, input_size)
    predicted_output = nn.forward_pass(sample_input)
    print(f"Predicted Output: {predicted_output}")

    print('\nNeural Network: \n')
    nn.print_internal_states()