
import torch
import torch.nn as nn
import gym
import numpy as np

class RNNAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNAutoencoder, self).__init__()
        
        self.encoder = nn.RNN(input_size, hidden_size, batch_first=True)
        self.decoder = nn.RNN(hidden_size, input_size, batch_first=True)
        
    def forward(self, x):
        encoded_output, hidden_state = self.encoder(x)
        decoded_output, _ = self.decoder(encoded_output)
        
        return encoded_output, decoded_output, hidden_state
    
# Example usage
input_size = 130
hidden_size = 512
seq_length = 10
batch_size = 1



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('Pong-ram-v0')

# Generate random input data

# input_data = torch.randn(batch_size, seq_length, input_size)

done = False

for step in range(9):
        if step == 0 or done:
            state = env.reset()
            state = state[0]
            # print("state1 :",state)
            # episode_data = []  # Reset the episode data
            episode_data = torch.empty(0, 130).to(device) 
        
        action = env.action_space.sample()
        next_state, reward, done, trun, info = env.step(action)
        # print("state2 :",next_state)

        state = np.array(state).flatten()/ 255
        # print("werw", state)
        single_data = torch.tensor(np.concatenate([state, [action, reward]]), dtype=torch.float32).to(device)
        # episode_data.append(single_data)
        # print(episode_data.shape, single_data.shape)
        episode_data = torch.cat((episode_data, single_data.unsqueeze(0)), dim=0) 


sequence_data = episode_data  # Add batch dimension

# Create an instance of the RNN autoencoder
model = RNNAutoencoder(input_size, hidden_size).to(device)

# Forward pass
encoded_output, decoded_output, hidden_state_ = model(sequence_data.unsqueeze(0))

print("Input shape:", sequence_data.unsqueeze(0).shape)
print("Encoded output shape:", encoded_output.shape)
print("Decoded output shape:", decoded_output.shape)