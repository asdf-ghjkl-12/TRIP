import numpy as np
import random
import torch


class ReplayMemory:
    def __init__(self, memory_size=100000, action_size=4, cartpole_env=True, per=False):
        if cartpole_env:
            self.states = np.zeros(shape=(memory_size, 4))
            self.next_states = np.zeros(shape=(memory_size, 4))

        self.actions = np.zeros(memory_size)
        self.rewards = np.zeros(memory_size)
        self.terminals = np.zeros(memory_size)

        self.count = 0
        self.current = 0
        self.memory_size = memory_size
        self.per = per  # Use prioritized experience replay
        self.taus = []
        self.seq = []

    # def add(self, state, reward, action, terminal, next_state):
    #     self.states[self.current] = state
    #     self.rewards[self.current] = reward
    #     self.actions[self.current] = action
    #     self.terminals[self.current] = terminal
    #     self.next_states[self.current] = next_state
    #     # self.taus.append(tau) 

    #     self.current = (self.current + 1) % self.memory_size
    #     self.count += 1
        
    #     self.seq.append([state, reward, action, terminal])
    #     if terminal: self.seq = []
    def add(self, state, reward, action, terminal, next_state):
        device = torch.device('cuda')  # Use the appropriate device
        # self.states[self.current] = state
        # self.states[self.current] = state.cpu().numpy()
        # self.states[self.current] = state if isinstance(state, np.ndarray) else state.numpy()
        # state_tensor = state.cpu()
        # state_tensor = torch.from_numpy(state).cpu()
        if torch.is_tensor(state):
            state_tensor = state.cpu()
        elif isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).cpu()
        else:
            raise ValueError("Unsupported data type: {}".format(type(state)))



        self.states[self.current] = state_tensor



        self.rewards[self.current] = reward
        self.actions[self.current] = action
        self.terminals[self.current] = terminal
        self.next_states[self.current] = next_state

        self.current = (self.current + 1) % self.memory_size
        self.count += 1

        # state_tensor = torch.from_numpy(state).float().to(device).unsqueeze(0)  # 2D tensor
        if torch.is_tensor(state):
            state_tensor = state.float().to(device).unsqueeze(0)
        elif isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().to(device).unsqueeze(0)
        else:
            raise ValueError("Unsupported data type: {}".format(type(state)))

        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device).unsqueeze(0)  # 2D tensor
        action_tensor = torch.tensor([action], dtype=torch.float32, device=device).unsqueeze(0)  # 2D tensor
        terminal_tensor = torch.tensor([terminal], dtype=torch.float32, device=device).unsqueeze(0)  # 2D tensor

        tau = torch.cat([state_tensor, reward_tensor, action_tensor, terminal_tensor], dim=-1)
        self.seq.append(tau)
        if terminal: 
            self.seq = []

    def sample(self, batch_size):
        state_batch = []
        reward_batch = []
        action_batch = []
        terminal_batch = []
        next_state_batch = []

        if self.per:
            a = 1  # TODO: implement PER
        else:  # randomly select samples from memory
            for i in range(batch_size):
                data_index = random.randint(0, self.current-1 if self.count < self.memory_size else self.memory_size-1)
                state_batch.append(self.states[data_index])
                reward_batch.append(self.rewards[data_index])
                action_batch.append(self.actions[data_index])
                terminal_batch.append(self.terminals[data_index])
                next_state_batch.append(self.next_states[data_index])

            return state_batch, reward_batch, action_batch, terminal_batch, next_state_batch
    
    def sequence(self):
        return self.seq