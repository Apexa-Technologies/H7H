import pandas as pd # type: ignore
import numpy as np
import time
from collections import deque
from Model import *
from Environment import Environment
import sys

MAX_MEMORY = 100_000
BATCH_SIZE = 64


class Agent:
    def __init__(self, df, args):
        self.args = args
        self.window_size = 30
        self.epsilon = 0.5
        self.gamma = 0.95

        self.lr = 1

        self.input_size = 14 # Make Dynamic
        self.hidden_size = 64
        self.output_size = 4

        self.realized_pnl = 0

        self.winning_streak = 0
        self.losing_streak = 0
        
        self.last_trade = 0

        self.memory = deque(maxlen=MAX_MEMORY)

        self.environment = Environment(df, self.window_size)
        self.model = LSTM_Q_Net(self.input_size, self.hidden_size, self.output_size)

        # Implement Dict Loading Logic

        self.trainer = QTrainer(self.model, self.lr, self.gamma, BATCH_SIZE, self.input_size, self.hidden_size, self.output_size, self.window_size)

    def run(self, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}")
            state0 = self.reset()
            done = False

            total_reward = 0
            
            while not done:
                action0 = self.get_action(state0)

                state1, profit, executed, done = self.environment.forward(action0)

                reward = self.calculate_reward(profit, executed)
                self.last_trade += 1
                self.store_experience(state1, action0, reward, state1, done)
                self.train()
                total_reward += reward
                time.sleep(0.01)
            
            self.environment.close()
            self.update_epsilon()
            self.save(epoch)
            print(f"\nEpoch: {epoch}\nReward: {total_reward}")

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return  # Not enough experiences to sample

        mini_batch = self.sample_prioritized()
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        # Convert to Tensors and reshape
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device).view(BATCH_SIZE, self.window_size, self.input_size)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device).view(BATCH_SIZE, self.window_size, self.input_size)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.bool).to(device)

        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def calculate_reward(self, profit, executed):
        reward = 0
        if profit is None:
            profit = 0

        if profit > 0:
            reward += 10
            self.losing_streak = 0
            self.winning_streak += 1

            if profit > 100:
                reward += 20

            if self.realized_pnl < 0:
                reward = reward * 0.5

        if profit < -0:
            reward += -10
            self.winning_streak = 0
            self.losing_streak += 1

            if profit < -100:
                reward += -20

            if self.losing_streak >= 5:
                self.losing_streak = 0
                reward += -15

        if executed == 2:
            self.last_trade = 0
        
        if self.last_trade >= 60:
            print("penalized")
            self.last_trade = 0
            reward += -20
        
        return reward

    def get_action(self, state):
        #state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = self.model.forward(state_tensor)
        #print(torch.tensor(q_values).numpy())
        return torch.argmax(q_values).item()

    #########################################################################################################################################

    def store_experience(self, state, action, reward, next_state, done):
        clipped_reward = np.clip(reward, -1, 1)
        self.memory.append((state, action, clipped_reward, next_state, done))

    def sample_prioritized(self):
        # Calculate absolute rewards as priorities
        priorities = np.array([abs(self.memory[i][2]) for i in range(len(self.memory))])
        # Convert NaNs or Infs to a small number (e.g., 1e-6 to avoid division by zero)
        priorities = np.nan_to_num(priorities, nan=1e-6, posinf=1e-6, neginf=1e-6)
        # If the sum of priorities is zero (rare but possible), assign uniform probability
        if np.sum(priorities) == 0:
            probs = np.ones_like(priorities) / len(priorities)  # Uniform distribution
        else:
            probs = priorities / np.sum(priorities)  # Normalize to get probabilities
        # Sample indices based on calculated probabilities
        idxs = np.random.choice(len(self.memory), BATCH_SIZE, p=probs)
        mini_batch = [self.memory[i] for i in idxs]
        return mini_batch

    def update_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * 0.99)  # Faster decay

    def reset(self):
        self.realized_pnl = 0

        self.winning_streak = 0
        self.losing_streak = 0
        
        self.last_trade = 0

        state0 = self.environment.reset()

        return state0
        
    def save(self, epoch):
        name = "Epoch-" + str(epoch) + ".pth"
        self.model.save(name)

if __name__ == "__main__":
    df = pd.read_csv('trade_data.csv')
    df = df.drop(columns=["Time"])
    
    agent = Agent(df, sys.argv)
    agent.run(epochs=100)