import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# Define the LSTM model for sales forecasting
class SalesForecastLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(SalesForecastLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

# Define the Q-Network for dynamic pricing
class PricingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PricingQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Dynamic Pricing Agent using Deep Q-Learning
class DynamicPricingAgent:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.data['Report Date'] = pd.to_datetime(self.data['Report Date'])
        self.data.sort_values('Report Date', inplace=True)

        # Feature Engineering
        self.data['Rolling_Median_Price'] = self.data['Product Price'].rolling(window=7).median().bfill()
        self.data.fillna(0, inplace=True)

        self.min_price = self.data['Product Price'].min()
        self.max_price = self.data['Product Price'].max()

        # Initialize LSTM Model for Sales Forecasting
        self.lstm_model = SalesForecastLSTM(input_size=2)
        self.lstm_optimizer = optim.Adam(self.lstm_model.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()

        if os.path.exists("models/lstm_model.pth"):
            self.lstm_model.load_state_dict(torch.load("models/lstm_model.pth"))
            print("LSTM model loaded from disk.")

        # Q-Learning Parameters
        self.discount_factor = 0.99
        self.batch_size = 32
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # Action space: percentage changes in price
        self.actions = [-0.1, -0.05, 0, 0.05, 0.1]

        # Initialize Q-Network
        self.q_network = PricingQNetwork(state_dim=4, action_dim=len(self.actions))
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.q_loss = nn.MSELoss()

        if os.path.exists("models/q_network.pth"):
            self.q_network.load_state_dict(torch.load("models/q_network.pth"))
            print("Q-Network model loaded from disk.")

        # Experience replay memory
        self.memory = deque(maxlen=5000)

    def forecast_sales(self, price, previous_sales):
        """Use LSTM to forecast next day sales"""
        self.lstm_model.eval()
        input_data = torch.FloatTensor([[price, previous_sales]]).unsqueeze(0)
        with torch.no_grad():
            predicted_sales = self.lstm_model(input_data)
        return predicted_sales.item()

    def select_action(self, state):
        """Select an action using epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.actions))
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.q_network(state_tensor)).item()

    def update_q_network(self):
        """Train the Q-Network using experience replay"""
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state in batch:
            state_tensor = torch.FloatTensor(state)
            next_state_tensor = torch.FloatTensor(next_state)

            with torch.no_grad():
                target = reward + self.discount_factor * torch.max(self.q_network(next_state_tensor))

            predicted = self.q_network(state_tensor)[action]
            loss = self.q_loss(predicted, target)

            self.q_optimizer.zero_grad()
            loss.backward()
            self.q_optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_agent(self, episodes=1000):
        """Run simulations to train the dynamic pricing agent"""
        for _ in range(episodes):
            record = self.data.sample(1).iloc[0]
            current_price = record['Product Price']
            previous_sales = record['Total Sales']
            predicted_sales = self.forecast_sales(current_price, previous_sales)

            state = [current_price, previous_sales, record['Organic Conversion Percentage'], predicted_sales]
            action_index = self.select_action(state)

            new_price = np.clip(current_price * (1 + self.actions[action_index]), self.min_price, self.max_price)
            new_sales = self.forecast_sales(new_price, previous_sales)

            reward = (new_sales - previous_sales) / max(previous_sales, 1)
            if new_price > current_price:
                reward += 0.2  # Bonus for price increase

            next_state = [new_price, new_sales, record['Organic Conversion Percentage'], predicted_sales]
            self.memory.append((state, action_index, reward, next_state))
            self.update_q_network()

        torch.save(self.lstm_model.state_dict(), "models/lstm_model.pth")
        torch.save(self.q_network.state_dict(), "models/q_network.pth")
        print("Models have been saved.")

    def determine_optimal_price(self):
        """Determine the optimal price for the next day"""
        prices = self.data['Product Price'].unique()
        best_price = max(prices, key=lambda price: self.q_network(torch.FloatTensor([price, 1, 1, 1])).max().item())
        return best_price

# Execute the dynamic pricing strategy
if __name__ == "__main__":
    agent = DynamicPricingAgent('woolballhistory.csv')
    agent.train_agent(episodes=5000)
    optimal_price = agent.determine_optimal_price()
    print(f"Optimal Price for Tomorrow: ${optimal_price:.2f}")
