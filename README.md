# Reinforcement Learning for Dynamic Price Optimization

## Overview
This project implements a reinforcement learning (RL) model to optimize the pricing strategy for a product based on historical sales data. The model utilizes a combination of:
- *LSTM (Long Short-Term Memory)* for sales forecasting.
- *Deep Q-Network (DQN)* for price optimization.

The goal is to dynamically adjust the product price to maximize total sales and conversion rates while encouraging price exploration beyond historical median values.

## Features
- *Data-Driven Pricing Strategy*: Uses historical sales data to predict optimal prices.
- *Deep Reinforcement Learning*: Implements a Deep Q-Network to optimize pricing actions.
- *Time-Series Forecasting*: Uses an LSTM model to predict future sales based on price changes.
- *Exploration vs. Exploitation*: Ensures balance between testing new price points and leveraging known profitable prices.
- *Model Persistence*: Saves trained models for future use, reducing the need for retraining.

## Project Structure

├── models/                    # Directory for saved models (LSTM & Q-Network)
├── data/                      # Directory for dataset storage
├── src/fine2.py   # Main RL implementation script
├── requirements.txt           # List of dependencies
├── README.md                  # Project documentation

## Project ppt
https://www.canva.com/design/DAGeQayMZ8Y/xmyYCz0-Xg5FeSqdp7kc4Q/edit?utm_content=DAGeQayMZ8Y&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton


## Installation & Setup
### Prerequisites
Ensure you have Python 3.8+ and the required dependencies installed.

### Installation
1. Clone this repository:
   bash
   git clone https://github.com/Roshani1416/RL_Project1
   cd RL_Project1
   
2. Install dependencies:
   bash
   pip install -r requirements.txt
   
3. Place the historical sales dataset (woolballhistory.csv) in the data/ folder.

## Usage
### Train and Run the Model
To execute the RL model and obtain the optimal price prediction, run:
bash
python rl_price_optimization.py

- The model will train for a specified number of episodes and save the trained LSTM and Q-Network models in the models/ directory.
- If trained models exist, they will be loaded instead of retraining.

## Results
After execution, the script will output the predicted *Optimal Product Price for Tomorrow*, which can be used for pricing strategy decisions.

## Future Enhancements
- Incorporate additional features such as seasonality trends.
- Implement a multi-agent RL approach for broader pricing optimization.
- Extend support for multiple products.

## License
This project is licensed under the MIT License.

## Contact
For queries or contributions, feel free to reach out at *roshanisingh7900@gmail.com*.
