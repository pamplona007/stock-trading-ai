import yfinance as yf
import tensorflow as tf
import time
from datetime import datetime
from datetime import timedelta
from collections import deque
import numpy as np
import random

stock_symbol = 'AAPL'
time_interval = '1m'

model = tf.keras.models.load_model('stock_trading_model')

polling_interval = 10

def reward_function(portfolio_value, stock_owned, stock_data):
    return (portfolio_value + stock_owned * stock_data.iloc[-1]['Open']) / 100000 - 1

class StockTradingEnv():
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.observation_space = self.stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        self.action_space = [0, 1, 2]
        self.available_currency = 100000

    def step(self, stock_data, action):
        self.stock_data = stock_data

        if action == 0:
            if self.available_currency >= self.stock_data.iloc[-1]['Open']*100:
                self.available_currency -= self.stock_data.iloc[-1]['Open']*100
                self.stock_owned += 100
        elif action == 1:
            if self.stock_owned > 0:
                self.available_currency += self.stock_data.iloc[-1]['Open']*100
                self.stock_owned -= 100
        else:
            pass
        reward = reward_function(self.available_currency, self.stock_owned, self.stock_data)
        return self.observation_space.iloc[-1], reward

    def reset(self):
        self.available_currency = 100000
        self.stock_owned = 0
        return self.observation_space.iloc[-1]

stock_data = yf.download(stock_symbol, datetime.now() - timedelta(days=1), datetime.now(), interval=time_interval)
stock_data = stock_data.reset_index()
stock_data = stock_data.dropna()

env = StockTradingEnv(stock_data)
obs = env.reset()

memory = deque(maxlen=1000)
iteractions = 0
epsilon = 0.1
batch_size = 32
discount_factor = 0.99
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

while True:
    stock_data = yf.download(stock_symbol, datetime.now() - timedelta(days=1), datetime.now(), interval=time_interval)
    stock_data = stock_data.reset_index()
    stock_data = stock_data.dropna()

    action_probs = model.predict(np.array([obs]))[0]

    if np.random.rand() < epsilon:
        action = np.random.choice(env.action_space)
    else:
        action = np.argmax(action_probs)

    next_obs, reward = env.step(stock_data, action)

    print(stock_data.iloc[-1])
    print('Iteration: {}, Action: {}, Portfolio Value: {}, Stock Owned: {}'.format(iteractions, action, env.available_currency + env.stock_owned * stock_data.iloc[-1]['Open'], env.stock_owned))

    memory.append((obs, action, reward, next_obs))

    obs = next_obs

    if len(memory) > batch_size:
        iteractions += 1

        batch = random.sample(memory, batch_size)

        obs_batch, action_batch, reward_batch, next_obs_batch = zip(*batch)

        obs_batch = np.array(obs_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_obs_batch = np.array(next_obs_batch)

        target = reward_batch + discount_factor * np.amax(model.predict(next_obs_batch), axis=1)
        target_full = model.predict(obs_batch)

        for i, action in enumerate(action_batch):
            target_full[i][action] = target[i]

        with tf.GradientTape() as tape:
            all_action_probs = model(obs_batch)
            loss_value = loss_fn(target_full, all_action_probs)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        model.save('real_time_trading_model')

        if iteractions % 100 == 0:
            epsilon = epsilon * 0.99
            print('Iteration: {}, Loss: {:.4f}'.format(iteractions, loss_value))

        if iteractions % 250 == 0:
            model.save('real_time_trading_model_{}'.format(iteractions))

    time.sleep(polling_interval)
