import yfinance as yf
import numpy as np
import tensorflow as tf
import random
from collections import deque

def reward_function(portfolio_value, stock_owned, stock_data, current_step):
    return (portfolio_value + stock_owned * stock_data.iloc[current_step]['Open']) / 100000 - 1

class StockTradingEnv():
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.observation_space = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        self.action_space = [0, 1, 2]
        self.current_step = 0
        self.available_currency = 100000

    def step(self, action):
        self.current_step += 1
        if action == 0:  # Buy
            if self.available_currency >= self.stock_data.iloc[self.current_step]['Open']*100:
                self.available_currency -= self.stock_data.iloc[self.current_step]['Open']*100
                self.stock_owned += 100
        elif action == 1:  # Sell
            if self.stock_owned > 0:
                self.available_currency += self.stock_data.iloc[self.current_step]['Open']*100
                self.stock_owned -= 100
        else:  # Hold
            pass
        done = self.current_step == len(self.stock_data)-1
        reward = reward_function(self.available_currency, self.stock_owned, self.stock_data, self.current_step)
        return self.observation_space.iloc[self.current_step], reward, done, {}

    def reset(self):
        self.current_step = 0
        self.available_currency = 100000
        self.stock_owned = 0
        return self.observation_space.iloc[self.current_step]

stock_data = yf.download('AAPL', '2023-01-19', '2023-01-20', interval='5m')
stock_data = stock_data.reset_index()
stock_data = stock_data.dropna()

print(stock_data.head())

env = StockTradingEnv(stock_data)

inputs = tf.keras.Input(shape=(5,))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer, loss=loss_fn)

memory = deque(maxlen=1000)
epsilon = 0.1
num_episodes = 20
batch_size = 32
discount_factor = 0.99

for episode in range(num_episodes):
    obs = env.reset()
    done = False

    while not done:
        action_probs = model.predict(np.array([obs]))[0]
        if np.random.rand() < epsilon:
            action = np.random.choice(env.action_space)
        else:
            action = np.argmax(action_probs)
        next_obs, reward, done, _ = env.step(action)
        memory.append((obs, action, reward, next_obs, done))
        obs = next_obs

    batch = random.sample(memory, batch_size)

    obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = map(np.array, zip(*batch))
    obs_batch = np.array(obs_batch.tolist())
    next_obs_batch = np.array(next_obs_batch.tolist())
    action_batch = np.array(action_batch.tolist())
    reward_batch = np.array(reward_batch.tolist())
    done_batch = np.array(done_batch.tolist())

    target = model.predict(obs_batch)
    target_next = model.predict(next_obs_batch)
    target_val = target_next[np.arange(batch_size), np.argmax(target, axis=1)]

    target[np.arange(batch_size), action_batch] = reward_batch + discount_factor * target_val * (1 - done_batch)

    action_one_hot = np.zeros((batch_size, 3))
    action_one_hot[np.arange(batch_size), action_batch] = 1

    with tf.GradientTape() as tape:
        logits = model(obs_batch)
        loss = loss_fn(action_one_hot, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    model.fit(obs_batch, target, epochs=1, verbose=0)
    if episode % 10 == 0:
        model.save('model_ep' + str(episode) + '.h5')

    print('Episode: {}, Reward: {}, Portfolio Value: {}'.format(episode, reward, env.available_currency + env.stock_owned * stock_data.iloc[env.current_step]['Open']))
