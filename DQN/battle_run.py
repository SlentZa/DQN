import gym

from dqn import DQNAgent
from utils import mini_batch_train

env_id = "CartPole-v1"
MAX_EPISODES = 10000
MAX_STEPS = 500
BATCH_SIZE = 64

env = gym.make(env_id)
agent = DQNAgent(env, use_conv=False)
episode_rewards = mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)