from stable_baselines3 import PPO
from env_v0 import *
import sys

num_agt = 1
num_box = 1


env = EnvPlot(num_agt,num_box,(10,10))
print("1")
model = PPO(policy = "CnnPolicy",env =  env, verbose=1)
print("2")
model.learn(total_timesteps=1000)
print("3")
obs = env.reset()
for i in range(100):
    action, _state = model.predict(obs, deterministic=True)
    print(action)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset() 