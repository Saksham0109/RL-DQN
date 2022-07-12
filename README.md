# Taxi
Implemented DQN on Taxi Environment and a custom made Environment using Stable Baselines3.

Being new to RL,I first decided to implement DQN(RL algorithm) to the well known Taxi environment.

Then i made a custom environment involving a delivery system that the agent had to learn.

I tried implementing DQN on this alongside finetuning the reward function.

The agent showed significant improvement which i have saved in the tensorboard files.

## TaxiEnv

Run.ipynb:
Can be used to run the pre built model.
If you just want to see the model work you just need to clone this entire repository and run this file.

img:
Contains the images required for rendering of the environment

Taxi.py:
Contains code of the Taxi environment

test.ipynb:
Training a DQN model on the environment

DQNTaxi.zip:
Contains the prebuilt DQN model trained by me on taxi environment

## CustomEnv

environment.yml:
Environment file that need to be replicated.

env_v0.py:
The custom environment.The environment consists of boxes which the agent has to pick and deliver at the delivery location

test.ipynb:
Training a DQN model on the environment

DQNEnvPlot:
Contains the tensorboard file showing improvement in the agent as it gets trained.
