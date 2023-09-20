import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import random
import math
import time
from grid import CustomGridEnv as GridEnv
from main import Policy
from main import get_observation_data

# Set up the environment
env = GridEnv(grid_size=5)
env.reset()

# Load the policy
policy = Policy()
model_path = 'C:/Users/jpedr/OneDrive/Área de Trabalho/projetos/Trabalho_3_RNP/modelos/policy9900.pth'
loaded_model = torch.load(model_path)
loaded_model.eval()  # Set the model to evaluation mode

# Interaction loop with the environment
state = env.reset()
state = get_observation_data(state)
state = torch.tensor(state, dtype=torch.float32)

for t in range(1000):
    with torch.no_grad():
        action = loaded_model(state)
        action = action.argmax().item()
    state, reward, done, _ = env.step(action)
    state = get_observation_data(state)
    state = torch.tensor(state, dtype=torch.float32)

    # Render the environment
    env.render()
    time.sleep(0.1)

    if done:
        break

env.close()
