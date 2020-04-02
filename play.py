import numpy as np

from unityagents import UnityEnvironment
from tqdm import tqdm

from agent import Agent
from utils import StateAggregator

# Create and setup environment
env = UnityEnvironment(file_name="./Banana.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)
max_moves = 300

# Print environment stats
print('Number of agents:', len(env_info.agents))
print('Number of actions:', action_size)
print('States look like:', state)
print('States have length:', state_size)

def get_action(state):
    global action_size
    return np.random.randint(action_size)

# Play
frames = 4
agent = Agent(state_size*frames, action_size)
env_info = env.reset(train_mode=False)[brain_name] 
state = env_info.vector_observations[0]             # get the current state
score = 0                                           # initialize the score
moves=0
state_agg = StateAggregator(state, frames)
progress = tqdm(desc='Playing', total=max_moves)
while True:
    action = agent.act(state_agg.to_input(), 0)    # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score

    state_agg.push(next_state) 

    progress.update()
    moves+=1

    if done:                                       # exit loop if episode finished
        break

env.close()
print(f"Score: {score}")
print(f"Moves made: {moves}")