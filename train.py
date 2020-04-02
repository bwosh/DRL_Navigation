import numpy as np

from tqdm import tqdm
from unityagents import UnityEnvironment

from agent import Agent
from utils import StateAggregator

# Parameters
max_moves = 400
max_episodes = 5000
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.999

lr=5e-4
update_every=4
batch_size=32
buffer_size=2**14
gamma = 0.99
tau = 2e-3
frames = 4
solve_value = 13.0

# Create and setup environment
env = UnityEnvironment(file_name="./Banana.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

def get_action(state):
    global action_size
    return np.random.randint(action_size)

# Play episodes
agent = Agent(state_size*frames, action_size, lr=lr, update_every=update_every, batch_size=batch_size, buffer_size=buffer_size, gamma = gamma,tau = tau)
scores = []
eps = eps_start
print("=== TRAINING ===")
progress = tqdm(range(max_episodes), desc='Episodes')
for episode in progress:
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                           # initialize the score
    moves = 0
    state_agg = StateAggregator(state, frames)
    next_state_agg= StateAggregator(state, frames)
    while True:
        # take & perform action
        action = agent.act(state_agg.to_input(), eps)                 
        env_info = env.step(action)[brain_name]        
        
        next_state_agg.push(env_info.vector_observations[0])
        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        # step
        agent.step(state_agg.to_input(), action, reward, next_state_agg.to_input(), done)
        
        # update information
        score += reward                               
        state_agg.push(env_info.vector_observations[0])                  
        moves+=1

        # check for end of episode
        if done or moves > max_moves:
            scores.append(score)
            mscore = np.mean(scores)
            mscore100 = np.mean(np.array(scores)[-100:])
            progress.set_description(f"Last:{score:.1f}, Mean:{mscore:.3f} Mean100:{mscore100:.3f} ")
            eps = max(eps_end, eps_decay*eps)
            break
    if (episode-1) % 100 ==0:
        agent.save()
    if mscore100 >= solve_value:
        print(f"Environment solved in:  {episode+1} episodes.")
        break

env.close()
np.save('scores.npy', np.array(scores))
print(f"Last score: {scores[-1]}")
agent.save()