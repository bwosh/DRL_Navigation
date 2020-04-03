# DQN algorithm
Deep Q network is an algorithm that tries to solve given environment using trial and error approach. The task is to approximate function that maximizes expected overall result.

The task of funcion Q is to return estimation of what move is the best given the current state of an environment.

While training implementation uses epsilon-greedy approarch that addresses probability of exploring new posibilities vs expoiting the best known results. The epsilon value (proportion of wht percent of moves is used for exploration) is decreasing over time since belief that function knows about many possibilities around increases.

Implementation uses two similar networks: 
* first that trains after set of moves inside of environment
* second that snapshots the knowledge and is used to protect training from staying in one place.

While training the network uses replay buffer that stores infrmation about N last moves. Stored values are sampled during training so they not tend to overfit the last seen sequence.

# Approach
Disclaimer: Used implementation uses some parts of DQN implementation from [Udacity sample](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn).

To send more information about the movement and dynamics the network is provided with de data from last 4 frames. 

# Model
Fully connected network is used to implement Q network. It's built using 3 fully connected layers:
* First : Input size of 4 * lenght of observation with ReLU activation
* Second : Using 64 neurons with ReLU activation
* Third : Using 32 neurons with ReLU activation, returning probability distribution in lenght of action space.


# Training parameters
Maximum numbers of episodes to train was set to:  
**max_episodes = 5000**

Epsilon-greedy method parameter was decaying over time starting from:
**eps_start = 1.0** ending at  **eps_end = 0.01** with decay over episodes equal to **eps_decay = 0.999**. The dacay values was chosen to approximately equal to 0.1 at last episode.


Parameters of training the network were:  
* Learning rate : 5e-4
* batch size: 32
* replay buffer size: 2**14
* backpropagation was used every **4** moves
* Target network was updated with local network parameter values with interpolation value of: 2e-3
* Number of frames on every input: 4
* Environment solve value 13.0 (over last 100 episodes)

# Training results
With parameters described above the environment was solved in 2204 episodes.  


![Play score results](./assets/train_DQN.png)  
(The plot was created using [this script](./report_preparation.ipynb))

Final model can be found here: [model.pth](model.pth)

# Play results
After the network was trained 100 episodes of play were launched (using [model.pth](model.pth)) achieving average score of 14.9 points.  

![Play score results](./assets/play_DQN.png)  
(The plot was created using [this script](./report_preparation.ipynb))

# Possible improvements
Generally current solution can be improved. Improvement can cause to either:
* solve environment quicker
* achieve better averag result

It could be done by:
* searching of the better hiperparameters (such as: number of frames)
* using bigger network that is able to store more spatial information
* use longer episodes to gather more information about the environment
*and* tweak penality function : add penalty for moving with no banana capture
* use RGB image that contains more information about visible bananas than sumilated probes used for this task
* more modern algorithms could be considered to train the network