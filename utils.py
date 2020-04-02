import numpy as np

def soft_update(local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class StateAggregator():
    def __init__(self, default, repeats):
        self.data = [default for i in range(repeats)]

    def push(self, value):
        self.data.append(value)
        self.data = self.data[1:]

    def to_input(self):
        return np.hstack(self.data)