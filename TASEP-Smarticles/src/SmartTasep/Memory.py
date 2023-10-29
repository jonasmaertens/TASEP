from collections import deque, namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def transpose(batch):
    return Transition(*zip(*batch))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.buffer = [None] * capacity
        self.max_size = capacity
        self.index = 0
        self.size = 0

    def push(self, *args):
        """Save a transition"""
        self.buffer[self.index] = Transition(*args)
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        # return Transition(*zip(*random.sample(self.memory, batch_size)))
        indices = random.sample(range(self.size), batch_size)
        return transpose([self.buffer[index] for index in indices])

    def __len__(self):
        return self.size


"""
TODO: (by GitHub Copilot)
One possible way to improve the speed of this class is to use a list instead of a deque for storing the transitions.
Lists are generally faster than deques for random access, which is important for the sample method. However, using a
list may require additional code to handle the case where the buffer is full and new transitions need to overwrite
old ones.

Another way to improve the speed of this class is to use a numpy array instead of a namedtuple for storing the
transitions. Numpy arrays are generally faster than namedtuples for numerical operations, which can be important for
large-scale reinforcement learning problems. However, using a numpy array may require additional code to handle the
case where the buffer is not yet full and the sample method needs to return a smaller batch size.

Finally, you could experiment with different data structures and algorithms for implementing the replay buffer,
such as using a priority queue or a binary heap for efficient sampling of transitions with high priority. However,
these approaches may be more complex to implement and may not be necessary for smaller-scale reinforcement
learning problems.

Also: https://discuss.pytorch.org/t/how-to-make-the-replay-buffer-more-efficient/80986
"""
