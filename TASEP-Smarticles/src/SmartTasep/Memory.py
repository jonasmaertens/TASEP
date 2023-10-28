from collections import deque, namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        return Transition(*zip(*random.sample(self.memory, batch_size)))

    def __len__(self):
        return len(self.memory)


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
