import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Tanh function
def tanh(x):
    return np.tanh(x)

# ReLU function
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU function
def leaky_relu(x):
    return np.maximum(0.01 * x, x)

# sign function
def sign(x):
    return np.sign(x)


# Plotting
root_path = os.path.dirname(os.path.abspath(__file__)) 
os.chdir(os.path.join(root_path, 'out'))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# remove duplicate tick label at 0
ax.yaxis.set_ticks([-1, 1, -0.5, 0.5])
ax.xaxis.set_ticks([-10, 10, -5, 5, 0])
zeroLabel = ax.xaxis.get_ticklabels()[4]
offset = matplotlib.transforms.ScaledTranslation(0.1,0, fig.dpi_scale_trans)
zeroLabel.set_transform(zeroLabel.get_transform() + offset)



z = np.linspace(-10, 10, 100)

a, = ax.plot(z, sigmoid(z), color='r')
plt.savefig('sigmoid.pdf', dpi=300, bbox_inches='tight')
a.remove()

b, = ax.plot(z, tanh(z), color='r')
plt.savefig('tanh.pdf', dpi=300, bbox_inches='tight')
b.remove()

ax.yaxis.set_ticks([5, 10, -5, -10])

c, = ax.plot(z, relu(z), color='r')
plt.savefig('relu.pdf', dpi=300, bbox_inches='tight')
c.remove()

d, = ax.plot(z, leaky_relu(z), color='r')
plt.savefig('leaky_relu.pdf', dpi=300, bbox_inches='tight')
d.remove()







