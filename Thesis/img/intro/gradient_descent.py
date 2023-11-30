import matplotlib.pyplot as plt
import numpy as np
import scipy
import os

f = lambda x: 0.0007307675257*x**(6) - 0.0127292063099*x**(5) + 0.0988657296407*x**(4) - 0.5463196429977*x**(3) + 2.0917848972624*x**(2) - 3.3192068383025*x

x = np.linspace(-1,8,100)
y = f(x)

# plot f
plt.plot(x,y)

# mark minima
minima = scipy.signal.argrelextrema(y, np.less)[0]
plt.plot(x[minima], y[minima], 'ro')

# name the minima A and B
plt.text(x[minima[0]]+0.2, y[minima[0]]-0.5, 'A', fontsize=20)
plt.text(x[minima[1]]+0.2, y[minima[1]]-0.5, 'B', fontsize=20)

# hide axis
plt.axis('off')

# visualize gradent descent with a small learning rate starting at x=7.5
gradient_descent_points = [(7.8, f(7.8))]
learning_rate = 0.05
for i in range(12):
    x_new = gradient_descent_points[-1][0] - learning_rate * scipy.misc.derivative(f, gradient_descent_points[-1][0])
    gradient_descent_points.append((x_new, f(x_new)))
  
# draw red lines 
for i in range(len(gradient_descent_points)-1):
    plt.plot([gradient_descent_points[i][0], gradient_descent_points[i+1][0]], [gradient_descent_points[i][1], gradient_descent_points[i+1][1]], color='red')

# draw points
for i in range(len(gradient_descent_points)):
    plt.plot(gradient_descent_points[i][0], gradient_descent_points[i][1], marker='o', markersize=4, color='black')

#plt.show()
root_path = os.path.dirname(os.path.abspath(__file__)) 
os.chdir(os.path.join(root_path, 'out'))
plt.savefig('gradient_descent_small_lr.pdf', dpi=300, bbox_inches='tight')