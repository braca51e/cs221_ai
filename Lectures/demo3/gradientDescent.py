import numpy as np

############################################################
# Modeling

#points = [
#        (np.array([1, 0]), 2),
#        (np.array([1, 0]), 4),
#        (np.array([0, 1]), -1),
#]
#d = 2  # dimensionality (number of features)

true_w = np.array([1, 2, 3, 4, 5])
d = len(true_w)
points = []
for i in range(100000):
    x = np.random.randn(d)
    y = true_w.dot(x) + np.random.randn()
    points.append((x, y))

def F(w):  # TrainLoss
    return sum((w.dot(x) - y)**2 for x, y in points) / len(points)

def dF(w):  # Gradient of TrainLoss
    return sum(2 * (w.dot(x) - y) * x for x, y in points) / len(points)

def sF(w, i):  # Loss on example i
    x, y = points[i]
    return (w.dot(x) - y)**2 

def sdF(w, i):  # Gradient of Loss on example i
    x, y = points[i]
    return 2 * (w.dot(x) - y) * x


############################################################
# Algorithms

def gradientDescent(F, dF, d):
    w = np.zeros(d)
    eta = 0.01
    for t in range(500):
        value = F(w)
        gradient = dF(w)
        w = w - eta * gradient  # KEY: take a step
        print('T=%d, w=%s, F(w)=%s, dF(w)=%s' % (t, w, value, gradient))

def stochasticGradientDescent(sF, sdF, d, n):
    w = np.zeros(d)
    num_updates = 0
    for t in range(500):
        for i in range(n):
            num_updates += 1
            eta = 1.0 / num_updates
            value = sF(w, i)
            gradient = sdF(w, i)
            w = w - eta * gradient
        print('T=%d, w=%s, F(w)=%s, dF(w)=%s' % (t, w, value, gradient))

############################################################
# Main
#gradientDescent(F, dF, d)
stochasticGradientDescent(sF, sdF, d, len(points))
