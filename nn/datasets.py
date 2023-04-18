import numpy as np
import random

def data_circle(num_samples, noise):
    """
    Generates the two circles dataset with the given number of samples and noise
    :param num_samples: total number of samples
    :param noise: noise percentage (0 .. 50)
    :return: None
    """
    radius = 5

    def get_circle_label(x, y, xc, yc):
        return 1 if np.sqrt((x - xc)**2 + (y - yc)**2) < (radius * 0.5) else 0

    noise *= 0.01
    points = np.zeros([num_samples, 2])
    labels = np.zeros(num_samples).astype(int)
    # Generate positive points inside the circle.
    for i in range(num_samples // 2):
        r = random.uniform(0, radius * 0.5)
        angle = random.uniform(0, 2 * np.pi)
        x = r * np.sin(angle)
        y = r * np.cos(angle)
        noise_x = random.uniform(-radius, radius) * noise
        noise_y = random.uniform(-radius, radius) * noise
        labels[i] = get_circle_label(x + noise_x, y + noise_y, 0, 0)
        points[i] = (x, y)
    # Generate negative points outside the circle.
    for i in range(num_samples // 2, num_samples):
        r = random.uniform(radius * 0.7, radius)
        angle = random.uniform(0, 2 * np.pi)
        x = r * np.sin(angle)
        y = r * np.cos(angle)
        noise_x = random.uniform(-radius, radius) * noise
        noise_y = random.uniform(-radius, radius) * noise
        labels[i] = get_circle_label(x + noise_x, y + noise_y, 0, 0)
        points[i] = (x, y)
    return points, labels

def data_spiral(num_samples, noise):
    """
    Generates the spiral dataset with the given number of samples and noise
    :param num_samples: total number of samples
    :param noise: noise percentage (0 .. 50)
    :return: None
    """
    noise *= 0.01
    '''
    half = num_samples // 2
    points = np.zeros([num_samples, 2])
    labels = np.zeros(num_samples).astype(int)
    for j in range(num_samples):
        i = j % half
        label = 1
        delta = 0
        if j >= half:  # negative examples
            label = 0
            delta = np.pi
        r = i / half * 5
        t = 1.75 * i / half * 2 * np.pi + delta
        x = r * np.sin(t) + random.uniform(-1, 1) * noise
        y = r * np.cos(t) + random.uniform(-1, 1) * noise
        labels[j] = label
        points[j] = (x, y)
    '''
    n = np.sqrt(np.random.rand(num_samples,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(num_samples,1) * noise
    d1y = np.sin(n)*n + np.random.rand(num_samples,1) * noise
    points = np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y))))
    labels = np.hstack((np.zeros(num_samples, dtype=np.int64),np.ones(num_samples, dtype=np.int64)))

    return points, labels

def grid_points():
    """
    Generates grid points as the test set
    """
    x_min, x_max = -15, 15  # grid x bounds
    y_min, y_max = -15, 15  # grid y bounds
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    x = np.c_[xx.ravel(),yy.ravel()]
    y = np.ones(shape=x.shape[0], dtype=np.int64)
    
    return x, y, xx, yy
    