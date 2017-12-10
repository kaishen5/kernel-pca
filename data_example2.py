# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 10:02:02 2017

@author: kashen
"""

import numpy as np
import math

def get_curves(points=1000, radius=2, noise=None, original=False, *args, **kwargs):
    """
    Generates syntethic data in the shape of two quarter circles, as in
    the example from the paper by Mika et al.
    Arguments:
        points: number of points in the generated dataset.
        noise: name of the distribution to be used as additive noise.
               Use one of the distribution from numpy.random, see
               https://docs.scipy.org/doc/numpy-1.10.0/reference/routines.random.html
               Default is 'uniform', with low=-0.5 and high=0.5. Noise is added to
               the semi-circle's radious.
        args, kwargs: Any arguments you want to pass to the corresponding
                      numpy.random sampling function, except size.
    Returns:
        Arrays with the X and Y coordinates for the new data.
    """
    if noise is None:
        noise = 'uniform'
        kwargs['low'] = -0.5
        kwargs['high'] = 0.5
    kwargs['size'] = points // 2
    dist = getattr(np.random, noise)

    angles = np.linspace(0, math.pi/2, num=points//2)
    cos = np.cos(angles)
    sin = np.sin(angles)
    left_center = -0.5
    if original==False:
        left_radius = radius + dist(*args, **kwargs)
    else:
        left_radius = radius
    left_x = -left_radius*cos + left_center
    left_y = left_radius*sin
    right_center = 0.5
    if original==False:
        right_radius = radius + dist(*args, **kwargs)
    else:
        right_radius = radius
    right_x = right_radius*cos[::-1] + right_center
    right_y = right_radius*sin[::-1]
    return np.concatenate((left_x, right_x)), np.concatenate((left_y, right_y))




def get_square(points=1000, length=4, noise=None, original=False, *args, **kwargs):
    """
    Generates syntethic data in the shape of two quarter circles, as in
    the example from the paper by Mika et al.
    Arguments:
        points: number of points in the generated dataset.
        noise: name of the distribution to be used as additive noise.
               Use one of the distribution from numpy.random, see
               https://docs.scipy.org/doc/numpy-1.10.0/reference/routines.random.html
               Default is 'uniform', with low=-0.5 and high=0.5. Noise is added
               in the direction orthogonal to the current side.
        args, kwargs: Any arguments you want to pass to the corresponding
                      numpy.random sampling function, except size.
    Returns:
        Arrays with the X and Y coordinates for the new data.
    """
    if noise is None:
        noise = 'uniform'
        kwargs['low'] = -0.5
        kwargs['high'] = 0.5
    kwargs['size'] = points // 4
    dist = getattr(np.random, noise)

    real_values = np.linspace(0, length, num=points//4)
    x_values = []
    y_values = []
    # Left side
    
    if original==False:
        # Left side
        x_values.append(dist(*args, **kwargs))
        y_values.append(real_values)
        # Right side
        x_values.append(dist(*args, **kwargs) + length)
        y_values.append(real_values)
        # Top side
        x_values.append(real_values)
        y_values.append(dist(*args, **kwargs) + length)
        # Bottom side
        x_values.append(real_values)
        y_values.append(dist(*args, **kwargs))
            
    else:
       # Left side
        x_values.append([0]*(points//4))
        y_values.append(real_values)
        # Right side
        x_values.append([length]*(points//4))
        y_values.append(real_values)
        # Top side
        x_values.append(real_values)
        y_values.append([length]*(points//4))
        # Bottom side
        x_values.append(real_values)
        y_values.append([0]*(points//4))
        
        
    

    return np.concatenate(x_values), np.concatenate(y_values)



def main():
    "Example data generation"
    X, Y = get_curves(noise='normal', original=True, scale=0.2)
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(5,10))
    plt.subplot2grid((2, 1), (0, 0))
    plt.axis('off')
    plt.plot(X, Y ,'k.')
    plt.subplot2grid((2, 1), (1, 0))
    plt.axis('off')
    X, Y = get_square(noise='normal', original=True, scale=0.2)
    plt.plot(X, Y ,'k.')
    plt.show()

if __name__ == '__main__':
    main()