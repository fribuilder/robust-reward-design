# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 10:59:29 2022

@author: 53055
"""

import numpy as np

def Projection(x):
    if np.linalg.norm(x) >= 1:
        return x
    else:
        return x/np.linalg.norm(x)
    
def generate_positive_definite_M(n):
    P = np.random.random((n, n))
    P = np.dot(P, P.T)
    return P

def gradient_descent(P, x):
    #Here x is 1*n, P n*n
    lr = 0.01
    diff_norm = np.inf
    eps = 0.001
    it_count = 1
#    while diff_norm > eps:
    while it_count <= 10000:
        print(it_count)
        gradient = x.dot(P + P.T)
        x_new = x - lr * gradient 
        x_new = Projection(x_new)
        diff_norm = np.linalg.norm(x-x_new)
        x = x_new
        it_count += 1
    return x

def obj_function(P, x):
    return np.dot(x, P).dot(x.T)
        
if __name__ == "__main__":
    n = 3
    x = np.random.random(n)
    P = generate_positive_definite_M(n)
    x = gradient_descent(P, x)
    eig, eig_V = np.linalg.eig(P)
    obj_x = obj_function(P, x)
    obj_0 = obj_function(P, eig_V[:, 0].T)
    obj_1 = obj_function(P, eig_V[:, 1].T)
    obj_2 = obj_function(P, eig_V[:, 2].T)
    print(x)
    print(eig)
    print(eig_V)
    print(obj_x)
    print(obj_0)
    print(obj_1)
    print(obj_2)