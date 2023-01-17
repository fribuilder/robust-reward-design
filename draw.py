# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 00:15:14 2022

@author: 53055
"""

import numpy as np
import matplotlib.pyplot as plt
def draw66():
    x = [1, 2, 3, 4, 5]
    y1 = [0, 0.3521, 0.3881, 0.3877, 0.3877]
    y2 = [0.3847, 0.3877, 0.3877, 0.3877, 0.3877]
    
#    plt.title("Defender's expected value at initial state")
    plt.plot(x, y2, marker = ".", markersize = 8, linewidth = 4, color = 'blue', label = "Initial policy 1")
    plt.plot(x, y1, marker = "*", markersize = 8, linewidth = 4, color = 'red', label = "Initial policy 2")
    plt.xticks([1, 2, 3, 4, 5])
    plt.xlabel("Iteration", fontsize = 16)
    plt.yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
    plt.ylabel("Defender's value", fontsize = 16) 
    plt.legend()
    plt.show()
    
def draw66Action():
    x = [1, 2, 3, 4, 5]
    y1 = [0.004, 0.3744, 0.3777, 0.3777, 0.3777]
    y2 = [0.3952, 0.395, 0.3943, 0.3941, 0.394]
    
#    plt.title("Defender's expected value at initial state")
    plt.plot(x, y2, marker = ".", markersize = 8, linewidth = 4, color = 'blue', label = "Initial policy 1")
    plt.plot(x, y1, marker = "*", markersize = 8, linewidth = 4, color = 'red', label = "Initial policy 2")
    plt.xticks([1, 2, 3, 4, 5])
    plt.xlabel("Iteration", fontsize = 16)
    plt.yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
    plt.ylabel("Defender's value", fontsize = 16) 
    plt.legend()
    plt.show()
    
    
def draw1010():
    x = [1, 2, 3, 4, 5, 6]
#    y1 = [0.3999, 0.4087, 0.4070, 0.4070, 0.4070, 0.4070, 0.4070]
#    y2 = [0.0247, 0.3715, 0.4117, 0.4075, 0.4084, 0.4083, 0.4083]
#    y3 = [0.0234, 0.3654, 0.4116, 0.4076, 0.4084, 0.4083, 0.4083]
#    y4 = [0.0248, 0.3720, 0.4118, 0.4076, 0.4084, 0.4083, 0.4083]
    y1 = [0.4165, 0.4464, 0.4456, 0.4457, 0.4457, 0.4457]
    y2 = [0.0243, 0.4176, 0.4466, 0.4453, 0.4457, 0.4457]
    y3 = [0.0268, 0.4194, 0.4466, 0.4453, 0.4457, 0.4457]
    
#    plt.title("Defender's expected value at initial state")
    plt.plot(x, y1, marker = "*", markersize = 8, linewidth = 2, color = 'blue', label = "Initial policy 1")
    plt.plot(x, y2, marker = "^", markersize = 8, linewidth = 2, color = 'red', label = "Initial policy 2")
    plt.plot(x, y3, marker = "o", markersize = 8, linewidth = 2, color = 'green', label = "Initial policy 3")
#    plt.plot(x, y4, marker = "s", markersize = 8, linewidth = 2, color = 'black', label = "Initial policy 4")
    plt.xticks([1, 2, 3, 4, 5, 6])
    plt.xlabel("Iteration", fontsize = 16)
    plt.yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
    plt.ylabel("Defender's value", fontsize = 16) 
    plt.legend()
    plt.show()

if __name__ == "__main__":
#    draw66()
#    draw66Action()
    draw1010()