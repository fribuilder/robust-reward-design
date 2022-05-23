# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
font = {'weight' : 'bold',
        'size'   : 10}

plt.rc('font', **font)
import numpy as np

def plot_h():
    h0 = [50, 60, 70, 80, 90, 100, 110]
    V0 = [-34.92, -34.5, -28.88, -2.87, -1.605, -1.231, -0.557]
    h1 = [110, 113]
    V1 = [-0.557, -0.159]
    subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    ylab = "Defender's value V1".translate(subscript)
    plt.plot(h0, V0, marker = 'o', markersize = 12, linewidth = 4, color = 'C0')
    plt.plot(h1, V1, marker = '*', markersize = 12, linewidth = 4, color = 'C0')
    plt.xlabel("Resource constraint h", fontsize = 14, fontweight = 'bold')
    plt.ylabel(ylab, fontsize = 14, fontweight = 'bold')
    plt.show()


def plot_chart():
    labels = [100, 120, 140, 160, 180, 200]
#    decoy_1 = [12.5713, 32.1900, 51.7228, 70.9155, 87.4072, 100.8890]
#    decoy_2 = [85.9988, 86.3756, 86.8306, 87.5661, 89.8515, 91.5199]
    decoy_1 = [12.6, 32.2, 51.7, 70.9, 87.4, 100.9]
    decoy_2 = [86, 86.3, 86.8, 87.6, 89.9, 91.5]

    x = np.arange(len(labels))  # the label locations
    width = 0.45 # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, decoy_1, width, label='Decoy (4, 6)')
    rects2 = ax.bar(x + width/2, decoy_2, width, label='Decoy (7, 4)')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Assigned resource', fontsize = 14, fontweight = 'bold')
    ax.set_xlabel('Resource constraint h', fontsize = 14, fontweight = 'bold')
#    ax.set_title('Scores by group and gender')
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 120)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()


def plot_h_2():
    subscript = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    ylab = "Defender's value V1".translate(subscript)
    h = [100, 120, 140, 160, 180, 200]
    V = [-6.7935, -6.7793, -6.7287, -6.4582, -3.6372, -1.1954]
    plt.plot(h, V, marker = 'o', markersize = 12, linewidth = 4)
    plt.xlabel("Resource constraint h", fontsize = 14, fontweight = 'bold')
    plt.ylabel(ylab, fontsize = 14, fontweight = 'bold')
    plt.show()


if __name__ == "__main__":
    plot_h()
#    plot_chart()
#    plot_h_2()