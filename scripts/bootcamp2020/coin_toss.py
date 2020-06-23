import numpy as np
import matplotlib
import matplotlib.style
matplotlib.use('agg')
matplotlib.style.use('paperstyle.mpl')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7,5))
ax.bar([0,1], 0.5, tick_label=['heads', 'tails'])
ax.set_ylabel('P(C)')
ax.set_title('T(0.5)')
plt.savefig('coin_toss_5.png')

fig, ax = plt.subplots(figsize=(7,5))
ax.bar([0,1], [0.2,0.8], tick_label=['heads', 'tails'])
ax.set_ylabel('P(C)')
ax.set_title('T(0.2)')
plt.savefig('coin_toss_2.png')
