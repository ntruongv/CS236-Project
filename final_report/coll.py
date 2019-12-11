import numpy as np
import matplotlib.pyplot as plt

itr = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
col = [112,180,326,105,87,84,117,124]
ade = [.94,.83,1.42,.48,.5,.74,.44,.46]
fde = [1.65,1.41,2.47,.91,.93,1.4,.82,.87]

fig, ax1 = plt.subplots()
ax1.set_xlabel("Iterations")
ax1.set_ylabel('ADE/FDE Score')
ax1.plot(itr, ade, label="ADE")
ax1.plot(itr, fde, label="FDE")
ax1.tick_params(axis='y')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('Num. of Collisions')  # we already handled the x-label with ax1
ax2.plot(itr, col, label="Collisions", color="g")
ax2.tick_params(axis='y')

fig.tight_layout()  # otherwise the right y-label is slightly clipped

ax1.legend(loc=2)
ax2.legend(loc=1)
plt.savefig("coll_itr.png")
