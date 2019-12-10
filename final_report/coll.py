import numpy as np
import matplotlib.pyplot as plt

itr = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
col = [112,180,326,105,87,84,117,124]
ade = [.94,.83,1.42,.48,.5,.74,.44,.46]
fde = [1.65,1.41,2.47,.91,.93,1.4,.82,.87]

plt.plot(itr, col, label="Collisions")
plt.plot(itr, ade, label="ADE")
plt.plot(itr, fde, label="FDE")
plt.legend()
plt.savefig("coll_itr.png")
