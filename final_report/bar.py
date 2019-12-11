import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['SGAN', 'OSGAN', 'VOSGAN-90', 'VOSGAN-10', 'GOAttGAN-10', 'GOAttGAN-5', 'Hybrid 1', 'Hybrid 2', 'Dis. Noise', 'Dis. Mod.']
ades = [.41,.39,2.18,.51,.48,.51,.59,.88,.6,.64]
fdes = [.8,.77,5.75,.95,.91,.95,1.09,1.56,1.07,1.05]
cols = [172,66,522,271,86, 185, 280, 365, 419, 250]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
rects1 = ax.bar(x - width/2, ades, width, label='ADE')
rects2 = ax.bar(x + width/2, fdes, width, label='FDE')
rects3 = ax2.bar(x, cols, width, label='Collisions', color="g")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax2.set_ylabel('Collisions')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend()

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1, ax)
autolabel(rects2, ax)
autolabel(rects3, ax2)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)



fig.tight_layout()

plt.show()
