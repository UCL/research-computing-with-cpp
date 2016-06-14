import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import sys
figure, axes = plt.subplots()
axes.set_frame_on(False)
axes.get_xaxis().tick_bottom()
axes.get_yaxis().grid( linestyle='-', linewidth=2)
axes.broken_barh([(0, 20),(20,1),(21, 10)], (0.6, 0.8), facecolors=('red','green', 'yellow'))
axes.broken_barh([(0, 10),(10,1),(11, 20)], (1.6, 0.8), facecolors=('red','blue','yellow'))
axes.set_yticks([1, 2])
axes.set_yticklabels(['Rank 1', 'Rank 0'])
red_patch = mpatches.Patch(color='red', label='Function 1')
blue_patch = mpatches.Patch(color='blue', label='Send')
green_patch = mpatches.Patch(color='green', label='Receive')
yellow_patch = mpatches.Patch(color='yellow', label='Function 2')
axes.legend(handles=[red_patch, blue_patch, green_patch, yellow_patch])
figure.savefig(sys.argv[1])
