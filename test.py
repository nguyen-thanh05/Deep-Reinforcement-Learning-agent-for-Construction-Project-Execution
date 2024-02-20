import numpy as np
import matplotlib.pyplot as plt

array = np.loadtxt("Hello.txt")
print(array.shape)

array = array.reshape(5, 5, 5)
array = array.transpose(0, 2, 1)

# prepare some coordinates

fig = plt.figure()

final_rendering_cube = array
# set the colors of each object
colors = np.empty(final_rendering_cube.shape, dtype=object)
colors[array == 1] = '#7A88CCC0'

#print(colors)

ax = fig.add_subplot(projection='3d')
ax.voxels(final_rendering_cube, facecolors=colors, edgecolor='k')

plt.show()