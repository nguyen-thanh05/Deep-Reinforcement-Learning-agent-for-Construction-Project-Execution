import os
import random
import sys

import numpy as np
import matplotlib.pyplot as plt


class TargetLoader:
    def __init__(self, path: str) -> None:
        self._path = path
        self.targets: list[np.ndarray] | None = None

    def load_all(self) -> list[np.ndarray]:
        if self.targets is not None:
            return self.targets

        self.targets = []
        for file in os.listdir(self._path):
            self.targets.append(self.load_file(os.path.join(self._path, file)))
        return self.targets

    @staticmethod
    def load_file(file_name: str) -> np.ndarray:
        array = np.fromfile(file_name, dtype=int, sep=' ')
        array = array[3:].reshape(array[0], array[1], array[2])
        return array.transpose(0, 2, 1)

    def visualize(self, idx=-1) -> None:
        if self.targets is None:
            self.load_all()

        if len(self.targets) == 0:
            print(f"No target found in {self._path}", file=sys.stderr)
            return
        elif idx >= len(self.targets):
            print(f"Index {idx} out of range", file=sys.stderr)
            return

        if idx == -1:
            idx = random.randint(0, len(self.targets) - 1)

        cubes = self.targets[idx]
        colors = np.empty(cubes.shape, dtype=object)
        colors[cubes == 1] = '#7A88CCC0'
        colors[cubes==2] = '#FF5733C0'
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.voxels(cubes, facecolors=colors, edgecolor='k')
        plt.show()


if __name__ == "__main__":
    loader = TargetLoader("targets")
    loader.visualize(1)
