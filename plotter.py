import matplotlib.pyplot as plt
import numpy as np
import sys


def load() -> list:
    values = list()
    with open("values.txt", "r") as f:
        data = f.read().split("),")

        for i in range(len(data)):
            new_data = data[i].replace("(", "").replace(")", "")
            t1, t2, t3, t4 = new_data.split(", ")
            values.append((int(t1), int(t2), float(t3), float(t4)))

    return values


def plot(curr_set, labels, heatmap):
    plt.title(f"Accuracy {curr_set}-Set")

    plt.xticks(np.arange(0, 23), labels["y"])
    plt.xlabel("Hidden Layer Size")
    plt.yticks(np.arange(0, 10), labels["x"])
    plt.ylabel("Learning Rate")

    plt.imshow(heatmap, cmap='viridis', interpolation="bessel")
    plt.colorbar()
    plt.show()


def run():
    values = load()
    axes = {"x": [], "y": [], "train_acc": [], "test_acc": []}

    for t1, t2, t3, t4 in values:
        axes["x"].append(t1)
        axes["y"].append(t2)
        axes["train_acc"].append(t3)
        axes["test_acc"].append(t4)

    X_SIZE = 10
    Y_SIZE = 22

    curr_set = sys.argv[1]

    heatmap = np.zeros(X_SIZE * Y_SIZE).reshape((X_SIZE, Y_SIZE))
    labels = {"x": set(), "y": set()}

    for i in range(len(values)):
        x = int((axes["y"][i] - 1) / 5)
        y = int((axes["x"][i] - 40) / 25)

        labels["x"].add(axes["y"][i] / 1000)
        labels["y"].add(axes["x"][i])

        heatmap[x - 1, y - 1] = axes[f"{curr_set.lower()}_acc"][i]

    labels["x"] = sorted(labels["x"], key=float, reverse=True)
    labels["y"] = sorted(labels["y"], key=float)

    plot(curr_set, labels, heatmap)


if __name__ == '__main__':
    run()
