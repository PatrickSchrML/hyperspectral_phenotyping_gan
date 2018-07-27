from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")


def plot_conti_code():
    data = pickle.load(open("/home/patrick/repositories/hyperspectral_phenotyping_gan/experiments/generated_code.p",
                            "rb"))
    labels = np.array(data["y"]).squeeze()
    code = np.array(data["z"]).copy()
    z = np.array(data["z"]).copy()
    code = code[:, -3:]
    signatures = np.array(data["x"])
    tsne = TSNE(n_jobs=26, n_components=2, learning_rate=100)
    Y = tsne.fit_transform(z)

    colors = ["red", "green", "blue"]
    for label in [0, 1, 2]:
        data = Y[labels == label]
        plt.scatter(data[:, 0], data[:, 1], c=colors[label], alpha=0.3,
                    label=str(label))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_conti_code()
