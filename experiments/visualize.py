from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")


def plot_conti_code():
    data = pickle.load(open("/home/patrick/repositories/hyperspectral_phenotyping_gan/experiments_hdr/generated_code_noise10_disc4_conti3.p",
                            "rb"))
    labels = np.array(data["y"]).squeeze()
    labels_unique = np.unique(labels)
    code = np.array(data["z"]).copy()
    z = np.array(data["z"]).copy()

    code = code[:, -6:-2]
    print(code[0])
    signatures = np.array(data["x"])
    tsne = TSNE(n_jobs=26, n_components=2, learning_rate=100)
    Y = tsne.fit_transform(code)

    colors = ["red", "green", "blue"]
    for idx, label in enumerate(labels_unique):
        data_tsne = Y[labels == label]
        plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=colors[idx], alpha=0.3,
                    label=str(label))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_conti_code()
