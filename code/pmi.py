import numpy as np
import matplotlib.pyplot as plt


def get_cooccurence_matrix(corpus, vocab, window_size=1):
    vocab_index = {word: i for i, word in enumerate(vocab)}
    cooccurrence_matrix = np.zeros((len(vocab), len(vocab)), dtype=int)

    for sentence in corpus:
        words = sentence.split()
        indices = [vocab_index[word] for word in words if word in vocab_index]

        for i in range(len(indices)):
            for j in range(
                max(0, i - window_size), min(len(indices), i + window_size + 1)
            ):
                if i != j:
                    cooccurrence_matrix[indices[i], indices[j]] += 1

    return cooccurrence_matrix


def pmi(cooccurrence_matrix, ppmi=False):
    total = np.sum(cooccurrence_matrix)
    row_sum = np.sum(cooccurrence_matrix, axis=0)
    col_sum = np.sum(cooccurrence_matrix, axis=1)
    pmi_matrix = np.zeros(cooccurrence_matrix.shape, dtype=float)

    for i in range(cooccurrence_matrix.shape[0]):
        for j in range(cooccurrence_matrix.shape[1]):
            pmi = np.log2(
                (cooccurrence_matrix[i, j] / total)
                / (row_sum[i] * col_sum[j] / total**2)
            )
            if ppmi:
                pmi = max(pmi, 0)
            pmi_matrix[i, j] = pmi

    return pmi_matrix


corpus = [
    "I love deep learning",
    "I love NLP",
    "I do programming",
]
vocab = ["I", "love", "do", "deep", "learning", "NLP", "programming"]
cooccurence_mat = get_cooccurence_matrix(corpus, vocab)
print("The Cooccurence Matrix is:")
print(cooccurence_mat)
print("The PMI Matrix is:")
print(pmi(cooccurence_mat))
print("The PPMI Matrix is:")
ppmi = pmi(cooccurence_mat, ppmi=True)
print(ppmi)

U, S, V = np.linalg.svd(ppmi)
# U, S, V = np.linalg.svd(cooccurence_mat)
# plot to 2D
plt.scatter(U[:, 0], U[:, 1])
for i, word in enumerate(vocab):
    plt.text(U[i, 0], U[i, 1], word)
plt.title("SVD via PPMI")
plt.savefig("svd.png", dpi=300, bbox_inches="tight")
