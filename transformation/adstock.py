import numpy as np

def adstock(x, alpha=0.5):

    result = np.zeros(len(x))

    result[0] = x[0]

    for i in range(1, len(x)):

        result[i] = (
            x[i]
            + alpha * result[i - 1]
        )

    return result