import numpy as np
import matplotlib.pyplot as plt


def get_gamma(N, problem="Problem 5"):
    """
    Get the Gamma matrix for the given problem and number of steps

    Parameters:
        N: number of steps
        problem: "Problem 5" or "Problem 4"
    Returns:
        Gamma: Gamma matrix
    """
    if problem == "Problem 5":
        data = np.load("Results/Problem5/Problem_5_estimates.npz")
    elif problem == "Problem 4":
        data = np.load("Results/Problem4/Problem_4_estimates.npz")
    else:
        raise ValueError("Invalid problem")

    markov_mat = data["markov_mat"]

    markov_mat_test = markov_mat[:, :, :N].reshape(2, -1)

    Gamma = np.zeros((2*N, 2*N))
    
    for i in range(N):
        markov_input = markov_mat_test[:, :2*(N-i)]
        markov_input = np.block([
            [np.zeros((2, 2*i)), markov_input]
        ])
        Gamma[:, 2*i:2*i+2] = markov_input.T

    return Gamma


Gamma = get_gamma(10, problem="Problem 5")
data = np.load("Results/Problem5/Problem_5_estimates.npz")
markov_mat = data["markov_mat"]
markov_mat_test = markov_mat[:, :, :10].reshape(2, -1)

fig, ax = plt.subplots(2, 1, figsize=(8, 12))
ax[0].matshow(Gamma, aspect='auto', cmap='viridis')
ax[1].matshow(markov_mat_test.T, aspect='auto', cmap='viridis')
ax[0].axis('equal')
ax[1].axis('equal')
plt.show()
plt.savefig('Figures/Problem8/Problem_8_Gamma.png')
plt.close()
