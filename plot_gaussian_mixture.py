import numpy as np
import matplotlib.pyplot as plt
from config import N, mu_1_true, mu_2_true

def plot_gaussian_mixture(X, mu_1_hat, mu_2_hat, sigma_1_hat, sigma_2_hat):
    # find eigen values of sigma_1_hat and sigma_2_hat
    sigma_1_eig_val, sigma_1_eig_vec = np.linalg.eig(sigma_1_hat)
    sigma_2_eig_val, sigma_2_eig_vec = np.linalg.eig(sigma_2_hat)

    # sort the eigen values and vectors
    order = np.argsort(sigma_1_eig_val)[::-1]
    sigma_1_eig_val = sigma_1_eig_val[order]
    sigma_1_eig_vec = sigma_1_eig_vec[:, order]

    order = np.argsort(sigma_2_eig_val)[::-1]
    sigma_2_eig_val = sigma_2_eig_val[order]
    sigma_2_eig_vec = sigma_2_eig_vec[:, order]

    # find radius of the ellipses
    a1 = np.sqrt(sigma_1_eig_val[0])
    b1 = np.sqrt(sigma_1_eig_val[1])
    a2 = np.sqrt(sigma_2_eig_val[0])
    b2 = np.sqrt(sigma_2_eig_val[1])

    # find the center of the ellipses
    center1 = mu_1_hat
    center2 = mu_2_hat

    # plot the ellipses
    theta1 = np.linspace(0, 2 * np.pi, 100)
    theta2 = np.linspace(0, 2 * np.pi, 100)
    ellipse1 = np.array([a1 * np.cos(theta1), b1 * np.sin(theta1)])
    ellipse2 = np.array([a2 * np.cos(theta2), b2 * np.sin(theta2)])

    ellipse1_rotated = sigma_1_eig_vec @ ellipse1
    ellipse2_rotated = sigma_2_eig_vec @ ellipse2

    ellipse1_final = ellipse1_rotated + center1[:, np.newaxis]
    ellipse2_final = ellipse2_rotated + center2[:, np.newaxis]

    plt.plot(ellipse1_final[0], ellipse1_final[1], 'r-', label='Estimated Ellipse 1', linewidth=2)
    plt.plot(ellipse2_final[0], ellipse2_final[1], 'g-', label='Estimated Ellipse 2', linewidth=2)

    # also plot true data points
    plt.scatter(X[:N//2, 0], X[:N//2, 1], c='r', alpha=0.5, label='Group 1')
    plt.scatter(X[N//2:, 0], X[N//2:, 1], c='g', alpha=0.5, label='Group 2')

    # also sign the true means
    plt.scatter(mu_1_true[0], mu_1_true[1], c='r', label='True Mean 1', marker='x', s=100)
    plt.scatter(mu_2_true[0], mu_2_true[1], c='g', label='True Mean 2', marker='x', s=100)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Estimated Ellipses')
    plt.legend()

    plt.show()