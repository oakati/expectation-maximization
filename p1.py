import numpy as np
from generate_data import generate_data
from EM import EM
from config import N, K, mu_1_true, mu_2_true, sigma_1_true, sigma_2_true
from plot_gaussian_mixture import plot_gaussian_mixture

X = generate_data()

# initial parameters
mu_1_0 = np.array([-10, 10])
mu_2_0 = np.array([10, -10])

sigma_1_0 = np.array([[1, 0], 
                      [0, 1]])
sigma_2_0 = np.array([[1, 0],
                      [0, 1]])
pi_1_0 = 0.5

# Initialize parameters
theta_0 = (mu_1_0, mu_2_0, sigma_1_0, sigma_2_0, pi_1_0)

# Set the convergence threshold
epsilon = 1e-3

# Run the EM algorithm
theta_hat = EM(X, theta_0, epsilon)

# Assign the results
mu_1_hat = theta_hat[0]
mu_2_hat = theta_hat[1]
sigma_1_hat = theta_hat[2]
sigma_2_hat = theta_hat[3]
pi_1_hat = theta_hat[4]

print("\nEstimated Parameters:")
print("-" * 50)
print(f"μ₁: {mu_1_hat}")
print(f"μ₂: {mu_2_hat}")
print("\nΣ₁:")
print(sigma_1_hat)
print("\nΣ₂:")
print(sigma_2_hat)
print(f"\nπ₁: {pi_1_hat:.3f}")
print(f"π₂: {1-pi_1_hat:.3f}")
print("-" * 50)

plot_gaussian_mixture(X, mu_1_hat, mu_2_hat, sigma_1_hat, sigma_2_hat)