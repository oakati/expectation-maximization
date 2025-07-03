from config import N, K
import numpy as np
from scipy.stats import multivariate_normal

def calculate_log_likelihood(pi_1, pi_2, pdf_k1, pdf_k2):
    return np.log(pi_1 * pdf_k1 + pi_2 * pdf_k2).sum()

def EM(X, theta_0, epsilon):
    """
    EM algorithm for Gaussian mixture model
    
    Parameters:
    X: set of observations
    theta_0: initial parameters
    epsilon: accuracy

    return:
        theta_hat: estimated parameters (mu_1_hat, mu_2_hat, sigma_1_hat, sigma_2_hat, pi_1_hat)
    """

    # Initialize parameters
    mu_1 = theta_0[0]
    mu_2 = theta_0[1]
    sigma_1 = theta_0[2]
    sigma_2 = theta_0[3]
    pi_1 = theta_0[4]
    pi_2 = 1 - pi_1

    # Calculate the initial log-likelihood
    pdf_k1 = multivariate_normal.pdf(X, mean=mu_1, cov=sigma_1)
    pdf_k2 = multivariate_normal.pdf(X, mean=mu_2, cov=sigma_2)

    L = np.log(pi_1 * pdf_k1 + pi_2 * pdf_k2).sum()

    # Store the previous log-likelihood
    L_prime = L - 2 * epsilon

    # Initialize responsibilities
    gama = np.zeros((N, K))

    # Iterate until the log-likelihood converges
    while L - L_prime > epsilon:

        # Calculate the pdf of the data
        pdf_k1 = multivariate_normal.pdf(X, mean=mu_1, cov=sigma_1)
        pdf_k2 = multivariate_normal.pdf(X, mean=mu_2, cov=sigma_2)

        # Evaluate responsibilities
        gama[:, 0] = np.divide( (pi_1 * pdf_k1) , (pi_1 * pdf_k1 + pi_2 * pdf_k2) )
        gama[:, 1] = np.divide( (pi_2 * pdf_k2) , (pi_1 * pdf_k1 + pi_2 * pdf_k2) )

        # Re-estimate parameters
        N_1 = gama[:, 0].sum()
        N_2 = gama[:, 1].sum()

        mu_1 = np.divide((gama[:, 0].reshape(-1, 1) * X).sum(axis=0), N_1)
        mu_2 = np.divide((gama[:, 1].reshape(-1, 1) * X).sum(axis=0), N_2)

        diff = X - mu_1
        weighted_diff = diff * gama[:, 0][:, np.newaxis]
        sigma_1 = (weighted_diff.T @ diff) / N_1

        diff = X - mu_2
        weighted_diff = diff * gama[:, 1][:, np.newaxis]
        sigma_2 = (weighted_diff.T @ diff) / N_2

        pi_1 = np.divide( N_1 , N )
        pi_2 = 1 - pi_1

        # Store the previous log-likelihood
        L_prime = L
        # Evaluate the new log-likelihood
        L = calculate_log_likelihood(pi_1, pi_2, pdf_k1, pdf_k2)

        print(f"Current log-likelihood: {L:.4f}")
        print(f"Previous log-likelihood: {L_prime:.4f}")
        print(f"Difference: {L - L_prime:.4f}")
        print("-" * 50)

    # Assign the results
    theta_hat = (mu_1, mu_2, sigma_1, sigma_2, pi_1)

    return theta_hat
