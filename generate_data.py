import numpy as np
from config import N, mu_1_true, mu_2_true, sigma_1_true, sigma_2_true

def generate_data():

    # generate data from both distributions
    n = N // 2
    X = np.concatenate([
        np.random.multivariate_normal(mu_1_true, sigma_1_true, size=n),
        np.random.multivariate_normal(mu_2_true, sigma_2_true, size=n)
    ])
    return X