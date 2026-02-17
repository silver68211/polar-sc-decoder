"""
Polar code construction methods.
"""

import numpy as np


def polar_code_construct(N, K, method='bhattacharyya', design_snr=0.0):
    """
    Construct a polar code by selecting frozen bit positions.
    
    Parameters
    ----------
    N : int
        Code length (must be a power of 2).
    K : int
        Number of information bits.
    method : str, optional
        Construction method. Options: 'bhattacharyya', 'ga' (default: 'bhattacharyya').
    design_snr : float, optional
        Design SNR in dB for construction (default: 0.0).
        
    Returns
    -------
    frozen_bits : ndarray
        Indices of frozen bit positions.
    info_bits : ndarray
        Indices of information bit positions.
    """
    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError(f"N must be a power of 2, got {N}")
    if K <= 0 or K > N:
        raise ValueError(f"K must be between 1 and {N}, got {K}")
    
    if method == 'bhattacharyya':
        # Bhattacharyya parameter based construction
        reliabilities = _bhattacharyya_construction(N, design_snr)
    elif method == 'ga':
        # Gaussian approximation based construction
        reliabilities = _ga_construction(N, design_snr)
    else:
        raise ValueError(f"Unknown construction method: {method}")
    
    # Select K most reliable positions for information bits
    info_bits = np.argsort(reliabilities)[-K:]
    frozen_bits = np.setdiff1d(np.arange(N), info_bits)
    
    return frozen_bits, info_bits


def _bhattacharyya_construction(N, design_snr):
    """
    Compute channel reliabilities using Bhattacharyya bounds.
    
    This is a simplified implementation for demonstration.
    """
    n = int(np.log2(N))
    
    # Convert SNR to noise variance
    snr_linear = 10 ** (design_snr / 10)
    sigma2 = 1 / (2 * snr_linear)
    
    # Initialize with AWGN channel reliability
    z = np.zeros(N)
    z[0] = np.exp(-1 / (4 * sigma2))
    
    # Recursive channel polarization
    for level in range(n):
        step = 2 ** (level + 1)
        for j in range(0, N, step):
            z_temp = z[j:j+step//2].copy()
            # Degraded (minus) channel
            z[j:j+step//2] = 2 * z_temp - z_temp ** 2
            # Upgraded (plus) channel  
            z[j+step//2:j+step] = z_temp ** 2
    
    # Return negative (lower is more reliable)
    return -z


def _ga_construction(N, design_snr):
    """
    Compute channel reliabilities using Gaussian approximation.
    
    This is a simplified implementation for demonstration.
    """
    n = int(np.log2(N))
    
    # Convert SNR to initial LLR mean
    snr_linear = 10 ** (design_snr / 10)
    mu = 4 * snr_linear
    
    # Initialize
    means = np.zeros(N)
    means[0] = mu
    
    # Recursive channel polarization using GA
    for level in range(n):
        step = 2 ** (level + 1)
        for j in range(0, N, step):
            mu_temp = means[j:j+step//2].copy()
            # Approximate update rules for GA
            means[j:j+step//2] = _phi_inv(1 - (1 - _phi(mu_temp)) ** 2)
            means[j+step//2:j+step] = 2 * mu_temp
    
    return means


def _phi(x):
    """Helper function for GA construction."""
    return np.where(x < 10, np.exp(-x) * (1 - np.exp(-x)), 
                    np.exp(-x))


def _phi_inv(y):
    """Inverse of phi function (approximate)."""
    return np.where(y < 0.5, -np.log(y), -np.log(1 - y))
