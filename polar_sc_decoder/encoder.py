"""
Polar code encoder implementation.
"""

import numpy as np


class PolarEncoder:
    """
    Polar code encoder using Kronecker construction.
    
    Parameters
    ----------
    N : int
        Code length (must be a power of 2).
    frozen_bits : array-like
        Indices of frozen bit positions.
    """
    
    def __init__(self, N, frozen_bits):
        if N <= 0 or (N & (N - 1)) != 0:
            raise ValueError(f"N must be a power of 2, got {N}")
        
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.array(frozen_bits, dtype=int)
        self.info_bits = np.setdiff1d(np.arange(N), self.frozen_bits)
        self.K = len(self.info_bits)
        
    def encode(self, message):
        """
        Encode a message using polar coding.
        
        Parameters
        ----------
        message : array-like
            Information bits to encode (length K).
            
        Returns
        -------
        codeword : ndarray
            Encoded codeword (length N).
        """
        message = np.array(message, dtype=int)
        if len(message) != self.K:
            raise ValueError(f"Message length must be {self.K}, got {len(message)}")
        
        # Create u vector with frozen bits set to 0
        u = np.zeros(self.N, dtype=int)
        u[self.info_bits] = message
        
        # Polar transform using Kronecker construction
        x = u.copy()
        for i in range(self.n):
            step = 2 ** (i + 1)
            for j in range(0, self.N, step):
                x[j:j+step//2] = (x[j:j+step//2] + x[j+step//2:j+step]) % 2
        
        return x
