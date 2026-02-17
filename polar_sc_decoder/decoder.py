"""
Successive Cancellation (SC) decoder for polar codes.
"""

import numpy as np


class SCDecoder:
    """
    Successive Cancellation (SC) decoder for polar codes.
    
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
        self.frozen_bits = set(frozen_bits)
        self.info_bits = np.setdiff1d(np.arange(N), list(self.frozen_bits))
        self.K = len(self.info_bits)
        
    def decode(self, llr):
        """
        Decode received LLRs using SC decoding.
        
        Parameters
        ----------
        llr : array-like
            Log-likelihood ratios of received symbols (length N).
            
        Returns
        -------
        message : ndarray
            Decoded information bits (length K).
        """
        llr = np.array(llr, dtype=float)
        if len(llr) != self.N:
            raise ValueError(f"LLR length must be {self.N}, got {len(llr)}")
        
        # SC decoding using recursive structure
        u_hat = self._sc_decode_recursive(llr, 0, self.N)
        
        return u_hat[self.info_bits]
    
    def _sc_decode_recursive(self, llr, start_index, block_size):
        """
        Recursive SC decoding.
        
        Parameters
        ----------
        llr : ndarray
            Current LLR values for this block.
        start_index : int
            Starting bit index for this block.
        block_size : int
            Size of current block.
            
        Returns
        -------
        u_hat : ndarray
            Decoded bits for this block.
        """
        u_hat = np.zeros(block_size, dtype=int)
        
        if block_size == 1:
            # Leaf node - make decision
            if start_index in self.frozen_bits:
                u_hat[0] = 0
            else:
                u_hat[0] = 0 if llr[0] >= 0 else 1
            return u_hat
        
        # Split block
        half = block_size // 2
        
        # Compute LLRs for left half (check operation)
        llr_left = np.zeros(half)
        for i in range(half):
            llr_left[i] = self._f_function(llr[i], llr[i + half])
        
        # Decode left half
        u_left = self._sc_decode_recursive(llr_left, start_index, half)
        
        # Compute LLRs for right half (xor operation)
        llr_right = np.zeros(half)
        for i in range(half):
            llr_right[i] = self._g_function(llr[i], llr[i + half], u_left[i])
        
        # Decode right half
        u_right = self._sc_decode_recursive(llr_right, start_index + half, half)
        
        # Combine results: u_hat is just concatenation
        u_hat[:half] = u_left
        u_hat[half:] = u_right
        
        return u_hat
    
    def _f_function(self, llr1, llr2):
        """Check node operation (min-sum approximation)."""
        return np.sign(llr1) * np.sign(llr2) * min(abs(llr1), abs(llr2))
    
    def _g_function(self, llr1, llr2, bit):
        """XOR node operation."""
        return llr2 + (1 - 2 * bit) * llr1


