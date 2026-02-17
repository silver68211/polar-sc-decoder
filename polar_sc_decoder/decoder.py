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
        
        # Initialize LLR and bit trees
        self.llr_tree = np.zeros((self.n + 1, self.N))
        self.bit_tree = np.zeros((self.n + 1, self.N), dtype=int)
        self.llr_tree[self.n] = llr
        
        # Decode bit by bit
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            llr_i = self._get_llr(i)
            
            if i in self.frozen_bits:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if llr_i >= 0 else 1
            
            self._update_bit(i, u_hat[i])
        
        return u_hat[self.info_bits]
    
    def _get_llr(self, index):
        """Get LLR for bit at index by traversing the tree."""
        path = self._get_path(index)
        
        for level in range(self.n - 1, -1, -1):
            step = 2 ** (level + 1)
            offset = (index // step) * step
            pos_in_block = index % step
            
            if pos_in_block < step // 2:
                # Left child - check node
                for j in range(step // 2):
                    llr1 = self.llr_tree[level + 1, offset + j]
                    llr2 = self.llr_tree[level + 1, offset + step // 2 + j]
                    self.llr_tree[level, offset + j] = self._f_function(llr1, llr2)
            else:
                # Right child - xor node
                for j in range(step // 2):
                    llr1 = self.llr_tree[level + 1, offset + j]
                    llr2 = self.llr_tree[level + 1, offset + step // 2 + j]
                    bit = self.bit_tree[level, offset + j]
                    self.llr_tree[level, offset + step // 2 + j] = self._g_function(llr1, llr2, bit)
        
        return self.llr_tree[0, index]
    
    def _update_bit(self, index, bit):
        """Update bit in the tree after decision."""
        self.bit_tree[0, index] = bit
        
        # Propagate bit decisions upward
        for level in range(1, self.n + 1):
            step = 2 ** level
            offset = (index // step) * step
            pos = index % step
            
            if pos < step // 2:
                # Update from left child
                self.bit_tree[level, offset + pos] = bit
            else:
                # Update from right child (XOR with left)
                left_pos = pos - step // 2
                left_bit = self.bit_tree[level - 1, offset + left_pos]
                self.bit_tree[level, offset + pos] = (left_bit + bit) % 2
    
    def _get_path(self, index):
        """Get path from root to leaf for given index."""
        path = []
        for level in range(self.n):
            path.append((index >> (self.n - 1 - level)) & 1)
        return path
    
    def _f_function(self, llr1, llr2):
        """Check node operation (min-sum approximation)."""
        return np.sign(llr1) * np.sign(llr2) * min(abs(llr1), abs(llr2))
    
    def _g_function(self, llr1, llr2, bit):
        """XOR node operation."""
        return llr2 + (1 - 2 * bit) * llr1

