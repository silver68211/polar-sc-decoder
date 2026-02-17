"""
Example usage of the polar-sc-decoder package.
"""

import numpy as np
from polar_sc_decoder import PolarEncoder, SCDecoder, polar_code_construct


def main():
    # Parameters
    N = 8  # Code length (must be power of 2)
    K = 4  # Number of information bits
    
    print("Polar Code Example")
    print("=" * 50)
    print(f"Code parameters: N={N}, K={K}, Rate={K/N:.2f}")
    print()
    
    # Step 1: Code construction
    print("Step 1: Constructing polar code...")
    frozen_bits, info_bits = polar_code_construct(N, K, method='bhattacharyya', design_snr=0.0)
    print(f"Frozen bit positions: {frozen_bits}")
    print(f"Info bit positions: {info_bits}")
    print()
    
    # Step 2: Encoding
    print("Step 2: Encoding...")
    encoder = PolarEncoder(N, frozen_bits)
    message = np.array([1, 0, 1, 1])  # K information bits
    print(f"Message: {message}")
    
    codeword = encoder.encode(message)
    print(f"Codeword: {codeword}")
    print()
    
    # Step 3: Channel (BPSK + AWGN)
    print("Step 3: Simulating channel...")
    snr_db = 2.0
    snr_linear = 10 ** (snr_db / 10)
    noise_var = 1 / (2 * snr_linear)
    
    # BPSK modulation: 0 -> +1, 1 -> -1
    tx_symbols = 1 - 2 * codeword
    
    # Add noise
    noise = np.sqrt(noise_var) * np.random.randn(N)
    rx_symbols = tx_symbols + noise
    
    # Compute LLRs
    llr = 2 * rx_symbols / noise_var
    print(f"Received LLRs: {llr}")
    print()
    
    # Step 4: Decoding
    print("Step 4: Decoding...")
    decoder = SCDecoder(N, frozen_bits)
    decoded_message = decoder.decode(llr)
    print(f"Decoded message: {decoded_message}")
    print()
    
    # Step 5: Verification
    print("Step 5: Verification...")
    errors = np.sum(message != decoded_message)
    print(f"Number of bit errors: {errors}")
    if errors == 0:
        print("✓ Decoding successful!")
    else:
        print("✗ Decoding failed")


if __name__ == "__main__":
    main()
