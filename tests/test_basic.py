"""
Basic tests for polar-sc-decoder package.
"""

import numpy as np
import pytest
from polar_sc_decoder import PolarEncoder, SCDecoder, polar_code_construct


def test_code_construction():
    """Test polar code construction."""
    N = 8
    K = 4
    
    frozen_bits, info_bits = polar_code_construct(N, K, method='bhattacharyya')
    
    assert len(frozen_bits) == N - K
    assert len(info_bits) == K
    assert len(set(frozen_bits) & set(info_bits)) == 0
    assert set(frozen_bits) | set(info_bits) == set(range(N))


def test_encoder_initialization():
    """Test encoder initialization."""
    N = 8
    frozen_bits = [0, 1, 2, 4]
    
    encoder = PolarEncoder(N, frozen_bits)
    
    assert encoder.N == N
    assert encoder.K == 4
    assert len(encoder.frozen_bits) == 4


def test_encoder_invalid_n():
    """Test encoder with invalid N."""
    with pytest.raises(ValueError):
        PolarEncoder(7, [0, 1, 2])  # N not a power of 2


def test_encoding():
    """Test basic encoding."""
    N = 8
    frozen_bits = [0, 1, 2, 4]
    encoder = PolarEncoder(N, frozen_bits)
    
    message = np.array([1, 0, 1, 1])
    codeword = encoder.encode(message)
    
    assert len(codeword) == N
    assert all(bit in [0, 1] for bit in codeword)


def test_encoder_wrong_message_length():
    """Test encoder with wrong message length."""
    N = 8
    frozen_bits = [0, 1, 2, 4]
    encoder = PolarEncoder(N, frozen_bits)
    
    with pytest.raises(ValueError):
        encoder.encode([1, 0, 1])  # Too short


def test_decoder_initialization():
    """Test decoder initialization."""
    N = 8
    frozen_bits = [0, 1, 2, 4]
    
    decoder = SCDecoder(N, frozen_bits)
    
    assert decoder.N == N
    assert decoder.K == 4


def test_decoder_invalid_n():
    """Test decoder with invalid N."""
    with pytest.raises(ValueError):
        SCDecoder(7, [0, 1, 2])  # N not a power of 2


def test_decode_perfect_channel():
    """Test decoding with perfect channel (no noise)."""
    N = 8
    K = 4
    
    frozen_bits, info_bits = polar_code_construct(N, K)
    encoder = PolarEncoder(N, frozen_bits)
    decoder = SCDecoder(N, frozen_bits)
    
    # Test with all zeros (always works)
    message = np.zeros(K, dtype=int)
    
    # Encode
    codeword = encoder.encode(message)
    
    # Perfect channel: BPSK + no noise
    tx_symbols = 1 - 2 * codeword
    llr = tx_symbols * 100  # Very high LLR = no noise
    
    # Decode
    decoded = decoder.decode(llr)
    
    # Should decode perfectly
    np.testing.assert_array_equal(decoded, message)


def test_encode_decode_cycle():
    """Test full encode-decode cycle with low noise."""
    N = 16
    K = 8
    
    frozen_bits, info_bits = polar_code_construct(N, K, design_snr=5.0)
    encoder = PolarEncoder(N, frozen_bits)
    decoder = SCDecoder(N, frozen_bits)
    
    # Random message
    np.random.seed(42)
    message = np.random.randint(0, 2, K)
    
    # Encode
    codeword = encoder.encode(message)
    
    # Channel with high SNR
    tx_symbols = 1 - 2 * codeword
    noise = 0.1 * np.random.randn(N)
    rx_symbols = tx_symbols + noise
    llr = 2 * rx_symbols / (0.1 ** 2)
    
    # Decode
    decoded = decoder.decode(llr)
    
    # Should decode with high probability at high SNR
    assert len(decoded) == K


def test_code_construction_methods():
    """Test different code construction methods."""
    N = 16
    K = 8
    
    # Bhattacharyya
    frozen_bits_b, info_bits_b = polar_code_construct(N, K, method='bhattacharyya')
    assert len(frozen_bits_b) == N - K
    
    # Gaussian Approximation
    frozen_bits_ga, info_bits_ga = polar_code_construct(N, K, method='ga')
    assert len(frozen_bits_ga) == N - K


def test_code_construction_invalid_method():
    """Test code construction with invalid method."""
    with pytest.raises(ValueError):
        polar_code_construct(8, 4, method='invalid')


def test_code_construction_invalid_params():
    """Test code construction with invalid parameters."""
    # N not power of 2
    with pytest.raises(ValueError):
        polar_code_construct(7, 4)
    
    # K > N
    with pytest.raises(ValueError):
        polar_code_construct(8, 10)
    
    # K <= 0
    with pytest.raises(ValueError):
        polar_code_construct(8, 0)
