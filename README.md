# polar-sc-decoder

A NumPy-based research toolkit for Polar Codes under Successive Cancellation (SC) decoding, featuring decoder-aware BLER analysis and SC-optimized code construction.

## Features

- **Polar Code Encoder**: Efficient encoding using Kronecker construction
- **SC Decoder**: Successive Cancellation decoder implementation
- **Code Construction**: Bhattacharyya and Gaussian Approximation methods for optimal frozen bit selection
- **Pure NumPy**: Lightweight implementation with minimal dependencies

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
import numpy as np
from polar_sc_decoder import PolarEncoder, SCDecoder, polar_code_construct

# Code parameters
N = 8  # Code length (power of 2)
K = 4  # Information bits

# Construct polar code
frozen_bits, info_bits = polar_code_construct(N, K, method='bhattacharyya')

# Create encoder and decoder
encoder = PolarEncoder(N, frozen_bits)
decoder = SCDecoder(N, frozen_bits)

# Encode message
message = np.array([1, 0, 1, 1])
codeword = encoder.encode(message)

# Decode (with LLRs from channel)
decoded = decoder.decode(llr)
```

## Example

See `example.py` for a complete example with channel simulation:

```bash
python example.py
```

## Requirements

- Python >= 3.7
- NumPy >= 1.20.0

## License

MIT License - see LICENSE file for details.
