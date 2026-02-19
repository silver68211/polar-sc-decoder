
import os
import numpy as np


def generate_5g_ranking(k: int, n: int, sort: bool = True):
    """
    Return frozen and information bit positions of the 5G NR Polar code.

    The channel reliability order is defined according to
    3GPP TS 38.212, Table 5.3.1.2-1.

    Parameters
    ----------
    k : int
        Number of information bits.
    n : int
        Codeword length (must be a power of two, 32 <= n <= 1024).
    sort : bool, optional
        If True, returned indices are sorted in ascending order.

    Returns
    -------
    frozen_pos : np.ndarray
        Array of shape (n-k,) containing frozen bit indices.
    info_pos : np.ndarray
        Array of shape (k,) containing information bit indices.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """

    # -----------------------------
    # Input validation
    # -----------------------------
    if not isinstance(k, int) or not isinstance(n, int):
        raise ValueError("k and n must be integers.")

    if not isinstance(sort, bool):
        raise ValueError("sort must be boolean.")

    if k < 0 or k > 1024:
        raise ValueError("k must satisfy 0 <= k <= 1024.")

    if n < 32 or n > 1024:
        raise ValueError("n must satisfy 32 <= n <= 1024.")

    if n < k:
        raise ValueError("Invalid code rate: k cannot exceed n.")

    if not np.log2(n).is_integer():
        raise ValueError("n must be a power of 2.")

    # -----------------------------
    # Load 5G reliability sequence
    # -----------------------------
    source = os.path.join("codes", "polar_5G.csv")
    ch_order = np.genfromtxt(source, delimiter=";").astype(int)

    # ch_order[:, 0] → reliability rank
    # ch_order[:, 1] → channel index

    # -----------------------------
    # Select first n most reliable channels
    # -----------------------------
    # Sort by channel index (ascending)
    ch_order_sorted = ch_order[np.argsort(ch_order[:, 1])]

    # Keep only first n channels
    ch_order_n = ch_order_sorted[:n]

    # Sort those n channels by reliability rank
    ch_order_n = ch_order_n[np.argsort(ch_order_n[:, 0])]

    # -----------------------------
    # Split into frozen and info bits
    # -----------------------------
    frozen_pos = ch_order_n[: n - k, 1]
    info_pos = ch_order_n[n - k :, 1]

    if sort:
        frozen_pos = np.sort(frozen_pos)
        info_pos = np.sort(info_pos)

    return frozen_pos.astype(int), info_pos.astype(int)
