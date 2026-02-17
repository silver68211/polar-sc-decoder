
import os
import numpy as np

from SCDecoder import PolarSCDecoder


def awgn_bpsk_channel(codeword_bits: np.ndarray, sigma: float) -> np.ndarray:
    """
    BPSK modulation + AWGN:
        x = 1 - 2*b  in {+1,-1}
        y = x + n,   n ~ N(0, sigma^2)
    """
    x = 1.0 - 2.0 * codeword_bits
    return x + np.random.randn(*codeword_bits.shape) * sigma


def llr_awgn_bpsk(y: np.ndarray, sigma: float) -> np.ndarray:
    """LLR for AWGN+BPSK with mapping 0->+1, 1->-1: L = 2y/sigma^2."""
    return 2.0 * y / (sigma * sigma)


def compute_mprim_alpha(llr_abs: np.ndarray, info_bits: np.ndarray) -> np.ndarray:
    """

    Parameters
    ----------
    llr_abs : ndarray, shape (B, N)
        Absolute LLRs at leaf after SC decoding (or comparable per-bit LLRs).
    info_bits : ndarray, shape (K_current,)
        Current information positions (0-based).

    Returns
    -------
    mprim_sum_over_batch : ndarray, shape (K_current,)
        Sum over batch of m_prim for each info bit x (aligned to info_bits order).
    """
    B, N = llr_abs.shape
    info_bits = np.asarray(info_bits, dtype=np.int64)

    # Gather |L| on current info bits: shape (B, Kc)
    llr_info = llr_abs[:, info_bits]  # (B, Kc)

    # term1: exp(-|L_x|) for each candidate x in info_bits: (B, Kc)
    term1 = np.exp(-llr_info)

    # term2: prod_{j<=x} 1/(1+exp(-|L_j|))
    # We'll compute this efficiently by ordering info_bits and taking a cumulative product.
    # Sort info_bits (ascending), but return results in original info_bits order.
    sort_idx = np.argsort(info_bits)
    inv_sort = np.empty_like(sort_idx)
    inv_sort[sort_idx] = np.arange(sort_idx.size)

    info_sorted = info_bits[sort_idx]             # (Kc,)
    llr_sorted = llr_abs[:, info_sorted]          # (B, Kc)

    p_corr_sorted = 1.0 / (1.0 + np.exp(-llr_sorted))  # (B, Kc)
    cumprod_sorted = np.cumprod(p_corr_sorted, axis=1) # (B, Kc)

    # map cumprod back to original info_bits order
    cumprod = cumprod_sorted[:, inv_sort]         # (B, Kc)

    mprim = term1 * cumprod                        # (B, Kc)
    return mprim.sum(axis=0)                       # (Kc,)


if __name__ == "__main__":
    # -----------------------------
    # Code parameters
    # -----------------------------
    n = 9
    K = 256
    N = 2 ** n

    batch_size = 1000
    esno_db = 3.0

    # Start with no frozen bits (rate-1); we will add frozen bits iteratively
    frozen_bits = np.array([], dtype=np.int32)

    os.makedirs("results", exist_ok=True)

    while True:
        print("*" * 80 + f"ESNO: {esno_db:.2f} dB" + "*" * 80)

        # Instantiate NumPy SC decoder
       
        decoder = PolarSCDecoder(frozen_bits, N)
        if decoder is None:
            raise RuntimeError("Instantiate decoder = PolarSCDecoderNP(frozen_bits, N).")

        # Current info positions
        info_bits = np.setdiff1d(np.arange(N, dtype=np.int32), frozen_bits).astype(np.int32)

        # All-zero information bits => all-zero u-vector
        b = np.zeros((batch_size, info_bits.shape[0]), dtype=np.float32)
        msg = np.zeros((batch_size, N), dtype=np.float32)
        msg[:, info_bits] = b

        sigma = np.sqrt(1.0 / (2.0 * 10 ** (esno_db / 10.0)))

        ber_cnt = 0.0
        fer_cnt = 0.0
        itr = 0

        # accumulate metric over all simulated blocks
        mprim_alpha_sum = np.zeros((info_bits.shape[0],), dtype=np.float64)

        while True:
            itr += 1

            # Channel + LLRs
            y = awgn_bpsk_channel(msg, sigma)
            llr_ch = llr_awgn_bpsk(y, sigma).astype(np.float32)

            # Decode (NumPy)
            u_hat_n, llr_leaf = decoder.decode(llr_ch)

            # Extract info bits
            u_hat = u_hat_n[:, info_bits]

            # |LLR| for metric
            llr_abs = np.abs(llr_leaf)

            # Metric accumulation: sum over batch, then accumulate over iterations
            mprim_alpha_sum += compute_mprim_alpha(llr_abs, info_bits)

            # FER/BER
            frame_err = (u_hat != b).any(axis=1)
            fer_cnt += frame_err.sum()
            ber_cnt += (u_hat != b).sum()

            fer_est = fer_cnt / (batch_size * itr)
            
            # for printing
            mprim_alpha = np.sum(mprim_alpha_sum / (batch_size * itr))
            
            msg_line = (
                f"ESNO: {esno_db:.2f}, "
                f"FER: {fer_est:1.4e}, "
                f"Eq. (8): {mprim_alpha:1.4e}, "
                f"BLError: {int(fer_cnt):4d} blocks: {batch_size*itr:10d}"
            )
            print(msg_line)

            # stop conditions
            if fer_cnt >= 100:
                break
            if itr * batch_size > 1e9:
                break

        # If we already froze enough bits to reach dimension K, stop.
        if len(frozen_bits) >= (N - K):
            break

        # Average metric over all processed blocks
        mprim_alpha = mprim_alpha_sum / (batch_size * itr)  # shape (K_current,)

        # Freeze the info bit with the largest metric (DESCENDING)
        metric_sorted_ind = np.argsort(mprim_alpha)[::-1]
        info_bits_sorted = info_bits[metric_sorted_ind]

        # Add the single worst bit to frozen set
        frozen_bits = np.concatenate([frozen_bits, np.array([info_bits_sorted[0]], dtype=np.int32)])

        # Print and save
        print("frozen_bits:\n", frozen_bits)
        print("frozen_len:\t", len(frozen_bits))

        np.savetxt(
            f"results/frozen_bits_{N}_{K}_SC_esno_{esno_db}.csv",
            frozen_bits.astype(np.int32),
            delimiter=",",
        )
