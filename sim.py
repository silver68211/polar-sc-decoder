
import os
import numpy as np

from SCDecoder import PolarSCDecoder
from utiles import generate_5g_ranking


def awgn_bpsk_channel(codeword_bits: np.ndarray, sigma: float) -> np.ndarray:
    """
    BPSK modulation + AWGN:
        x = 1 - 2*b  in {+1,-1}
        y = x + n,   n ~ N(0, sigma^2)

    Parameters
    ----------
    codeword_bits : ndarray, shape (B, N), entries in {0,1}
    sigma : float, noise std

    Returns
    -------
    y : ndarray, shape (B, N), float
    """
    x = 1.0 - 2.0 * codeword_bits
    y = x + np.random.randn(*codeword_bits.shape) * sigma
    return y


def llr_awgn_bpsk(y: np.ndarray, sigma: float) -> np.ndarray:
    """
    LLR for AWGN+BPSK with mapping 0->+1, 1->-1:
        L = 2y/sigma^2
    """
    return 2.0 * y / (sigma * sigma)


def simulate_sc_ber_fer(
    decoder,
    frozen_bits: np.ndarray,
    info_bits: np.ndarray,
    K: int,
    N: int,
    esno_db_grid: np.ndarray,
    batch_size: int = 100000,
    target_frame_errors: int = 50,
    save_dir: str = "results",
    file_prefix: str = "SC",
):
    """
    Monte-Carlo simulation of BER/FER for a polar code under SC decoding .

    Parameters
    ----------
    decoder : PolarSCDecoderNP (or compatible)
        Must implement: u_hat_n, llr_leaf = decoder.decode(llr_ch)
        where llr_ch has shape (B, N).
    frozen_bits : ndarray
        Frozen positions (0-based). Shape (N-K,).
    info_bits : ndarray
        Information positions (0-based). Shape (K,).
    K, N : int
        Code dimension and length.
    snr_db_grid : ndarray
        SNR points in dB (interpreted as Es/N0 in your original code).
    batch_size : int
        Number of codewords per iteration.
    target_frame_errors : int
        Stop criterion per SNR: accumulate at least this many frame errors.
    save_dir : str
        Directory to save BER/FER CSV.
    file_prefix : str
        Prefix used in filenames.

    Returns
    -------
    BER_list, FER_list : list[float], list[float]
    """
    os.makedirs(save_dir, exist_ok=True)

    esno_lin = 10 ** (esno_db_grid / 10.0)
    sigma_grid = np.sqrt(1.0 / (2.0 * esno_lin))  # matches your original formula

    # all-zero information bits
    b = np.zeros((batch_size, K), dtype=np.float32)

    # transmitted u (length N), with zeros on info bits too => all-zero u
    # this matches the "assume all-zero" setting used in your simulations
    msg = np.zeros((batch_size, N), dtype=np.float32)
    msg[:, info_bits] = b  # (still all zeros)

    BER_list = []
    FER_list = []

    for idx, sigma in enumerate(sigma_grid):
        print("*" * 50 + f"ESNO: {esno_db_grid[idx]:.2f} dB" + "*" * 50)

        ber_cnt = 0.0
        fer_cnt = 0.0
        itr = 0

        while True:
            itr += 1

            # channel: BPSK + AWGN
            y = awgn_bpsk_channel(msg, sigma)

            # LLRs
            llr_ch = llr_awgn_bpsk(y, sigma).astype(np.float32)

            # decode
            # NOTE: In your TF decoder you call model(-llr_ch) and inside it does llr_ch = -inputs.
            # Net effect: it decodes with +llr_ch. Here we directly pass +llr_ch.
            u_hat_n, _llr_leaf = decoder.decode(llr_ch)

            # take info bits only
            u_hat = u_hat_n[:, info_bits]

            # frame errors: at least one wrong info bit
            frame_err = (u_hat != b).any(axis=1)
            fer_cnt += frame_err.sum()

            # bit errors
            ber_cnt += (u_hat != b).sum()

            ber_est = ber_cnt / (batch_size * K * itr)
            fer_est = fer_cnt / (batch_size * itr)

            msg_line = (
                f"ESNO: {esno_db_grid[idx]:.2f}, "
                f"BER: {ber_est:1.4e}, FER: {fer_est:1.4e}, "
                f"BLError: {int(fer_cnt):4d} blocks: {batch_size*itr:10d}"
            )
            print(msg_line)

            if fer_cnt > target_frame_errors:
                break

        BER_list.append(ber_est)
        FER_list.append(fer_est)

        np.savetxt(os.path.join(save_dir, f"BER_{N}_{K}_{file_prefix}.csv"),
                   np.array(BER_list), delimiter=",")
        np.savetxt(os.path.join(save_dir, f"FER_{N}_{K}_{file_prefix}.csv"),
                   np.array(FER_list), delimiter=",")

    return BER_list, FER_list


if __name__ == "__main__":
    # -----------------------------
    # Code parameters
    # -----------------------------
    n = 8
    K = 128
    N = 2 ** n
    batch_size = 1000

    # -----------------------------
    # Frozen/info positions (0-based)
    # -----------------------------
    frozen_pos, info_pos = generate_5g_ranking(K, N)
    frozen_bits = np.array(frozen_pos[: N - K], dtype=np.int32)
    info_bits = np.setdiff1d(np.arange(N, dtype=np.int32), frozen_bits)

    if len(info_bits) != K:
        raise ValueError(
            "info_bits length must equal K. "
            "Replace the placeholder frozen_bits with your 5G ranking output."
        )

    decoder = PolarSCDecoder(frozen_bits, N)

    if decoder is None:
        raise RuntimeError("Instantiate decoder = PolarSCDecoder(frozen_bits, N).")

    # -----------------------------
    # Simulation grid
    # -----------------------------
    esno_db = np.arange(0.0, 1.5, 0.5)

    simulate_sc_ber_fer(
        decoder=decoder,
        frozen_bits=frozen_bits,
        info_bits=info_bits,
        K=K,
        N=N,
        esno_db_grid=esno_db,
        batch_size=batch_size,
        target_frame_errors=100,
        save_dir="results",
        file_prefix="SC",
    )





    
    


