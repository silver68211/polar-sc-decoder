
import numpy as np
import tensorflow as tf 

class PolarSCDecoder:
    """
    Successive Cancellation (SC) decoder for polar codes.

    Notes
    -----
    - Input: channel LLRs of shape (..., n)
    - Output:
        u_hat_n : estimated source bits u_1^n (same shape as input)
        llr_out : the per-bit decision LLRs at leaves (same shape as input)
    - frozen_pos are 0-based indices in [0, n-1]
    """

    def __init__(self, frozen_pos, n, output_dtype=np.float32):
        if output_dtype not in (np.float16, np.float32, np.float64):
            raise ValueError("output_dtype must be one of {np.float16, np.float32, np.float64}.")

        n = int(n)
        if len(frozen_pos) > n:
            raise ValueError("Num. of elements in frozen_pos cannot be greater than n.")
        if not (np.log2(n).is_integer()):
            raise ValueError("n must be a power of 2.")

        self._n = n
        self._frozen_pos = np.array(frozen_pos, dtype=np.int64)
        self._k = self._n - len(self._frozen_pos)
        self._info_pos = np.setdiff1d(np.arange(self._n, dtype=np.int64), self._frozen_pos)

        if self._k != len(self._info_pos):
            raise RuntimeError("Internal error: invalid info_pos generated.")

        self._output_dtype = output_dtype
        self._llr_max = 80.0  # internal clipping for numerical stability

        # frozen indicator vector: 1 for frozen, 0 for info
        self._frozen_ind = np.zeros(self._n, dtype=np.int8)
        self._frozen_ind[self._frozen_pos] = 1

    # -----------------------------
    # Public properties
    # -----------------------------
    @property
    def n(self):
        return self._n

    @property
    def k(self):
        return self._k

    @property
    def frozen_pos(self):
        return self._frozen_pos

    @property
    def info_pos(self):
        return self._info_pos

    @property
    def llr_max(self):
        return self._llr_max

    @property
    def output_dtype(self):
        return self._output_dtype

    # -----------------------------
    # Internal LLR update rules
    # -----------------------------
    def _clip(self, x):
        return np.clip(x, -self._llr_max, self._llr_max)

    def _cn_op(self, x, y):
        """
        Check-node update (boxplus) for LLR inputs using stable log-domain form:
            f(x,y) = log(1+exp(x+y)) - log(exp(x)+exp(y))
        """
        x_in = self._clip(x)
        y_in = self._clip(y)

        # log(1+exp(x+y))
        a = np.log1p(np.exp(x_in + y_in))
        # log(exp(x)+exp(y)) = logsumexp([x,y])
        m = np.maximum(x_in, y_in)
        b = m + np.log(np.exp(x_in - m) + np.exp(y_in - m))

        return a - b

    def _vn_op(self, x, y, u_hat):
        """Variable-node update for LLR inputs: g(x,y,u) = (1-2u)*x + y"""
        return (1.0 - 2.0 * u_hat) * x + y

    # -----------------------------
    # Recursive SC decoder
    # -----------------------------
    def _polar_decode_sc(self, llr_ch, frozen_ind):
        """
        Recursive SC decoding.

        Parameters
        ----------
        llr_ch : np.ndarray, shape (B, n_local)
            LLRs at current node for a batch of B codewords.
        frozen_ind : np.ndarray, shape (n_local,)
            Frozen indicator for local indices: 1 frozen, 0 info.

        Returns
        -------
        u_hat : np.ndarray, shape (B, n_local)
            Estimated u bits at this node.
        llr_out : np.ndarray, shape (B, n_local)
            Leaf decision LLRs aligned to local ordering.
        u_hat_up : np.ndarray, shape (B, n_local)
            Re-encoded (partial-sum) vector to pass upward.
        """
        n_local = frozen_ind.shape[0]

        if n_local > 1:
            h = n_local // 2

            llr1 = llr_ch[:, :h]
            llr2 = llr_ch[:, h:]
            fr1 = frozen_ind[:h]
            fr2 = frozen_ind[h:]

            # upper branch
            x1_in = self._cn_op(llr1, llr2)
            u1, llr1_out, u1_up = self._polar_decode_sc(x1_in, fr1)

            # lower branch
            x2_in = self._vn_op(llr1, llr2, u1_up)
            u2, llr2_out, u2_up = self._polar_decode_sc(x2_in, fr2)

            # combine estimates
            u_hat = np.concatenate([u1, u2], axis=1)
            llr_out = np.concatenate([llr1_out, llr2_out], axis=1)

            # compute partial sums for upward pass:
            # u_up = [u1 xor u2, u2]
            u1_int = u1_up.astype(np.int8)
            u2_int = u2_up.astype(np.int8)
            u1_xor = np.bitwise_xor(u1_int, u2_int).astype(np.float32)
            u_hat_up = np.concatenate([u1_xor, u2_up], axis=1)

            return u_hat, llr_out, u_hat_up

        # leaf node: hard decision (or frozen)
        if int(frozen_ind[0]) == 1:
            u_hat = np.zeros((llr_ch.shape[0], 1), dtype=np.float32)
            u_hat_up = u_hat
            llr_out = llr_ch
            return u_hat, llr_out, u_hat_up

        # hard decision: u = 0 if L>=0 else 1
        # match your TF rule: u = 0.5*(1 - sign(L)), with sign(0)->0 => u=0.5, then set to 1
        s = np.sign(llr_ch)
        u_hat = 0.5 * (1.0 - s)
        u_hat = np.where(u_hat == 0.5, 1.0, u_hat).astype(np.float32)

        llr_out = llr_ch
        u_hat_up = u_hat
        return u_hat, llr_out, u_hat_up

    # -----------------------------
    # Public decode API
    # -----------------------------
    def decode(self, inputs):
        """
        SC decode.

        Parameters
        ----------
        inputs : np.ndarray, shape (..., n)
            Channel logits/LLRs. To match your TF code, we negate them once:
            llr_ch = -inputs

        Returns
        -------
        u_hat_n : np.ndarray, shape (..., n)
        llr_leaf : np.ndarray, shape (..., n)
        """
        x = np.asarray(inputs)
        if x.shape[-1] != self._n:
            raise ValueError(f"Last input dimension must be of length n={self._n}. Got {x.shape[-1]}.")
        if x.ndim < 2:
            raise ValueError("Inputs must have at least 2 dimensions: (batch, n) or (..., n).")

        # flatten batch dimensions
        orig_shape = x.shape
        b = int(np.prod(orig_shape[:-1]))
        llr_ch = x.reshape(b, self._n).astype(np.float32)

        # match your TF: logits -> "true" LLRs
        llr_ch =  llr_ch

        u_hat, llr_leaf, _ = self._polar_decode_sc(llr_ch, self._frozen_ind)
        llr_leaf = self._clip(llr_leaf).astype(self._output_dtype)
        u_hat = u_hat.astype(self._output_dtype)

        # reshape back
        u_hat = u_hat.reshape(*orig_shape[:-1], self._n)
        llr_leaf = llr_leaf.reshape(*orig_shape[:-1], self._n)
        return u_hat, llr_leaf


