import numpy as np



def create_causal_mask(seq_len):
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    return mask[np.newaxis, np.newaxis, :, :]


class Embedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weights = np.random.randn(vocab_size, d_model) * (1.0 / np.sqrt(d_model))
        self.grad_weights = np.zeros_like(self.weights)
        self._cache_x = None

    def forward(self, x):
        self._cache_x = x
        return self.weights[x] * np.sqrt(self.d_model)

    def backward(self, dout):
        self.grad_weights.fill(0.0)
        scaled = dout * np.sqrt(self.d_model)
        np.add.at(self.grad_weights, self._cache_x, scaled)


class PositionalEncoding:
    def __init__(self, seq_len, d_model):
        self.seq_len = seq_len
        self.d_model = d_model
        self.positional_encoding = self._create_positional_encoding()

    def _create_positional_encoding(self):
        positions = np.arange(self.seq_len).reshape(-1, 1)
        pe = np.zeros((self.seq_len, self.d_model))
        i = np.arange(0, self.d_model, 2)
        denominator = np.power(10000, i / self.d_model)
        angles = positions / denominator
        pe[:, 0::2] = np.sin(angles)
        pe[:, 1::2] = np.cos(angles)
        return pe

    def forward(self, x):
        seq_len = x.shape[1]
        return x + self.positional_encoding[np.newaxis, :seq_len, :]


class LayerNorm:
    def __init__(self, d_model):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.grad_gamma = np.zeros_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)
        self.eps = 1e-5
        self._cache = None

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        x_hat = (x - mean) / std
        out = self.gamma * x_hat + self.beta
        self._cache = (x, x_hat, mean, var, std)
        return out

    def backward(self, dout):
        x, x_hat, mean, var, std = self._cache
        n = x.shape[-1]

        self.grad_gamma = np.sum(dout * x_hat, axis=(0, 1))
        self.grad_beta = np.sum(dout, axis=(0, 1))

        dx_hat = dout * self.gamma
        dvar = np.sum(dx_hat * (x - mean) * -0.5 * (var + self.eps) ** (-1.5), axis=-1, keepdims=True)
        dmean = (
            np.sum(dx_hat * -1.0 / std, axis=-1, keepdims=True)
            + dvar * np.mean(-2.0 * (x - mean), axis=-1, keepdims=True)
        )
        dx = dx_hat / std + dvar * 2.0 * (x - mean) / n + dmean / n
        return dx


class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.w_q = np.random.randn(d_model, d_model) * (1.0 / np.sqrt(d_model))
        self.w_k = np.random.randn(d_model, d_model) * (1.0 / np.sqrt(d_model))
        self.w_v = np.random.randn(d_model, d_model) * (1.0 / np.sqrt(d_model))
        self.w_o = np.random.randn(d_model, d_model) * (1.0 / np.sqrt(d_model))

        self.grad_w_q = np.zeros_like(self.w_q)
        self.grad_w_k = np.zeros_like(self.w_k)
        self.grad_w_v = np.zeros_like(self.w_v)
        self.grad_w_o = np.zeros_like(self.w_o)

        self._cache = None

    def _split_heads(self, x):
        b, t, _ = x.shape
        x = x.reshape(b, t, self.num_heads, self.depth)
        return x.transpose(0, 2, 1, 3)

    def _combine_heads(self, x):
        b, h, t, d = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, t, h * d)

    def forward(self, x, mask=None):
        q_lin = np.matmul(x, self.w_q)
        k_lin = np.matmul(x, self.w_k)
        v_lin = np.matmul(x, self.w_v)

        q = self._split_heads(q_lin)
        k = self._split_heads(k_lin)
        v = self._split_heads(v_lin)

        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.depth)
        if mask is not None:
            scores = np.where(mask, -1e9, scores)

        shifted = scores - np.max(scores, axis=-1, keepdims=True)
        attn = np.exp(shifted)
        attn /= np.sum(attn, axis=-1, keepdims=True)

        context = np.matmul(attn, v)
        combined = self._combine_heads(context)
        out = np.matmul(combined, self.w_o)

        self._cache = (x, q, k, v, attn, combined, mask)
        return out

    def backward(self, dout):
        x, q, k, v, attn, combined, mask = self._cache

        self.grad_w_o = np.einsum("bti,btj->ij", combined, dout)
        dcombined = np.matmul(dout, self.w_o.T)
        dcontext = self._split_heads(dcombined)

        dattn = np.matmul(dcontext, v.transpose(0, 1, 3, 2))
        dv = np.matmul(attn.transpose(0, 1, 3, 2), dcontext)

        sum_term = np.sum(dattn * attn, axis=-1, keepdims=True)
        dscores = attn * (dattn - sum_term)
        if mask is not None:
            dscores = np.where(mask, 0.0, dscores)

        scale = 1.0 / np.sqrt(self.depth)
        dq = np.matmul(dscores, k) * scale
        dk = np.matmul(dscores.transpose(0, 1, 3, 2), q) * scale

        dq_lin = self._combine_heads(dq)
        dk_lin = self._combine_heads(dk)
        dv_lin = self._combine_heads(dv)

        self.grad_w_q = np.einsum("bti,btj->ij", x, dq_lin)
        self.grad_w_k = np.einsum("bti,btj->ij", x, dk_lin)
        self.grad_w_v = np.einsum("bti,btj->ij", x, dv_lin)

        dx_q = np.matmul(dq_lin, self.w_q.T)
        dx_k = np.matmul(dk_lin, self.w_k.T)
        dx_v = np.matmul(dv_lin, self.w_v.T)
        return dx_q + dx_k + dx_v


class FeedForward:
    def __init__(self, d_model, d_ff):
        self.w1 = np.random.randn(d_model, d_ff) * (1.0 / np.sqrt(d_model))
        self.b1 = np.zeros(d_ff)
        self.w2 = np.random.randn(d_ff, d_model) * (1.0 / np.sqrt(d_ff))
        self.b2 = np.zeros(d_model)

        self.grad_w1 = np.zeros_like(self.w1)
        self.grad_b1 = np.zeros_like(self.b1)
        self.grad_w2 = np.zeros_like(self.w2)
        self.grad_b2 = np.zeros_like(self.b2)

        self._cache = None

    def forward(self, x):
        z1 = np.matmul(x, self.w1) + self.b1
        a1 = np.maximum(0, z1)
        out = np.matmul(a1, self.w2) + self.b2
        self._cache = (x, z1, a1)
        return out

    def backward(self, dout):
        x, z1, a1 = self._cache
        self.grad_w2 = np.einsum("bti,btj->ij", a1, dout)
        self.grad_b2 = np.sum(dout, axis=(0, 1))

        da1 = np.matmul(dout, self.w2.T)
        dz1 = da1 * (z1 > 0)

        self.grad_w1 = np.einsum("bti,btj->ij", x, dz1)
        self.grad_b1 = np.sum(dz1, axis=(0, 1))
        return np.matmul(dz1, self.w1.T)


class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        self._x = x
        attn_out = self.attention.forward(x, mask)
        self._res1 = x + attn_out
        out1 = self.norm1.forward(self._res1)

        ffn_out = self.ffn.forward(out1)
        self._res2 = out1 + ffn_out
        out2 = self.norm2.forward(self._res2)
        return out2

    def backward(self, dout):
        dres2 = self.norm2.backward(dout)

        dout1_res = dres2
        dffn = dres2
        dout1_ffn = self.ffn.backward(dffn)
        dout1 = dout1_res + dout1_ffn

        dres1 = self.norm1.backward(dout1)
        dx_res = dres1
        dattn = dres1
        dx_attn = self.attention.backward(dattn)
        return dx_res + dx_attn


class AdamOptimizer:
    def __init__(
        self,
        learning_rate=3e-4,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        grad_clip_norm=None,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.m = {}
        self.v = {}
        self.t = 0
    
    def step(self, params_and_grads):
        self.t += 1
        for name, param, grad in params_and_grads:
            grad_used = grad

            # L2 regularization on matrix-like weights (not biases/norm vectors).
            if self.weight_decay > 0.0 and param.ndim > 1:
                grad_used = grad_used + self.weight_decay * param

            # Optional per-parameter gradient clipping for training stability.
            if self.grad_clip_norm is not None:
                grad_norm = np.linalg.norm(grad_used)
                if grad_norm > self.grad_clip_norm:
                    grad_used = grad_used * (self.grad_clip_norm / (grad_norm + 1e-12))

            if name not in self.m:
                self.m[name] = np.zeros_like(grad_used)
                self.v[name] = np.zeros_like(grad_used)

            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad_used
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad_used ** 2)

            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


class FullModel:
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len):
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)
        self.num_layers = int(num_layers)
        self.max_seq_len = int(max_seq_len)

        self.embedding = Embedding(self.vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(self.max_seq_len, self.d_model)
        self.transformer_blocks = [TransformerBlock(self.d_model, self.num_heads, self.d_ff) for _ in range(self.num_layers)]
        self.output_layer = np.random.randn(self.d_model, self.vocab_size) * (1.0 / np.sqrt(self.d_model))
        self.grad_output_layer = np.zeros_like(self.output_layer)
        self._cache_hidden = None

    def forward(self, x, return_hidden=False):
        seq_len = x.shape[1]
        mask = create_causal_mask(seq_len)

        hidden = self.embedding.forward(x)
        hidden = self.positional_encoding.forward(hidden)
        for block in self.transformer_blocks:
            hidden = block.forward(hidden, mask)

        logits = np.matmul(hidden, self.output_layer)
        self._cache_hidden = hidden
        if return_hidden:
            return logits, hidden
        return logits

    def backward(self, dlogits):
        hidden = self._cache_hidden
        self.grad_output_layer = np.einsum("bti,btj->ij", hidden, dlogits)
        dhidden = np.matmul(dlogits, self.output_layer.T)

        for block in reversed(self.transformer_blocks):
            dhidden = block.backward(dhidden)

        self.embedding.backward(dhidden)

    def parameters_and_grads(self):
        params = [("output_layer", self.output_layer, self.grad_output_layer)]
        params.append(("embedding.weights", self.embedding.weights, self.embedding.grad_weights))

        for i, block in enumerate(self.transformer_blocks):
            params.extend(
                [
                    (f"blocks.{i}.attn.w_q", block.attention.w_q, block.attention.grad_w_q),
                    (f"blocks.{i}.attn.w_k", block.attention.w_k, block.attention.grad_w_k),
                    (f"blocks.{i}.attn.w_v", block.attention.w_v, block.attention.grad_w_v),
                    (f"blocks.{i}.attn.w_o", block.attention.w_o, block.attention.grad_w_o),
                    (f"blocks.{i}.ffn.w1", block.ffn.w1, block.ffn.grad_w1),
                    (f"blocks.{i}.ffn.b1", block.ffn.b1, block.ffn.grad_b1),
                    (f"blocks.{i}.ffn.w2", block.ffn.w2, block.ffn.grad_w2),
                    (f"blocks.{i}.ffn.b2", block.ffn.b2, block.ffn.grad_b2),
                    (f"blocks.{i}.norm1.gamma", block.norm1.gamma, block.norm1.grad_gamma),
                    (f"blocks.{i}.norm1.beta", block.norm1.beta, block.norm1.grad_beta),
                    (f"blocks.{i}.norm2.gamma", block.norm2.gamma, block.norm2.grad_gamma),
                    (f"blocks.{i}.norm2.beta", block.norm2.beta, block.norm2.grad_beta),
                ]
            )
        return params

    def save_block_checkpoints(self, file_path="block_checkpoints.npz"):
        state = {
            "attn_w_q": np.array([block.attention.w_q for block in self.transformer_blocks]),
            "attn_w_k": np.array([block.attention.w_k for block in self.transformer_blocks]),
            "attn_w_v": np.array([block.attention.w_v for block in self.transformer_blocks]),
            "attn_w_o": np.array([block.attention.w_o for block in self.transformer_blocks]),
            "ffn_w1": np.array([block.ffn.w1 for block in self.transformer_blocks]),
            "ffn_b1": np.array([block.ffn.b1 for block in self.transformer_blocks]),
            "ffn_w2": np.array([block.ffn.w2 for block in self.transformer_blocks]),
            "ffn_b2": np.array([block.ffn.b2 for block in self.transformer_blocks]),
            "norm_gamma": np.array([block.norm1.gamma for block in self.transformer_blocks]),
            "norm_beta": np.array([block.norm1.beta for block in self.transformer_blocks]),
            "norm2_gamma": np.array([block.norm2.gamma for block in self.transformer_blocks]),
            "norm2_beta": np.array([block.norm2.beta for block in self.transformer_blocks]),
            "num_layers": np.array([self.num_layers], dtype=np.int64),
        }
        np.savez(file_path, **state)

    def load_block_checkpoints(self, file_path="block_checkpoints.npz"):
        ckpt = np.load(file_path)
        num_layers = int(ckpt["num_layers"][0])
        if num_layers != self.num_layers:
            raise ValueError(f"Block checkpoint mismatch. checkpoint={num_layers}, model={self.num_layers}")

        for i, block in enumerate(self.transformer_blocks):
            block.attention.w_q = ckpt["attn_w_q"][i]
            block.attention.w_k = ckpt["attn_w_k"][i]
            block.attention.w_v = ckpt["attn_w_v"][i]
            block.attention.w_o = ckpt["attn_w_o"][i]
            block.ffn.w1 = ckpt["ffn_w1"][i]
            block.ffn.b1 = ckpt["ffn_b1"][i]
            block.ffn.w2 = ckpt["ffn_w2"][i]
            block.ffn.b2 = ckpt["ffn_b2"][i]
            block.norm1.gamma = ckpt["norm_gamma"][i]
            block.norm1.beta = ckpt["norm_beta"][i]
            block.norm2.gamma = ckpt["norm2_gamma"][i]
            block.norm2.beta = ckpt["norm2_beta"][i]

    def save_weights(self, file_path="model_weights.npz"):
        state = {
            "embedding_weights": self.embedding.weights,
            "output_layer": self.output_layer,
            "meta": np.array(
                [self.vocab_size, self.d_model, self.num_heads, self.d_ff, self.num_layers, self.max_seq_len],
                dtype=np.int64,
            ),
        }
        state.update(
            {
                "attn_w_q": np.array([block.attention.w_q for block in self.transformer_blocks]),
                "attn_w_k": np.array([block.attention.w_k for block in self.transformer_blocks]),
                "attn_w_v": np.array([block.attention.w_v for block in self.transformer_blocks]),
                "attn_w_o": np.array([block.attention.w_o for block in self.transformer_blocks]),
                "ffn_w1": np.array([block.ffn.w1 for block in self.transformer_blocks]),
                "ffn_b1": np.array([block.ffn.b1 for block in self.transformer_blocks]),
                "ffn_w2": np.array([block.ffn.w2 for block in self.transformer_blocks]),
                "ffn_b2": np.array([block.ffn.b2 for block in self.transformer_blocks]),
                "norm_gamma": np.array([block.norm1.gamma for block in self.transformer_blocks]),
                "norm_beta": np.array([block.norm1.beta for block in self.transformer_blocks]),
                "norm2_gamma": np.array([block.norm2.gamma for block in self.transformer_blocks]),
                "norm2_beta": np.array([block.norm2.beta for block in self.transformer_blocks]),
            }
        )
        np.savez(file_path, **state)

    def load_weights(self, file_path="model_weights.npz"):
        ckpt = np.load(file_path)
        meta = ckpt["meta"].astype(np.int64)
        expected = np.array(
            [self.vocab_size, self.d_model, self.num_heads, self.d_ff, self.num_layers, self.max_seq_len],
            dtype=np.int64,
        )
        if not np.array_equal(meta, expected):
            raise ValueError(
                "Checkpoint architecture mismatch. "
                f"checkpoint={meta.tolist()}, model={expected.tolist()}"
            )

        self.embedding.weights = ckpt["embedding_weights"]
        self.output_layer = ckpt["output_layer"]

        for i, block in enumerate(self.transformer_blocks):
            block.attention.w_q = ckpt["attn_w_q"][i]
            block.attention.w_k = ckpt["attn_w_k"][i]
            block.attention.w_v = ckpt["attn_w_v"][i]
            block.attention.w_o = ckpt["attn_w_o"][i]
            block.ffn.w1 = ckpt["ffn_w1"][i]
            block.ffn.b1 = ckpt["ffn_b1"][i]
            block.ffn.w2 = ckpt["ffn_w2"][i]
            block.ffn.b2 = ckpt["ffn_b2"][i]
            block.norm1.gamma = ckpt["norm_gamma"][i]
            block.norm1.beta = ckpt["norm_beta"][i]
            block.norm2.gamma = ckpt["norm2_gamma"][i]
            block.norm2.beta = ckpt["norm2_beta"][i]
