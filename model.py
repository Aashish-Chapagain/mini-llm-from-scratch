import numpy as np 

class Embedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weights = np.random.rand(vocab_size, d_model) * (1/np.sqrt(d_model))
    
    def forward(self, x):
        return self.weights[x] * np.sqrt(self.d_model)

class PositionalEncoding:
    def __init__(self, seq_len, d_model):
        self.seq_len = seq_len
        self.d_model = d_model
        self.positional_encoding = self._create_positional_encoding()

    def _create_positional_encoding(self):
        positions = np.arange(self.seq_len).reshape(-1, 1)
        pe = np.zeros((self.seq_len, self.d_model))
        i = np.arange(0, self.d_model, 2)
        denominator = np.power(10000,  i/self.d_model)
        angles = positions / denominator

        pe[:, 0::2] = np.sin(angles)
        pe[:, 1::2] = np.cos(angles)

        return pe 



    def forward(self, x):
      return x + self.positional_encoding[np.newaxis, :, :]



class LayerNorm: 
    def __init__(self,d_model):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.d_model = d_model

    def forward(self, x ):
        mean = np.mean(x, axis=-1, keepdims=True)
        standard_deviation = np.sqrt(np.var(x, axis=-1, keepdims=True) + 1e-5)
        x_normalized = (x - mean) / standard_deviation
        return self.gamma * x_normalized + self.beta



class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model 
        self.num_heads = num_heads 
        self.depth = d_model // num_heads
        self.w_q = np.random.rand(d_model, d_model) * (1/np.sqrt(d_model))
        self.w_k = np.random.rand(d_model, d_model) * (1/np.sqrt(d_model))
        self.w_v = np.random.rand(d_model, d_model) * (1/np.sqrt(d_model))
        self.w_o = np.random.rand(d_model, d_model) * (1/np.sqrt(d_model))
    


    