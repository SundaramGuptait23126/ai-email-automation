from dataclasses import dataclass

@dataclass
class SLMConfig:
    vocab_size: int = 4096       # Vocabulary size
    d_model: int = 256           # Embedding dimension (small & fast for local inference)
    n_layers: int = 6            # Number of Transformer layers
    n_heads: int = 8             # Number of attention heads
    d_ff: int = 768              # Feed-forward hidden dimension (SwiGLU)
    max_seq_len: int = 1024      # Max sequence length (Email threads + response)
    dropout: float = 0.1
    eps: float = 1e-6            # RMSNorm epsilon
    rope_theta: float = 10000.0  # Rotary position embedding base
    tie_word_embeddings: bool = True
