"""
TRANSFORMER ARCHITECTURE: A COMPREHENSIVE GUIDE
================================================
This file explains Transformers from first principles with mathematical foundations
and code implementations showing what happens under the hood.

Author: Educational Tutorial
Purpose: Understand Transformers deeply without relying on black-box libraries
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import math

# ============================================================================
# PART 1: FUNDAMENTAL BUILDING BLOCKS
# ============================================================================

class TransformerMath:
    """
    Mathematical foundations of Transformers explained step by step.
    """
    
    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Softmax function: Converts logits to probabilities.
        
        Mathematics:
        ------------
        softmax(x_i) = exp(x_i) / Σ exp(x_j)
        
        This creates a probability distribution where:
        - All values are between 0 and 1
        - All values sum to 1
        - Larger inputs get exponentially larger probabilities
        
        Statistical Interpretation:
        - Transforms raw scores into a normalized probability distribution
        - Used in attention to weight the importance of different positions
        
        Numerical Stability:
        - We subtract max(x) to prevent overflow: softmax(x) = softmax(x - c)
        """
        # Subtract max for numerical stability
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    @staticmethod
    def layer_norm(x: np.ndarray, epsilon: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Layer Normalization: Normalize activations across features.
        
        Mathematics:
        ------------
        μ = (1/d) Σ x_i                    (mean)
        σ² = (1/d) Σ (x_i - μ)²           (variance)
        x_norm = (x - μ) / sqrt(σ² + ε)   (normalization)
        
        Statistical Interpretation:
        - Centers the distribution (zero mean)
        - Standardizes the scale (unit variance)
        - Makes training more stable by preventing internal covariate shift
        
        Why Layer Norm vs Batch Norm?
        - Layer norm normalizes across features (independent of batch)
        - Better for sequences where batch statistics are unreliable
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x_normalized = (x - mean) / np.sqrt(variance + epsilon)
        return x_normalized, mean, variance


# ============================================================================
# PART 2: POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding:
    """
    Positional Encoding: Add position information to embeddings.
    
    THE PROBLEM:
    ------------
    Transformers process all tokens in parallel (unlike RNNs).
    They have no inherent notion of sequence order!
    
    THE SOLUTION:
    -------------
    Add position-dependent patterns to embeddings so the model knows:
    - Where each token is in the sequence
    - The relative distances between tokens
    
    Mathematics:
    ------------
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    where:
    - pos: position in sequence (0, 1, 2, ...)
    - i: dimension index (0, 1, 2, ..., d_model/2)
    - d_model: embedding dimension
    
    Why Sinusoidal?
    ---------------
    1. Unique encoding for each position
    2. Can generalize to longer sequences than seen in training
    3. Relative positions can be expressed as linear functions
    4. Different frequencies capture different scales of position
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Args:
            d_model: Dimension of embeddings
            max_len: Maximum sequence length to pre-compute
        """
        self.d_model = d_model
        
        # Create position encodings
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]  # (max_len, 1)
        
        # Compute the division term: 10000^(2i/d_model)
        # This creates different frequencies for different dimensions
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = np.sin(position * div_term)
        
        # Apply cos to odd indices
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe
    
    def encode(self, seq_len: int) -> np.ndarray:
        """
        Get positional encodings for a sequence.
        
        Returns:
            Positional encodings of shape (seq_len, d_model)
        """
        return self.pe[:seq_len, :]
    
    def visualize(self, max_pos: int = 100):
        """
        Visualize the positional encoding patterns.
        """
        plt.figure(figsize=(15, 5))
        plt.imshow(self.pe[:max_pos, :].T, aspect='auto', cmap='RdBu')
        plt.colorbar()
        plt.xlabel('Position in Sequence')
        plt.ylabel('Embedding Dimension')
        plt.title('Positional Encoding Patterns\n(Different frequencies for different dimensions)')
        plt.tight_layout()
        return plt


# ============================================================================
# PART 3: ATTENTION MECHANISM - THE HEART OF TRANSFORMERS
# ============================================================================

class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention: The core mechanism of Transformers.
    
    THE INTUITION:
    --------------
    "Which parts of the input should I pay attention to?"
    
    For each position (query), we:
    1. Compare it with all positions (keys)
    2. Get similarity scores
    3. Use scores to weight the values
    4. Return weighted combination
    
    Mathematics:
    ------------
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    where:
    - Q (Query): "What am I looking for?" (seq_len, d_k)
    - K (Key): "What do I contain?" (seq_len, d_k)
    - V (Value): "What do I actually output?" (seq_len, d_v)
    - d_k: Dimension of queries and keys
    
    Step-by-Step Process:
    ---------------------
    1. Compute similarity: QK^T gives (seq_len, seq_len) scores
    2. Scale by √d_k to prevent large magnitudes
    3. Apply softmax to get attention weights (probabilities)
    4. Multiply by V to get weighted combination
    
    Statistical Interpretation:
    ---------------------------
    - QK^T: Dot product measures cosine similarity (when normalized)
    - Softmax: Converts similarities to probability distribution
    - Weighted sum: Expected value under attention distribution
    
    Why Scale by √d_k?
    ------------------
    Without scaling, dot products grow large for high dimensions.
    Large values push softmax into saturation (gradients → 0).
    Scaling keeps variance approximately 1.
    """
    
    def __init__(self, dropout: float = 0.1):
        self.dropout = dropout
    
    def __call__(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                 mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Queries (batch, [heads], seq_len, d_k)
            K: Keys (batch, [heads], seq_len, d_k)
            V: Values (batch, [heads], seq_len, d_v)
            mask: Optional mask (batch, seq_len, seq_len)
        
        Returns:
            output: Attention output (batch, [heads], seq_len, d_v)
            attention_weights: Attention probabilities (batch, [heads], seq_len, seq_len)
        """
        d_k = Q.shape[-1]
        
        # Step 1: Compute attention scores
        # QK^T: How similar is each query to each key?
        # Shape: (batch, [heads], seq_len_q, seq_len_k)
        # Need to transpose last two dimensions of K
        scores = np.matmul(Q, K.swapaxes(-2, -1))
        
        # Step 2: Scale by √d_k
        # This prevents the dot products from growing too large
        scores = scores / np.sqrt(d_k)
        
        # Step 3: Apply mask (if provided)
        # Masking is used to:
        # - Prevent attending to padding tokens
        # - Implement causal attention (for autoregressive models)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Step 4: Apply softmax to get attention weights
        # This converts scores to a probability distribution
        # Each row sums to 1 and represents "how much to attend to each position"
        attention_weights = TransformerMath.softmax(scores, axis=-1)
        
        # Step 5: Apply dropout (in training)
        # Randomly zero out some attention weights for regularization
        if self.dropout > 0:
            # In actual implementation, apply dropout here
            pass
        
        # Step 6: Apply attention weights to values
        # This is a weighted sum: we combine values according to attention weights
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    @staticmethod
    def visualize_attention(attention_weights: np.ndarray, 
                           tokens: list = None,
                           title: str = "Attention Weights"):
        """
        Visualize attention patterns.
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(attention_weights, cmap='viridis')
        plt.colorbar(label='Attention Weight')
        
        if tokens is not None:
            plt.xticks(range(len(tokens)), tokens, rotation=90)
            plt.yticks(range(len(tokens)), tokens)
        
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.title(title)
        plt.tight_layout()
        return plt


# ============================================================================
# PART 4: MULTI-HEAD ATTENTION
# ============================================================================

class MultiHeadAttention:
    """
    Multi-Head Attention: Multiple attention mechanisms in parallel.
    
    THE MOTIVATION:
    ---------------
    Single attention is like having one "view" of the relationships.
    Multiple heads allow the model to attend to different aspects:
    - Head 1: Might learn syntactic relationships
    - Head 2: Might learn semantic relationships
    - Head 3: Might learn positional patterns
    - etc.
    
    Mathematics:
    ------------
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    
    where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
    
    Process:
    --------
    1. Project Q, K, V into h different subspaces (h = number of heads)
    2. Apply attention in each subspace independently
    3. Concatenate the results
    4. Project back to d_model dimensions
    
    Key Insight:
    ------------
    Instead of one attention with d_model dimensions,
    we use h attentions with d_model/h dimensions each.
    This is MORE EXPRESSIVE at the same computational cost!
    
    Parameter Count:
    ----------------
    - W^Q: (d_model, d_model)
    - W^K: (d_model, d_model)
    - W^V: (d_model, d_model)
    - W^O: (d_model, d_model)
    Total: 4 * d_model^2 parameters
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Initialize projection matrices
        # In practice, these are learned parameters
        self.W_q = self._init_weights((d_model, d_model))
        self.W_k = self._init_weights((d_model, d_model))
        self.W_v = self._init_weights((d_model, d_model))
        self.W_o = self._init_weights((d_model, d_model))
        
        self.attention = ScaledDotProductAttention(dropout)
    
    def _init_weights(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Initialize weights using Xavier/Glorot initialization.
        
        Why Xavier?
        -----------
        Maintains variance of activations across layers.
        Variance(output) ≈ Variance(input)
        
        Formula: W ~ N(0, 2/(n_in + n_out))
        """
        fan_in, fan_out = shape
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)
    
    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Split the last dimension into (num_heads, d_k).
        
        Input: (batch, seq_len, d_model)
        Output: (batch, num_heads, seq_len, d_k)
        
        This reshaping allows us to process all heads in parallel.
        """
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Inverse of split_heads.
        
        Input: (batch, num_heads, seq_len, d_k)
        Output: (batch, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def __call__(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                 mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply multi-head attention.
        
        What happens under the hood:
        ----------------------------
        1. Linear projections: Q, K, V → multiple subspaces
        2. Split into heads: Reshape for parallel processing
        3. Attention in each head: Independently compute attention
        4. Concatenate heads: Combine information from all heads
        5. Final projection: Mix information across heads
        """
        batch_size = Q.shape[0]
        
        # Step 1: Linear projections
        # This is what happens when you call a linear layer in PyTorch/TF
        Q = np.matmul(Q, self.W_q)  # (batch, seq_len, d_model)
        K = np.matmul(K, self.W_k)
        V = np.matmul(V, self.W_v)
        
        # Step 2: Split into multiple heads
        Q = self._split_heads(Q)  # (batch, num_heads, seq_len, d_k)
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        # Step 3: Apply attention in each head
        # All heads are processed in parallel via broadcasting
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        
        # Step 4: Concatenate heads
        attn_output = self._combine_heads(attn_output)
        
        # Step 5: Final linear projection
        output = np.matmul(attn_output, self.W_o)
        
        return output, attention_weights


# ============================================================================
# PART 5: FEED-FORWARD NETWORK
# ============================================================================

class FeedForward:
    """
    Position-wise Feed-Forward Network.
    
    Architecture:
    -------------
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    
    This is applied to each position independently and identically.
    
    Key Points:
    -----------
    1. Two linear transformations with ReLU activation
    2. Typically d_ff = 4 * d_model (expansion then compression)
    3. Applied position-wise: same network for each position
    
    Why is this needed?
    -------------------
    - Attention is linear in values (weighted sum)
    - FFN adds non-linearity and increases model capacity
    - The expansion (d_model → d_ff) allows learning complex transformations
    
    What happens under the hood:
    ----------------------------
    1. Expand: Project to higher dimension (d_model → d_ff)
    2. Non-linearity: Apply ReLU (introduces non-linear transformations)
    3. Compress: Project back to model dimension (d_ff → d_model)
    
    This is equivalent to a 1x1 convolution in CNNs!
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (typically 4 * d_model)
            dropout: Dropout rate
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Initialize weights
        self.W_1 = self._init_weights((d_model, d_ff))
        self.b_1 = np.zeros(d_ff)
        self.W_2 = self._init_weights((d_ff, d_model))
        self.b_2 = np.zeros(d_model)
    
    def _init_weights(self, shape: Tuple[int, int]) -> np.ndarray:
        """Xavier initialization."""
        fan_in, fan_out = shape
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """
        ReLU activation: max(0, x)
        
        Why ReLU?
        ---------
        - Simple and efficient
        - Doesn't saturate for positive values
        - Introduces sparsity (many zeros)
        """
        return np.maximum(0, x)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply feed-forward network.
        
        What happens:
        -------------
        When you call nn.Linear() in PyTorch or Dense() in TensorFlow:
        1. Matrix multiplication: x @ W
        2. Add bias: result + b
        3. Apply activation: ReLU(result)
        """
        # First linear transformation + ReLU
        hidden = self.relu(np.matmul(x, self.W_1) + self.b_1)
        
        # Dropout (in training)
        if self.dropout > 0:
            # mask = np.random.binomial(1, 1-self.dropout, hidden.shape)
            # hidden = hidden * mask / (1 - self.dropout)
            pass
        
        # Second linear transformation
        output = np.matmul(hidden, self.W_2) + self.b_2
        
        return output


# ============================================================================
# PART 6: ENCODER LAYER
# ============================================================================

class EncoderLayer:
    """
    Single Transformer Encoder Layer.
    
    Architecture:
    -------------
    x → [Multi-Head Attention] → Add & Norm → [Feed-Forward] → Add & Norm → output
         ↑________________________↓              ↑__________________↓
              (residual connection)                (residual connection)
    
    Components:
    -----------
    1. Multi-Head Self-Attention
    2. Add & Normalize (Residual + Layer Norm)
    3. Feed-Forward Network
    4. Add & Normalize (Residual + Layer Norm)
    
    Residual Connections:
    ---------------------
    output = LayerNorm(x + Sublayer(x))
    
    Why residual connections?
    - Enable gradient flow in deep networks
    - Allow lower layers to be bypassed if needed
    - Stabilize training
    
    Why Layer Normalization?
    - Stabilizes training by normalizing activations
    - Reduces internal covariate shift
    - Makes the model less sensitive to initialization
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout = dropout
        
        # Layer norm parameters (learned)
        self.gamma_1 = np.ones(d_model)
        self.beta_1 = np.zeros(d_model)
        self.gamma_2 = np.ones(d_model)
        self.beta_2 = np.zeros(d_model)
    
    def layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """
        Layer normalization with learnable parameters.
        
        output = gamma * ((x - mean) / std) + beta
        
        gamma and beta are learned, allowing the model to:
        - Adjust the scale of normalization
        - Shift the normalized values
        """
        x_norm, _, _ = TransformerMath.layer_norm(x)
        return gamma * x_norm + beta
    
    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Process input through encoder layer.
        
        What happens under the hood when you call encoder_layer(x):
        -----------------------------------------------------------
        1. Self-attention: Each position attends to all positions
        2. Residual: Add input to attention output
        3. Normalize: Stabilize the distribution
        4. Feed-forward: Apply non-linear transformation
        5. Residual: Add previous output to FFN output
        6. Normalize: Final stabilization
        """
        # Self-attention sublayer
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.layer_norm(x + attn_output, self.gamma_1, self.beta_1)
        
        # Feed-forward sublayer
        ff_output = self.feed_forward(x)
        x = self.layer_norm(x + ff_output, self.gamma_2, self.beta_2)
        
        return x


# ============================================================================
# PART 7: COMPLETE ENCODER
# ============================================================================

class TransformerEncoder:
    """
    Complete Transformer Encoder.
    
    Architecture:
    -------------
    Input → Embedding → Positional Encoding → [N × Encoder Layers] → Output
    
    This is used in:
    - BERT (Bidirectional Encoder Representations from Transformers)
    - Encoder portion of Encoder-Decoder models (e.g., Translation)
    - Classification tasks
    """
    
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 d_ff: int, num_layers: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            num_layers: Number of encoder layers
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        self.d_model = d_model
        
        # Embedding layer
        # This converts token IDs to dense vectors
        self.embedding = self._init_embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Stack of encoder layers
        self.layers = [EncoderLayer(d_model, num_heads, d_ff, dropout) 
                      for _ in range(num_layers)]
        
        self.dropout = dropout
    
    def _init_embedding(self, vocab_size: int, d_model: int) -> np.ndarray:
        """
        Initialize embedding matrix.
        
        What this does:
        ---------------
        Creates a lookup table: token_id → vector
        Shape: (vocab_size, d_model)
        
        When you call embedding(token_ids):
        - It's just indexing: embedding_matrix[token_ids]
        - No matrix multiplication needed!
        """
        return np.random.randn(vocab_size, d_model) * 0.02
    
    def embed(self, x: np.ndarray) -> np.ndarray:
        """
        Convert token IDs to embeddings.
        
        Args:
            x: Token IDs (batch, seq_len)
        
        Returns:
            Embeddings (batch, seq_len, d_model)
        """
        return self.embedding[x]
    
    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Encode input sequence.
        
        What happens when you call transformer_encoder(input_ids):
        ----------------------------------------------------------
        1. Embedding lookup: Convert IDs to vectors
        2. Scale embeddings: Multiply by √d_model (stabilizes training)
        3. Add positional encoding: Inject position information
        4. Apply dropout: Regularization
        5. Pass through encoder layers: Transform representations
        """
        seq_len = x.shape[1]
        
        # Step 1: Embed tokens
        x = self.embed(x)
        
        # Step 2: Scale embeddings
        # This scaling is used in the original paper
        x = x * np.sqrt(self.d_model)
        
        # Step 3: Add positional encoding
        x = x + self.pos_encoder.encode(seq_len)
        
        # Step 4: Dropout (in training)
        
        # Step 5: Apply encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


# ============================================================================
# PART 8: DECODER LAYER (FOR SEQUENCE GENERATION)
# ============================================================================

class DecoderLayer:
    """
    Single Transformer Decoder Layer.
    
    Architecture:
    -------------
    x → [Masked Self-Attention] → Add & Norm 
      → [Cross-Attention] → Add & Norm 
      → [Feed-Forward] → Add & Norm → output
    
    Key Differences from Encoder:
    ------------------------------
    1. Masked Self-Attention: Prevents attending to future positions
    2. Cross-Attention: Attends to encoder output (for translation, etc.)
    
    Masked Attention:
    -----------------
    In autoregressive generation (like GPT), we can't see future tokens.
    We apply a causal mask:
    
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]
    
    Position i can only attend to positions <= i.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer norm parameters
        self.gamma_1 = np.ones(d_model)
        self.beta_1 = np.zeros(d_model)
        self.gamma_2 = np.ones(d_model)
        self.beta_2 = np.zeros(d_model)
        self.gamma_3 = np.ones(d_model)
        self.beta_3 = np.zeros(d_model)
    
    def layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        x_norm, _, _ = TransformerMath.layer_norm(x)
        return gamma * x_norm + beta
    
    def __call__(self, x: np.ndarray, encoder_output: np.ndarray,
                 src_mask: Optional[np.ndarray] = None,
                 tgt_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Process input through decoder layer.
        
        Args:
            x: Target sequence
            encoder_output: Output from encoder
            src_mask: Mask for encoder output
            tgt_mask: Causal mask for target sequence
        """
        # Masked self-attention
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.layer_norm(x + attn_output, self.gamma_1, self.beta_1)
        
        # Cross-attention to encoder output
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.layer_norm(x + attn_output, self.gamma_2, self.beta_2)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.layer_norm(x + ff_output, self.gamma_3, self.beta_3)
        
        return x


# ============================================================================
# PART 9: PRACTICAL EXAMPLE - TEXT CLASSIFICATION
# ============================================================================

def example_sentiment_analysis():
    """
    Example: Using Transformer Encoder for sentiment analysis.
    
    This shows what happens when you use transformers in practice.
    """
    print("=" * 80)
    print("EXAMPLE: SENTIMENT ANALYSIS WITH TRANSFORMERS")
    print("=" * 80)
    
    # Hyperparameters
    vocab_size = 10000
    d_model = 128
    num_heads = 4
    d_ff = 512
    num_layers = 2
    max_len = 50
    
    print(f"\nModel Configuration:")
    print(f"  Vocabulary Size: {vocab_size}")
    print(f"  Model Dimension: {d_model}")
    print(f"  Number of Heads: {num_heads}")
    print(f"  FF Dimension: {d_ff}")
    print(f"  Number of Layers: {num_layers}")
    
    # Create encoder
    encoder = TransformerEncoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
    
    # Simulate input: batch of 2 sentences, each 10 tokens
    # In practice, these would be token IDs from a tokenizer
    batch_size = 2
    seq_len = 10
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape}")
    print(f"  batch_size: {batch_size}")
    print(f"  sequence_length: {seq_len}")
    
    # Encode
    print("\nEncoding sequence...")
    output = encoder(input_ids)
    
    print(f"Output shape: {output.shape}")
    print(f"  Shape: (batch_size, seq_len, d_model)")
    print(f"  Each token is now represented as a {d_model}-dimensional vector")
    print(f"  that contains contextual information from the entire sequence!")
    
    # For classification, we typically use the first token ([CLS] in BERT)
    cls_output = output[:, 0, :]  # (batch_size, d_model)
    print(f"\n[CLS] token representation shape: {cls_output.shape}")
    print("This would be fed to a classification head (linear layer)")
    
    # Parameter count
    total_params = 0
    total_params += vocab_size * d_model  # Embeddings
    total_params += num_layers * (4 * d_model * d_model)  # Attention projections
    total_params += num_layers * (d_model * d_ff + d_ff * d_model)  # FFN
    total_params += num_layers * (4 * d_model)  # Layer norm parameters
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"  This is what gets trained during fine-tuning!")
    
    return encoder, output


# ============================================================================
# PART 10: UNDERSTANDING ATTENTION PATTERNS
# ============================================================================

def demonstrate_attention():
    """
    Demonstrate how attention works with a simple example.
    """
    print("\n" + "=" * 80)
    print("ATTENTION MECHANISM DEMONSTRATION")
    print("=" * 80)
    
    # Create a simple sequence
    # Let's say we have 5 positions
    seq_len = 5
    d_k = 8
    batch = 1
    
    # Create simple query, key, value matrices
    # In practice, these come from linear projections of embeddings
    np.random.seed(42)
    Q = np.random.randn(batch, seq_len, d_k)
    K = np.random.randn(batch, seq_len, d_k)
    V = np.random.randn(batch, seq_len, d_k)
    
    print(f"\nInput shapes:")
    print(f"  Q (Queries): {Q.shape}")
    print(f"  K (Keys): {K.shape}")
    print(f"  V (Values): {V.shape}")
    
    # Apply attention
    attention = ScaledDotProductAttention()
    output, weights = attention(Q, K, V)
    
    print(f"\nAttention weights shape: {weights.shape}")
    print(f"  This is a {seq_len}x{seq_len} matrix")
    print(f"  Each row shows how much each position attends to all positions")
    
    # Show attention weights for first position
    print(f"\nAttention weights for position 0:")
    print(f"  {weights[0, 0, :]}")
    print(f"  Sum: {weights[0, 0, :].sum():.6f} (should be 1.0)")
    
    # Visualize
    tokens = ['The', 'cat', 'sat', 'on', 'mat']
    attention.visualize_attention(weights[0], tokens, 
                                  "Self-Attention: How each word attends to others")
    plt.savefig('/home/claude/attention_pattern.png', dpi=150, bbox_inches='tight')
    print(f"\nAttention pattern saved to: attention_pattern.png")
    
    return weights


# ============================================================================
# PART 11: COMPARING WITH LIBRARY IMPLEMENTATION
# ============================================================================

def library_comparison():
    """
    Show what happens when you use PyTorch/TensorFlow vs our implementation.
    """
    print("\n" + "=" * 80)
    print("COMPARING WITH LIBRARY IMPLEMENTATIONS")
    print("=" * 80)
    
    print("\nWhat happens under the hood when you write:\n")
    
    print("PyTorch Example:")
    print("-" * 40)
    print("""
import torch
import torch.nn as nn

# Create a transformer encoder layer
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512, 
    nhead=8,
    dim_feedforward=2048
)

# What this ACTUALLY does:
# 1. Initializes Multi-Head Attention:
#    - W_q, W_k, W_v, W_o matrices (4 × 512×512)
#    - Uses Xavier initialization
# 
# 2. Initializes Feed-Forward Network:
#    - First linear: 512 → 2048
#    - ReLU activation
#    - Second linear: 2048 → 512
#
# 3. Initializes Layer Normalization:
#    - Learnable gamma and beta parameters
#
# 4. Sets up residual connections automatically

# When you call:
output = encoder_layer(input_tensor)

# It executes:
# 1. attn_output = multi_head_attention(x, x, x)
# 2. x = layer_norm(x + attn_output)  # Residual + Norm
# 3. ff_output = feed_forward(x)
# 4. x = layer_norm(x + ff_output)    # Residual + Norm
# 5. return x
    """)
    
    print("\nTensorFlow/Keras Example:")
    print("-" * 40)
    print("""
import tensorflow as tf

# Create multi-head attention
attention = tf.keras.layers.MultiHeadAttention(
    num_heads=8,
    key_dim=64  # d_model / num_heads
)

# Under the hood, this:
# 1. Creates dense layers for Q, K, V projections
# 2. Implements scaled dot-product attention
# 3. Concatenates heads and projects output

# When you call:
output = attention(query, value, key)

# It does:
# 1. Q = query @ W_q   (linear projection)
# 2. K = key @ W_k     (linear projection)
# 3. V = value @ W_v   (linear projection)
# 4. Split into heads: reshape to (batch, heads, seq, d_k)
# 5. scores = (Q @ K.T) / sqrt(d_k)  (attention scores)
# 6. weights = softmax(scores)        (attention probabilities)
# 7. output = weights @ V             (weighted sum)
# 8. Concatenate heads and project
    """)
    
    print("\nOur Implementation:")
    print("-" * 40)
    print("""
# We've built the EXACT same thing from scratch!
# The only difference is:
# - Libraries optimize with GPU kernels
# - Libraries handle automatic differentiation
# - Libraries have batching optimizations
#
# But the MATHEMATICS is identical!
    """)


# ============================================================================
# PART 12: KEY STATISTICS AND MATHEMATICAL PROPERTIES
# ============================================================================

def mathematical_properties():
    """
    Explain the statistical properties of transformers.
    """
    print("\n" + "=" * 80)
    print("STATISTICAL PROPERTIES OF TRANSFORMERS")
    print("=" * 80)
    
    print("""
1. ATTENTION AS SOFT DICTIONARY LOOKUP
---------------------------------------
Think of attention as a differentiable key-value store:
- Keys: What information is available
- Values: The actual information
- Queries: What we're looking for

Attention weight: similarity between query and key
Output: weighted sum of values based on similarity

Statistical interpretation:
- Attention weights form a probability distribution
- Output is the expected value of V under this distribution


2. SELF-ATTENTION IS PERMUTATION EQUIVARIANT
--------------------------------------------
If you permute the input, the output is permuted the same way:
  Attention(Perm(X)) = Perm(Attention(X))

This is why we need positional encodings!
Without them, "cat sat mat" = "mat sat cat"


3. COMPLEXITY ANALYSIS
----------------------
For sequence length n and dimension d:

Self-Attention:
  Time: O(n² × d)     [dominant term is n²]
  Space: O(n²)        [attention matrix]

Feed-Forward:
  Time: O(n × d²)     [applied to each position]
  Space: O(d²)        [weight matrices]

For long sequences (n >> d), attention is the bottleneck!
This is why we have efficient attention variants:
- Sparse Attention (O(n × sqrt(n)))
- Linformer (O(n × k) where k << n)
- Performer (O(n × d))


4. GRADIENT FLOW
----------------
Residual connections enable direct gradient paths:
  
  ∂L/∂x = ∂L/∂output × (1 + ∂sublayer/∂x)
  
The "+1" ensures gradients can flow even if sublayer gradients vanish!

Layer normalization keeps gradients stable:
- Prevents explosion/vanishing
- Makes training less sensitive to learning rate


5. REPRESENTATION CAPACITY
--------------------------
Transformer with:
- L layers
- h heads  
- d_model dimensions

Can represent 2^(L×h×d_model) different functions!
This exponential capacity comes from:
- Multiple heads learn different patterns
- Layer stacking composes representations
- Non-linearity (ReLU) creates decision boundaries


6. INDUCTIVE BIASES
-------------------
Unlike CNNs (locality) or RNNs (sequentiality):
- Transformers have MINIMAL inductive bias
- They learn patterns purely from data
- This requires more data but generalizes better

The only inductive biases are:
- Positional encodings (position matters)
- Multi-head attention (multiple perspectives useful)
    """)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TRANSFORMER ARCHITECTURE: COMPREHENSIVE TUTORIAL")
    print("=" * 80)
    print("\nThis script demonstrates transformers from first principles.")
    print("Everything is implemented from scratch using only NumPy!")
    print("\n" + "=" * 80)
    
    # 1. Demonstrate positional encoding
    print("\n1. POSITIONAL ENCODING")
    print("-" * 40)
    pe = PositionalEncoding(d_model=128, max_len=100)
    pe.visualize()
    plt.savefig('/home/claude/positional_encoding.png', dpi=150, bbox_inches='tight')
    print("Positional encoding visualization saved.")
    
    # 2. Run sentiment analysis example
    encoder, output = example_sentiment_analysis()
    
    # 3. Demonstrate attention
    weights = demonstrate_attention()
    
    # 4. Compare with libraries
    library_comparison()
    
    # 5. Explain mathematical properties
    mathematical_properties()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key Takeaways:
--------------
1. Transformers are built on attention mechanism (weighted averaging)
2. Multi-head attention learns multiple relationship patterns
3. Position encoding adds sequence order information
4. Feed-forward networks add non-linearity and capacity
5. Residual connections + layer norm enable deep training
6. Libraries abstract away the details, but it's just linear algebra!

When you use transformers in practice:
- transformer_model(input) calls all these components
- Each component is differentiable (can be trained)
- Gradients flow through the entire network
- Parameters are updated to minimize loss

The "magic" is just mathematics + optimization!
    """)
    print("=" * 80)
