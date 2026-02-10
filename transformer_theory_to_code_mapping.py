"""
THEORY-TO-CODE MAPPING GUIDE
=============================

This guide shows exactly how theoretical concepts map to code implementation.
Use this to understand how mathematics translates to working code.

Format:
-------
THEORY: Mathematical concept or formula
CODE: Corresponding implementation
LOCATION: File and line reference
EXPLANATION: Why this implements the theory
"""

# ============================================================================
# MAPPING 1: SOFTMAX FUNCTION
# ============================================================================

print("=" * 80)
print("MAPPING 1: SOFTMAX FUNCTION")
print("=" * 80)

theory_to_code_softmax = """
THEORY: Softmax Function
========================

Mathematical Formula:
    softmax(x_i) = exp(x_i) / Σⱼ exp(x_j)

With numerical stability:
    softmax(x_i) = exp(x_i - c) / Σⱼ exp(x_j - c)
    where c = max(x)

↓↓↓ MAPS TO CODE ↓↓↓

CODE LOCATION: transformer_architecture_explained.py, lines ~45-60
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        # THEORY: Subtract max for numerical stability
        # WHY: Prevents exp(large_number) from causing overflow
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        
        # THEORY: Compute exp(x_i)
        # WHY: Exponential emphasizes differences between values
        exp_x = np.exp(x_shifted)
        
        # THEORY: Normalize by sum
        # WHY: Creates probability distribution (sums to 1)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

LINE-BY-LINE MAPPING:
---------------------
Line: x_shifted = x - np.max(x, axis=axis, keepdims=True)
Theory: softmax(x - c) = softmax(x)
Purpose: Numerical stability
Math: Prevents overflow while preserving result

Line: exp_x = np.exp(x_shifted)
Theory: exp(x_i) in numerator and denominator
Purpose: Convert to positive values
Math: e^x is strictly positive

Line: return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
Theory: Division by Σⱼ exp(x_j)
Purpose: Normalization
Math: Creates probability distribution

PRACTICAL EXAMPLE:
------------------
Input: x = [2.0, 1.0, 0.1]

Step 1 (shift): x_shifted = [0, -1.0, -1.9]  (subtracted max=2.0)
Step 2 (exp): exp_x = [1.0, 0.368, 0.150]
Step 3 (sum): sum = 1.518
Step 4 (normalize): [0.659, 0.242, 0.099]

Verify: 0.659 + 0.242 + 0.099 = 1.0 ✓

WHEN USED IN TRANSFORMERS:
--------------------------
Location: Attention weights computation
Context: Converting attention scores to probabilities
Input: Raw similarity scores (can be any real number)
Output: Attention weights (probabilities summing to 1)
"""

print(theory_to_code_softmax)

# ============================================================================
# MAPPING 2: LAYER NORMALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("MAPPING 2: LAYER NORMALIZATION")
print("=" * 80)

theory_to_code_layernorm = """
THEORY: Layer Normalization
============================

Mathematical Formula:
    μ = (1/d) Σᵢ xᵢ                    (mean)
    σ² = (1/d) Σᵢ (xᵢ - μ)²           (variance)
    x̂ᵢ = (xᵢ - μ) / √(σ² + ε)        (normalize)
    yᵢ = γᵢx̂ᵢ + βᵢ                    (scale and shift)

↓↓↓ MAPS TO CODE ↓↓↓

CODE LOCATION: transformer_architecture_explained.py, lines ~65-85
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def layer_norm(x: np.ndarray, epsilon: float = 1e-6):
        # THEORY: Compute mean μ = (1/d) Σᵢ xᵢ
        # WHY: Center the distribution
        mean = np.mean(x, axis=-1, keepdims=True)
        
        # THEORY: Compute variance σ² = (1/d) Σᵢ (xᵢ - μ)²
        # WHY: Measure spread of distribution
        variance = np.var(x, axis=-1, keepdims=True)
        
        # THEORY: Normalize x̂ᵢ = (xᵢ - μ) / √(σ² + ε)
        # WHY: Create zero-mean, unit-variance distribution
        x_normalized = (x - mean) / np.sqrt(variance + epsilon)
        
        return x_normalized, mean, variance

And with learnable parameters (in EncoderLayer):

    def layer_norm(self, x, gamma, beta):
        x_norm, _, _ = TransformerMath.layer_norm(x)
        
        # THEORY: yᵢ = γᵢx̂ᵢ + βᵢ
        # WHY: Allow model to learn optimal scale and shift
        return gamma * x_norm + beta

LINE-BY-LINE MAPPING:
---------------------
Line: mean = np.mean(x, axis=-1, keepdims=True)
Theory: μ = (1/d) Σᵢ xᵢ
Purpose: Center the data
Math: Average across features (last dimension)
Shape: (batch, seq_len, 1) - one mean per position

Line: variance = np.var(x, axis=-1, keepdims=True)
Theory: σ² = (1/d) Σᵢ (xᵢ - μ)²
Purpose: Measure spread
Math: Average squared deviation from mean
Shape: (batch, seq_len, 1) - one variance per position

Line: x_normalized = (x - mean) / np.sqrt(variance + epsilon)
Theory: x̂ = (x - μ) / √(σ² + ε)
Purpose: Standardize distribution
Math: Z-score normalization
Why epsilon: Prevent division by zero

Line: return gamma * x_norm + beta
Theory: y = γx̂ + β
Purpose: Learnable scale and shift
Math: Affine transformation
Why: Model can "undo" normalization if needed

PRACTICAL EXAMPLE:
------------------
Input: x = [1.0, 2.0, 3.0, 4.0]

Step 1 (mean): μ = 2.5
Step 2 (variance): σ² = 1.25
Step 3 (std): σ = 1.118
Step 4 (normalize): x̂ = [-1.342, -0.447, 0.447, 1.342]
Step 5 (scale/shift): y = 1.0 * x̂ + 0.0 = x̂

Verify mean: (-1.342 - 0.447 + 0.447 + 1.342) / 4 = 0.0 ✓
Verify variance: (1.342² + 0.447² + 0.447² + 1.342²) / 4 ≈ 1.0 ✓

AXIS PARAMETER:
---------------
axis=-1: Normalize across features (LayerNorm)
- Each position normalized independently
- Mean/variance computed over d_model dimensions

Compare to BatchNorm (axis=0):
- Normalize across batch
- Mean/variance computed over batch dimension

WHY LAYER NORM FOR TRANSFORMERS:
--------------------------------
1. Sequence length varies → batch stats unreliable
2. Independent of batch size
3. Same behavior train/test
4. Works with small batches
"""

print(theory_to_code_layernorm)

# ============================================================================
# MAPPING 3: POSITIONAL ENCODING
# ============================================================================

print("\n" + "=" * 80)
print("MAPPING 3: POSITIONAL ENCODING")
print("=" * 80)

theory_to_code_posenc = """
THEORY: Positional Encoding
============================

Mathematical Formula:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

↓↓↓ MAPS TO CODE ↓↓↓

CODE LOCATION: transformer_architecture_explained.py, lines ~150-180
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def __init__(self, d_model: int, max_len: int = 5000):
        self.d_model = d_model
        
        # THEORY: Create position encodings matrix
        # SHAPE: (max_len, d_model)
        pe = np.zeros((max_len, d_model))
        
        # THEORY: Position indices [0, 1, 2, ..., max_len-1]
        # WHY: Each position gets unique encoding
        position = np.arange(0, max_len)[:, np.newaxis]
        
        # THEORY: Compute division term 10000^(2i/d_model)
        # WHY: Different frequencies for different dimensions
        div_term = np.exp(
            np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
        )
        
        # THEORY: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # WHY: Even dimensions use sine
        pe[:, 0::2] = np.sin(position * div_term)
        
        # THEORY: PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        # WHY: Odd dimensions use cosine
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe

LINE-BY-LINE MAPPING:
---------------------
Line: position = np.arange(0, max_len)[:, np.newaxis]
Theory: pos ∈ {0, 1, 2, ..., max_len-1}
Purpose: Position in sequence
Shape: (max_len, 1)
Math: Column vector of positions

Line: div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
Theory: 10000^(2i/d_model)
Purpose: Wavelength of sinusoid
Math Derivation:
  10000^(2i/d_model) = exp(log(10000^(2i/d_model)))
                      = exp((2i/d_model) × log(10000))
  We compute: exp(-2i/d_model × log(10000))
  Then use: 1/10000^(2i/d_model)
Result: Different frequency for each dimension pair

Line: pe[:, 0::2] = np.sin(position * div_term)
Theory: sin(pos / 10000^(2i/d_model))
Purpose: Even dimensions (0, 2, 4, ...)
Math: position × div_term = pos / 10000^(2i/d_model)

Line: pe[:, 1::2] = np.cos(position * div_term)
Theory: cos(pos / 10000^(2i/d_model))
Purpose: Odd dimensions (1, 3, 5, ...)
Math: Uses same div_term as corresponding sine

PRACTICAL EXAMPLE:
------------------
d_model = 4, max_len = 3

Positions: [0, 1, 2]
Dimensions: [0, 1, 2, 3]

div_term for dim 0,1: 10000^(0/4) = 1.0
div_term for dim 2,3: 10000^(2/4) = 100.0

Position 0:
  PE(0, 0) = sin(0/1.0) = 0
  PE(0, 1) = cos(0/1.0) = 1
  PE(0, 2) = sin(0/100) = 0
  PE(0, 3) = cos(0/100) = 1

Position 1:
  PE(1, 0) = sin(1/1.0) = 0.841
  PE(1, 1) = cos(1/1.0) = 0.540
  PE(1, 2) = sin(1/100) = 0.010
  PE(1, 3) = cos(1/100) = 0.999

Position 2:
  PE(2, 0) = sin(2/1.0) = 0.909
  PE(2, 1) = cos(2/1.0) = -0.416
  PE(2, 2) = sin(2/100) = 0.020
  PE(2, 3) = cos(2/100) = 0.998

Result: Each position has unique pattern!

FREQUENCY INTERPRETATION:
--------------------------
Low dimensions (i=0,1): High frequency
  - Changes rapidly with position
  - Fine-grained position information

High dimensions (i=d_model/2): Low frequency
  - Changes slowly with position
  - Coarse-grained position information

Analogous to Fourier basis:
  - Multiple frequency components
  - Capture patterns at different scales

WHY SIN/COS PAIR:
-----------------
Using both sin and cos for same frequency:
1. Provides phase offset
2. Allows relative position encoding
3. Full information about frequency component

Mathematical property:
  PE(pos+k) = Linear_function(PE(pos))
  This allows model to learn relative positions!
"""

print(theory_to_code_posenc)

# ============================================================================
# MAPPING 4: SCALED DOT-PRODUCT ATTENTION
# ============================================================================

print("\n" + "=" * 80)
print("MAPPING 4: SCALED DOT-PRODUCT ATTENTION")
print("=" * 80)

theory_to_code_attention = """
THEORY: Scaled Dot-Product Attention
=====================================

Mathematical Formula:
    Attention(Q, K, V) = softmax(QK^T / √d_k) V

Step by step:
    1. scores = QK^T              (similarity)
    2. scaled_scores = scores / √d_k   (scaling)
    3. weights = softmax(scaled_scores) (normalization)
    4. output = weights × V       (weighted sum)

↓↓↓ MAPS TO CODE ↓↓↓

CODE LOCATION: transformer_architecture_explained.py, lines ~220-265
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def __call__(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]
        
        # THEORY: Step 1 - Compute attention scores QK^T
        # WHY: Measure similarity between queries and keys
        # MATH: Dot product = cosine similarity (for normalized vectors)
        scores = np.matmul(Q, K.swapaxes(-2, -1))
        
        # THEORY: Step 2 - Scale by √d_k
        # WHY: Prevent large magnitudes in high dimensions
        # MATH: Maintain variance ≈ 1
        scores = scores / np.sqrt(d_k)
        
        # THEORY: Step 3 (optional) - Apply mask
        # WHY: Prevent attending to certain positions
        # MATH: Set to -∞ so softmax gives 0 probability
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # THEORY: Step 4 - Apply softmax
        # WHY: Convert scores to probability distribution
        # MATH: Each row sums to 1, all values in (0,1)
        attention_weights = TransformerMath.softmax(scores, axis=-1)
        
        # THEORY: Step 5 - Apply attention weights to values
        # WHY: Weighted combination of all values
        # MATH: Expected value under attention distribution
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights

LINE-BY-LINE MAPPING:
---------------------
Line: d_k = Q.shape[-1]
Theory: Dimension of query/key vectors
Purpose: Needed for scaling
Math: Last dimension of Q or K

Line: scores = np.matmul(Q, K.swapaxes(-2, -1))
Theory: QK^T where K^T means transpose of K
Purpose: Compute similarity matrix
Math: scores[i,j] = Q[i] · K[j] (dot product)
Shape: (batch, [heads], seq_len_q, seq_len_k)

Line: scores = scores / np.sqrt(d_k)
Theory: Division by √d_k
Purpose: Prevent saturation in softmax
Math Justification:
  Var(Q·K) = d_k for unit variance Q,K
  Var(Q·K/√d_k) = 1
  Keeps scores in reasonable range

Line: scores = np.where(mask == 0, -1e9, scores)
Theory: Masking (set to -∞)
Purpose: Exclude certain positions
Math: softmax(-∞) = 0
Usage: Padding mask, causal mask

Line: attention_weights = TransformerMath.softmax(scores, axis=-1)
Theory: softmax along key dimension
Purpose: Normalize similarities to probabilities
Math: weights[i] sums to 1 over all keys
Interpretation: How much query i attends to each key

Line: output = np.matmul(attention_weights, V)
Theory: Weighted sum of values
Purpose: Combine information from all positions
Math: output[i] = Σⱼ weights[i,j] × V[j]
Interpretation: Aggregate relevant information

PRACTICAL EXAMPLE:
------------------
Sequence: "The cat sat"
Q, K, V from: ["The", "cat", "sat"]

Step 1: Compute scores (similarity between all pairs)
  scores = [[The·The, The·cat, The·sat],
            [cat·The, cat·cat, cat·sat],
            [sat·The, sat·cat, sat·sat]]
  
  Example values:
  scores = [[0.8, 0.3, 0.1],
            [0.3, 0.9, 0.5],
            [0.2, 0.6, 0.7]]

Step 2: Scale by √d_k (assume d_k=64)
  scaled = scores / 8
  scaled = [[0.10, 0.04, 0.01],
            [0.04, 0.11, 0.06],
            [0.03, 0.08, 0.09]]

Step 3: Apply softmax (per row)
  weights = [[0.37, 0.35, 0.28],
             [0.28, 0.39, 0.33],
             [0.29, 0.35, 0.36]]

Interpretation:
  Row 0: "The" attends to: 37% The, 35% cat, 28% sat
  Row 1: "cat" attends to: 28% The, 39% cat, 33% sat
  Row 2: "sat" attends to: 29% The, 35% cat, 36% sat

Step 4: Weighted sum of values
  output[0] = 0.37×V[The] + 0.35×V[cat] + 0.28×V[sat]
  output[1] = 0.28×V[The] + 0.39×V[cat] + 0.33×V[sat]
  output[2] = 0.29×V[The] + 0.35×V[cat] + 0.36×V[sat]

Result: Each position's output is mixture of all values!

SHAPE TRACKING:
---------------
Input shapes (batch=2, seq_len=4, d_k=8):
  Q: (2, 4, 8)
  K: (2, 4, 8)
  V: (2, 4, 8)

After QK^T:
  scores: (2, 4, 4)  # 4x4 similarity matrix per batch

After scaling:
  scaled_scores: (2, 4, 4)  # Same shape

After softmax:
  attention_weights: (2, 4, 4)  # Probabilities

After multiplying by V:
  output: (2, 4, 8)  # Same shape as V

COMPUTATIONAL COST:
-------------------
For sequence length n, dimension d:

QK^T: n × d × n = n²d multiplications
Softmax: n² operations
Attention × V: n × n × d = n²d multiplications

Total: O(n²d) - quadratic in sequence length!
"""

print(theory_to_code_attention)

# ============================================================================
# MAPPING 5: MULTI-HEAD ATTENTION
# ============================================================================

print("\n" + "=" * 80)
print("MAPPING 5: MULTI-HEAD ATTENTION")
print("=" * 80)

theory_to_code_multihead = """
THEORY: Multi-Head Attention
=============================

Mathematical Formula:
    MultiHead(Q, K, V) = Concat(head₁, ..., head_h)W^O
    
    where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)

Parameters:
    W^Q_i, W^K_i, W^V_i: Projection matrices for head i
    W^O: Output projection matrix
    h: Number of heads
    d_k = d_model / h: Dimension per head

↓↓↓ MAPS TO CODE ↓↓↓

CODE LOCATION: transformer_architecture_explained.py, lines ~360-450
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def __init__(self, d_model, num_heads, dropout=0.1):
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # THEORY: Projection matrices W^Q, W^K, W^V
        # WHY: Transform input to query/key/value spaces
        # SHAPE: (d_model, d_model) each
        self.W_q = self._init_weights((d_model, d_model))
        self.W_k = self._init_weights((d_model, d_model))
        self.W_v = self._init_weights((d_model, d_model))
        
        # THEORY: Output projection W^O
        # WHY: Combine all heads
        # SHAPE: (d_model, d_model)
        self.W_o = self._init_weights((d_model, d_model))
    
    def _split_heads(self, x):
        # THEORY: Reshape for parallel head computation
        # IN: (batch, seq_len, d_model)
        # OUT: (batch, num_heads, seq_len, d_k)
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def _combine_heads(self, x):
        # THEORY: Concatenate all heads
        # IN: (batch, num_heads, seq_len, d_k)
        # OUT: (batch, seq_len, d_model)
        batch_size, num_heads, seq_len, d_k = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def __call__(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        
        # THEORY: Step 1 - Linear projections
        # MATH: QW^Q, KW^K, VW^V
        # WHY: Transform to query/key/value spaces
        Q = np.matmul(Q, self.W_q)
        K = np.matmul(K, self.W_k)
        V = np.matmul(V, self.W_v)
        
        # THEORY: Step 2 - Split into multiple heads
        # MATH: Reshape to (batch, heads, seq_len, d_k)
        # WHY: Process heads in parallel
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        # THEORY: Step 3 - Apply attention in each head
        # MATH: Attention(Q_head, K_head, V_head) for each head
        # WHY: Each head learns different patterns
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        
        # THEORY: Step 4 - Concatenate heads
        # MATH: [head₁; head₂; ...; head_h]
        # WHY: Combine information from all heads
        attn_output = self._combine_heads(attn_output)
        
        # THEORY: Step 5 - Final projection
        # MATH: Concat(heads)W^O
        # WHY: Mix information across heads
        output = np.matmul(attn_output, self.W_o)
        
        return output, attention_weights

LINE-BY-LINE MAPPING:
---------------------
Line: self.d_k = d_model // num_heads
Theory: d_k = d_model / h
Purpose: Dimension per head
Math: Each head works in lower-dimensional subspace
Example: d_model=512, h=8 → d_k=64

Line: self.W_q = self._init_weights((d_model, d_model))
Theory: Query projection matrix W^Q
Purpose: Project input to query space
Shape: (d_model, d_model) - not (d_model, d_k)!
Why: Projects all heads together, split later

Line: Q = np.matmul(Q, self.W_q)
Theory: QW^Q
Purpose: Linear transformation to query space
Input: (batch, seq_len, d_model)
Output: (batch, seq_len, d_model)
Note: Projects to full dimension, will split next

Line: Q = self._split_heads(Q)
Theory: Reshape for h heads
Math: (batch, seq_len, d_model) → (batch, h, seq_len, d_k)
Purpose: Separate different heads
Implementation:
  1. Reshape: (batch, seq_len, h, d_k)
  2. Transpose: (batch, h, seq_len, d_k)

Line: attn_output, _ = self.attention(Q, K, V, mask)
Theory: Attention(Q_head, K_head, V_head) for each head
Math: Same attention formula, applied to each head
Broadcasting: All h heads computed in parallel
Shape: (batch, h, seq_len, d_k)

Line: attn_output = self._combine_heads(attn_output)
Theory: Concat(head₁, head₂, ..., head_h)
Math: Concatenate along head dimension
Implementation:
  1. Transpose: (batch, h, seq_len, d_k) → (batch, seq_len, h, d_k)
  2. Reshape: (batch, seq_len, h×d_k) = (batch, seq_len, d_model)

Line: output = np.matmul(attn_output, self.W_o)
Theory: Concat(heads)W^O
Purpose: Mix information from all heads
Input: (batch, seq_len, d_model)
Output: (batch, seq_len, d_model)

PRACTICAL EXAMPLE:
------------------
Configuration:
  d_model = 8
  num_heads = 2
  d_k = 4
  seq_len = 3

Input shape: (1, 3, 8)

Step 1: Project
  Q = input @ W_q  →  (1, 3, 8)
  K = input @ W_k  →  (1, 3, 8)
  V = input @ W_v  →  (1, 3, 8)

Step 2: Split heads
  Q →  (1, 2, 3, 4)  [2 heads, seq_len=3, d_k=4]
  K →  (1, 2, 3, 4)
  V →  (1, 2, 3, 4)

Step 3: Attention (per head in parallel)
  Head 0: Attention on Q[0], K[0], V[0]  →  (1, 3, 4)
  Head 1: Attention on Q[1], K[1], V[1]  →  (1, 3, 4)
  Combined: (1, 2, 3, 4)

Step 4: Concatenate
  (1, 2, 3, 4)  →  (1, 3, 8)  [concat 2 heads of 4 dims each]

Step 5: Output projection
  (1, 3, 8) @ W_o  →  (1, 3, 8)

WHY THIS WORKS:
---------------
Same total computation as single head:
  Single: 4 × (d_model × d_model) = 4 × 64 parameters
  Multi: 4 × (d_model × d_model) = 4 × 64 parameters
  Same! But multi-head is more expressive.

Each head learns different patterns:
  Head 1: Maybe syntactic relationships
  Head 2: Maybe semantic similarities
  W^O learns how to combine them

PARAMETER COUNT:
----------------
For d_model=512, h=8:
  W_q: 512 × 512 = 262,144
  W_k: 512 × 512 = 262,144
  W_v: 512 × 512 = 262,144
  W_o: 512 × 512 = 262,144
  Total: 1,048,576 parameters per multi-head attention layer
"""

print(theory_to_code_multihead)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("MAPPING SUMMARY")
print("=" * 80)

mapping_summary = """
HOW TO USE THIS GUIDE
=====================

1. Read the theory in transformer_theory.py
2. Look up the corresponding code section here
3. See the exact line-by-line mapping
4. Run transformer_practical_comparison.py to see numerical examples
5. Study transformer_architecture_explained.py for complete implementation

KEY PATTERNS
============

Pattern 1: Mathematical Formula → NumPy Operation
--------------------------------------------------
Theory: Matrix multiplication AB
Code: np.matmul(A, B) or A @ B

Theory: Element-wise operations
Code: Standard operators (+, -, *, /)

Theory: Summation Σ
Code: np.sum(..., axis=...)

Theory: Mean μ = (1/n)Σx
Code: np.mean(..., axis=...)

Pattern 2: Shape Transformations
---------------------------------
Theory: "For each position"
Code: Operations on axis=1 (sequence dimension)

Theory: "For each feature"
Code: Operations on axis=-1 (model dimension)

Theory: "Across batch"
Code: Operations on axis=0 (batch dimension)

Pattern 3: Broadcasting
------------------------
Theory: Apply same operation to all examples
Code: NumPy broadcasting (automatic expansion)

Example:
  (batch, seq, d) + (1, 1, d) → element-wise add
  Broadcast happens automatically!

Pattern 4: Probability Distributions
-------------------------------------
Theory: Normalize to sum to 1
Code: softmax function

Theory: Sample from distribution
Code: argmax for deterministic, multinomial for stochastic

COMPLETE MAPPING TABLE
=======================

┌─────────────────────────────┬──────────────────────────────┐
│ Mathematical Concept        │ Code Implementation          │
├─────────────────────────────┼──────────────────────────────┤
│ Softmax function            │ exp(x) / sum(exp(x))         │
│ Layer normalization         │ (x - mean) / std             │
│ Positional encoding         │ sin/cos with frequencies     │
│ Attention QK^T              │ np.matmul(Q, K.T)            │
│ Scaling by √d_k             │ / np.sqrt(d_k)               │
│ Weighted sum                │ np.matmul(weights, V)        │
│ Multi-head split            │ reshape + transpose          │
│ Head concatenation          │ transpose + reshape          │
│ Feed-forward                │ ReLU(xW₁)W₂                  │
│ Residual connection         │ x + sublayer(x)              │
├─────────────────────────────┼──────────────────────────────┤
│ Batch dimension             │ axis=0, shape[0]             │
│ Sequence dimension          │ axis=1, shape[1]             │
│ Model dimension             │ axis=-1, shape[-1]           │
│ Number of heads             │ Custom axis after split      │
└─────────────────────────────┴──────────────────────────────┘

FILE ORGANIZATION
=================

1. transformer_theory.py
   → Pure mathematics and concepts
   → No code, just formulas and explanations

2. transformer_architecture_explained.py
   → Complete implementation from scratch
   → Only NumPy, no frameworks
   → Heavily commented

3. transformer_practical_comparison.py
   → Side-by-side: library vs manual
   → Numerical examples
   → Small matrices for clarity

4. transformer_libraries_explained.py
   → PyTorch, TensorFlow, HuggingFace
   → What library calls do internally
   → Production usage

5. THIS FILE (transformer_theory_to_code_mapping.py)
   → Connects theory to code
   → Line-by-line explanations
   → Shape tracking

RECOMMENDED WORKFLOW
====================

For Learning:
1. Read theory for a component
2. Look up mapping in this file
3. Run practical comparison
4. Study full implementation
5. Try modifying the code

For Implementation:
1. Understand theory first
2. Use full implementation as template
3. Consult mapping for specific operations
4. Test with small examples from practical comparison
5. Optimize using library patterns

For Debugging:
1. Print shapes at each step
2. Compare with expected shapes in mapping
3. Check numerical values against examples
4. Verify gradients flow correctly
5. Use theoretical properties for sanity checks

NEXT STEPS
==========

1. Try implementing a component yourself
2. Compare your code with the reference
3. Test on small examples
4. Verify theoretical properties hold
5. Optimize and add error handling

Remember: Understanding the mapping between theory and code
is key to mastering transformers!
"""

print(mapping_summary)

print("\n" + "=" * 80)
print("END OF THEORY-TO-CODE MAPPING")
print("=" * 80)
