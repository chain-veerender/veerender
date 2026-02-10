"""
TRANSFORMER ARCHITECTURE: THEORETICAL FOUNDATIONS
==================================================

This document provides the mathematical theory, statistical foundations, and
conceptual explanations that support the code implementations.

Read this alongside the code files to understand the "why" behind the "how".
"""

# ============================================================================
# PART 1: FUNDAMENTAL MATHEMATICAL OPERATIONS
# ============================================================================

"""
1.1 SOFTMAX FUNCTION
====================

Mathematical Definition:
------------------------
For a vector x = [x₁, x₂, ..., xₙ], the softmax function is:

    softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)

Properties:
-----------
1. Output range: (0, 1) for each element
2. Sum property: Σᵢ softmax(xᵢ) = 1
3. Monotonicity: If xᵢ > xⱼ, then softmax(xᵢ) > softmax(xⱼ)
4. Temperature scaling: softmax(x/T) where T controls "sharpness"

Statistical Interpretation:
---------------------------
Softmax converts arbitrary real values into a probability distribution.

Connection to Maximum Entropy:
- Softmax is the maximum entropy distribution subject to constraints
- Maximizes uncertainty while matching expected features
- Derived from Boltzmann distribution in statistical mechanics

Why Use Softmax in Attention?
------------------------------
1. Normalization: Ensures attention weights sum to 1
2. Differentiability: Smooth gradients for learning
3. Amplification: Exponential emphasizes differences
4. Probabilistic: Weights can be interpreted as probabilities

Numerical Stability:
--------------------
Problem: exp(xᵢ) can overflow for large xᵢ (e.g., exp(1000))

Solution: Shift by max value
    softmax(x - c) = softmax(x) for any constant c
    
Proof:
    exp(xᵢ - c) / Σⱼ exp(xⱼ - c)
    = exp(xᵢ)·exp(-c) / [Σⱼ exp(xⱼ)·exp(-c)]
    = exp(xᵢ) / Σⱼ exp(xⱼ)

Choose c = max(x) to prevent overflow.

Gradient of Softmax:
--------------------
∂softmax(xᵢ)/∂xⱼ = softmax(xᵢ)[δᵢⱼ - softmax(xⱼ)]

where δᵢⱼ is the Kronecker delta (1 if i=j, 0 otherwise)

This Jacobian has useful properties for backpropagation:
- Diagonal elements are positive
- Off-diagonal elements are negative
- Well-conditioned for typical values
"""

print("=" * 80)
print("PART 1: FUNDAMENTAL MATHEMATICAL OPERATIONS")
print("=" * 80)

# ============================================================================
# PART 2: LAYER NORMALIZATION
# ============================================================================

"""
1.2 LAYER NORMALIZATION
========================

Mathematical Definition:
------------------------
For a d-dimensional input x:

    μ = (1/d) Σᵢ xᵢ                    (mean)
    σ² = (1/d) Σᵢ (xᵢ - μ)²           (variance)
    x̂ᵢ = (xᵢ - μ) / √(σ² + ε)        (normalize)
    yᵢ = γᵢx̂ᵢ + βᵢ                    (scale and shift)

where:
- μ: mean across features
- σ²: variance across features
- ε: small constant for numerical stability (typically 10⁻⁶)
- γ: learned scale parameter (initialized to 1)
- β: learned shift parameter (initialized to 0)

Statistical Interpretation:
---------------------------
Layer normalization standardizes the distribution of activations:
1. Centers the data (zero mean)
2. Normalizes the scale (unit variance)
3. Allows model to learn optimal scale and shift

Why Normalization Helps:
-------------------------
1. Reduces Internal Covariate Shift
   - Problem: Distribution of layer inputs changes during training
   - Effect: Earlier layers must adapt to changing distributions
   - Solution: Normalize to maintain consistent statistics

2. Improves Gradient Flow
   - Prevents gradients from vanishing or exploding
   - Makes loss surface smoother
   - Allows higher learning rates

3. Reduces Sensitivity to Initialization
   - Less dependent on weight initialization
   - More robust training

Layer Norm vs Batch Norm:
--------------------------
                Layer Norm          Batch Norm
Normalizes      Across features     Across batch
Independence    Independent         Depends on batch
Sequence        Better              Worse
Batch size      Any size OK         Needs large batch
Inference       Same as training    Needs running stats

Why Layer Norm for Transformers?
---------------------------------
1. Sequence lengths vary → batch statistics unreliable
2. Small batch sizes common in NLP
3. Independence between examples desired
4. Consistent behavior train/test

Mathematical Properties:
------------------------
1. Affine Invariance:
   LayerNorm(Wx + b) depends only on direction of Wx + b, not scale

2. Reparameterization:
   Allows learning meaningful γ and β without constraints

3. Gradient Properties:
   ∂L/∂x has bounded norm when activations are normalized

Backpropagation Through Layer Norm:
------------------------------------
Gradient computation requires chain rule through:
1. Scale/shift: ∂L/∂γ, ∂L/∂β
2. Normalization: ∂L/∂x̂
3. Variance: ∂L/∂σ²
4. Mean: ∂L/∂μ

Key gradient:
∂L/∂xᵢ = γ/σ · [∂L/∂x̂ᵢ - mean(∂L/∂x̂) - x̂ᵢ·mean(x̂·∂L/∂x̂)]

This involves computing covariances, making it computationally expensive.
"""

print("\n" + "=" * 80)
print("PART 2: LAYER NORMALIZATION THEORY")
print("=" * 80)

# ============================================================================
# PART 3: POSITIONAL ENCODING
# ============================================================================

"""
2. POSITIONAL ENCODING
=======================

The Problem:
------------
Transformers process sequences in parallel. Without positional information:
- "The cat sat on the mat" = "mat the on sat cat The"
- Order-invariant = permutation equivariant
- Need to inject position information

Mathematical Definition:
------------------------
For position pos and dimension i:

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

where:
- pos ∈ {0, 1, 2, ..., max_len-1}: position in sequence
- i ∈ {0, 1, 2, ..., d_model/2-1}: dimension index
- d_model: embedding dimension

Why Sinusoidal Functions?
--------------------------
1. Unique Encoding:
   Each position gets a unique encoding vector
   PE(pos₁) ≠ PE(pos₂) for pos₁ ≠ pos₂

2. Bounded Values:
   All values in [-1, 1]
   Prevents dominating the token embeddings

3. Relative Positions:
   For any fixed offset k:
   PE(pos + k) can be expressed as linear function of PE(pos)
   
   Specifically:
   PE(pos + k) = T_k · PE(pos)
   
   where T_k is a transformation matrix dependent only on k

4. Extrapolation:
   Can generalize to sequences longer than training
   No learned parameters → no overfitting to sequence length

Mathematical Proof of Relative Position Property:
--------------------------------------------------
Using trigonometric identities:

sin(α + β) = sin(α)cos(β) + cos(α)sin(β)
cos(α + β) = cos(α)cos(β) - sin(α)sin(β)

Let ωᵢ = 1/10000^(2i/d_model), then:

PE(pos+k, 2i) = sin(ωᵢ(pos+k))
              = sin(ωᵢ·pos)cos(ωᵢ·k) + cos(ωᵢ·pos)sin(ωᵢ·k)
              = PE(pos,2i)·cos(ωᵢ·k) + PE(pos,2i+1)·sin(ωᵢ·k)

This is a linear combination of PE(pos,2i) and PE(pos,2i+1)!

Frequency Interpretation:
--------------------------
Different dimensions use different frequencies:
- Lower dimensions (small i): high frequency, fine-grained position
- Higher dimensions (large i): low frequency, coarse-grained position

This creates a "positional fingerprint" similar to:
- Binary encoding: but continuous and differentiable
- Fourier basis: multiple frequency components

Geometric Interpretation:
-------------------------
Each position traces a curve in d_model-dimensional space:
- Smooth, continuous trajectory
- No two positions share the same location
- Distances encode relative positions

Why Not Learned Positional Embeddings?
---------------------------------------
Learned embeddings are common alternatives, but:

Sinusoidal Pros:
+ Extrapolate to longer sequences
+ No parameters to learn
+ Relative position encoding
+ Deterministic

Learned Pros:
+ Can adapt to data
+ Might capture task-specific patterns
+ Often work better in practice

Trade-off: Inductive bias vs flexibility

Addition vs Concatenation:
---------------------------
Position encoding is ADDED to token embeddings, not concatenated.

Why addition?
1. Preserves dimension d_model
2. Allows model to weight importance
3. Simpler architecture
4. Token and position info can interact

Mathematical justification:
- Addition creates additive mixture
- Attention can learn to separate if needed
- Gradient flows to both components

Alternative Positional Encodings:
----------------------------------
1. Learned absolute: PE_pos as learned vectors
2. Relative (T5, Transformer-XL): attention bias based on distance
3. Rotary (RoPE): rotation in complex space
4. ALiBi: attention bias proportional to distance
"""

print("\n" + "=" * 80)
print("PART 3: POSITIONAL ENCODING THEORY")
print("=" * 80)

# ============================================================================
# PART 4: ATTENTION MECHANISM - MATHEMATICAL FOUNDATIONS
# ============================================================================

"""
3. SCALED DOT-PRODUCT ATTENTION
================================

3.1 BASIC FORMULATION
----------------------

Mathematical Definition:
------------------------
    Attention(Q, K, V) = softmax(QK^T / √d_k) V

where:
- Q: Query matrix (n × d_k) - "What am I looking for?"
- K: Key matrix (n × d_k) - "What do I offer?"
- V: Value matrix (n × d_v) - "What do I actually contain?"
- n: sequence length
- d_k: dimension of queries/keys
- d_v: dimension of values

Step-by-Step Interpretation:
-----------------------------
1. Similarity Scores: QK^T
   - Computes dot product between every query and key
   - Result: (n × n) matrix of similarities
   - scores[i,j] = how much query i attends to key j

2. Scaling: / √d_k
   - Prevents large values when d_k is high
   - Keeps variance approximately 1

3. Attention Weights: softmax(...)
   - Converts similarities to probabilities
   - Each row sums to 1
   - Normalized attention distribution

4. Weighted Sum: (...) V
   - Combines values using attention weights
   - Output: contextual representation

3.2 WHY DOT PRODUCT?
--------------------

Similarity Measures:
--------------------
Several options for measuring similarity:

1. Dot Product: q · k = Σᵢ qᵢkᵢ
   - Fast to compute (matrix multiplication)
   - Measures alignment
   - Used in transformers

2. Cosine Similarity: (q · k) / (||q|| ||k||)
   - Normalized dot product
   - Range: [-1, 1]
   - Not used (unnecessary normalization)

3. Additive: v^T tanh(W_q q + W_k k)
   - Used in older attention mechanisms (Bahdanau)
   - More parameters
   - Slower

Why Dot Product Wins:
---------------------
1. Computational Efficiency:
   - Matrix multiplication is highly optimized
   - GPU/TPU acceleration
   - O(n²d) complexity same as additive, but faster constants

2. Theoretical Properties:
   - Natural similarity measure in vector spaces
   - Directly measures alignment
   - Works well with learned representations

3. Empirical Success:
   - Performs as well or better than additive
   - Simpler architecture

3.3 WHY SCALE BY √d_k?
-----------------------

Problem Without Scaling:
------------------------
For random vectors q, k with unit variance:
    E[q · k] = 0
    Var[q · k] = d_k

As d_k increases:
- Dot products have higher variance
- Softmax saturates (gradients → 0)
- Training becomes unstable

Mathematical Analysis:
----------------------
Let q, k ~ N(0, I_d_k) (standard normal, independent)

Then q · k = Σᵢ₌₁^d_k qᵢkᵢ

Since qᵢ and kᵢ are independent N(0,1):
    E[qᵢkᵢ] = E[qᵢ]E[kᵢ] = 0
    Var[qᵢkᵢ] = E[qᵢ²]E[kᵢ²] = 1

By independence:
    Var[q · k] = Σᵢ Var[qᵢkᵢ] = d_k

Standard deviation grows as √d_k!

Effect on Softmax:
------------------
For large d_k, dot products become large in magnitude.

Example with d_k = 512:
- Typical dot product magnitude: ±√512 ≈ ±22.6
- After softmax: exp(22.6) ≈ 7×10⁹ vs exp(-22.6) ≈ 1×10⁻¹⁰
- Extreme probabilities (nearly 0 or 1)
- Gradients vanish

Solution: Scale by √d_k
------------------------
Divide by √d_k to maintain unit variance:
    scores = QK^T / √d_k
    Var[scores] ≈ 1

Benefits:
1. Softmax inputs have reasonable magnitude
2. Attention weights not too peaked
3. Gradients flow well
4. Training stable

Empirical Validation:
---------------------
Original Transformer paper shows:
- Without scaling: poor performance for large d_k
- With scaling: consistent performance across d_k

3.4 ATTENTION AS SOFT DICTIONARY LOOKUP
----------------------------------------

Conceptual Framework:
---------------------
Think of attention as a differentiable key-value store:

Dictionary = {(k₁, v₁), (k₂, v₂), ..., (kₙ, vₙ)}

Query q:
1. Compare q with all keys: similarity(q, kᵢ)
2. Get attention weights: softmax(similarities)
3. Return weighted average of values

Key Differences from Hard Lookup:
----------------------------------
1. Soft Matching:
   - Not exact key match
   - Similarity-based weighting
   - Multiple values contribute

2. Differentiable:
   - Gradients flow through attention
   - Can learn key/value representations
   - End-to-end training

3. Context-Dependent:
   - Query determines what's retrieved
   - Same keys/values → different output for different queries

Statistical Interpretation:
---------------------------
Attention weights form a probability distribution p over positions.

Output = E_p[V] (expected value)

This is optimal under:
- Minimize squared error
- Distribution determined by query-key similarity

Information-Theoretic View:
----------------------------
Attention selects relevant information:
- High attention = high relevance
- Low attention = low relevance
- Information bottleneck: compress input to relevant parts

3.5 MASKING IN ATTENTION
-------------------------

Purpose of Masking:
-------------------
1. Padding Mask:
   - Prevent attending to padding tokens
   - Sequences have different lengths
   - Padded to same length for batching

2. Causal Mask (Look-ahead mask):
   - Prevent attending to future positions
   - For autoregressive generation (GPT, decoder)
   - Position i can only attend to positions ≤ i

Implementation:
---------------
Add large negative number to scores before softmax:

    mask = [0 if valid else -∞]
    scores = scores + mask
    attention_weights = softmax(scores)

After softmax: exp(-∞) = 0
- Invalid positions get zero attention
- Only valid positions contribute

Causal Mask Pattern:
--------------------
For sequence length 4:

    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]

Position i can attend to positions 0, 1, ..., i.

Why -∞ and Not 0?
------------------
Setting scores to 0 would still give probability mass after softmax.
Using -∞ ensures probability 0 after softmax.

In practice, use large negative number (e.g., -10000) to avoid numerical issues.
"""

print("\n" + "=" * 80)
print("PART 4: ATTENTION MECHANISM THEORY")
print("=" * 80)

# ============================================================================
# PART 5: MULTI-HEAD ATTENTION
# ============================================================================

"""
4. MULTI-HEAD ATTENTION
========================

4.1 MOTIVATION
--------------

Why Multiple Heads?
-------------------
Single attention mechanism learns one type of relationship.
Multiple heads allow learning different types:

Examples:
- Head 1: Syntactic dependencies (subject-verb)
- Head 2: Semantic similarities (synonyms)
- Head 3: Positional relationships (adjacent words)
- Head 4: Long-range dependencies (across clauses)

Analogy:
--------
Like ensemble learning:
- Each head is a "weak learner"
- Combination is stronger than any individual
- Diversity improves robustness

4.2 MATHEMATICAL FORMULATION
-----------------------------

Definition:
-----------
    MultiHead(Q, K, V) = Concat(head₁, ..., head_h)W^O

where:
    head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)

Parameters:
-----------
For each head i:
- W^Q_i ∈ ℝ^(d_model × d_k): Query projection
- W^K_i ∈ ℝ^(d_model × d_k): Key projection  
- W^V_i ∈ ℝ^(d_model × d_v): Value projection

Output projection:
- W^O ∈ ℝ^(hd_v × d_model): Combines all heads

Typical choices:
- h = 8 (number of heads)
- d_k = d_v = d_model / h
- Total dimension: h × d_k = d_model

4.3 WHY THIS PARAMETERIZATION?
-------------------------------

Key Insight: Subspace Projections
----------------------------------
Instead of:
- One attention in d_model-dimensional space

We use:
- h attentions in d_k-dimensional subspaces
- d_k = d_model / h

Benefits:
---------
1. Same Computational Cost:
   - Single head: 4 × (d_model × d_model) parameters
   - Multi-head: 4 × h × (d_model × d_k) = 4 × (d_model × d_model)
   - Same total parameters!

2. More Expressive:
   - Each head can specialize
   - Learns different representations
   - Combined view is richer

3. Parallelization:
   - All heads computed in parallel
   - GPU-friendly
   - No sequential dependency

Mathematical Perspective:
-------------------------
Multi-head attention learns h different similarity functions:

    sim_i(q, k) = (qW^Q_i)(kW^K_i)^T / √d_k

Each W^Q_i and W^K_i defines a subspace where similarity is computed.

Different subspaces capture different aspects of relationships.

4.4 CONCATENATION VS SUMMATION
-------------------------------

Why Concatenate Heads?
----------------------
After computing h head outputs, we concatenate rather than sum.

Options:
--------
1. Concatenation: [head₁; head₂; ...; head_h]
   - Preserves all information from each head
   - Output dimension: h × d_v
   - Used in Transformers

2. Summation: head₁ + head₂ + ... + head_h
   - Output dimension: d_v
   - Simpler but loses information

Why Concatenation Wins:
------------------------
1. Information Preservation:
   - Each head's output kept separate
   - Final projection W^O can learn how to combine
   - More flexible than fixed summation

2. Learning Combinations:
   - W^O learns optimal weighting
   - Can emphasize different heads for different outputs
   - Adaptive combination

Final Projection W^O:
---------------------
After concatenation, apply linear transformation:
    output = Concat(heads) W^O

Purpose:
1. Mix information across heads
2. Project back to d_model dimension
3. Learn head importance
4. Introduce additional parameters for expressiveness

4.5 THEORETICAL ANALYSIS
------------------------

Representational Capacity:
--------------------------
Multi-head attention can represent any single-head attention:
- Set h-1 heads to zero
- Use remaining head as single-head

But also represents combinations impossible with single-head.

Proof sketch:
For any single-head attention with parameters (W^Q, W^K, W^V, W^O),
we can construct multi-head parameters that reproduce it exactly.

Universal Approximation:
------------------------
With sufficient heads and dimensions, multi-head attention can approximate
any continuous function on sequences (under mild conditions).

Not proven rigorously, but supported by:
1. Empirical success
2. Connection to kernel methods
3. Representation learning theory

Attention Patterns:
-------------------
Different heads learn different patterns:
- Local attention: nearby positions
- Global attention: distant positions
- Positional attention: specific relative positions
- Content-based attention: semantic similarity

Visualization reveals:
- Clear specialization in trained models
- Some heads very interpretable
- Others more distributed

4.6 COMPUTATIONAL COMPLEXITY
-----------------------------

For sequence length n, model dimension d_model, h heads:

Per Head:
---------
- Projection: O(nd_model × d_k) = O(nd_model²/h)
- Attention: O(n²d_k) = O(n²d_model/h)
- Total per head: O(nd_model²/h + n²d_model/h)

All Heads Combined:
--------------------
- Projections: O(nd_model²)
- Attention: O(n²d_model)
- Output projection: O(nd_model²)
- Total: O(nd_model² + n²d_model)

Comparison to Single Head:
---------------------------
Same asymptotic complexity!
But multi-head:
+ More expressive
+ Better parallelization
- Slightly more memory for intermediate values

Memory Usage:
-------------
- Attention matrices: h × n²
- Intermediate projections: h × n × d_k
- Total extra memory: O(hn² + nd_model)

For typical values (n=512, d_model=768, h=8):
- Single attention matrix: 512² ≈ 250K floats ≈ 1MB
- All heads: 8MB
- Manageable on modern GPUs
"""

print("\n" + "=" * 80)
print("PART 5: MULTI-HEAD ATTENTION THEORY")
print("=" * 80)

# ============================================================================
# PART 6: FEED-FORWARD NETWORKS
# ============================================================================

"""
5. POSITION-WISE FEED-FORWARD NETWORKS
=======================================

5.1 ARCHITECTURE
----------------

Definition:
-----------
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

Or more generally:
    FFN(x) = σ(xW₁ + b₁)W₂ + b₂

where:
- W₁ ∈ ℝ^(d_model × d_ff): First layer weights
- b₁ ∈ ℝ^d_ff: First layer bias
- W₂ ∈ ℝ^(d_ff × d_model): Second layer weights
- b₂ ∈ ℝ^d_model: Second layer bias
- σ: Activation function (ReLU in original, GELU in modern)
- d_ff: Hidden dimension (typically 4 × d_model)

Position-Wise Property:
-----------------------
Applied independently and identically to each position.

For sequence [x₁, x₂, ..., xₙ]:
    output = [FFN(x₁), FFN(x₂), ..., FFN(xₙ)]

Same parameters for all positions!

5.2 WHY FEED-FORWARD NETWORKS?
-------------------------------

Problem: Attention is Linear in Values
---------------------------------------
Attention output is weighted sum of values:
    output = Σᵢ αᵢvᵢ

This is a linear operation!
- Cannot learn non-linear transformations
- Limited expressiveness
- Need non-linearity

Solution: Feed-Forward Network
-------------------------------
1. Introduces non-linearity (ReLU/GELU)
2. Increases model capacity
3. Learns complex transformations

Theoretical Role:
-----------------
- Attention: Aggregates information (where to look)
- FFN: Processes information (what to do with it)

Complementary functions:
- Attention is "routing" mechanism
- FFN is "computation" mechanism

5.3 WHY d_ff = 4 × d_model?
----------------------------

Architecture Choice:
--------------------
Original Transformer uses:
- d_model = 512
- d_ff = 2048 (4×)

Why 4× expansion?
------------------
1. Computational Budget:
   - Most computation in FFN
   - Expansion ratio controls capacity

2. Empirical Tuning:
   - 4× works well across tasks
   - Sweet spot between underfitting and overfitting

3. Parameter Distribution:
   - ~2/3 of parameters in FFN
   - ~1/3 in attention
   - Balances capacity

Trade-offs:
-----------
Larger d_ff:
+ More capacity
+ Can learn complex transformations
- More parameters
- More computation
- Risk of overfitting

Smaller d_ff:
+ Fewer parameters
+ Faster
- Less capacity
- May underfit

Modern Variants:
----------------
- GLU/SwiGLU: Different activation, similar expansion
- Experts: Mixture of experts with larger total capacity
- Adapter layers: Small FFN for fine-tuning

5.4 ACTIVATION FUNCTIONS
------------------------

ReLU (Original Transformer):
----------------------------
    ReLU(x) = max(0, x)

Properties:
+ Simple, fast
+ Non-saturating for x > 0
+ Sparse activation (many zeros)
- Not smooth at 0
- Dead neurons (always 0)

GELU (Modern, e.g., BERT, GPT):
-------------------------------
    GELU(x) = x·Φ(x)

where Φ is CDF of standard normal.

Approximation:
    GELU(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])

Properties:
+ Smooth everywhere
+ Non-monotonic (slight negative for x < 0)
+ Better empirical performance
- Slightly more expensive

SwiGLU (LLaMA, etc.):
---------------------
    SwiGLU(x) = Swish(xW)⊙(xV)

where:
- Swish(x) = x·sigmoid(βx)
- ⊙: element-wise multiplication
- W, V: separate weight matrices

Properties:
+ State-of-the-art performance
+ Gating mechanism
- More parameters

5.5 THEORETICAL ANALYSIS
------------------------

Universal Approximation:
------------------------
Two-layer FFN with sufficient width can approximate any continuous function
(on bounded domain).

Proof sketch (for ReLU):
1. ReLU network can approximate any continuous piecewise linear function
2. Any continuous function can be approximated by piecewise linear
3. Therefore, sufficiently wide ReLU network is universal approximator

Practical implications:
- d_ff = 2048 is enough for most tasks
- Depth more important than width
- Stacking layers increases expressiveness

Capacity vs Depth:
------------------
For same total parameters:
- Wider FFN: More capacity per layer
- More layers: Hierarchical representations

Trade-off depends on task:
- Language: Benefit from depth (many layers)
- Vision: Can use wider FFNs

Gradient Flow:
--------------
FFN can cause gradient issues:

1. Vanishing Gradients:
   - If ReLU neurons die (always negative input)
   - Gradient is zero
   - Solution: Residual connections

2. Exploding Gradients:
   - Large weights amplify gradients
   - Solution: Gradient clipping, normalization

3. Dead ReLU:
   - Neuron always outputs 0
   - No learning
   - Solution: GELU, LeakyReLU, careful initialization

5.6 POSITION-WISE: WHY?
-----------------------

Independence Across Positions:
------------------------------
Each position processed independently.

Benefits:
---------
1. Parallelization:
   - All positions computed simultaneously
   - Extremely GPU-efficient
   - No sequential dependency

2. Parameter Sharing:
   - Same weights for all positions
   - Reduces parameters
   - Generalizes across position

3. Translation Invariance:
   - Same transformation everywhere
   - Learns position-independent features
   - Complements positional encoding

Comparison to Alternatives:
----------------------------
1. Shared FFN (used):
   - Same weights everywhere
   - O(d_model × d_ff) parameters

2. Position-specific FFN:
   - Different weights per position
   - O(n × d_model × d_ff) parameters
   - Too many parameters, poor generalization

3. Convolutional:
   - Local interactions
   - Not used in standard transformers
   - See in some variants (ConvBERT)

Connection to 1D Convolution:
------------------------------
Position-wise FFN is equivalent to:
- 1×1 convolution (in CNN terminology)
- Pointwise convolution
- Applied to each position independently

This perspective helpful for:
- Understanding computational patterns
- Implementing efficiently
- Connecting to CNN literature
"""

print("\n" + "=" * 80)
print("PART 6: FEED-FORWARD NETWORK THEORY")
print("=" * 80)

# ============================================================================
# PART 7: RESIDUAL CONNECTIONS AND LAYER NORMALIZATION
# ============================================================================

"""
6. RESIDUAL CONNECTIONS
========================

6.1 BASIC FORMULATION
---------------------

Definition:
-----------
    output = x + F(x)

where:
- x: input
- F(x): some transformation (attention, FFN, etc.)
- +: element-wise addition

Also called "skip connections" or "shortcut connections".

In Transformers:
----------------
Applied around each sublayer:
1. After self-attention: x + Attention(x)
2. After feed-forward: x + FFN(x)

Both followed by layer normalization.

6.2 WHY RESIDUAL CONNECTIONS?
------------------------------

Problem: Degradation in Deep Networks
--------------------------------------
Without residuals, deeper networks often perform WORSE than shallow ones.

Not due to overfitting (training error increases!)

Hypothesis: Optimization difficulty
- Deep networks hard to optimize
- Gradients vanish/explode
- Hard to learn identity mapping

Solution: Residual Learning
----------------------------
Instead of learning H(x), learn residual F(x) = H(x) - x

    H(x) = x + F(x)

Benefits:
---------
1. Learning identity is easy: set F(x) = 0
2. If deeper layer not needed, network can learn to skip it
3. Gradient flow: direct path through residual

Mathematical Analysis:
----------------------
Forward pass:
    H(x) = x + F(x)

Backward pass (chain rule):
    ∂L/∂x = ∂L/∂H · ∂H/∂x
          = ∂L/∂H · (1 + ∂F/∂x)

Key insight: The "+1" term!
- Even if ∂F/∂x → 0 (vanishing gradient in F)
- We still have ∂L/∂x ≈ ∂L/∂H
- Gradient flows through shortcut

6.3 GRADIENT FLOW ANALYSIS
---------------------------

Consider L-layer network with residuals:
    x_l = x_{l-1} + F_l(x_{l-1})

Expanding:
    x_L = x_0 + Σ_{l=1}^L F_l(x_{l-1})

Gradient:
    ∂L/∂x_0 = ∂L/∂x_L · ∂x_L/∂x_0
            = ∂L/∂x_L · (1 + Σ_{l=1}^L ∂F_l/∂x_0)

Two paths for gradients:
1. Through residuals: ∂L/∂x_L (always present!)
2. Through layers: ∂L/∂x_L · ∂F_l/∂x_0

Benefits:
---------
1. No vanishing: First term always present
2. No explosion: Bounded by 1 + sum, not product
3. Ensembling: Sum of gradients from multiple paths

Comparison to Feedforward:
---------------------------
Feedforward: ∂L/∂x_0 = ∂L/∂x_L · ∏_{l=1}^L ∂F_l/∂x_{l-1}
- Product of L terms
- Vanishes if any term small
- Explodes if any term large

Residual: ∂L/∂x_0 = ∂L/∂x_L · (1 + ...)
- Sum preserves gradients
- Much more stable

6.4 RESIDUAL FUNCTION LEARNING
-------------------------------

Hypothesis: Learning Residuals is Easier
-----------------------------------------
Consider desired mapping H*(x).

Option 1: Learn H(x) directly
- Network must learn full function
- Difficult optimization

Option 2: Learn F(x) = H*(x) - x
- Network learns difference from identity
- Often smaller, simpler function
- Easier optimization

Empirical Evidence:
-------------------
In trained networks:
- Residuals F(x) often have smaller norm than inputs x
- Suggests small refinements to identity
- Validates hypothesis

Mathematical Justification:
---------------------------
If optimal function close to identity:
    H*(x) ≈ x

Then F*(x) = H*(x) - x ≈ 0

Learning F ≈ 0 is easier than learning H ≈ I (identity).

6.5 INTERACTION WITH LAYER NORM
--------------------------------

Standard Transformer Block:
---------------------------
Two variants:

Post-LN (original):
    x = x + Attention(x)
    x = LayerNorm(x)
    x = x + FFN(x)
    x = LayerNorm(x)

Pre-LN (modern):
    x = x + Attention(LayerNorm(x))
    x = x + FFN(LayerNorm(x))

Differences:
------------
Post-LN:
+ Exact residual connection
- Can have gradient issues for very deep networks
- Original Transformer, BERT use this

Pre-LN:
+ More stable training
+ Can train deeper networks
- Slightly different function
- GPT-2, GPT-3 use this

Why Pre-LN Works Better:
-------------------------
1. Gradients:
   - Layer norm after residual can disrupt gradient flow
   - Layer norm before keeps residual pure

2. Initialization:
   - Pre-LN starts closer to identity
   - Post-LN needs careful initialization

3. Depth:
   - Pre-LN scales to 100+ layers
   - Post-LN struggles beyond 12-24 layers

Mathematical Analysis:
----------------------
Pre-LN gradient:
    ∂L/∂x = ∂L/∂output · (1 + ∂F/∂LayerNorm(x) · ∂LayerNorm/∂x)

Post-LN gradient:
    ∂L/∂x = ∂L/∂LayerNorm(x+F(x)) · ∂LayerNorm/∂(x+F(x)) · (1 + ∂F/∂x)

Pre-LN has cleaner gradient path through residual.
"""

print("\n" + "=" * 80)
print("PART 7: RESIDUAL CONNECTIONS THEORY")
print("=" * 80)

# ============================================================================
# PART 8: COMPLETE TRANSFORMER ARCHITECTURE
# ============================================================================

"""
7. COMPLETE TRANSFORMER ARCHITECTURE
=====================================

7.1 ENCODER-DECODER ARCHITECTURE
---------------------------------

Full Transformer:
-----------------
Input → Encoder → Decoder → Output

Encoder:
--------
N identical layers, each with:
1. Multi-head self-attention
2. Add & Norm
3. Feed-forward network
4. Add & Norm

Encoder processes source sequence (e.g., English sentence).

Decoder:
--------
N identical layers, each with:
1. Masked multi-head self-attention (causal)
2. Add & Norm
3. Multi-head cross-attention (to encoder output)
4. Add & Norm
5. Feed-forward network
6. Add & Norm

Decoder generates target sequence (e.g., French sentence).

7.2 INFORMATION FLOW
--------------------

Encoder:
--------
1. Input Embedding:
   - Token IDs → dense vectors
   - Add positional encoding
   - Dropout

2. For each encoder layer:
   - Self-attention: each position attends to all positions
   - Residual + Norm
   - FFN: position-wise transformation
   - Residual + Norm

3. Output:
   - Contextualized representations
   - Each token has information from entire input

Decoder:
--------
1. Output Embedding (shifted right):
   - Previous output tokens
   - Add positional encoding
   - Dropout

2. For each decoder layer:
   - Masked self-attention: causal (can't see future)
   - Residual + Norm
   - Cross-attention: attend to encoder output
   - Residual + Norm
   - FFN: position-wise transformation
   - Residual + Norm

3. Output:
   - Linear projection to vocabulary
   - Softmax to get probabilities

7.3 ENCODER-ONLY (BERT-style)
------------------------------

Architecture:
-------------
- Only encoder stack
- Bidirectional attention (see entire sequence)
- Used for understanding tasks

Applications:
-------------
- Text classification
- Named entity recognition
- Question answering
- Sentence similarity

Key Difference:
---------------
Can attend to entire sequence (including "future").
- Better for understanding
- Cannot generate autoregressively

Training:
---------
Masked Language Modeling (MLM):
- Mask 15% of tokens
- Predict masked tokens
- Learn bidirectional context

7.4 DECODER-ONLY (GPT-style)
-----------------------------

Architecture:
-------------
- Only decoder stack
- Causal (masked) attention
- Used for generation tasks

Applications:
-------------
- Text generation
- Language modeling
- Few-shot learning
- Creative writing

Key Difference:
---------------
Can only attend to previous positions.
- Natural for autoregressive generation
- Can generate text left-to-right

Training:
---------
Causal Language Modeling:
- Predict next token given previous
- Maximize P(x_t | x_<t)
- Standard language modeling objective

7.5 CROSS-ATTENTION
-------------------

Definition:
-----------
    CrossAttention(Q, K, V) = Attention(Q_decoder, K_encoder, V_encoder)

Where:
- Q comes from decoder
- K, V come from encoder

Purpose:
--------
Allows decoder to "attend to" encoder output.
- Queries: what decoder is looking for
- Keys/Values: what encoder provides

In Translation:
---------------
- Decoder generating French word
- Attends to relevant English words
- Dynamic "alignment" learned automatically

Mathematical:
-------------
Same as self-attention, but:
- Q from one sequence (decoder)
- K, V from another sequence (encoder)

This enables:
- Conditioning generation on input
- Learning alignments
- Transferring information across sequences

7.6 ARCHITECTURAL VARIANTS
---------------------------

1. Transformer-XL:
------------------
- Relative positional encoding
- Segment-level recurrence
- Handles longer sequences

2. Reformer:
------------
- LSH attention (approximate)
- Reversible layers
- Memory efficient

3. Longformer:
--------------
- Sparse attention patterns
- Local + global attention
- Linear complexity

4. Vision Transformer (ViT):
----------------------------
- Patches instead of tokens
- Same architecture
- Competitive with CNNs

5. Switch Transformer:
----------------------
- Mixture of Experts
- Sparse activation
- Massive scale

Common Modifications:
---------------------
- Different attention patterns (sparse, local, etc.)
- Different normalization (LayerNorm, RMSNorm)
- Different activations (GELU, SwiGLU)
- Different positional encodings (learned, RoPE, ALiBi)
"""

print("\n" + "=" * 80)
print("PART 8: COMPLETE ARCHITECTURE THEORY")
print("=" * 80)

# ============================================================================
# PART 9: TRAINING DYNAMICS
# ============================================================================

"""
8. TRAINING TRANSFORMERS
=========================

8.1 LOSS FUNCTIONS
------------------

For Language Modeling:
----------------------
Cross-Entropy Loss:
    L = -Σ_t log P(x_t | x_<t)

where:
- x_t: true token at position t
- P(x_t | x_<t): predicted probability

Per Position:
    L_t = -log(softmax(o_t)_{y_t})

where:
- o_t: logits at position t
- y_t: true label at position t

For Masked LM (BERT):
---------------------
    L = -Σ_{t∈masked} log P(x_t | x_{\t})

Only compute loss on masked positions.

For Sequence-to-Sequence:
-------------------------
    L = -Σ_t log P(y_t | y_<t, x)

where:
- x: source sequence
- y: target sequence

8.2 OPTIMIZATION
----------------

Adam Optimizer:
---------------
Most common choice for transformers.

Update rule:
    m_t = β₁m_{t-1} + (1-β₁)g_t
    v_t = β₂v_{t-1} + (1-β₂)g_t²
    m̂_t = m_t / (1-β₁^t)
    v̂_t = v_t / (1-β₂^t)
    θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)

where:
- g_t: gradient at step t
- m_t: first moment (mean)
- v_t: second moment (variance)
- β₁, β₂: decay rates (typically 0.9, 0.999)
- α: learning rate
- ε: numerical stability (10⁻⁸)

Why Adam?
---------
1. Adaptive learning rates per parameter
2. Momentum helps escape local minima
3. Works well out-of-the-box
4. Handles sparse gradients (embeddings)

Alternatives:
-------------
- AdamW: Adam with decoupled weight decay
- Adafactor: Memory-efficient for large models
- LAMB: Layer-wise adaptive for large batches

8.3 LEARNING RATE SCHEDULES
----------------------------

Warmup + Decay:
---------------
Original Transformer schedule:
    lr = d_model^{-0.5} · min(step^{-0.5}, step · warmup^{-1.5})

Properties:
- Linear increase during warmup
- Inverse square root decay after

Why Warmup?
-----------
1. Initial Training Instability:
   - Random initialization
   - Large gradients early
   - Warmup stabilizes

2. Adam Statistics:
   - Need time to accumulate momentum
   - Warmup allows this

3. Empirical Success:
   - Better final performance
   - More stable training

Typical Values:
---------------
- Warmup steps: 4000-10000
- Peak learning rate: 10⁻⁴ to 10⁻³
- Final learning rate: 10⁻⁵ to 10⁻⁴

Modern Schedules:
-----------------
1. Cosine Decay:
    lr = lr_min + 0.5(lr_max - lr_min)(1 + cos(πt/T))

2. Linear Decay:
    lr = lr_max - (lr_max - lr_min) · t/T

3. Constant with Warmup:
    lr = lr_max after warmup

8.4 REGULARIZATION
------------------

Dropout:
--------
Randomly zero activations with probability p.

Applied to:
1. Embeddings: After adding positional encoding
2. Attention: After softmax
3. FFN: After first activation
4. Residual: Before adding to main path

Typical rate: p = 0.1

Why Dropout?
------------
1. Prevents co-adaptation of neurons
2. Ensemble effect (averaging many subnetworks)
3. Robust to missing features

Label Smoothing:
----------------
Instead of hard targets [0, 0, 1, 0]:
Use soft targets [ε/V, ε/V, 1-ε, ε/V]

where:
- ε: smoothing parameter (typically 0.1)
- V: vocabulary size

Benefits:
---------
1. Prevents overconfidence
2. Better calibration
3. Improved generalization

Weight Decay:
-------------
Add L2 penalty to loss:
    L = L_task + λ||θ||²

Or in AdamW, decouple from gradient:
    θ_t = θ_{t-1} - α·(m̂_t/(√v̂_t + ε) + λθ_{t-1})

Benefits:
---------
1. Prevents large weights
2. Better generalization
3. Implicit regularization

8.5 INITIALIZATION
------------------

Xavier/Glorot Initialization:
------------------------------
For weight matrix W ∈ ℝ^{n_in × n_out}:

    W ~ U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))

Or normal version:
    W ~ N(0, 2/(n_in + n_out))

Why This Works:
---------------
Maintains variance of activations:
    Var(output) ≈ Var(input)

Proof (simplified):
For y = Wx where x has unit variance:
    Var(y_i) = Σ_j Var(W_{ij})Var(x_j)
             = n_in · Var(W) · 1

Want Var(y) = 1, so:
    Var(W) = 1/n_in

Similar analysis for backward pass gives 1/n_out.
Compromise: use average.

Transformer-Specific:
---------------------
1. Embeddings: Small random (std = 0.02)
2. Attention: Xavier
3. FFN: Xavier, but scale down by √(2N) for N layers
4. LayerNorm: γ = 1, β = 0

8.6 GRADIENT CLIPPING
---------------------

Problem:
--------
Gradients can explode, causing:
- NaN in parameters
- Unstable training
- Divergence

Solution:
---------
Clip gradients if norm exceeds threshold:

    if ||g|| > threshold:
        g = g · threshold / ||g||

Typical threshold: 1.0 or 5.0

Why It Works:
-------------
- Limits step size
- Prevents catastrophic updates
- Maintains direction, scales magnitude

8.7 MIXED PRECISION TRAINING
-----------------------------

Idea:
-----
Use float16 for most operations, float32 for critical parts.

Benefits:
---------
1. 2× faster (fp16 ops faster on modern GPUs)
2. 2× less memory
3. Larger batch sizes possible

Challenges:
-----------
1. Underflow: Small gradients → 0 in fp16
2. Overflow: Large values → inf in fp16

Solutions:
----------
1. Loss Scaling:
   - Multiply loss by large factor
   - Scale gradients up (prevent underflow)
   - Scale down before optimizer step

2. Mixed Precision:
   - FP16 for forward/backward
   - FP32 for parameter updates
   - FP32 master copy of weights

Implementation:
---------------
Modern libraries (PyTorch AMP, TensorFlow mixed_precision) handle automatically.
"""

print("\n" + "=" * 80)
print("PART 9: TRAINING DYNAMICS THEORY")
print("=" * 80)

# ============================================================================
# PART 10: COMPUTATIONAL COMPLEXITY
# ============================================================================

"""
9. COMPUTATIONAL COMPLEXITY ANALYSIS
=====================================

9.1 SELF-ATTENTION COMPLEXITY
------------------------------

Operations:
-----------
1. QK^T: (n × d_k) @ (d_k × n) = n² × d_k multiplications
2. Softmax: n² exponentials + divisions
3. Attention @ V: (n × n) @ (n × d_v) = n² × d_v multiplications

Total: O(n² × d_k + n² + n² × d_v) = O(n²d)

where:
- n: sequence length
- d: model dimension

Memory:
-------
Attention matrix: n × n
Intermediate QKV: 3 × n × d

Total: O(n² + nd)

Bottleneck:
-----------
For long sequences (n >> d), the n² term dominates!

Example:
- n = 1024, d = 768
- Attention matrix: 1M entries
- O(n²) = 1M operations per dimension

This is why attention is expensive for long sequences.

9.2 FEED-FORWARD COMPLEXITY
----------------------------

Operations:
-----------
1. First linear: (n × d) @ (d × d_ff) = n × d × d_ff
2. Activation: n × d_ff
3. Second linear: (n × d_ff) @ (d_ff × d) = n × d_ff × d

Total: O(2nd × d_ff) = O(nd²) for d_ff = 4d

Memory:
-------
Intermediate: n × d_ff = n × 4d

Total: O(nd)

Comparison:
-----------
Attention: O(n²d)
FFN: O(nd²)

For typical transformers (d ≈ 768):
- Short sequences (n < d): FFN dominates
- Long sequences (n > d): Attention dominates

9.3 TOTAL COMPLEXITY PER LAYER
-------------------------------

One Transformer Layer:
----------------------
- Self-attention: O(n²d)
- FFN: O(nd²)
- LayerNorm: O(nd)
- Residual: O(nd)

Total: O(n²d + nd²)

Full Model (L layers):
-----------------------
- Encoder/Decoder: L × O(n²d + nd²)
- Embeddings: O(nd)
- Output: O(ndV) where V is vocabulary size

Total: O(L(n²d + nd²) + ndV)

9.4 MEMORY REQUIREMENTS
------------------------

Parameters:
-----------
For BERT-base (L=12, d=768, d_ff=3072, V=30k):

- Embeddings: V × d = 30k × 768 ≈ 23M
- Each layer:
  - Attention: 4 × d² = 4 × 768² ≈ 2.4M
  - FFN: 2 × d × d_ff = 2 × 768 × 3072 ≈ 4.7M
  - LayerNorm: 4d ≈ 3k
  - Total per layer: ≈ 7.1M
- All layers: 12 × 7.1M ≈ 85M
- Total: ≈ 110M parameters

Storage: 110M × 4 bytes (fp32) = 440MB

Activations:
------------
For batch size B, sequence length n:

Per layer:
- Self-attention:
  - QKV: 3 × B × n × d
  - Attention matrix: B × h × n × n (h heads)
  - Output: B × n × d

- FFN:
  - Hidden: B × n × d_ff

Total per layer: B × n × (4d + d_ff + hn)

All layers: L × B × n × (4d + d_ff + hn)

Example (B=32, n=512, L=12, d=768, h=12, d_ff=3072):
≈ 12 × 32 × 512 × 4608 ≈ 900M floats ≈ 3.6GB

9.5 EFFICIENT ATTENTION VARIANTS
---------------------------------

Problem: O(n²) scaling
Solution: Approximate attention

1. Sparse Attention (O(n√n)):
------------------------------
- Only attend to subset of positions
- Local + global patterns
- Example: Longformer, BigBird

Complexity: O(n × k) where k << n

2. Linear Attention (O(nd²)):
------------------------------
- Factorize attention matrix
- Approximate with low-rank
- Example: Performer, FNet

Complexity: O(nd) or O(nd²)

3. Kernel-Based (O(nd)):
------------------------
- Replace softmax with kernel
- Compute in linear time
- Example: Linear Transformer

Complexity: O(nd)

Trade-offs:
-----------
Approximations:
+ Faster for long sequences
+ Lower memory
- May lose some expressiveness
- Quality depends on task

9.6 PARALLELIZATION
-------------------

GPU Efficiency:
---------------
Transformers are highly parallel:

1. Attention:
   - All positions computed simultaneously
   - Matrix multiplication: GPU-optimized
   - Batch parallelism: process multiple examples

2. FFN:
   - Independent per position
   - Embarrassingly parallel
   - Can process all positions at once

3. Multi-head:
   - All heads independent
   - Computed in parallel
   - Just reshape operations

Optimization Techniques:
------------------------
1. Kernel Fusion:
   - Combine multiple operations
   - Reduce memory transfers
   - Example: Flash Attention

2. Mixed Precision:
   - FP16 for speed
   - 2× faster on modern GPUs

3. Gradient Checkpointing:
   - Trade computation for memory
   - Recompute activations during backward
   - Enables larger models

Model Parallelism:
------------------
For very large models:
1. Layer parallelism: Different layers on different GPUs
2. Tensor parallelism: Split matrices across GPUs
3. Pipeline parallelism: Process different micro-batches

Example (GPT-3, 175B parameters):
- Requires model parallelism
- Split across hundreds of GPUs
- Sophisticated scheduling needed
"""

print("\n" + "=" * 80)
print("PART 10: COMPUTATIONAL COMPLEXITY THEORY")
print("=" * 80)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("THEORETICAL SUMMARY")
print("=" * 80)

summary_text = """
KEY THEORETICAL INSIGHTS
========================

1. ATTENTION AS SOFT LOOKUP
----------------------------
- Attention = differentiable key-value store
- Queries find relevant keys
- Values weighted by similarity
- Statistical: expected value under learned distribution

2. MULTI-HEAD = ENSEMBLE
-------------------------
- Multiple parallel attention mechanisms
- Each head learns different patterns
- Combination more powerful than single view
- Same cost as single-head!

3. POSITIONAL ENCODING
----------------------
- Sinusoidal = unique fingerprint per position
- Multiple frequencies = multi-scale information
- Allows extrapolation to longer sequences
- Alternative: learned, relative, rotary

4. RESIDUAL CONNECTIONS
-----------------------
- Enable gradient flow in deep networks
- Learn residual (difference from identity)
- Sum of paths, not product
- Critical for training deep transformers

5. LAYER NORMALIZATION
----------------------
- Stabilize training dynamics
- Reduce internal covariate shift
- Make optimization easier
- Pre-LN vs Post-LN affects depth

6. FEED-FORWARD NETWORKS
------------------------
- Add non-linearity (attention is linear in values)
- Expand then compress (bottleneck)
- Position-wise = parallel
- 2/3 of parameters typically

7. COMPLEXITY TRADE-OFFS
------------------------
- Attention: O(n²d) - bottleneck for long sequences
- FFN: O(nd²) - bottleneck for high dimensions
- Total: O(L(n²d + nd²)) for L layers
- Memory: activations often exceed parameters

8. TRAINING STABILITY
---------------------
- Warmup prevents early instability
- Gradient clipping prevents explosions
- Mixed precision enables larger models
- Careful initialization essential

DESIGN PRINCIPLES
=================

1. Parallelization First
   - All positions processed simultaneously
   - GPU-friendly operations
   - Enables large-scale training

2. Depth Through Residuals
   - Stack many layers
   - Gradient flow maintained
   - Hierarchical representations

3. Multi-Head Diversity
   - Learn multiple perspectives
   - Ensemble-like benefits
   - Specialization emerges

4. Normalization for Stability
   - Layer norm after sublayers
   - Stabilizes training
   - Enables adaptive learning rates

5. Position-wise Processing
   - FFN independent per position
   - Parameter sharing
   - Translation invariance

MATHEMATICAL FOUNDATIONS
=========================

Core Operations:
1. Matrix Multiplication: Attention, FFN
2. Softmax: Probability distribution
3. Layer Norm: Standardization
4. Residual: Skip connections
5. Non-linearity: ReLU, GELU

All differentiable → end-to-end learning!

Statistical View:
- Attention: Conditional expectation
- Softmax: Maximum entropy distribution
- Layer Norm: Standardization
- Dropout: Bayesian approximation
- Ensemble: Multiple models averaged

Information Theory:
- Attention: Information routing
- Bottleneck: Compression
- Cross-entropy: Information content
- Perplexity: Uncertainty measure

WHEN TO USE TRANSFORMERS
=========================

Advantages:
+ Parallel processing (fast training)
+ Long-range dependencies
+ State-of-the-art performance
+ Transfer learning (pre-training)
+ Flexible architecture

Disadvantages:
- O(n²) memory for attention
- Requires lots of data
- Computationally expensive
- Position encoding needed

Best For:
- NLP: Text classification, generation, QA
- Vision: Image classification, detection
- Speech: Recognition, synthesis
- Multi-modal: Image captioning, VQA

Consider Alternatives When:
- Sequence length > 10k (use efficient variants)
- Limited compute (use smaller models)
- Real-time inference (use distilled models)
- Structured data (may benefit from specialized architectures)

THE BIG PICTURE
===============

Transformers succeeded by:
1. Eliminating recurrence (parallelization)
2. Direct connections (attention)
3. Deep stacking (residuals + norm)
4. Large scale (billions of parameters)
5. Pre-training (transfer learning)

They are not "magic" - just:
- Well-designed architecture
- Effective use of compute
- Large-scale training data
- Careful engineering

Understanding theory helps:
- Debug issues
- Design improvements
- Make informed choices
- Push boundaries

The transformer revolution continues!
"""

print(summary_text)

print("\n" + "=" * 80)
print("END OF THEORETICAL FOUNDATIONS")
print("=" * 80)
print("\nThis theory supports the implementations in:")
print("- transformer_architecture_explained.py")
print("- transformer_libraries_explained.py")
print("- transformer_practical_comparison.py")
print("\nRefer to code for concrete examples of these concepts!")
