"""
PRACTICAL COMPARISON: Library APIs vs Manual Implementation
============================================================

This file contains side-by-side comparisons of common transformer operations
using libraries vs manual implementation, so you can see exactly what happens.
"""

import numpy as np

print("=" * 80)
print("PRACTICAL TRANSFORMER OPERATIONS - SIDE BY SIDE COMPARISON")
print("=" * 80)

# ============================================================================
# EXAMPLE 1: SOFTMAX
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 1: SOFTMAX OPERATION")
print("=" * 80)

# Sample attention scores
scores = np.array([
    [2.5, 1.0, 0.5, -1.0],
    [1.5, 2.0, 0.0, -0.5],
    [3.0, 2.5, 1.5, 0.5]
])

print("\nInput scores (attention logits):")
print(scores)

# ===== LIBRARY WAY =====
print("\n--- Using Library (Conceptual PyTorch) ---")
print("Code: F.softmax(scores, dim=-1)")
print("What it does: Calls optimized CUDA kernel")

# ===== MANUAL WAY =====
print("\n--- Manual Implementation ---")
print("Step-by-step execution:")

# Step 1: Subtract max for numerical stability
scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
print(f"\n1. Subtract max: scores - {np.max(scores, axis=-1, keepdims=True).flatten()}")
print(scores_shifted)

# Step 2: Exponentiate
exp_scores = np.exp(scores_shifted)
print(f"\n2. Exponentiate: e^x")
print(exp_scores)

# Step 3: Normalize
softmax_output = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
print(f"\n3. Normalize by sum: {np.sum(exp_scores, axis=-1, keepdims=True).flatten()}")
print(softmax_output)

print("\n✓ Verification: Each row sums to 1.0")
print(f"Row sums: {np.sum(softmax_output, axis=-1)}")

# ============================================================================
# EXAMPLE 2: LAYER NORMALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 2: LAYER NORMALIZATION")
print("=" * 80)

# Sample hidden states
hidden_states = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [0.5, 1.5, 2.5, 3.5],
    [2.0, 4.0, 6.0, 8.0]
])

print("\nInput hidden states:")
print(hidden_states)
print(f"Shape: {hidden_states.shape} (batch=3, d_model=4)")

# ===== LIBRARY WAY =====
print("\n--- Using Library (PyTorch/TensorFlow) ---")
print("Code: nn.LayerNorm(normalized_shape=4)(hidden_states)")
print("What it does: Normalize, then scale & shift with learned params")

# ===== MANUAL WAY =====
print("\n--- Manual Implementation ---")

# Learnable parameters (typically learned during training)
gamma = np.array([1.0, 1.0, 1.0, 1.0])  # Scale parameter
beta = np.array([0.0, 0.0, 0.0, 0.0])   # Shift parameter
epsilon = 1e-6

print("\n1. Compute mean for each example (across features):")
mean = np.mean(hidden_states, axis=-1, keepdims=True)
print(f"   Means: {mean.flatten()}")

print("\n2. Compute variance for each example:")
variance = np.var(hidden_states, axis=-1, keepdims=True)
print(f"   Variances: {variance.flatten()}")

print("\n3. Normalize (subtract mean, divide by std):")
normalized = (hidden_states - mean) / np.sqrt(variance + epsilon)
print(normalized)

print("\n4. Scale and shift with learned parameters:")
output = gamma * normalized + beta
print(output)

print("\n✓ Verification: Mean ≈ 0, Variance ≈ 1 for each row")
print(f"Output means: {np.mean(output, axis=-1)}")
print(f"Output variances: {np.var(output, axis=-1)}")

# ============================================================================
# EXAMPLE 3: SINGLE-HEAD ATTENTION
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 3: SINGLE-HEAD ATTENTION")
print("=" * 80)

# Small sequence for demonstration
seq_len = 4
d_k = 8

# Random embeddings for 4 tokens
X = np.random.randn(seq_len, d_k)
print(f"\nInput embeddings for {seq_len} tokens:")
print(f"Shape: {X.shape} (seq_len={seq_len}, d_k={d_k})")
print(X)

# Weight matrices (normally learned)
W_q = np.random.randn(d_k, d_k) * 0.1
W_k = np.random.randn(d_k, d_k) * 0.1
W_v = np.random.randn(d_k, d_k) * 0.1

# ===== LIBRARY WAY =====
print("\n--- Using Library ---")
print("Code: attention_layer(query=X, key=X, value=X)")
print("Returns: (attention_output, attention_weights)")

# ===== MANUAL WAY =====
print("\n--- Manual Implementation ---")
print("\nSTEP 1: Project to Q, K, V spaces")
Q = X @ W_q
K = X @ W_k
V = X @ W_v
print(f"Q = X @ W_q, shape: {Q.shape}")
print(f"K = X @ W_k, shape: {K.shape}")
print(f"V = X @ W_v, shape: {V.shape}")

print("\nSTEP 2: Compute attention scores (Q @ K^T)")
scores = Q @ K.T
print(f"Attention scores shape: {scores.shape} ({seq_len}×{seq_len})")
print("Attention scores:")
print(scores)
print("\nInterpretation: scores[i,j] = similarity between position i and j")

print("\nSTEP 3: Scale by √d_k")
scaled_scores = scores / np.sqrt(d_k)
print(f"Scaling factor: 1/√{d_k} = {1/np.sqrt(d_k):.4f}")
print("Scaled scores:")
print(scaled_scores)

print("\nSTEP 4: Apply softmax to get attention weights")
# Subtract max for stability
exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
print("Attention weights (probabilities):")
print(attention_weights)
print(f"\n✓ Each row sums to 1: {np.sum(attention_weights, axis=-1)}")

print("\nSTEP 5: Apply attention weights to values")
attention_output = attention_weights @ V
print(f"Output shape: {attention_output.shape}")
print("Attention output:")
print(attention_output)

print("\n📊 INTERPRETATION:")
print("For each position (row), we computed a weighted average of all values,")
print("where weights come from the similarity between queries and keys.")
print(f"\nExample: Position 0 attends most to positions {np.argmax(attention_weights[0])} and {np.argsort(attention_weights[0])[-2]}")

# ============================================================================
# EXAMPLE 4: MULTI-HEAD ATTENTION
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 4: MULTI-HEAD ATTENTION")
print("=" * 80)

d_model = 16
num_heads = 4
d_k = d_model // num_heads  # 4 per head
seq_len = 3

X = np.random.randn(seq_len, d_model)
print(f"\nInput: {seq_len} tokens, {d_model} dimensions each")

# ===== LIBRARY WAY =====
print("\n--- Using Library ---")
print(f"Code: MultiHeadAttention(num_heads={num_heads}, key_dim={d_k})(X, X, X)")
print("Internally: Creates 4 sets of Q,K,V projections, computes 4 attentions in parallel")

# ===== MANUAL WAY =====
print("\n--- Manual Implementation ---")

# Single set of projection matrices (library would have separate for each head)
W_q = np.random.randn(d_model, d_model) * 0.1
W_k = np.random.randn(d_model, d_model) * 0.1
W_v = np.random.randn(d_model, d_model) * 0.1
W_o = np.random.randn(d_model, d_model) * 0.1

print("\nSTEP 1: Linear projections")
Q = X @ W_q  # (seq_len, d_model)
K = X @ W_k
V = X @ W_v
print(f"Q, K, V shapes: {Q.shape}")

print(f"\nSTEP 2: Reshape for {num_heads} heads")
print(f"Original: ({seq_len}, {d_model})")
print(f"Reshape to: ({seq_len}, {num_heads}, {d_k})")
Q_heads = Q.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)  # (heads, seq, d_k)
K_heads = K.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
V_heads = V.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
print(f"After reshape: {Q_heads.shape} (heads, seq_len, d_k)")

print("\nSTEP 3: Compute attention for each head independently")
head_outputs = []
for i in range(num_heads):
    print(f"\n  Head {i}:")
    Q_head = Q_heads[i]
    K_head = K_heads[i]
    V_head = V_heads[i]
    
    # Attention computation
    scores = (Q_head @ K_head.T) / np.sqrt(d_k)
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    output = weights @ V_head
    head_outputs.append(output)
    print(f"    Attention output shape: {output.shape}")

print("\nSTEP 4: Concatenate all heads")
concat_output = np.concatenate(head_outputs, axis=-1)
print(f"Concatenated shape: {concat_output.shape} (seq_len, num_heads × d_k)")

print("\nSTEP 5: Final linear projection")
final_output = concat_output @ W_o
print(f"Final output shape: {final_output.shape}")

print("\n📊 KEY INSIGHT:")
print(f"We ran {num_heads} separate attention mechanisms in parallel!")
print(f"Each head learned to focus on different aspects of the relationships.")
print(f"Total cost: same as single-head with dimension {d_model}")

# ============================================================================
# EXAMPLE 5: POSITION-WISE FEED-FORWARD
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 5: POSITION-WISE FEED-FORWARD NETWORK")
print("=" * 80)

d_model = 8
d_ff = 32
seq_len = 3

X = np.random.randn(seq_len, d_model)
print(f"\nInput: {seq_len} positions, {d_model} dimensions")
print(X)

# ===== LIBRARY WAY =====
print("\n--- Using Library ---")
print("Code: Sequential([Dense(d_ff), ReLU(), Dense(d_model)])(X)")
print("Applied independently to each position")

# ===== MANUAL WAY =====
print("\n--- Manual Implementation ---")

W1 = np.random.randn(d_model, d_ff) * 0.1
b1 = np.zeros(d_ff)
W2 = np.random.randn(d_ff, d_model) * 0.1
b2 = np.zeros(d_model)

print("\nSTEP 1: First linear layer (expansion)")
print(f"X @ W1 + b1")
print(f"({d_model}) → ({d_ff})")
hidden = X @ W1 + b1
print(f"Hidden shape: {hidden.shape}")
print(hidden)

print("\nSTEP 2: ReLU activation")
print("ReLU(x) = max(0, x)")
activated = np.maximum(0, hidden)
print(f"After ReLU (negative values → 0):")
print(activated)

print("\nSTEP 3: Second linear layer (compression)")
print(f"hidden @ W2 + b2")
print(f"({d_ff}) → ({d_model})")
output = activated @ W2 + b2
print(f"Output shape: {output.shape}")
print(output)

print("\n📊 KEY INSIGHT:")
print("Each position is transformed independently!")
print(f"Position 0: {d_model} → {d_ff} → {d_model}")
print(f"Position 1: {d_model} → {d_ff} → {d_model}")
print(f"Position 2: {d_model} → {d_ff} → {d_model}")
print("Same weights for all positions (shared parameters)")

# ============================================================================
# EXAMPLE 6: COMPLETE FORWARD PASS
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 6: COMPLETE ENCODER LAYER FORWARD PASS")
print("=" * 80)

d_model = 8
seq_len = 3
X = np.random.randn(seq_len, d_model)

print(f"\nInput sequence: {seq_len} tokens × {d_model} dims")
print(X)

print("\n" + "-" * 80)
print("SUBLAYER 1: MULTI-HEAD ATTENTION")
print("-" * 80)

# Simplified single-head attention for clarity
W_q = np.random.randn(d_model, d_model) * 0.1
W_k = np.random.randn(d_model, d_model) * 0.1
W_v = np.random.randn(d_model, d_model) * 0.1

Q = X @ W_q
K = X @ W_k
V = X @ W_v

scores = (Q @ K.T) / np.sqrt(d_model)
attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
attn_output = attention_weights @ V

print(f"Attention output shape: {attn_output.shape}")

print("\nResidual Connection: X + attention_output")
x_after_attn = X + attn_output

print("\nLayer Normalization")
mean = np.mean(x_after_attn, axis=-1, keepdims=True)
var = np.var(x_after_attn, axis=-1, keepdims=True)
x_normalized = (x_after_attn - mean) / np.sqrt(var + 1e-6)

print(f"After attention + residual + norm: {x_normalized.shape}")

print("\n" + "-" * 80)
print("SUBLAYER 2: FEED-FORWARD NETWORK")
print("-" * 80)

d_ff = 32
W1 = np.random.randn(d_model, d_ff) * 0.1
W2 = np.random.randn(d_ff, d_model) * 0.1

ff_output = np.maximum(0, x_normalized @ W1) @ W2
print(f"FFN output shape: {ff_output.shape}")

print("\nResidual Connection: x_normalized + ff_output")
x_after_ffn = x_normalized + ff_output

print("\nLayer Normalization")
mean = np.mean(x_after_ffn, axis=-1, keepdims=True)
var = np.var(x_after_ffn, axis=-1, keepdims=True)
final_output = (x_after_ffn - mean) / np.sqrt(var + 1e-6)

print("\n" + "=" * 80)
print("FINAL OUTPUT")
print("=" * 80)
print(f"Shape: {final_output.shape}")
print(final_output)

print("\n📊 SUMMARY OF WHAT HAPPENED:")
print("1. Self-attention: Each token looked at all tokens")
print("2. Residual + Norm: Added input, normalized")
print("3. Feed-forward: Non-linear transformation per position")
print("4. Residual + Norm: Added previous, normalized")
print("5. Result: Contextualized representations!")

# ============================================================================
# COMPARISON TABLE
# ============================================================================

print("\n" + "=" * 80)
print("OPERATION COMPLEXITY COMPARISON")
print("=" * 80)

comparison = """
┌─────────────────────────┬─────────────────────┬──────────────────────┐
│ Operation               │ Library Call        │ Actual Computation   │
├─────────────────────────┼─────────────────────┼──────────────────────┤
│ Linear Layer            │ layer(x)            │ x @ W + b            │
│ Complexity              │                     │ O(n × d_in × d_out)  │
├─────────────────────────┼─────────────────────┼──────────────────────┤
│ Layer Norm              │ layer_norm(x)       │ (x-μ)/σ × γ + β      │
│ Complexity              │                     │ O(n × d)             │
├─────────────────────────┼─────────────────────┼──────────────────────┤
│ Attention               │ attn(Q,K,V)         │ softmax(QK^T/√d)V    │
│ Complexity              │                     │ O(n² × d)            │
├─────────────────────────┼─────────────────────┼──────────────────────┤
│ Feed-Forward            │ ffn(x)              │ ReLU(xW₁)W₂          │
│ Complexity              │                     │ O(n × d × d_ff)      │
├─────────────────────────┼─────────────────────┼──────────────────────┤
│ Encoder Layer           │ encoder_layer(x)    │ All of the above     │
│ Complexity              │                     │ O(n² × d + n × d²)   │
└─────────────────────────┴─────────────────────┴──────────────────────┘

For BERT-base (n=512, d=768, 12 layers):
- Attention: 12 × (512² × 768) ≈ 2.4B ops per example
- FFN: 12 × (512 × 768 × 3072) ≈ 14.5B ops per example
- Total: ~17B floating point operations per forward pass!
"""

print(comparison)

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
Key Takeaways:
--------------
1. Library calls hide the matrix multiplications and numerical operations
2. Everything boils down to basic linear algebra operations
3. The complexity comes from:
   - Number of operations (O(n²) for attention)
   - Number of parameters (millions to billions)
   - Number of layers (depth)
4. Understanding the underlying math helps you:
   - Debug performance issues
   - Optimize memory usage
   - Design better architectures
   - Estimate computational costs

When you call transformer_model(input):
→ Hundreds of matrix multiplications
→ Billions of floating point operations
→ All managed by optimized library code

The "magic" is systematic application of linear algebra + calculus!
""")
