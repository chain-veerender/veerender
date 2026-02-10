"""
TRANSFORMER LIBRARIES: WHAT HAPPENS UNDER THE HOOD
===================================================

This document shows exactly what happens when you use popular transformer libraries
(PyTorch, TensorFlow, HuggingFace) and maps them to the underlying mathematics.
"""

# ============================================================================
# PART 1: PyTorch Transformers
# ============================================================================

print("=" * 80)
print("PyTorch Transformer API - Under the Hood")
print("=" * 80)

pytorch_example = """
# ============================================================================
# 1. BASIC TRANSFORMER MODEL IN PYTORCH
# ============================================================================

import torch
import torch.nn as nn

# When you write this:
model = nn.Transformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1
)

# What happens under the hood:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TransformerUnderTheHood:
    def __init__(self):
        # 1. CREATE ENCODER LAYERS (6 layers)
        self.encoder_layers = []
        for i in range(6):
            layer = {
                # Multi-head attention
                'self_attn': {
                    'W_q': torch.randn(512, 512) * 0.02,  # Query projection
                    'W_k': torch.randn(512, 512) * 0.02,  # Key projection
                    'W_v': torch.randn(512, 512) * 0.02,  # Value projection
                    'W_o': torch.randn(512, 512) * 0.02,  # Output projection
                },
                # Feed-forward network
                'ffn': {
                    'W1': torch.randn(512, 2048) * 0.02,   # First layer
                    'b1': torch.zeros(2048),
                    'W2': torch.randn(2048, 512) * 0.02,   # Second layer
                    'b2': torch.zeros(512),
                },
                # Layer normalization parameters
                'norm1': {'gamma': torch.ones(512), 'beta': torch.zeros(512)},
                'norm2': {'gamma': torch.ones(512), 'beta': torch.zeros(512)},
            }
            self.encoder_layers.append(layer)
        
        # 2. CREATE DECODER LAYERS (6 layers, similar structure)
        # Each decoder layer has:
        # - Masked self-attention
        # - Cross-attention (to encoder output)
        # - Feed-forward network
        # - 3 layer normalizations
        
        # Total parameters per encoder layer:
        # - Attention: 4 × (512 × 512) = 1,048,576
        # - FFN: (512 × 2048) + (2048 × 512) + biases = 2,099,200
        # - LayerNorm: 2 × (512 + 512) = 2,048
        # Total per layer: ~3,149,824 parameters
        # Total for 6 encoder + 6 decoder: ~37,797,888 parameters!

# ============================================================================
# 2. USING THE MODEL
# ============================================================================

# Input preparation
src = torch.randn(10, 32, 512)  # (seq_len, batch, d_model)
tgt = torch.randn(20, 32, 512)  # (seq_len, batch, d_model)

# When you call:
output = model(src, tgt)

# What happens step by step:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def forward_pass_detailed(src, tgt):
    # ENCODER PATH
    x = src  # Start with source sequence
    
    for encoder_layer in range(6):
        # 1. Multi-Head Self-Attention
        Q = x @ W_q  # Project to query space
        K = x @ W_k  # Project to key space
        V = x @ W_v  # Project to value space
        
        # Split into 8 heads (each head has dimension 512/8 = 64)
        Q = Q.reshape(batch, seq, 8, 64).transpose(1, 2)  # (batch, heads, seq, 64)
        K = K.reshape(batch, seq, 8, 64).transpose(1, 2)
        V = V.reshape(batch, seq, 8, 64).transpose(1, 2)
        
        # Attention computation for each head
        scores = (Q @ K.transpose(-2, -1)) / sqrt(64)  # (batch, heads, seq, seq)
        attn_weights = softmax(scores, dim=-1)
        attn_output = attn_weights @ V  # (batch, heads, seq, 64)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).reshape(batch, seq, 512)
        attn_output = attn_output @ W_o  # Final projection
        
        # 2. Add & Norm
        x = layer_norm(x + attn_output)
        
        # 3. Feed-Forward Network
        ff_output = relu(x @ W1 + b1) @ W2 + b2
        
        # 4. Add & Norm
        x = layer_norm(x + ff_output)
    
    encoder_output = x
    
    # DECODER PATH
    y = tgt  # Start with target sequence
    
    for decoder_layer in range(6):
        # 1. Masked Multi-Head Self-Attention (causal mask)
        # Same as encoder attention, but with mask preventing future positions
        
        # 2. Add & Norm
        
        # 3. Cross-Attention (attend to encoder output)
        Q = y @ W_q  # Queries from decoder
        K = encoder_output @ W_k  # Keys from encoder
        V = encoder_output @ W_v  # Values from encoder
        # ... same attention computation ...
        
        # 4. Add & Norm
        
        # 5. Feed-Forward Network
        
        # 6. Add & Norm
    
    return y  # Final decoder output

# ============================================================================
# 3. SPECIFIC OPERATIONS
# ============================================================================

# A. Multi-Head Attention Layer
attention = nn.MultiheadAttention(
    embed_dim=512,
    num_heads=8,
    dropout=0.1
)

# Under the hood:
class MultiheadAttentionDetail:
    def __init__(self):
        # Linear layers for Q, K, V projections
        self.q_proj = nn.Linear(512, 512)  # Actually just stores W_q matrix
        self.k_proj = nn.Linear(512, 512)  # W_k matrix
        self.v_proj = nn.Linear(512, 512)  # W_v matrix
        self.out_proj = nn.Linear(512, 512)  # W_o matrix
        
        # When initialized:
        # - W_q, W_k, W_v, W_o are initialized with Xavier uniform
        # - Biases are initialized to zero
    
    def forward(self, query, key, value):
        # 1. Project inputs
        Q = self.q_proj(query)  # Equivalent to: query @ self.q_proj.weight.T + bias
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # 2. Reshape for multi-head
        batch_size = Q.size(0)
        seq_len = Q.size(1)
        
        Q = Q.view(batch_size, seq_len, 8, 64).transpose(1, 2)
        K = K.view(batch_size, seq_len, 8, 64).transpose(1, 2)
        V = V.view(batch_size, seq_len, 8, 64).transpose(1, 2)
        
        # 3. Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(64)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Optional: Apply dropout
        attn_weights = torch.dropout(attn_weights, p=0.1, training=True)
        
        attn_output = torch.matmul(attn_weights, V)
        
        # 4. Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, 512)
        
        # 5. Final projection
        output = self.out_proj(attn_output)
        
        return output, attn_weights

# B. Layer Normalization
layer_norm = nn.LayerNorm(512)

# Under the hood:
class LayerNormDetail:
    def __init__(self, d_model):
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(d_model))   # Scale
        self.beta = nn.Parameter(torch.zeros(d_model))   # Shift
        self.eps = 1e-6  # For numerical stability
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        
        # 1. Compute mean and variance across features (last dimension)
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # (batch, seq_len, 1)
        
        # 2. Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 3. Scale and shift
        output = self.gamma * x_norm + self.beta
        
        return output

# C. Feed-Forward Network
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        
        # Under the hood:
        # 1. First linear: (batch, seq_len, d_model) @ (d_model, d_ff)
        hidden = x @ self.linear1.weight.T + self.linear1.bias
        # Result: (batch, seq_len, d_ff)
        
        # 2. ReLU activation
        hidden = torch.maximum(hidden, torch.zeros_like(hidden))
        
        # 3. Second linear: (batch, seq_len, d_ff) @ (d_ff, d_model)
        output = hidden @ self.linear2.weight.T + self.linear2.bias
        # Result: (batch, seq_len, d_model)
        
        return output
"""

print(pytorch_example)

# ============================================================================
# PART 2: HuggingFace Transformers
# ============================================================================

print("\n" + "=" * 80)
print("HuggingFace Transformers - Under the Hood")
print("=" * 80)

huggingface_example = """
# ============================================================================
# 1. LOADING A PRE-TRAINED MODEL
# ============================================================================

from transformers import BertModel, BertTokenizer

# When you write:
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# What happens under the hood:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 1. Downloads pre-trained weights from HuggingFace Hub
#    File: pytorch_model.bin (size: ~440MB for BERT-base)
#    Contains: ~110 million parameters

# 2. Loads model configuration
#    - vocab_size: 30,522
#    - hidden_size: 768 (d_model)
#    - num_hidden_layers: 12
#    - num_attention_heads: 12
#    - intermediate_size: 3072 (d_ff)
#    - max_position_embeddings: 512

# 3. Initializes model architecture
class BERTModelStructure:
    def __init__(self):
        # Embedding Layer
        self.embeddings = {
            'word_embeddings': torch.randn(30522, 768),  # Vocab → hidden
            'position_embeddings': torch.randn(512, 768),  # Position → hidden
            'token_type_embeddings': torch.randn(2, 768),  # Segment → hidden
            'LayerNorm': {'gamma': torch.ones(768), 'beta': torch.zeros(768)},
        }
        
        # 12 Encoder Layers
        self.encoder_layers = []
        for i in range(12):
            layer = {
                'attention': {
                    'self': {
                        'query': torch.randn(768, 768),
                        'key': torch.randn(768, 768),
                        'value': torch.randn(768, 768),
                    },
                    'output': {
                        'dense': torch.randn(768, 768),
                        'LayerNorm': {'gamma': torch.ones(768), 'beta': torch.zeros(768)},
                    }
                },
                'intermediate': {
                    'dense': torch.randn(768, 3072),  # Expansion
                },
                'output': {
                    'dense': torch.randn(3072, 768),  # Compression
                    'LayerNorm': {'gamma': torch.ones(768), 'beta': torch.zeros(768)},
                }
            }
            self.encoder_layers.append(layer)
        
        # Pooler (for [CLS] token)
        self.pooler = {
            'dense': torch.randn(768, 768),
        }

# 4. Loads pre-trained weights into this structure
#    Each matrix/vector is filled with learned values from training

# ============================================================================
# 2. USING THE MODEL
# ============================================================================

# Input text
text = "Transformers are amazing!"
inputs = tokenizer(text, return_tensors="pt")

# What tokenizer does:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 1. Tokenization
tokens = ["[CLS]", "transform", "##ers", "are", "amazing", "!", "[SEP]"]

# 2. Convert to IDs using vocabulary
input_ids = [101, 10938, 2869, 2024, 6429, 999, 102]  # Vocab lookup

# 3. Add attention mask (1 for real tokens, 0 for padding)
attention_mask = [1, 1, 1, 1, 1, 1, 1]

# 4. Add token type IDs (0 for first sentence, 1 for second in pairs)
token_type_ids = [0, 0, 0, 0, 0, 0, 0]

# Result:
inputs = {
    'input_ids': tensor([[101, 10938, 2869, 2024, 6429, 999, 102]]),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]]),
    'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0]]),
}

# Forward pass
outputs = model(**inputs)

# What happens in forward pass:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def bert_forward_detailed(input_ids, attention_mask, token_type_ids):
    batch_size = input_ids.size(0)
    seq_length = input_ids.size(1)
    
    # 1. EMBEDDING LAYER
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    # Token embeddings: lookup from word_embeddings table
    token_embeds = word_embeddings[input_ids]  # (batch, seq_len, 768)
    
    # Position embeddings: lookup from position_embeddings table
    position_ids = torch.arange(seq_length).unsqueeze(0)  # (1, seq_len)
    position_embeds = position_embeddings[position_ids]  # (1, seq_len, 768)
    
    # Token type embeddings: lookup from token_type_embeddings table
    token_type_embeds = token_type_embeddings[token_type_ids]  # (batch, seq_len, 768)
    
    # Combine all embeddings
    embeddings = token_embeds + position_embeds + token_type_embeds
    
    # Apply layer normalization and dropout
    embeddings = layer_norm(embeddings)
    embeddings = dropout(embeddings)
    
    # 2. ENCODER LAYERS (12 layers)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    hidden_states = embeddings
    
    for layer_module in encoder_layers:
        # Self-Attention
        Q = hidden_states @ query_weight  # (batch, seq_len, 768)
        K = hidden_states @ key_weight
        V = hidden_states @ value_weight
        
        # Reshape for 12 heads (768 / 12 = 64 per head)
        Q = Q.view(batch_size, seq_length, 12, 64).transpose(1, 2)
        K = K.view(batch_size, seq_length, 12, 64).transpose(1, 2)
        V = V.view(batch_size, seq_length, 12, 64).transpose(1, 2)
        
        # Attention computation
        scores = (Q @ K.transpose(-2, -1)) / sqrt(64)
        
        # Apply attention mask (prevent attending to padding)
        # attention_mask is converted to large negative numbers for masked positions
        extended_mask = (1.0 - attention_mask) * -10000.0
        scores = scores + extended_mask
        
        attn_probs = softmax(scores, dim=-1)
        attn_probs = dropout(attn_probs)
        
        context = attn_probs @ V  # (batch, 12, seq_len, 64)
        
        # Concatenate heads
        context = context.transpose(1, 2).reshape(batch_size, seq_length, 768)
        
        # Output projection
        attn_output = context @ output_dense_weight
        attn_output = dropout(attn_output)
        
        # Residual connection + Layer Norm
        hidden_states = layer_norm(hidden_states + attn_output)
        
        # Feed-Forward Network
        intermediate = hidden_states @ intermediate_dense_weight
        intermediate = gelu(intermediate)  # BERT uses GELU, not ReLU
        
        ff_output = intermediate @ output_dense_weight
        ff_output = dropout(ff_output)
        
        # Residual connection + Layer Norm
        hidden_states = layer_norm(hidden_states + ff_output)
    
    # 3. POOLER (for classification tasks)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    # Take [CLS] token representation (first token)
    cls_token = hidden_states[:, 0, :]  # (batch, 768)
    
    # Apply pooler dense layer + tanh
    pooled_output = tanh(cls_token @ pooler_dense_weight)
    
    return {
        'last_hidden_state': hidden_states,  # All token representations
        'pooler_output': pooled_output,      # [CLS] token for classification
    }

# ============================================================================
# 3. FINE-TUNING FOR SPECIFIC TASKS
# ============================================================================

from transformers import BertForSequenceClassification

# When you write:
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# What happens:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BertForSequenceClassificationStructure:
    def __init__(self):
        # 1. Load pre-trained BERT (110M parameters)
        self.bert = BERTModelStructure()
        
        # 2. Add classification head (NEW, randomly initialized)
        self.classifier = torch.randn(768, 2) * 0.02  # Hidden → num_labels
        
        # Total: 110M + 1,536 parameters
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        # 1. Get BERT outputs
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs['pooler_output']  # (batch, 768)
        
        # 2. Classification
        logits = pooled_output @ self.classifier  # (batch, 2)
        
        # 3. Compute loss if labels provided
        if labels is not None:
            loss = cross_entropy(logits, labels)
            return loss, logits
        
        return logits

# During fine-tuning:
# - All 110M BERT parameters are updated (unless frozen)
# - Classification head learns from scratch
# - Typically use lower learning rate for BERT, higher for head
"""

print(huggingface_example)

# ============================================================================
# PART 3: TensorFlow/Keras Implementation
# ============================================================================

print("\n" + "=" * 80)
print("TensorFlow/Keras Transformers - Under the Hood")
print("=" * 80)

tensorflow_example = """
# ============================================================================
# 1. BUILDING A TRANSFORMER IN TENSORFLOW
# ============================================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# When you write:
attention_layer = layers.MultiHeadAttention(
    num_heads=8,
    key_dim=64,
    dropout=0.1
)

# Under the hood:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MultiHeadAttentionTF:
    def __init__(self, num_heads=8, key_dim=64):
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.d_model = num_heads * key_dim  # 512
        
        # Initialize weight matrices as Dense layers
        self.query_dense = layers.Dense(self.d_model)  # Stores W_q
        self.key_dense = layers.Dense(self.d_model)    # Stores W_k
        self.value_dense = layers.Dense(self.d_model)  # Stores W_v
        self.output_dense = layers.Dense(self.d_model) # Stores W_o
        
        # When you call layer.build(), TensorFlow creates:
        # - W_q: shape (input_dim, d_model)
        # - W_k: shape (input_dim, d_model)
        # - W_v: shape (input_dim, d_model)
        # - W_o: shape (d_model, d_model)
        # - Biases for each
        # All initialized with Glorot uniform initialization
    
    def call(self, query, value, key=None):
        if key is None:
            key = value
        
        batch_size = tf.shape(query)[0]
        
        # 1. Linear projections
        # Dense layer call does: input @ kernel + bias
        Q = self.query_dense(query)  # (batch, seq_len, d_model)
        K = self.key_dense(key)
        V = self.value_dense(value)
        
        # 2. Split into multiple heads
        def split_heads(x, batch_size):
            # Reshape: (batch, seq_len, d_model) → (batch, seq_len, num_heads, key_dim)
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.key_dim))
            # Transpose: → (batch, num_heads, seq_len, key_dim)
            return tf.transpose(x, perm=[0, 2, 1, 3])
        
        Q = split_heads(Q, batch_size)
        K = split_heads(K, batch_size)
        V = split_heads(V, batch_size)
        
        # 3. Scaled dot-product attention
        # Shape: (batch, num_heads, seq_len_q, seq_len_k)
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
        
        # 4. Apply softmax
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # 5. Apply dropout
        attention_weights = layers.Dropout(0.1)(attention_weights, training=True)
        
        # 6. Apply attention to values
        # Shape: (batch, num_heads, seq_len_q, key_dim)
        attention_output = tf.matmul(attention_weights, V)
        
        # 7. Concatenate heads
        # Transpose: (batch, num_heads, seq_len, key_dim) → (batch, seq_len, num_heads, key_dim)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        # Reshape: → (batch, seq_len, d_model)
        attention_output = tf.reshape(attention_output, (batch_size, -1, self.d_model))
        
        # 8. Final linear projection
        output = self.output_dense(attention_output)
        
        return output

# ============================================================================
# 2. COMPLETE TRANSFORMER ENCODER BLOCK
# ============================================================================

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),  # Expansion
            layers.Dense(d_model)                   # Compression
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x, training, mask=None):
        # Under the hood, this executes:
        
        # 1. Multi-head attention
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            attention_mask=mask,
            training=training
        )
        # TensorFlow internally:
        # - Calls query_dense, key_dense, value_dense (matrix multiplications)
        # - Splits into heads (reshape + transpose)
        # - Computes attention scores (batched matrix multiply)
        # - Applies softmax and dropout
        # - Multiplies by values
        # - Concatenates heads
        # - Applies output projection
        
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # Residual connection
        
        # 2. Feed-forward network
        ffn_output = self.ffn(out1)
        # TensorFlow internally:
        # - First Dense: out1 @ W1 + b1
        # - ReLU: max(0, x)
        # - Second Dense: x @ W2 + b2
        
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection
        
        return out2

# ============================================================================
# 3. MODEL TRAINING - WHAT HAPPENS UNDER THE HOOD
# ============================================================================

# When you compile and train:
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=10)

# What happens in each training step:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def training_step_detailed(x_batch, y_batch):
    # 1. Forward pass with gradient tape
    with tf.GradientTape() as tape:
        # Record all operations for automatic differentiation
        
        # Forward through the model
        predictions = model(x_batch, training=True)
        # This calls all the layers in sequence:
        # - Embedding lookup
        # - Positional encoding addition
        # - Each transformer block (attention + FFN)
        # - Output projection
        
        # Compute loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, predictions)
        loss = tf.reduce_mean(loss)
    
    # 2. Backward pass
    # TensorFlow automatically computes gradients using chain rule
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # For each parameter W:
    # ∂L/∂W = ∂L/∂output × ∂output/∂W
    # TensorFlow traces back through all operations!
    
    # Example gradient computation for attention:
    # ∂L/∂W_q = ∂L/∂attention_output × ∂attention_output/∂Q × ∂Q/∂W_q
    #         = ∂L/∂attention_output × attention_weights × query_input^T
    
    # 3. Optimizer step (Adam in this case)
    # Adam maintains moving averages of gradients and squared gradients
    for param, gradient in zip(model.trainable_variables, gradients):
        # Update first moment estimate
        m = beta1 * m + (1 - beta1) * gradient
        
        # Update second moment estimate
        v = beta2 * v + (1 - beta2) * gradient^2
        
        # Bias correction
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        
        # Update parameters
        param = param - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
    
    return loss

# ============================================================================
# 4. SAVING AND LOADING MODELS
# ============================================================================

# When you save:
model.save('my_transformer_model')

# TensorFlow saves:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 1. saved_model.pb
#    - Model architecture (computation graph)
#    - All layer configurations
#    - Input/output specifications

# 2. variables/
#    - variables.data-00000-of-00001: All parameter values
#    - variables.index: Parameter index
#    
#    For each layer, saves:
#    - Weight matrices (W_q, W_k, W_v, W_o, FFN weights)
#    - Biases
#    - LayerNorm parameters (gamma, beta)
#    - Optimizer states (Adam m and v)

# When you load:
loaded_model = keras.models.load_model('my_transformer_model')

# TensorFlow:
# 1. Reconstructs architecture from saved_model.pb
# 2. Creates all layers and allocates memory
# 3. Loads all parameter values from variables/
# 4. Restores optimizer state (if training)
"""

print(tensorflow_example)

# ============================================================================
# PART 4: Summary of Key Differences
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY: Library Abstractions vs Reality")
print("=" * 80)

summary = """
# ============================================================================
# WHAT LIBRARIES HIDE FROM YOU
# ============================================================================

1. MATRIX OPERATIONS
   Library call:        output = layer(input)
   Reality:             output = input @ weights + bias
   
   The @ operator performs batched matrix multiplication:
   - For (batch, seq, d_in) @ (d_in, d_out)
   - Result: (batch, seq, d_out)
   - This is O(batch × seq × d_in × d_out) floating point operations!

2. MEMORY MANAGEMENT
   Library call:        model.to('cuda')
   Reality:             - Allocates GPU memory for all parameters
                        - Copies 110M+ parameters (440MB+) to GPU
                        - Sets up CUDA kernels for operations
                        - Manages memory pools for intermediate activations

3. AUTOMATIC DIFFERENTIATION
   Library call:        loss.backward()
   Reality:             - Traverses computational graph in reverse
                        - Applies chain rule at each operation
                        - Accumulates gradients for all parameters
                        - Example: For W in attention, computes:
                          ∂L/∂W = (∂L/∂output) @ (input^T)

4. BATCHING AND PADDING
   Library call:        DataLoader(dataset, batch_size=32)
   Reality:             - Groups sequences into batches
                        - Pads shorter sequences to match longest
                        - Creates attention masks (1 for real, 0 for pad)
                        - Prevents model from attending to padding

5. OPTIMIZATION
   Library call:        optimizer.step()
   Reality:             For each parameter W:
                        - Computes update based on gradient
                        - Updates momentum terms
                        - Applies weight decay
                        - Clips gradients if needed
                        - Updates parameter: W = W - lr * gradient

# ============================================================================
# COMPUTATIONAL COSTS (for BERT-base: 110M parameters)
# ============================================================================

Forward Pass (single example):
- Embedding lookup: O(1) - just indexing
- 12 Attention layers: 12 × (seq_len² × 768) multiplications
- 12 FFN layers: 12 × (seq_len × 768 × 3072 × 2) multiplications
- Total: ~billions of FLOPs for seq_len=512

Backward Pass:
- Approximately 2× the cost of forward pass
- Requires storing all intermediate activations
- Memory: ~GB-scale for large batches

Training One Epoch (1M examples):
- 1M forward passes
- 1M backward passes
- 1M optimizer updates
- Time: Hours to days on modern GPUs
- Cost: $10-$1000s depending on hardware

# ============================================================================
# WHAT "PRE-TRAINED" REALLY MEANS
# ============================================================================

When you load 'bert-base-uncased':
1. Model was trained on:
   - 3.3B words (Wikipedia + BookCorpus)
   - 1M training steps
   - 4-16 TPUs/GPUs
   - Total cost: ~$10,000-$50,000
   - Training time: Days to weeks

2. You download:
   - pytorch_model.bin: 440MB
   - Contains 110,023,680 float32 numbers
   - Each is the result of billions of gradient updates

3. Fine-tuning costs:
   - Your task dataset: 1K-1M examples
   - Training time: Minutes to hours
   - Cost: $1-$100
   - Updates all 110M parameters slightly

# ============================================================================
# EFFICIENCY TRICKS LIBRARIES USE
# ============================================================================

1. KERNEL FUSION
   Instead of:  x = layer_norm(x + attention(x))
   Library:     Fused kernel that does both in one GPU pass
   Speedup:     2-3× faster

2. MIXED PRECISION
   Instead of:  float32 (4 bytes per number)
   Library:     float16 (2 bytes) for forward/backward
                float32 for parameter updates
   Speedup:     2× faster, 2× less memory

3. GRADIENT CHECKPOINTING
   Instead of:  Store all activations (OOM for long sequences)
   Library:     Store only some, recompute others during backward
   Trade-off:   20% slower but 10× less memory

4. FLASH ATTENTION
   Instead of:  Materialize full attention matrix (seq_len²)
   Library:     Compute attention in blocks, never store full matrix
   Speedup:     3-5× faster for long sequences

5. MODEL PARALLELISM
   Instead of:  Run entire model on one GPU
   Library:     Split layers across multiple GPUs
   Enables:     Training models larger than single GPU memory
"""

print(summary)

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
Libraries like PyTorch, TensorFlow, and HuggingFace provide enormous value:
- Handle complex low-level details
- Optimize performance with CUDA kernels
- Automatic differentiation
- Distributed training
- Pre-trained models

BUT, under the hood, it's all just:
- Matrix multiplications
- Non-linear functions (softmax, ReLU, etc.)
- Gradient computations
- Parameter updates

Understanding the fundamentals helps you:
- Debug issues
- Design better architectures
- Optimize performance
- Make informed decisions about model design

The "magic" is just mathematics + engineering!
""")
