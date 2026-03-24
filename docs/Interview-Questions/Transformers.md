---
title: Transformer Interview Questions
description: 55+ Transformer architecture interview questions - self-attention, multi-head attention, positional encoding, RoPE, Flash Attention, MoE, KV cache, GQA, MLA, and modern LLM innovations for ML and AI interviews.
---

# Transformer Interview Questions

This comprehensive guide covers **55+ Transformer interview questions** commonly asked at top AI labs and tech companies like Google, OpenAI, Meta, DeepMind, Anthropic, and Amazon. From foundational self-attention to cutting-edge innovations like Multi-head Latent Attention and Flash Attention, each question includes detailed explanations, mathematical formulations, and code examples.

---

## Premium Interview Questions

---

### What is the Transformer architecture and why was it introduced? - Google, OpenAI Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Transformer`, `Architecture`, `Fundamentals` | **Asked by:** Google, OpenAI, Meta, Amazon

??? success "View Answer"

    The **Transformer** is a neural network architecture introduced in the 2017 paper *"Attention Is All You Need"* (Vaswani et al.) that relies entirely on **attention mechanisms** instead of recurrence or convolution to process sequential data.

    **Why it was introduced:**

    | Problem with RNNs/LSTMs | How Transformers Solve It |
    |------------------------|--------------------------|
    | Sequential processing (can't parallelize) | Processes all positions simultaneously |
    | Vanishing/exploding gradients over long sequences | Direct attention to any position regardless of distance |
    | O(n) path length for long-range dependencies | O(1) path length via self-attention |
    | Slow training due to sequential nature | Highly parallelizable on GPUs/TPUs |

    **Core idea:** Instead of processing tokens one-by-one (RNN) or with fixed-size windows (CNN), the Transformer lets every token directly attend to every other token in a single step.

    ```
    ┌─────────────────────────────────────────────┐
    │           TRANSFORMER ARCHITECTURE           │
    ├─────────────────────────────────────────────┤
    │                                             │
    │   Input ──► [Token Embedding]               │
    │             + [Positional Encoding]          │
    │                    │                        │
    │          ┌─────────▼──────────┐             │
    │          │     ENCODER (Nx)    │             │
    │          │  ┌───────────────┐  │             │
    │          │  │ Multi-Head    │  │             │
    │          │  │ Self-Attention│  │             │
    │          │  └──────┬────────┘  │             │
    │          │  ┌──────▼────────┐  │             │
    │          │  │ Feed-Forward  │  │             │
    │          │  │ Network       │  │             │
    │          │  └───────────────┘  │             │
    │          └─────────┬──────────┘             │
    │                    │                        │
    │          ┌─────────▼──────────┐             │
    │          │     DECODER (Nx)    │             │
    │          │  ┌───────────────┐  │             │
    │          │  │ Masked Self-  │  │             │
    │          │  │ Attention     │  │             │
    │          │  ├───────────────┤  │             │
    │          │  │ Cross-        │  │             │
    │          │  │ Attention     │  │             │
    │          │  ├───────────────┤  │             │
    │          │  │ Feed-Forward  │  │             │
    │          │  │ Network       │  │             │
    │          │  └───────────────┘  │             │
    │          └─────────┬──────────┘             │
    │                    │                        │
    │             [Linear + Softmax]              │
    │                    │                        │
    │                 Output                      │
    └─────────────────────────────────────────────┘
    ```

    The original Transformer used **6 encoder layers** and **6 decoder layers**, with model dimension $d_{model} = 512$ and 8 attention heads.

---

### What is the self-attention mechanism? Explain the complete computation step by step - Google, DeepMind Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Self-Attention`, `Scaled Dot-Product`, `QKV` | **Asked by:** Google, DeepMind, OpenAI, Anthropic

??? success "View Answer"

    **Self-attention** (also called intra-attention) allows each position in a sequence to attend to all other positions, computing a weighted sum of values where weights are determined by the compatibility between queries and keys.

    **Step-by-step computation:**

    **Step 1: Create Q, K, V matrices**

    For input matrix $X \in \mathbb{R}^{n \times d_{model}}$:

    $$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

    Where $W^Q, W^K \in \mathbb{R}^{d_{model} \times d_k}$ and $W^V \in \mathbb{R}^{d_{model} \times d_v}$

    **Step 2: Compute attention scores**

    $$\text{scores} = QK^T \in \mathbb{R}^{n \times n}$$

    Each element $(i, j)$ represents how much token $i$ should attend to token $j$.

    **Step 3: Scale**

    $$\text{scaled\_scores} = \frac{QK^T}{\sqrt{d_k}}$$

    **Step 4: Apply softmax**

    $$\text{attention\_weights} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

    **Step 5: Weighted sum of values**

    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

    ```python
    import torch
    import torch.nn.functional as F
    import math

    def scaled_dot_product_attention(Q, K, V, mask=None):
        """
        Q: (batch, seq_len, d_k)
        K: (batch, seq_len, d_k)
        V: (batch, seq_len, d_v)
        """
        d_k = Q.size(-1)

        # Step 1: Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, seq_len, seq_len)

        # Step 2: Scale
        scores = scores / math.sqrt(d_k)

        # Step 3: Apply mask (optional, for decoder)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Step 4: Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Step 5: Weighted sum of values
        output = torch.matmul(attention_weights, V)  # (batch, seq_len, d_v)

        return output, attention_weights
    ```

    **Intuition with analogy:** Think of a library search:

    - **Query (Q):** Your search question
    - **Key (K):** Index card labels for each book
    - **Value (V):** The actual book content
    - **Attention score:** How relevant each book is to your question
    - **Output:** A weighted blend of all relevant book contents

---

### Why do we scale the dot product by $\sqrt{d_k}$? What happens if we don't? - Google, Anthropic Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Scaled Dot-Product`, `Numerical Stability`, `Softmax` | **Asked by:** Google, Anthropic, DeepMind

??? success "View Answer"

    We scale by $\sqrt{d_k}$ to **prevent the dot products from growing too large**, which would push the softmax into regions with **extremely small gradients**.

    **Mathematical justification:**

    If $q$ and $k$ are vectors with components drawn independently from a distribution with mean 0 and variance 1, then their dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has:

    - **Mean:** $E[q \cdot k] = 0$
    - **Variance:** $\text{Var}(q \cdot k) = d_k$

    As $d_k$ grows, the dot products grow in magnitude proportional to $\sqrt{d_k}$, pushing softmax outputs toward one-hot vectors (saturation).

    **What happens without scaling:**

    | $d_k$ | Typical dot product magnitude | Softmax behavior |
    |-------|------------------------------|-----------------|
    | 16 | ~4 | Reasonable distribution |
    | 64 | ~8 | Starting to peak |
    | 512 | ~22.6 | Nearly one-hot, vanishing gradients |
    | 2048 | ~45.3 | Completely saturated |

    ```python
    import torch
    import torch.nn.functional as F

    d_k = 512
    q = torch.randn(1, d_k)
    k = torch.randn(10, d_k)

    # Without scaling - softmax saturates
    scores_unscaled = q @ k.T
    print(f"Unscaled scores std: {scores_unscaled.std():.2f}")      # ~22.6
    print(f"Unscaled softmax: {F.softmax(scores_unscaled, dim=-1)}")  # Nearly one-hot

    # With scaling - healthy gradient flow
    scores_scaled = (q @ k.T) / (d_k ** 0.5)
    print(f"Scaled scores std: {scores_scaled.std():.2f}")            # ~1.0
    print(f"Scaled softmax: {F.softmax(scores_scaled, dim=-1)}")      # Smooth distribution
    ```

    After dividing by $\sqrt{d_k}$, the variance of the dot product becomes 1, keeping the softmax in a region with healthy gradients.

---

### What is multi-head attention and why use multiple heads instead of a single attention? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Multi-Head Attention`, `Representation`, `Subspaces` | **Asked by:** Google, Meta, OpenAI, Amazon

??? success "View Answer"

    **Multi-Head Attention (MHA)** runs $h$ parallel attention functions, each operating on a different learned linear projection of Q, K, and V into lower-dimensional subspaces.

    $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

    Where each head is:

    $$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

    With $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$, and $W^O \in \mathbb{R}^{hd_v \times d_{model}}$.

    Typically $d_k = d_v = d_{model} / h$.

    **Why multiple heads?**

    1. **Different representation subspaces:** Each head can learn to attend to different types of relationships:
        - Head 1: Syntactic dependencies (subject-verb agreement)
        - Head 2: Positional proximity (nearby tokens)
        - Head 3: Semantic similarity (synonyms, related concepts)
        - Head 4: Coreference resolution (pronoun-antecedent)

    2. **Richer representation:** A single head computes one set of attention weights — it can only focus on one type of relationship per position. Multiple heads allow attending to multiple positions for different reasons simultaneously.

    3. **Computational equivalence:** MHA with $h$ heads of dimension $d_k = d_{model}/h$ has roughly the same computational cost as single-head attention with dimension $d_{model}$.

    ```python
    import torch
    import torch.nn as nn

    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model=512, n_heads=8):
            super().__init__()
            self.n_heads = n_heads
            self.d_k = d_model // n_heads

            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)

        def forward(self, Q, K, V, mask=None):
            batch_size = Q.size(0)

            # Linear projections and reshape to (batch, n_heads, seq_len, d_k)
            Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

            # Scaled dot-product attention per head
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            context = torch.matmul(attn, V)

            # Concatenate heads and project
            context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
            return self.W_o(context)
    ```

    In the original Transformer: $d_{model} = 512$, $h = 8$ heads, $d_k = d_v = 64$.

---

### What is positional encoding and why is it needed? Explain the sinusoidal formulation - OpenAI, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Positional Encoding`, `Sinusoidal`, `Sequence Order` | **Asked by:** OpenAI, Google, DeepMind, Anthropic

??? success "View Answer"

    **Positional encoding** injects information about the position of each token in the sequence because self-attention is **permutation-equivariant** — without it, the model cannot distinguish "The cat sat on the mat" from "mat the on sat cat The".

    **Sinusoidal positional encoding** (original Transformer):

    $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

    $$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

    Where $pos$ is the position and $i$ is the dimension index.

    **Why sinusoidal functions?**

    1. **Bounded values:** Outputs are always in $[-1, 1]$ regardless of sequence length
    2. **Unique encoding:** Each position gets a unique pattern across dimensions
    3. **Relative position via linear transformation:** For any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$, enabling the model to learn relative positioning:

    $$\begin{bmatrix} \sin(pos + k) \\ \cos(pos + k) \end{bmatrix} = \begin{bmatrix} \cos(k) & \sin(k) \\ -\sin(k) & \cos(k) \end{bmatrix} \begin{bmatrix} \sin(pos) \\ \cos(pos) \end{bmatrix}$$

    4. **Generalization to unseen lengths:** Can extrapolate to longer sequences than seen during training (in theory)

    ```python
    import torch
    import math

    def sinusoidal_positional_encoding(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        return pe

    # Each row is a unique positional "fingerprint"
    pe = sinusoidal_positional_encoding(100, 512)
    print(pe.shape)  # (100, 512)
    ```

    **Alternative approaches:** Learned positional embeddings (BERT, GPT-2), Rotary Position Embeddings (RoPE in LLaMA), ALiBi (BLOOM).

---

### What is the role of the feed-forward network (FFN) in each Transformer layer? - Meta, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `FFN`, `MLP`, `Transformer Sublayer` | **Asked by:** Meta, Amazon, Google

??? success "View Answer"

    The **position-wise feed-forward network** is applied independently and identically to each position. It is a two-layer MLP with a non-linear activation:

    $$\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$$

    Where $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$, and $\sigma$ is the activation function (ReLU in the original paper).

    In the original Transformer: $d_{model} = 512$, $d_{ff} = 2048$ (4x expansion).

    **Why is it needed?**

    | Component | What it does | Analogy |
    |-----------|-------------|---------|
    | Self-Attention | Mixes information **across positions** (inter-token) | "Who should I talk to?" |
    | FFN | Transforms information **within each position** (intra-token) | "What should I think about what I heard?" |

    Without FFN, the Transformer would only be able to compute weighted averages of value vectors — a **linear** operation. The FFN adds **non-linearity** and **per-position transformation capacity**.

    **Key properties:**

    1. **Position-wise:** The same FFN is applied to each position independently (like a 1x1 convolution)
    2. **Parameter heavy:** FFN accounts for roughly **2/3 of all parameters** in a Transformer layer
    3. **Knowledge storage:** Research (Geva et al., 2021) shows FFN layers act as **key-value memories**, storing factual knowledge learned during pre-training

    ```python
    import torch.nn as nn

    class PositionWiseFFN(nn.Module):
        def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
            super().__init__()
            self.w1 = nn.Linear(d_model, d_ff)
            self.w2 = nn.Linear(d_ff, d_model)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # x: (batch, seq_len, d_model)
            return self.w2(self.dropout(self.relu(self.w1(x))))
    ```

    **Modern variants** replace ReLU with SwiGLU (LLaMA, PaLM) which uses a gated linear unit:

    $$\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot xV) W_2$$

---

### Explain residual connections and layer normalization in the Transformer. What is pre-norm vs post-norm? - Google, DeepMind Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Residual Connection`, `Layer Norm`, `Pre-Norm`, `Post-Norm` | **Asked by:** Google, DeepMind, Anthropic

??? success "View Answer"

    **Residual (skip) connections** and **layer normalization** are critical for training deep Transformers.

    **Residual connections** add the input of a sublayer to its output:

    $$\text{output} = x + \text{Sublayer}(x)$$

    This creates a "gradient highway" that allows gradients to flow directly through the network, enabling training of very deep models (100+ layers).

    **Layer normalization** normalizes across the feature dimension:

    $$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

    Where $\mu$ and $\sigma^2$ are the mean and variance computed across the feature dimension, and $\gamma$, $\beta$ are learned scale and shift parameters.

    **Post-Norm (original Transformer):**

    $$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

    ```
    x ──┬──► [Sublayer] ──► (+) ──► [LayerNorm] ──► output
        └─────────────────────┘
    ```

    **Pre-Norm (GPT-2, LLaMA, most modern LLMs):**

    $$\text{output} = x + \text{Sublayer}(\text{LayerNorm}(x))$$

    ```
    x ──┬──► [LayerNorm] ──► [Sublayer] ──► (+) ──► output
        └──────────────────────────────────────┘
    ```

    | Aspect | Post-Norm | Pre-Norm |
    |--------|----------|---------|
    | Training stability | Requires careful warmup | More stable, easier to train |
    | Final performance | Slightly better (with proper training) | Slightly worse |
    | Gradient flow | Gradients can explode without warmup | Residual path is unobstructed |
    | Used in | Original Transformer, BERT | GPT-2/3, LLaMA, PaLM, most modern LLMs |

    Pre-Norm is preferred for large models because it is much **easier to train** — the residual stream is the "main trunk" and sublayers are additive updates to it.

---

### What is masked attention in the Transformer decoder and why is it necessary? - OpenAI, Anthropic Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Masked Attention`, `Causal`, `Autoregressive` | **Asked by:** OpenAI, Anthropic, Google, Meta

??? success "View Answer"

    **Masked (causal) attention** prevents the decoder from attending to future positions during training. This is essential for **autoregressive generation** — the model must predict token $t$ using only tokens $1, 2, \ldots, t-1$.

    The mask is applied before the softmax:

    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

    Where $M$ is an upper-triangular matrix of $-\infty$:

    $$M = \begin{bmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{bmatrix}$$

    After softmax, $-\infty$ values become 0, so future tokens receive zero attention weight.

    ```python
    import torch

    def create_causal_mask(seq_len):
        """Lower triangular mask for autoregressive decoding."""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask  # 1 = attend, 0 = block

    # For sequence length 5:
    # tensor([[1, 0, 0, 0, 0],
    #         [1, 1, 0, 0, 0],
    #         [1, 1, 1, 0, 0],
    #         [1, 1, 1, 1, 0],
    #         [1, 1, 1, 1, 1]])
    ```

    **Why is it necessary?**

    - During **training**, we feed the entire target sequence at once (teacher forcing) for efficiency. Without masking, the model could "cheat" by looking at the answer.
    - During **inference**, tokens are generated one at a time, so the mask is naturally satisfied.
    - The mask ensures training and inference behavior are **consistent**.

    **All decoder-only models** (GPT, LLaMA, Claude) use causal masking throughout — every layer is masked self-attention.

---

### What is cross-attention and how does it differ from self-attention? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Cross-Attention`, `Encoder-Decoder`, `Conditioning` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Cross-attention** (also called encoder-decoder attention) allows the decoder to attend to the encoder's output. Unlike self-attention where Q, K, V all come from the same sequence, in cross-attention:

    - **Queries (Q):** Come from the **decoder** (previous decoder layer's output)
    - **Keys (K) and Values (V):** Come from the **encoder** output

    $$\text{CrossAttention} = \text{softmax}\left(\frac{Q_{dec} K_{enc}^T}{\sqrt{d_k}}\right) V_{enc}$$

    ```
    Encoder Output ──────────────────┐
         │                           │
         ├──► K (Keys)               │
         └──► V (Values)             │
                                     │
    Decoder Hidden ──► Q (Queries)   │
                           │         │
                           ▼         ▼
                    [Attention Scores]
                           │
                           ▼
                    [Weighted Sum of Encoder Values]
    ```

    | Aspect | Self-Attention | Cross-Attention |
    |--------|---------------|-----------------|
    | Q, K, V source | All from same sequence | Q from decoder, K/V from encoder |
    | Purpose | Model internal relationships | Condition on input context |
    | Masking | Causal mask (decoder) or none (encoder) | No causal mask needed |
    | Used in | Every Transformer layer | Only encoder-decoder models (T5, BART, Whisper) |

    **Cross-attention is used in:**

    - **Machine translation:** Decoder attends to source language
    - **Summarization:** Decoder attends to input document
    - **Speech recognition (Whisper):** Text decoder attends to audio encoder
    - **Image captioning:** Text decoder attends to image encoder
    - **Diffusion models:** U-Net cross-attends to text embeddings (Stable Diffusion)

---

### What are the three types of Transformer architectures? Compare encoder-only, decoder-only, and encoder-decoder models - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `BERT`, `GPT`, `T5`, `Architecture Variants` | **Asked by:** Google, Meta, Amazon, OpenAI

??? success "View Answer"

    ```
    ┌──────────────────┬──────────────────┬──────────────────┐
    │  ENCODER-ONLY    │  DECODER-ONLY    │  ENCODER-DECODER │
    │  (Bidirectional) │  (Autoregressive)│  (Seq-to-Seq)    │
    ├──────────────────┼──────────────────┼──────────────────┤
    │                  │                  │                  │
    │  ┌──────────┐   │  ┌──────────┐   │  ┌────┐  ┌────┐ │
    │  │ Encoder  │   │  │ Decoder  │   │  │Enc │→│Dec │ │
    │  │ (bidir.) │   │  │ (causal) │   │  │    │  │    │ │
    │  └──────────┘   │  └──────────┘   │  └────┘  └────┘ │
    │                  │                  │                  │
    │  BERT           │  GPT-2/3/4      │  T5              │
    │  RoBERTa        │  LLaMA          │  BART            │
    │  DeBERTa        │  Claude         │  Whisper         │
    │  ELECTRA        │  Mistral        │  mBART           │
    │                  │  DeepSeek       │  FLAN-T5         │
    ├──────────────────┼──────────────────┼──────────────────┤
    │ Attention: Full  │ Attention:      │ Encoder: Full    │
    │ (all-to-all)     │ Causal mask     │ Decoder: Causal  │
    │                  │ (left-to-right) │ + Cross-attention│
    ├──────────────────┼──────────────────┼──────────────────┤
    │ Pre-training:    │ Pre-training:   │ Pre-training:    │
    │ MLM (mask &      │ CLM (next token │ Span corruption  │
    │ predict)         │ prediction)     │ (T5) or denoise  │
    ├──────────────────┼──────────────────┼──────────────────┤
    │ Best for:        │ Best for:       │ Best for:        │
    │ Classification   │ Text generation │ Translation      │
    │ NER, embeddings  │ Code, reasoning │ Summarization    │
    │ Retrieval        │ Chat, agents    │ ASR, multimodal  │
    └──────────────────┴──────────────────┴──────────────────┘
    ```

    **Why decoder-only dominates today:**

    - Simpler architecture (one stack of layers)
    - Scales better to very large models
    - In-context learning and instruction following emerge with scale
    - A single model can handle many tasks via prompting (no task-specific head needed)

---

### Explain the complete training objective for a Transformer language model. What is teacher forcing? - OpenAI, Anthropic Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Training`, `Cross-Entropy`, `Teacher Forcing`, `Autoregressive` | **Asked by:** OpenAI, Anthropic, Google

??? success "View Answer"

    **Autoregressive language modeling** trains the model to predict the next token given all previous tokens. The objective is to maximize:

    $$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_1, x_2, \ldots, x_{t-1}; \theta)$$

    This is equivalent to minimizing the **cross-entropy loss** between the predicted probability distribution and the one-hot true token at each position.

    **Teacher forcing** is the training strategy where the ground truth sequence is fed as input to the decoder at every timestep, rather than the model's own predictions:

    ```
    WITHOUT teacher forcing (autoregressive, slow, error propagation):
    Input:  <BOS>
    Pred:   "The"  →  Input: "The"
    Pred:   "cat"  →  Input: "The cat"
    Pred:   "sit"  →  Input: "The cat sit"  ← error propagates!

    WITH teacher forcing (parallel, fast, ground truth input):
    Input:  <BOS>   "The"    "cat"    "sat"    "on"
    Target: "The"   "cat"    "sat"    "on"     "the"
    Loss:   L1      L2       L3       L4       L5
    ```

    **Advantages of teacher forcing:**

    1. **Parallelization:** All positions computed simultaneously (the key speed advantage of Transformers)
    2. **Stable training:** No error accumulation from incorrect predictions
    3. **Efficient:** One forward pass gives loss for all positions

    **Disadvantage — exposure bias:** During training, the model always sees ground truth context, but during inference it sees its own (possibly incorrect) predictions. This mismatch is called exposure bias.

---

### What are the different tokenization methods used with Transformers? Compare BPE, WordPiece, and SentencePiece - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Tokenization`, `BPE`, `WordPiece`, `SentencePiece` | **Asked by:** Google, Meta, OpenAI, Amazon

??? success "View Answer"

    Transformers operate on **sub-word tokens**, not raw characters or whole words. This balances vocabulary size with representation efficiency.

    **Byte-Pair Encoding (BPE):**

    Used in: GPT-2/3/4, LLaMA, Claude, RoBERTa

    1. Start with character-level vocabulary
    2. Iteratively merge the most frequent adjacent pair
    3. Repeat for a fixed number of merges (determines vocab size)

    ```
    Corpus: "low lower lowest"
    Step 1: Characters: l, o, w, e, r, s, t, ...
    Step 2: Most frequent pair: (l, o) → merge to "lo"
    Step 3: Most frequent pair: (lo, w) → merge to "low"
    Step 4: Continue until vocab size reached...
    ```

    **WordPiece:**

    Used in: BERT, DistilBERT, ELECTRA

    Similar to BPE, but selects merges that **maximize the language model likelihood** of the training data, not just frequency. Uses `##` prefix for continuation tokens: `"playing" → ["play", "##ing"]`

    **SentencePiece:**

    Used in: T5, LLaMA, ALBERT, XLNet

    Treats the input as a raw byte stream (language-agnostic, no pre-tokenization needed). Supports both BPE and Unigram algorithms. Can handle any language without language-specific preprocessing.

    | Method | Merge criterion | Prefix | Language-agnostic | Used in |
    |--------|----------------|--------|-------------------|---------|
    | BPE | Frequency | None (Ġ for space) | No (needs pre-tokenization) | GPT, Claude |
    | WordPiece | LM likelihood | ## | No | BERT |
    | SentencePiece | BPE or Unigram | ▁ (for word start) | Yes | T5, LLaMA |
    | Byte-level BPE | Frequency on bytes | None | Yes | GPT-2+ |

---

### What is the KV Cache? Why is it critical for inference? - OpenAI, Anthropic Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `KV Cache`, `Inference`, `Autoregressive`, `Memory` | **Asked by:** OpenAI, Anthropic, Google, Meta

??? success "View Answer"

    The **KV cache** stores the Key and Value matrices from all previous tokens during autoregressive generation, avoiding redundant recomputation.

    **Without KV cache** — generating token $t$ requires computing attention over tokens $1$ through $t-1$, recomputing K and V for ALL previous tokens every step:

    ```
    Step 1: Compute K,V for [tok1]                    → predict tok2
    Step 2: Compute K,V for [tok1, tok2]              → predict tok3
    Step 3: Compute K,V for [tok1, tok2, tok3]        → predict tok4
    Total K,V computations: 1 + 2 + 3 + ... + n = O(n²)
    ```

    **With KV cache** — only compute K, V for the new token and append to cached values:

    ```
    Step 1: Compute K,V for [tok1], cache them        → predict tok2
    Step 2: Compute K,V for [tok2], append to cache   → predict tok3
    Step 3: Compute K,V for [tok3], append to cache   → predict tok4
    Total K,V computations: 1 + 1 + 1 + ... + 1 = O(n)
    ```

    **Memory cost of KV cache:**

    $$\text{KV cache size} = 2 \times n_{layers} \times n_{heads} \times d_{head} \times \text{seq\_len} \times \text{bytes\_per\_param}$$

    For LLaMA-2 70B with 4K context:

    - 80 layers × 64 heads × 128 $d_{head}$ × 4096 seq × 2 (K+V) × 2 bytes (FP16)
    - = **10.7 GB** just for KV cache of one sequence!

    This is why KV cache optimization (GQA, MQA, MLA, quantization, PagedAttention) is one of the most active areas in LLM serving.

---

### What is Rotary Position Embedding (RoPE) and why is it preferred over sinusoidal encoding? - Meta, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `RoPE`, `Positional Encoding`, `Rotation`, `Relative Position` | **Asked by:** Meta, Google, DeepMind, Anthropic

??? success "View Answer"

    **Rotary Position Embedding (RoPE)**, introduced by Su et al. (2021), encodes position by **rotating** the query and key vectors in 2D subspaces. It naturally encodes **relative position** in the attention dot product.

    **Core idea:** Instead of adding positional encoding, rotate Q and K vectors by an angle proportional to their position:

    For a 2D subspace at dimension pair $(2i, 2i+1)$:

    $$R_\theta(pos) = \begin{bmatrix} \cos(pos \cdot \theta_i) & -\sin(pos \cdot \theta_i) \\ \sin(pos \cdot \theta_i) & \cos(pos \cdot \theta_i) \end{bmatrix}$$

    Where $\theta_i = 10000^{-2i/d}$ (similar base frequency as sinusoidal).

    The rotated query and key:

    $$\tilde{q}_m = R_\theta(m) \cdot q_m, \quad \tilde{k}_n = R_\theta(n) \cdot k_n$$

    The attention score between positions $m$ and $n$:

    $$\tilde{q}_m^T \tilde{k}_n = q_m^T R_\theta(m)^T R_\theta(n) k_n = q_m^T R_\theta(n - m) k_n$$

    The dot product **only depends on the relative position** $(n - m)$, not the absolute positions!

    **Why RoPE is preferred:**

    | Property | Sinusoidal | Learned | RoPE |
    |----------|-----------|---------|------|
    | Relative position aware | Indirectly (model must learn) | No | Yes (by construction) |
    | Length extrapolation | Poor | Poor | Better (with NTK-aware scaling) |
    | Added to embeddings | Yes (additive) | Yes (additive) | No (multiplicative via rotation) |
    | Decays with distance | No | No | Yes (natural decay in dot product) |
    | Computational overhead | Minimal | Minimal | Minimal |

    **Used in:** LLaMA 1/2/3, PaLM, Mistral, DeepSeek, Qwen, Yi, and nearly all modern LLMs.

    ```python
    import torch

    def apply_rope(x, freqs_cos, freqs_sin):
        """Apply RoPE to input tensor x of shape (batch, seq_len, n_heads, d_head)."""
        # Split into pairs: (x0, x1), (x2, x3), ...
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Apply rotation
        x_rotated_even = x_even * freqs_cos - x_odd * freqs_sin
        x_rotated_odd = x_even * freqs_sin + x_odd * freqs_cos

        # Interleave back
        x_out = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        return x_out.flatten(-2)
    ```

---

### What is Grouped Query Attention (GQA) and how does it compare to MHA and MQA? - Meta, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `GQA`, `MQA`, `MHA`, `KV Cache`, `Efficiency` | **Asked by:** Meta, Google, Anthropic

??? success "View Answer"

    **Grouped Query Attention (GQA)** is a compromise between Multi-Head Attention (MHA) and Multi-Query Attention (MQA). It groups query heads to share a single key-value head, reducing KV cache size while retaining most of MHA's quality.

    ```
    Multi-Head Attention (MHA):          8 Q heads, 8 K heads, 8 V heads
    ┌──┬──┬──┬──┬──┬──┬──┬──┐  Q
    ├──┼──┼──┼──┼──┼──┼──┼──┤  K  (1:1 ratio)
    ├──┼──┼──┼──┼──┼──┼──┼──┤  V
    └──┴──┴──┴──┴──┴──┴──┴──┘

    Grouped Query Attention (GQA):       8 Q heads, 2 K heads, 2 V heads
    ┌──┬──┬──┬──┬──┬──┬──┬──┐  Q
    ├─────┼─────┼─────┼─────┤  K  (4:1 ratio, groups of 4)
    ├─────┼─────┼─────┼─────┤  V
    └─────┴─────┴─────┴─────┘

    Multi-Query Attention (MQA):         8 Q heads, 1 K head, 1 V head
    ┌──┬──┬──┬──┬──┬──┬──┬──┐  Q
    ├──────────────────────────┤  K  (all share 1)
    ├──────────────────────────┤  V
    └──────────────────────────┘
    ```

    | Method | KV Heads | KV Cache Size | Quality | Used in |
    |--------|----------|---------------|---------|---------|
    | MHA | $h$ | $2 \times L \times h \times d_k \times s$ | Best | BERT, GPT-2, original Transformer |
    | GQA | $g$ (e.g., $h/4$) | $2 \times L \times g \times d_k \times s$ | Near-MHA | LLaMA 2 70B, Mistral, Gemma |
    | MQA | 1 | $2 \times L \times 1 \times d_k \times s$ | Slightly worse | PaLM, Falcon, StarCoder |

    **Example — LLaMA 2 70B:**

    - 64 query heads, 8 KV heads (GQA with group size 8)
    - KV cache reduced by **8x** compared to MHA
    - Quality nearly matches the MHA baseline

    GQA achieves **MHA-level quality** with **MQA-level speed** — the best of both worlds. The key insight is that attention heads within a group often learn similar patterns, so sharing KV projections has minimal quality impact.

---

### What is Multi-head Latent Attention (MLA) from DeepSeek? How does it differ from GQA? - DeepSeek, Research Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `MLA`, `DeepSeek`, `KV Compression`, `Latent` | **Asked by:** DeepSeek, Google, Research Labs

??? success "View Answer"

    **Multi-head Latent Attention (MLA)**, introduced in DeepSeek-V2, compresses the KV cache into a low-rank **latent representation** instead of sharing KV heads (as in GQA/MQA).

    **Key idea:** Instead of caching full K and V tensors, compress them into a much smaller latent vector:

    $$c_t^{KV} = W^{DKV} h_t \quad \text{(compress to latent, } d_c \ll d_{model}\text{)}$$

    $$k_t^h = W_h^{UK} c_t^{KV}, \quad v_t^h = W_h^{UV} c_t^{KV} \quad \text{(up-project per head for attention)}$$

    Instead of caching $n_{heads} \times d_{head}$ for both K and V, MLA caches only the latent vector $c_t^{KV}$ of dimension $d_c$.

    ```
    Standard MHA:
    Hidden ──► W_K ──► K (cache: n_heads × d_head)
    Hidden ──► W_V ──► V (cache: n_heads × d_head)
    Total cached per token: 2 × n_heads × d_head

    GQA (g groups):
    Hidden ──► W_K ──► K (cache: g × d_head)
    Hidden ──► W_V ──► V (cache: g × d_head)
    Total cached per token: 2 × g × d_head

    MLA:
    Hidden ──► W_DKV ──► c_KV (cache: d_c, very small)
             ┌──► W_UK ──► K (computed on the fly from c_KV)
    c_KV ────┤
             └──► W_UV ──► V (computed on the fly from c_KV)
    Total cached per token: d_c (93.3% less than MHA in DeepSeek-V2)
    ```

    **Comparison:**

    | Method | What is cached | Cache size per token per layer | Quality |
    |--------|---------------|-------------------------------|---------|
    | MHA | Full K, V | $2 \times n_h \times d_h$ | Best |
    | GQA | Shared K, V | $2 \times g \times d_h$ | Near-MHA |
    | MQA | Single K, V | $2 \times d_h$ | Good |
    | MLA | Latent $c^{KV}$ | $d_c$ (very small) | Matches or beats MHA |

    **Why MLA is significant:**

    - DeepSeek-V2 (236B params, 21B active) achieved performance comparable to LLaMA 2 70B
    - KV cache compressed by **93.3%** compared to standard MHA
    - Unlike GQA which restricts which heads share KV, MLA allows **all heads** to reconstruct rich KV from a shared latent — more expressive
    - The up-projection matrices ($W_h^{UK}$, $W_h^{UV}$) are absorbed into the attention computation, adding minimal overhead

---

### What is Flash Attention and what problem does it solve? - Meta, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Flash Attention`, `IO-Aware`, `Memory`, `GPU` | **Asked by:** Meta, Google, Anthropic, OpenAI

??? success "View Answer"

    **Flash Attention** (Dao et al., 2022) is an **IO-aware** exact attention algorithm that is 2-4x faster and uses significantly less memory than standard attention, without any approximation.

    **The problem:** Standard attention materializes the full $N \times N$ attention matrix in GPU High Bandwidth Memory (HBM), which is:

    1. **Memory-bound:** $O(N^2)$ memory for the attention matrix
    2. **IO-bound:** Repeatedly reading/writing the large matrix between fast SRAM and slow HBM

    ```
    Standard Attention (memory bottleneck):
    ┌─────────┐      ┌─────────────────────────┐
    │ GPU SRAM│◄────►│   GPU HBM (slow)        │
    │ (fast,  │      │                         │
    │  small) │      │  Q: N×d    (read)       │
    │         │      │  K: N×d    (read)       │
    │         │      │  S: N×N    (write+read) │ ← bottleneck!
    │         │      │  P: N×N    (write+read) │ ← bottleneck!
    │         │      │  V: N×d    (read)       │
    │         │      │  O: N×d    (write)      │
    └─────────┘      └─────────────────────────┘

    Flash Attention (IO-aware tiling):
    ┌─────────┐      ┌─────────────────────────┐
    │ GPU SRAM│◄────►│   GPU HBM               │
    │ (fast)  │      │                         │
    │  Q_tile │      │  Q: N×d    (read once)  │
    │  K_tile │      │  K: N×d    (read once)  │
    │  V_tile │      │  O: N×d    (write once) │
    │  [S,P   │      │  (no N×N matrix!)       │
    │   tiles]│      │                         │
    └─────────┘      └─────────────────────────┘
    ```

    **How it works:**

    1. **Tiling:** Split Q, K, V into blocks that fit in SRAM
    2. **Kernel fusion:** Compute attention per block entirely in SRAM (no HBM writes for S, P)
    3. **Online softmax:** Use the online softmax trick (numerically stable running softmax) to compute exact attention without materializing the full matrix
    4. **Recomputation:** In the backward pass, recompute attention from Q, K, V blocks instead of storing the $N \times N$ matrix — trading compute for memory

    **Results:**

    | Metric | Standard Attention | Flash Attention |
    |--------|-------------------|-----------------|
    | Memory | $O(N^2)$ | $O(N)$ |
    | HBM reads/writes | $O(N^2 d + N^2)$ | $O(N^2 d^2 / M)$ where $M$ = SRAM size |
    | Wall-clock speed | Baseline | 2-4x faster |
    | Exactness | Exact | Exact (not an approximation!) |

    Flash Attention is now the default in PyTorch (`torch.nn.functional.scaled_dot_product_attention`), Hugging Face, and virtually all modern LLM training/inference frameworks.

---

### What is the Mixture of Experts (MoE) architecture? How does it work in Transformers? - Google, Mistral Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `MoE`, `Sparse`, `Router`, `Expert` | **Asked by:** Google, Mistral, Meta, DeepSeek

??? success "View Answer"

    **Mixture of Experts (MoE)** replaces the dense FFN in each Transformer layer with multiple "expert" FFNs, and a **router** selects a subset of experts per token. This dramatically increases model capacity (total parameters) while keeping compute (active parameters) constant.

    ```
    Standard Transformer Layer:
    Input ──► Self-Attention ──► [One FFN] ──► Output

    MoE Transformer Layer:
                                  ┌──► Expert 1 (FFN)
    Input ──► Self-Attention ──► Router ──► Expert 2 (FFN)  ──► Weighted Sum ──► Output
                                  ├──► Expert 3 (FFN)
                                  ├──► ...
                                  └──► Expert N (FFN)
                               (selects top-k, e.g. k=2)
    ```

    **Router mechanism:**

    $$g(x) = \text{TopK}(\text{softmax}(W_r x))$$

    The router produces a probability over all experts, and only the top-$k$ experts (typically $k=1$ or $k=2$) are activated for each token.

    **Key components:**

    1. **Expert networks:** Identical FFN architectures with independent parameters
    2. **Router (gating network):** Small linear layer that decides which experts to use
    3. **Load balancing loss:** Auxiliary loss to prevent all tokens from routing to the same few experts

    $$\mathcal{L}_{balance} = N \sum_{i=1}^{N} f_i \cdot p_i$$

    Where $f_i$ is the fraction of tokens routed to expert $i$ and $p_i$ is the average routing probability for expert $i$.

    **Notable MoE models:**

    | Model | Total Params | Active Params | Experts | Top-k |
    |-------|-------------|---------------|---------|-------|
    | Mixtral 8x7B | 47B | 13B | 8 | 2 |
    | DeepSeek-V2 | 236B | 21B | 160 | 6 |
    | DeepSeek-V3 | 671B | 37B | 256 | 8 |
    | Switch Transformer | 1.6T | ~equivalent to T5-Base | 2048 | 1 |

    **Trade-offs:**

    - **Advantage:** 4-8x more parameters at same compute cost
    - **Disadvantage:** Higher memory for storing all expert weights, load balancing challenges, communication overhead in distributed training

---

### Explain RMSNorm and why modern Transformers prefer it over LayerNorm - Meta, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `RMSNorm`, `LayerNorm`, `Normalization` | **Asked by:** Meta, Google, DeepMind

??? success "View Answer"

    **RMSNorm** (Root Mean Square Normalization) is a simplification of LayerNorm that removes the mean-centering step:

    **LayerNorm:**

    $$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

    **RMSNorm:**

    $$\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}}$$

    or equivalently:

    $$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}$$

    **Key differences:**

    | Aspect | LayerNorm | RMSNorm |
    |--------|----------|---------|
    | Mean subtraction | Yes ($x - \mu$) | No |
    | Learnable shift ($\beta$) | Yes | No |
    | Computation | Mean + Variance | Only RMS |
    | Speed | Baseline | ~7-10% faster |
    | Quality | Baseline | Equivalent |

    **Why RMSNorm is preferred:**

    1. The re-centering ($x - \mu$) and shift ($\beta$) in LayerNorm are often unnecessary — the re-scaling is what provides the regularization effect
    2. Removing mean computation reduces one reduction operation per normalization — significant at scale
    3. Empirically matches or exceeds LayerNorm quality

    ```python
    import torch
    import torch.nn as nn

    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            return (x / rms) * self.weight
    ```

    **Used in:** LLaMA 1/2/3, PaLM, Mistral, DeepSeek, Gemma — nearly all modern LLMs.

---

### What is the SwiGLU activation function? Why do modern LLMs use it instead of ReLU? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `SwiGLU`, `Activation`, `GLU`, `FFN` | **Asked by:** Google, Meta, DeepMind

??? success "View Answer"

    **SwiGLU** combines the **Swish** activation with a **Gated Linear Unit (GLU)** and is used in the FFN of most modern LLMs.

    **Standard FFN (ReLU):**

    $$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x)$$

    **SwiGLU FFN:**

    $$\text{FFN}_{SwiGLU}(x) = W_2 \cdot (\text{Swish}(W_1 x) \odot W_3 x)$$

    Where:

    - $\text{Swish}(x) = x \cdot \sigma(x)$ (also called SiLU)
    - $\sigma(x)$ is the sigmoid function
    - $\odot$ is element-wise multiplication
    - $W_3$ is an additional "gate" projection

    **Breakdown:**

    ```
    Standard FFN:
    x ──► W1 ──► ReLU ──► W2 ──► output
    Parameters: d × 4d + 4d × d = 8d²

    SwiGLU FFN:
         ┌──► W1 ──► Swish ──┐
    x ───┤                    ├──► ⊙ ──► W2 ──► output
         └──► W3 ─────────────┘
    Parameters: d × (8/3)d × 3 ≈ 8d²  (d_ff reduced to 8/3 × d to match param count)
    ```

    **Why SwiGLU is better:**

    | Property | ReLU | GELU | SwiGLU |
    |----------|------|------|--------|
    | Dead neurons | Yes (negative inputs → 0 forever) | Rare | No (gating is smooth) |
    | Smoothness | Not smooth at 0 | Smooth | Smooth |
    | Gating | No | No | Yes (multiplicative) |
    | Perplexity | Baseline | Better | Best |

    The gating mechanism allows the network to learn **which features to pass through**, making it more expressive than a simple non-linearity.

    **Used in:** LLaMA 1/2/3, PaLM, PaLM 2, Mistral, DeepSeek, Gemma.

---

### What is ALiBi (Attention with Linear Biases)? How does it encode position? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `ALiBi`, `Positional Encoding`, `Length Extrapolation` | **Asked by:** Google, Meta, Hugging Face

??? success "View Answer"

    **ALiBi** (Press et al., 2022) is a positional encoding method that adds a **linear bias** to attention scores proportional to the distance between tokens. It uses **no positional embeddings** at all.

    $$\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + m \cdot \text{distance\_matrix}\right)$$

    Where the distance matrix is:

    $$\text{distance}(i, j) = \begin{cases} 0 & \text{if } j \leq i \\ -(i - j) & \text{if } j > i \end{cases}$$

    And $m$ is a head-specific slope. For 8 heads: $m \in \{2^{-1}, 2^{-2}, 2^{-3}, \ldots, 2^{-8}\}$

    ```
    For a causal model with 4 positions, the bias matrix looks like:

    Head with m=0.5:       Head with m=0.125:
    [ 0.0  -∞   -∞   -∞ ]  [ 0.0    -∞     -∞     -∞   ]
    [-0.5  0.0  -∞   -∞ ]  [-0.125  0.0    -∞     -∞   ]
    [-1.0 -0.5  0.0  -∞ ]  [-0.250 -0.125  0.0    -∞   ]
    [-1.5 -1.0 -0.5  0.0]  [-0.375 -0.250 -0.125  0.0  ]
    ```

    **Key insight:** Different heads have different slopes — steep slopes create very local attention, gentle slopes allow attending far back. The model gets a "spectrum" of locality.

    **Advantages over sinusoidal and RoPE:**

    | Property | Sinusoidal | RoPE | ALiBi |
    |----------|-----------|------|-------|
    | Learned parameters | None | None | None |
    | Length extrapolation | Poor | Moderate (with scaling) | Strong |
    | Inductive bias | None | Relative distance | Recency bias (nearer = stronger) |
    | Implementation | Add to embeddings | Rotate Q, K | Add to attention scores |
    | Where applied | Input layer only | Every attention layer | Every attention layer |

    **Used in:** BLOOM, MPT.

---

### Explain the concept of attention heads learning different patterns. What do individual heads specialize in? - DeepMind, Anthropic Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Attention Patterns`, `Interpretability`, `Head Specialization` | **Asked by:** DeepMind, Anthropic, Google

??? success "View Answer"

    Research on Transformer interpretability (Clark et al., 2019; Voita et al., 2019) has shown that individual attention heads learn specialized roles:

    **Common head patterns:**

    | Pattern | Description | Example |
    |---------|------------|---------|
    | **Positional** | Attends to fixed relative positions (previous token, next token) | Head always looks at position $i-1$ |
    | **Syntactic** | Tracks grammatical dependencies | Subject attends to its verb |
    | **Rare token** | Attends to low-frequency or delimiter tokens | Focuses on punctuation marks |
    | **Induction** | Copies patterns seen earlier in context ($[A][B]...[A] \rightarrow [B]$) | Key mechanism for in-context learning |
    | **Previous token** | Always attends to immediately preceding token | Simple bigram-like pattern |
    | **BOS/Delimiter** | Attends to the beginning-of-sequence or special tokens | Acts as a "no-op" or default |
    | **Duplicate token** | Attends to earlier occurrences of the same token | "The cat... the" → head links both "the" |

    **Induction heads** (Olsson et al., 2022) are particularly important:

    ```
    Context: "The cat sat on the mat. The cat"
    Induction head detects: "The cat" appeared before → predicts "sat"

    Pattern: [A][B] ... [A] → predicts [B]

    This is the core mechanism behind in-context learning in LLMs.
    ```

    **Observations:**

    - Not all heads are equally important — pruning 20-40% of heads often has minimal quality impact
    - Earlier layers tend to learn local/syntactic patterns; later layers learn more abstract/semantic patterns
    - Some heads are "redundant" and learn overlapping patterns (motivating GQA)

---

### What is the difference between pre-training, fine-tuning, and in-context learning? - Google, OpenAI Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Pre-training`, `Fine-tuning`, `In-Context Learning`, `Transfer Learning` | **Asked by:** Google, OpenAI, Amazon, Meta

??? success "View Answer"

    These are the three main paradigms for adapting Transformers to tasks:

    **Pre-training:** Training on a massive corpus (web text, books, code) with a self-supervised objective. This is where the model learns general language understanding.

    - **Cost:** Millions of dollars, weeks on thousands of GPUs
    - **Data:** Trillions of tokens
    - **Objective:** Next token prediction (CLM) or masked language modeling (MLM)

    **Fine-tuning:** Further training the pre-trained model on task-specific data with task-specific supervision.

    - **Full fine-tuning:** Update all parameters
    - **LoRA / QLoRA:** Update only low-rank adapters (0.1-1% of parameters)
    - **Cost:** Hours to days on a few GPUs
    - **Data:** Thousands to millions of labeled examples

    **In-context learning (ICL):** No parameter updates — provide examples in the prompt and the model learns from them on the fly.

    ```
    Prompt:
    "Translate English to French:
     sea otter => loutre de mer
     peppermint => menthe poivrée
     cheese => "

    Model output: "fromage"
    ```

    | Aspect | Pre-training | Fine-tuning | In-context Learning |
    |--------|-------------|-------------|-------------------|
    | Parameters updated | All | All or subset | None |
    | Training data needed | Trillions of tokens | Thousands+ examples | 0-few examples |
    | Compute cost | Enormous | Moderate | Inference only |
    | Task specificity | General | Task-specific | Task-adaptive |
    | Deployment | Base model | New model checkpoint | Same base model |

    **Key insight:** As models scale, in-context learning becomes increasingly powerful, reducing the need for fine-tuning on many tasks.

---

### What is the computational complexity of self-attention? Why is it a bottleneck for long sequences? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Complexity`, `Quadratic`, `Long Context`, `Efficiency` | **Asked by:** Google, Meta, Anthropic

??? success "View Answer"

    Self-attention has **quadratic complexity** in sequence length:

    **Time complexity:** $O(N^2 \cdot d)$ — computing the $N \times N$ attention matrix with $d$-dimensional vectors.

    **Memory complexity:** $O(N^2 + Nd)$ — storing the attention matrix plus the Q, K, V, output matrices.

    ```
    Sequence Length    Attention Matrix     Memory (FP16)
    ───────────────    ────────────────     ─────────────
    512                262K entries         512 KB
    2,048              4.2M entries         8 MB
    8,192              67M entries          128 MB
    32,768             1.07B entries        2 GB
    131,072            17.2B entries        32 GB
    1,000,000          1T entries           ~2 TB
    ```

    **Why it's a bottleneck:**

    1. Memory grows quadratically — can't fit in GPU HBM for long sequences
    2. Compute grows quadratically — most operations are on the attention matrix
    3. For a 4K context model, attention is manageable. For 128K or 1M context, it becomes dominant

    **Solutions and their trade-offs:**

    | Approach | Complexity | Exact? | Examples |
    |----------|-----------|--------|---------|
    | Flash Attention | $O(N^2 d)$ time, $O(N)$ memory | Yes | Standard in all modern LLMs |
    | Sliding Window | $O(Nw)$ where $w$ = window size | No (local only) | Mistral, Longformer |
    | Linear Attention | $O(Nd^2)$ | No (approximation) | Performer, Linear Transformers |
    | Ring Attention | $O(N^2 d)$ but distributed | Yes | Used for very long contexts |
    | Sparse Attention | $O(N\sqrt{N})$ | No | BigBird, Longformer |

---

### What is Sliding Window Attention? How does Mistral use it? - Mistral, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Sliding Window`, `Local Attention`, `Mistral`, `Efficiency` | **Asked by:** Mistral, Google, Meta

??? success "View Answer"

    **Sliding Window Attention (SWA)** restricts each token to only attend to a fixed window of $w$ preceding tokens, rather than the full context.

    ```
    Full Causal Attention (seq_len=6):    Sliding Window (w=3):
    ┌───────────────┐                     ┌───────────────┐
    │ 1 0 0 0 0 0   │                     │ 1 0 0 0 0 0   │
    │ 1 1 0 0 0 0   │                     │ 1 1 0 0 0 0   │
    │ 1 1 1 0 0 0   │                     │ 1 1 1 0 0 0   │
    │ 1 1 1 1 0 0   │                     │ 0 1 1 1 0 0   │
    │ 1 1 1 1 1 0   │                     │ 0 0 1 1 1 0   │
    │ 1 1 1 1 1 1   │                     │ 0 0 0 1 1 1   │
    └───────────────┘                     └───────────────┘
    Complexity: O(N²)                     Complexity: O(N×w)
    ```

    **How Mistral 7B uses it:**

    - Window size $w = 4096$ tokens
    - Information propagates beyond the window through **stacking layers**: after $L$ layers, the effective receptive field is $L \times w$ tokens (Mistral: 32 layers × 4096 = 131K effective context)
    - Combined with **rolling KV cache**: only the last $w$ tokens' KV pairs are stored, enabling constant memory inference regardless of sequence length

    ```
    Layer 1: Token 10 sees tokens 7-10    (window=4)
    Layer 2: Token 10 sees tokens 4-10    (through layer 1 representations)
    Layer 3: Token 10 sees tokens 1-10    (full effective context)
    ```

    **Trade-offs:**

    | Aspect | Full Attention | Sliding Window |
    |--------|---------------|----------------|
    | Attention per token | $O(N)$ | $O(w)$ |
    | KV cache size | Grows with sequence | Fixed at $w$ |
    | Direct access to distant tokens | Yes | No (only through stacked layers) |
    | Information loss for very long range | None | Possible (depends on depth) |

---

### What are learned positional embeddings? Compare them to sinusoidal encoding - Google, OpenAI Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Positional Embeddings`, `Learned`, `Sinusoidal` | **Asked by:** Google, OpenAI, Meta

??? success "View Answer"

    **Learned positional embeddings** treat each position as a learnable parameter vector (like a lookup table) rather than computing it with a fixed formula.

    ```python
    import torch.nn as nn

    # Learned (BERT, GPT-2)
    position_embedding = nn.Embedding(max_seq_len, d_model)
    # pe = position_embedding(position_ids)  # Lookup table

    # Sinusoidal (original Transformer)
    # pe = sin/cos(position / 10000^(2i/d))  # Fixed formula
    ```

    | Aspect | Sinusoidal | Learned |
    |--------|-----------|---------|
    | Parameters | 0 | $\text{max\_seq\_len} \times d_{model}$ |
    | Fixed or trainable | Fixed | Trained with model |
    | Extrapolation to unseen lengths | Yes (formula works for any position) | No (no embedding for position > max) |
    | Quality | Good | Slightly better (within training length) |
    | Used in | Original Transformer | BERT, GPT-2, GPT-3 |
    | Modern preference | Neither — RoPE is preferred | Neither — RoPE is preferred |

    **Why both are largely replaced by RoPE:**

    - Both are **absolute** position encodings — they encode $pos$ directly
    - RoPE encodes **relative** positions in the attention computation, which is more natural for language
    - RoPE extrapolates better to longer sequences (with techniques like NTK-aware scaling or YaRN)

---

### What is the "Attention Is All You Need" paper's key contribution beyond the architecture itself? - Research, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Vaswani`, `Attention`, `Historical`, `Impact` | **Asked by:** Google, DeepMind, Research Labs

??? success "View Answer"

    The 2017 paper (Vaswani et al.) made several critical contributions:

    **1. Eliminated recurrence entirely**

    Previous models (like the Transformer's predecessors) used attention **alongside** RNNs. The key insight was that attention alone is sufficient — no recurrence needed. This unlocked massive parallelism during training.

    **2. Established the Transformer block as a universal building block**

    The pattern `[Multi-Head Attention → Add & Norm → FFN → Add & Norm]` became the standard building block for nearly all subsequent models in NLP, computer vision (ViT), speech (Whisper), protein folding (AlphaFold), and more.

    **3. Demonstrated scaling effectiveness**

    The paper showed that increasing model size ($d_{model}$, number of layers, number of heads) directly improved translation quality, foreshadowing the scaling laws that would drive the LLM revolution.

    **4. Practical training recipe**

    - Label smoothing (0.1)
    - Adam optimizer with custom warmup learning rate schedule: $lr = d_{model}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup}^{-1.5})$
    - Dropout (0.1) on sublayers and attention weights
    - Residual connections + Layer Normalization

    **5. Benchmark results**

    Achieved state-of-the-art on WMT 2014 English-to-German and English-to-French translation, training in a fraction of the time of previous best models.

    **Impact:** Every major model since — BERT, GPT, T5, ViT, DALL-E, AlphaFold 2, Whisper, Stable Diffusion — is built on the Transformer.

---

### How does the Transformer handle variable-length sequences? What is padding and attention masking? - Amazon, Google Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Padding`, `Attention Mask`, `Batching`, `Variable Length` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    Transformers process fixed-size tensors, so variable-length sequences in a batch must be **padded** to the same length and **masked** to ignore padding tokens.

    ```
    Batch of 3 sequences (different lengths):
    "The cat"       → [The, cat, PAD, PAD, PAD]
    "A big dog ran" → [A, big, dog, ran, PAD]
    "Hello"         → [Hello, PAD, PAD, PAD, PAD]

    Attention mask (1 = real token, 0 = padding):
    [1, 1, 0, 0, 0]
    [1, 1, 1, 1, 0]
    [1, 0, 0, 0, 0]
    ```

    **Two types of masking:**

    1. **Padding mask:** Prevents attention to PAD tokens

    ```python
    # Padding mask: (batch, 1, 1, seq_len) for broadcasting
    padding_mask = (input_ids != pad_token_id).unsqueeze(1).unsqueeze(2)
    # Applied: scores = scores.masked_fill(padding_mask == 0, float('-inf'))
    ```

    2. **Causal mask + Padding mask combined** (decoder):

    ```python
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))  # Lower triangular
    combined_mask = causal_mask & padding_mask  # Both must be 1 to attend
    ```

    **Efficiency concerns:**

    - Padding wastes compute on meaningless tokens
    - Solutions: **Packing** (concatenate short sequences to fill the batch dimension), or **Flash Attention** with variable-length support (no padding needed)

---

### What is Gradient Checkpointing and why is it used in Transformer training? - Meta, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Gradient Checkpointing`, `Memory`, `Training`, `Recomputation` | **Asked by:** Meta, Google, Anthropic

??? success "View Answer"

    **Gradient checkpointing** (also called activation checkpointing) trades **compute for memory** during training by not storing all intermediate activations for the backward pass, instead recomputing them as needed.

    **Without checkpointing:**

    - Store activations at every layer during forward pass
    - Memory: $O(L \times N \times d)$ for $L$ layers
    - For a 70B parameter model with long context, this can require hundreds of GB

    **With checkpointing:**

    - Only store activations at selected "checkpoint" layers (e.g., every $\sqrt{L}$ layers)
    - During backward pass, recompute activations between checkpoints
    - Memory: $O(\sqrt{L} \times N \times d)$ — typically **60-70% memory reduction**
    - Cost: ~33% more compute (one extra forward pass per segment)

    ```
    Without checkpointing (12 layers):
    Forward:  L1 → L2 → L3 → L4 → L5 → ... → L12
    Stored:   [a1] [a2] [a3] [a4] [a5] ... [a12]  ← stores ALL
    Memory:   12 × activation_size

    With checkpointing (every 4 layers):
    Forward:  L1 → L2 → L3 → L4 → L5 → ... → L12
    Stored:   [a1]           [a4]           [a8]  [a12]  ← stores 4 only
    Backward: Recompute L9-L12 from a8, then L5-L8 from a4, etc.
    Memory:   4 × activation_size + 4 × activation_size (recompute buffer)
    ```

    ```python
    # PyTorch gradient checkpointing
    from torch.utils.checkpoint import checkpoint

    class TransformerBlock(nn.Module):
        def forward(self, x):
            # Wrap sublayers with checkpoint to save memory
            x = x + checkpoint(self.attention, x)
            x = x + checkpoint(self.ffn, x)
            return x
    ```

    Used in virtually all large-scale Transformer training runs.

---

### What is mixed-precision training and why is it important for Transformers? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Mixed Precision`, `FP16`, `BF16`, `Training Efficiency` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Mixed-precision training** uses lower-precision formats (FP16 or BF16) for most operations while keeping a master copy of weights in FP32 for numerical stability.

    ```
    Full precision (FP32):          Mixed precision (BF16/FP16 + FP32):
    Weights: FP32 (4 bytes)         Master weights: FP32 (4 bytes)
    Activations: FP32               Working weights: BF16 (2 bytes)
    Gradients: FP32                 Activations: BF16 (2 bytes)
    Optimizer: FP32                 Gradients: BF16 (2 bytes)
                                    Optimizer states: FP32 (4 bytes)
    ```

    **Number formats:**

    | Format | Bits | Exponent | Mantissa | Range | Precision |
    |--------|------|----------|----------|-------|-----------|
    | FP32 | 32 | 8 | 23 | $\pm 3.4 \times 10^{38}$ | High |
    | FP16 | 16 | 5 | 10 | $\pm 65504$ | Medium |
    | BF16 | 16 | 8 | 7 | $\pm 3.4 \times 10^{38}$ | Lower but same range as FP32 |

    **BF16 is preferred** for LLM training because it has the same exponent range as FP32, avoiding overflow/underflow that can plague FP16.

    **Benefits:**

    1. **~2x memory reduction** for activations and gradients
    2. **~2x throughput** on Tensor Cores (NVIDIA GPUs have specialized hardware for FP16/BF16)
    3. Enables training larger models that wouldn't fit in memory at FP32

    **The training loop:**

    ```python
    # Simplified mixed-precision training with PyTorch
    scaler = torch.amp.GradScaler()  # For FP16 (not needed for BF16)

    for data, target in dataloader:
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(data)       # Forward in BF16
            loss = criterion(output, target)

        scaler.scale(loss).backward()  # Backward in BF16
        scaler.step(optimizer)         # Update FP32 master weights
        scaler.update()
    ```

---

### What is the learning rate warmup schedule used in Transformer training? Why is it necessary? - Google, DeepMind Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Learning Rate`, `Warmup`, `Cosine Decay`, `Training` | **Asked by:** Google, DeepMind, OpenAI

??? success "View Answer"

    Transformer training uses a **warmup-then-decay** learning rate schedule because Adam optimizer with randomly initialized parameters can be unstable with a high initial learning rate.

    **Original Transformer schedule:**

    $$lr = d_{model}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup\_steps}^{-1.5})$$

    This increases linearly during warmup, then decays proportionally to $1/\sqrt{\text{step}}$.

    **Modern LLM schedule (cosine with warmup):**

    ```
    Learning Rate
    │
    │        ╱╲
    │       ╱  ╲
    │      ╱    ╲
    │     ╱      ╲╲
    │    ╱         ╲╲
    │   ╱            ╲╲╲
    │  ╱                ╲╲╲╲
    │ ╱ warmup              ╲╲╲╲╲╲___  min_lr
    │╱
    └─────────────────────────────── Training Steps
    │←warmup→│←──── cosine decay ────→│
    ```

    **Why warmup is necessary:**

    1. **Adam's second moment is not yet estimated:** At initialization, Adam has no history of gradient magnitudes. Without warmup, the initial steps can have disproportionately large updates.
    2. **Random initialization instability:** Attention patterns are random at start — large updates can push them into degenerate states.
    3. **Layer normalization sensitivity:** LayerNorm parameters interact with the learning rate; starting too high can cause training divergence.

    **Typical hyperparameters for modern LLMs:**

    - Warmup: 1-2% of total training steps (e.g., 2000 steps)
    - Peak learning rate: $1 \times 10^{-4}$ to $3 \times 10^{-4}$
    - Minimum learning rate: Peak × 0.1
    - Decay: Cosine schedule to minimum

---

### What is PagedAttention (vLLM)? How does it optimize LLM serving? - Meta, Anthropic Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `PagedAttention`, `vLLM`, `Serving`, `KV Cache`, `Memory` | **Asked by:** Meta, Anthropic, Google

??? success "View Answer"

    **PagedAttention** (Kwon et al., 2023, used in vLLM) applies **virtual memory paging** concepts from operating systems to manage the KV cache during LLM inference.

    **The problem:** During serving, each request needs a contiguous block of GPU memory for its KV cache. Due to variable output lengths, this leads to:

    - **Internal fragmentation:** Pre-allocated cache blocks are larger than needed
    - **External fragmentation:** Memory gaps between freed blocks can't be reused
    - Typically **60-80% of KV cache memory is wasted**

    **PagedAttention solution:**

    ```
    Traditional KV Cache:
    ┌──────────────────────────────────────────────────┐
    │ Request 1 KV [████████░░░░░]  ← internal frag   │
    │ Request 2 KV [█████░░░░░░░░]  ← more waste      │
    │ [gap - external fragmentation]                    │
    │ Request 3 KV [██████████░░░]                      │
    └──────────────────────────────────────────────────┘

    PagedAttention (paged KV cache):
    ┌─────────────────────────────────────────────────────┐
    │ Block Table (per request):                          │
    │ Req 1: [blk5] → [blk2] → [blk9]                   │
    │ Req 2: [blk1] → [blk7]                             │
    │ Req 3: [blk3] → [blk8] → [blk4] → [blk6]         │
    │                                                     │
    │ Physical blocks (non-contiguous, fixed size):       │
    │ [blk1][blk2][blk3][blk4][blk5][blk6][blk7][blk8]  │
    │  ████   ████  ████  ████  ████  ████  ████  ████   │
    │  (all blocks fully utilized, <4% waste)             │
    └─────────────────────────────────────────────────────┘
    ```

    **Key ideas:**

    1. **Fixed-size blocks:** KV cache is divided into small fixed-size blocks (like memory pages)
    2. **Block table:** Each request has a virtual block table mapping logical → physical blocks
    3. **Non-contiguous storage:** Blocks don't need to be adjacent in GPU memory
    4. **Copy-on-write:** For beam search or parallel sampling, share KV cache blocks and only copy when modified

    **Results:** vLLM achieves **2-4x higher throughput** than Hugging Face's text-generation-inference, primarily by eliminating memory waste.

---

### What are Chinchilla scaling laws and how do they affect Transformer training decisions? - DeepMind, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Scaling Laws`, `Chinchilla`, `Compute-Optimal`, `Training` | **Asked by:** DeepMind, Google, Anthropic

??? success "View Answer"

    **Chinchilla scaling laws** (Hoffmann et al., 2022, DeepMind) showed that many LLMs were **significantly undertrained** — you should scale model size and training data equally.

    **Key finding:** For a given compute budget $C$, the optimal model size $N$ and training tokens $D$ scale as:

    $$N_{opt} \propto C^{0.5}, \quad D_{opt} \propto C^{0.5}$$

    The **optimal ratio** is approximately **20 tokens per parameter**.

    | Model | Parameters | Training Tokens | Tokens/Param | Optimal? |
    |-------|-----------|----------------|--------------|----------|
    | GPT-3 | 175B | 300B | 1.7 | Undertrained |
    | Gopher | 280B | 300B | 1.1 | Very undertrained |
    | Chinchilla | 70B | 1.4T | 20 | Optimal |
    | LLaMA | 65B | 1.4T | 21.5 | Optimal |
    | LLaMA 2 | 70B | 2T | 28.6 | Over-trained (for better inference) |

    **Impact on the field:**

    1. **Shifted focus from bigger models to more data:** Instead of training a 500B model on 300B tokens, train a 70B model on 1.4T tokens — same compute, much better performance
    2. **LLaMA family:** Meta's LLaMA models followed Chinchilla scaling, achieving GPT-3 level performance at 4x fewer parameters
    3. **Inference efficiency:** Smaller, well-trained models are cheaper to serve than larger, undertrained ones
    4. **Data becomes the bottleneck:** High-quality training data is now the scarce resource, not compute

    **However:** In practice, many modern models are intentionally "over-trained" beyond Chinchilla-optimal because a smaller model trained on more data is cheaper to **serve** (inference cost scales with model size, not training tokens).

---

### What is RLHF (Reinforcement Learning from Human Feedback) and how is it applied to Transformers? - OpenAI, Anthropic Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `RLHF`, `Alignment`, `PPO`, `Reward Model` | **Asked by:** OpenAI, Anthropic, Google, DeepMind

??? success "View Answer"

    **RLHF** is the technique that transforms a base language model into a helpful, harmless assistant (like ChatGPT or Claude). It has three stages:

    ```
    Stage 1: Supervised Fine-Tuning (SFT)
    ┌─────────────────────────────────────┐
    │ Base Model ──► Fine-tune on human-  │
    │                written examples of  │
    │                helpful responses    │
    │                                     │
    │ Input: "Explain gravity"            │
    │ Target: [High-quality explanation]  │
    └────────────────┬────────────────────┘
                     │
    Stage 2: Reward Model Training
    ┌────────────────▼────────────────────┐
    │ For each prompt, generate multiple  │
    │ responses. Humans rank them.        │
    │                                     │
    │ Prompt → Response A (rank 1) ✓     │
    │       → Response B (rank 2)        │
    │       → Response C (rank 3)        │
    │                                     │
    │ Train reward model: R(prompt, resp) │
    │ to predict human preferences        │
    └────────────────┬────────────────────┘
                     │
    Stage 3: RL Optimization (PPO)
    ┌────────────────▼────────────────────┐
    │ Optimize the SFT model to maximize  │
    │ reward model score using PPO:       │
    │                                     │
    │ objective = E[R(x, y)] - β·KL(π||π_ref)│
    │                                     │
    │ The KL penalty prevents the model   │
    │ from diverging too far from the SFT │
    │ model (reward hacking prevention)   │
    └─────────────────────────────────────┘
    ```

    **The PPO objective:**

    $$\mathcal{L} = \mathbb{E}_{x \sim D, y \sim \pi_\theta(x)} \left[ R_\phi(x, y) - \beta \cdot \text{KL}(\pi_\theta(y|x) \| \pi_{ref}(y|x)) \right]$$

    **Alternatives to RLHF:**

    - **DPO (Direct Preference Optimization):** Eliminates the reward model and PPO entirely — directly optimizes the policy from preference pairs. Simpler, more stable.
    - **RLAIF:** Uses AI feedback instead of human feedback (constitutional AI approach).

---

### What is the difference between sparse and dense attention? Describe approaches like BigBird and Longformer - Google, Research Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Sparse Attention`, `BigBird`, `Longformer`, `Efficiency` | **Asked by:** Google, Research Labs, Meta

??? success "View Answer"

    **Dense attention** computes attention scores between all $N^2$ token pairs. **Sparse attention** only computes a subset of these pairs.

    **Longformer attention pattern** (Beltagy et al., 2020):

    ```
    Combines three patterns:

    1. Sliding window (local):     2. Dilated window:       3. Global tokens:
    ┌─────────────┐               ┌─────────────┐          ┌─────────────┐
    │ █ █ ░ ░ ░ ░ │               │ █ ░ █ ░ █ ░ │          │ █ █ █ █ █ █ │ ← [CLS]
    │ █ █ █ ░ ░ ░ │               │ ░ █ ░ █ ░ █ │          │ █ █ ░ ░ ░ ░ │
    │ ░ █ █ █ ░ ░ │               │ █ ░ █ ░ █ ░ │          │ █ ░ █ ░ ░ ░ │
    │ ░ ░ █ █ █ ░ │               │ ░ █ ░ █ ░ █ │          │ █ ░ ░ █ ░ ░ │
    │ ░ ░ ░ █ █ █ │               │ █ ░ █ ░ █ ░ │          │ █ ░ ░ ░ █ ░ │
    │ ░ ░ ░ ░ █ █ │               │ ░ █ ░ █ ░ █ │          │ █ ░ ░ ░ ░ █ │
    └─────────────┘               └─────────────┘          └─────────────┘
    O(N×w)                        O(N×w)                   O(N×g)
    ```

    **BigBird** (Zaheer et al., 2020, Google) combines:

    1. **Random attention:** Each token attends to $r$ random tokens
    2. **Window attention:** Local sliding window of size $w$
    3. **Global attention:** Selected tokens attend to all others

    This achieves $O(N)$ complexity while theoretically being a **universal approximator** of sequence functions (proven to be Turing complete).

    | Model | Attention Pattern | Complexity | Max Length |
    |-------|------------------|-----------|-----------|
    | Standard Transformer | Full | $O(N^2)$ | ~512-2K |
    | Longformer | Window + Global | $O(N \times w)$ | 4K-16K |
    | BigBird | Window + Random + Global | $O(N)$ | 4K-16K |
    | Mistral | Sliding Window | $O(N \times w)$ | 32K+ |

    **Modern trend:** Rather than using sparse attention patterns, most current LLMs use **full attention + Flash Attention** for efficiency, combined with RoPE for length extrapolation. Sparse patterns are less common in the latest models.

---

### Explain the concept of "emergent abilities" in large Transformers. Are they real? - Google, Anthropic Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Emergence`, `Scaling`, `Capabilities`, `Evaluation` | **Asked by:** Google, Anthropic, DeepMind

??? success "View Answer"

    **Emergent abilities** (Wei et al., 2022) are capabilities that appear only at a certain model scale — seemingly absent in smaller models and suddenly present in larger ones.

    **Classic examples:**

    | Ability | Absent below | Emerges around |
    |---------|-------------|---------------|
    | Multi-step arithmetic | ~10B params | ~100B params |
    | Chain-of-thought reasoning | ~10B | ~60B |
    | Word unscrambling | ~10B | ~100B |
    | In-context learning (many-shot) | ~1B | ~10B |

    **The debate:**

    **"Emergence is real" view:**

    - Performance on certain benchmarks shows a sharp phase transition at specific scales
    - Qualitatively new capabilities appear (e.g., GPT-4 can pass bar exams, GPT-3 cannot)
    - Analogous to phase transitions in physics

    **"Emergence is a mirage" view** (Schaeffer et al., 2023):

    - Emergence disappears when using **continuous** metrics instead of **discontinuous** ones (e.g., exact-match accuracy shows emergence; token-level accuracy shows smooth improvement)
    - The "sudden" appearance is an artifact of the evaluation metric, not the model's capabilities
    - Per-token performance improves smoothly with scale

    ```
    Discontinuous metric (exact match):
    Accuracy │
             │                        ●●●●
             │                    ●●●●
             │               ●
             │          ●
             │     ●  ●
             │  ●●●
             └─────────────────── Scale (log)
             Looks like emergence!

    Continuous metric (token accuracy):
    Accuracy │                         ●●
             │                      ●●
             │                   ●●
             │                ●●
             │             ●●
             │          ●●
             │       ●●
             │    ●●
             │ ●●
             └─────────────────── Scale (log)
             Smooth improvement, no emergence!
    ```

    **Interview takeaway:** The truth is likely nuanced — some capabilities do emerge more suddenly than others, but the sharp "phase transition" narrative is partly an artifact of evaluation methodology.

---

### What is LoRA (Low-Rank Adaptation)? How does it enable efficient fine-tuning of large Transformers? - Meta, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `LoRA`, `Fine-tuning`, `Low-Rank`, `PEFT` | **Asked by:** Meta, Google, Amazon, OpenAI

??? success "View Answer"

    **LoRA** (Hu et al., 2021) freezes the pre-trained weights and injects **trainable low-rank decomposition matrices** into attention layers, reducing trainable parameters by 10,000x while matching full fine-tuning quality.

    **Core idea:** Instead of updating a weight matrix $W \in \mathbb{R}^{d \times d}$, learn a low-rank update $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times d}$ with rank $r \ll d$:

    $$h = Wx + \Delta Wx = Wx + BAx$$

    ```
    Full Fine-tuning:
    x ──► [W + ΔW] ──► h       (ΔW: d × d parameters to train)
          d×d trainable

    LoRA:
              ┌──► [W] (frozen) ──────┐
    x ──►─────┤                        ├──► h = Wx + BAx
              └──► [A]──►[B] (trainable)┘
                   d×r    r×d
                   (r << d, e.g. r=16)
    ```

    **Parameter savings example (LLaMA 65B):**

    | Method | Trainable Params | Memory |
    |--------|-----------------|--------|
    | Full fine-tuning | 65B | ~780 GB (with optimizer) |
    | LoRA (r=16) | ~10M (0.015%) | ~30 GB |
    | QLoRA (4-bit + LoRA) | ~10M | ~15 GB |

    **Why it works:**

    - Research shows that weight updates during fine-tuning have **low intrinsic rank** — the update $\Delta W$ is approximately low-rank even in full fine-tuning
    - Rank $r = 8$ to $r = 64$ is usually sufficient
    - At inference, $BA$ can be merged into $W$: $W' = W + BA$, so there is **zero additional latency**

    **QLoRA** extends this by quantizing the frozen weights to 4-bit, enabling fine-tuning of 70B+ models on a single 48GB GPU.

---

### What is the difference between additive attention and dot-product attention? - Google, Research Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Additive Attention`, `Dot-Product`, `Bahdanau`, `Luong` | **Asked by:** Google, Research Labs

??? success "View Answer"

    These are two fundamental attention mechanisms:

    **Additive attention** (Bahdanau et al., 2015):

    $$e_{ij} = v^T \tanh(W_1 h_i + W_2 s_j)$$

    $$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}$$

    Uses a learned feed-forward network to compute compatibility between query and key.

    **Dot-product (multiplicative) attention** (Luong et al., 2015 / Vaswani et al., 2017):

    $$e_{ij} = \frac{q_i^T k_j}{\sqrt{d_k}}$$

    $$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}$$

    Uses a simple dot product between query and key vectors.

    | Aspect | Additive | Dot-Product (Scaled) |
    |--------|---------|---------------------|
    | Scoring function | $v^T \tanh(W_1 h + W_2 s)$ | $q^T k / \sqrt{d_k}$ |
    | Parameters | $W_1$, $W_2$, $v$ | None (projections are in Q, K matrices) |
    | Computational cost | Higher (matrix multiply + tanh) | Lower (single matrix multiply) |
    | Performance at high $d_k$ | Better (doesn't suffer from scaling) | Needs scaling by $\sqrt{d_k}$ |
    | Parallelism | Lower | Higher (pure matrix ops) |
    | Used in | Original seq2seq | All Transformers |

    **Why Transformers use dot-product attention:** It can be computed as a single batched matrix multiplication ($QK^T$), which is extremely efficient on GPU hardware. Additive attention requires element-wise operations that are harder to parallelize.

---

### What is the "residual stream" view of Transformers? Why is it useful for interpretability? - Anthropic, DeepMind Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Residual Stream`, `Interpretability`, `Mechanistic` | **Asked by:** Anthropic, DeepMind, Research Labs

??? success "View Answer"

    The **residual stream** view (popularized by Elhage et al., 2021, Anthropic) reframes the Transformer as a series of **additive updates** to a shared "stream" of information flowing through the network.

    ```
    Standard view:          Residual stream view:
    x → [Attn] → [FFN] →   x ──────────────────────────────────► output
        [Attn] → [FFN] →        +Attn1  +FFN1  +Attn2  +FFN2
        ...                      │       │      │       │
                                 ▼       ▼      ▼       ▼
                            x₀ + a₁   + f₁   + a₂    + f₂  = final

    Each component READS from the stream and WRITES an additive update.
    ```

    Mathematically, the final output is:

    $$x_{final} = x_0 + \sum_{l=1}^{L} \text{Attn}_l(x_{l-1}) + \sum_{l=1}^{L} \text{FFN}_l(x_{l-1})$$

    **Why this view is powerful:**

    1. **Linear decomposition:** The output is a sum of contributions from each layer and sublayer, making it possible to attribute predictions to specific components

    2. **Superposition:** The residual stream acts as a shared "bandwidth" through which different features communicate across layers

    3. **Path analysis:** Any output logit can be decomposed into contributions from every attention head and FFN layer via path expansion

    4. **Skip connections matter:** Information can "skip" layers entirely — a token embedding at layer 0 directly influences the final output

    **Key implication:** Attention heads and FFN layers don't transform a hidden state in sequence — they each read from and write to a shared communication channel. This is the foundation of **mechanistic interpretability** research.

---

### How do Transformers handle multi-modal inputs (text + images)? Explain vision-language architectures - Google, OpenAI Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Multi-Modal`, `Vision-Language`, `ViT`, `Cross-Attention` | **Asked by:** Google, OpenAI, Anthropic, Meta

??? success "View Answer"

    Multi-modal Transformers process different input types (text, images, audio) by converting them into a shared token representation space.

    **Common approaches:**

    **1. Separate encoders + cross-attention (Flamingo, early approaches):**

    ```
    Image ──► [Vision Encoder] ──► Image tokens ──┐
                                                    ├──► [Cross-Attention in LLM] ──► Output
    Text  ──► [Text Tokenizer] ──► Text tokens  ──┘
    ```

    **2. Early fusion — interleaved tokens (GPT-4V, Gemini):**

    ```
    Image ──► [Vision Encoder] ──► [Projector] ──► Image tokens
                                                     ↓
    Input sequence: [text_1] [text_2] [img_1] [img_2] ... [img_N] [text_3] ...
                                                     ↓
                                            [Standard Transformer Decoder]
    ```

    **3. Vision Transformer (ViT) as encoder:**

    An image is split into patches, each treated as a "token":

    ```
    Image (224×224) → Split into 16×16 patches → 196 patches
    Each patch → Linear projection → Patch embedding (like a word embedding)
    + Positional embedding → Feed into standard Transformer encoder
    ```

    **LLaVA architecture (popular open-source approach):**

    ```
    Image ──► CLIP ViT ──► MLP Projector ──► Visual tokens
                                              ↓
    Text  ──► Tokenizer ─────────────────► Text tokens
                                              ↓
    Combined: [visual tokens] + [text tokens] ──► LLaMA decoder ──► Response
    ```

    **Key insight:** The Transformer's self-attention is **modality-agnostic** — it operates on sequences of vectors regardless of whether they originated from text, images, audio, or video. The "magic" is in the tokenization/projection step that maps different modalities into a shared vector space.

---

### What is speculative decoding and how does it speed up Transformer inference? - Google, Anthropic Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Speculative Decoding`, `Inference`, `Draft Model`, `Speed` | **Asked by:** Google, Anthropic, Meta

??? success "View Answer"

    **Speculative decoding** (Leviathan et al., 2023; Chen et al., 2023) uses a small "draft" model to predict multiple tokens, then verifies them in parallel with the large model — achieving 2-3x speedup with **zero quality loss**.

    **The problem:** Autoregressive generation is **memory-bound**, not compute-bound. Each step requires loading the entire model's weights from memory to generate a single token. The GPU compute is vastly underutilized.

    **How it works:**

    ```
    Standard decoding (1 token per forward pass):
    Step 1: Large model → token 1
    Step 2: Large model → token 2
    Step 3: Large model → token 3
    ...N steps for N tokens

    Speculative decoding:
    Step 1: Draft model (fast) → [tok1, tok2, tok3, tok4, tok5] (γ draft tokens)
    Step 2: Large model verifies ALL 5 in ONE forward pass
            → tok1 ✓, tok2 ✓, tok3 ✓, tok4 ✗ (reject, sample from large model)
    Result: 4 tokens in 2 forward passes instead of 4!
    ```

    **Acceptance criterion:** For each draft token, accept it if the large model's probability is at least as high as the draft model's. If rejected, sample from a modified distribution that corrects for the draft model's error. This guarantees the **exact same output distribution** as the large model alone.

    $$P(\text{accept token } x) = \min\left(1, \frac{p_{large}(x)}{p_{draft}(x)}\right)$$

    **Key requirements:**

    1. Draft model must be much faster (e.g., 7B drafting for 70B)
    2. Draft model should have reasonable agreement with the large model (higher acceptance rate → more speedup)
    3. Verification is parallel — verifying $\gamma$ tokens costs the same as generating 1 token (single forward pass)

    **Typical speedup:** 2-3x with ~70-80% acceptance rate.

---

### What is the Transformer's position in the broader history of sequence models? Compare to RNNs, LSTMs, and SSMs - Google, Research Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `RNN`, `LSTM`, `SSM`, `Mamba`, `History` | **Asked by:** Google, Research Labs, DeepMind

??? success "View Answer"

    ```
    Timeline of Sequence Models:

    1986        1997        2014          2017           2023+
    ┌────┐    ┌──────┐    ┌──────────┐  ┌────────────┐  ┌──────┐
    │ RNN│───►│ LSTM │───►│ Attention│─►│ Transformer│─►│ SSMs │
    └────┘    └──────┘    │ + RNN    │  │            │  │(Mamba)│
                          └──────────┘  └────────────┘  └──────┘
    ```

    | Property | RNN | LSTM | Transformer | SSM (Mamba) |
    |----------|-----|------|-------------|-------------|
    | Parallelizable (training) | No | No | Yes | Yes |
    | Long-range dependencies | Poor | Better | Excellent | Good |
    | Inference (per token) | O(1) | O(1) | O(N) attention | O(1) |
    | Training complexity | O(N) | O(N) | O(N²) | O(N log N) |
    | Memory at inference | O(d) fixed | O(d) fixed | O(N × d) grows | O(d) fixed |
    | Dominant since | 1986 | 1997 | 2017 | Emerging (2023+) |

    **State Space Models (Mamba):**

    SSMs (Gu & Dao, 2023) are the main challenger to Transformers. They process sequences with a **fixed-size recurrent state**, like RNNs, but can be **parallelized** during training via a convolution formulation:

    - **Training:** Computed as a convolution (parallelizable like Transformers)
    - **Inference:** Computed as a recurrence (O(1) per step, like RNNs)
    - **No attention matrix:** Memory doesn't grow with sequence length

    **Current status:** Transformers still dominate for most tasks, but hybrid architectures (Transformer + SSM layers, e.g., Jamba) and pure SSMs are rapidly improving. The key advantage of SSMs is **linear scaling with sequence length** at inference.

---

### What is the softmax bottleneck in language modeling? How does it affect Transformers? - Research, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Softmax Bottleneck`, `Output Layer`, `Expressiveness` | **Asked by:** Research Labs, Google, DeepMind

??? success "View Answer"

    The **softmax bottleneck** (Yang et al., 2018) is a fundamental expressiveness limitation: the final softmax layer restricts the model to producing log-probability matrices of rank at most $d_{model}$, even when the true distribution requires higher rank.

    **The problem:**

    The output probability for token $w$ given context $c$ is:

    $$P(w|c) = \text{softmax}(h_c^T e_w)$$

    Where $h_c \in \mathbb{R}^d$ is the context representation and $e_w \in \mathbb{R}^d$ is the token embedding. The log-probability matrix $A$ where $A_{c,w} = \log P(w|c)$ has rank at most $d$ — the hidden dimension.

    If the true language distribution requires a rank higher than $d$, no model with this architecture can perfectly represent it.

    **Practical implications:**

    - Languages have highly **context-dependent** word distributions that may require very high rank
    - The bottleneck is more severe for smaller $d_{model}$ and larger vocabularies
    - This is one reason why increasing $d_{model}$ improves perplexity

    **Solutions:**

    1. **Mixture of Softmaxes (MoS):** Use multiple softmax outputs and mix them
    2. **Larger hidden dimensions:** Modern LLMs use $d_{model}$ = 4096-8192
    3. **Tied vs untied embeddings:** Tying input/output embeddings saves parameters but worsens the bottleneck (used in smaller models; larger models often untie)

---

### What is the difference between absolute and relative positional encodings? Give examples of each - Meta, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Positional Encoding`, `Absolute`, `Relative`, `Comparison` | **Asked by:** Meta, Google, DeepMind

??? success "View Answer"

    **Absolute positional encodings** assign a fixed vector to each position index. **Relative positional encodings** encode the distance between token pairs.

    **Absolute examples:**

    - **Sinusoidal** (Vaswani et al.): $PE(pos, i) = \sin(pos / 10000^{2i/d})$
    - **Learned** (BERT, GPT-2): Lookup table trained with the model

    **Relative examples:**

    - **RoPE** (LLaMA, Mistral): Rotates Q, K so dot product depends on $(m-n)$ not $m$ or $n$ separately
    - **ALiBi** (BLOOM, MPT): Adds $-m|i-j|$ bias to attention scores
    - **Relative position bias** (T5): Learned scalar bias per relative distance, added to attention logits
    - **XL-style** (Transformer-XL): Modifies attention score to include relative position embedding

    | Property | Absolute | Relative |
    |----------|---------|---------|
    | Encodes | Position index $pos$ | Distance between positions $i - j$ |
    | Added to | Token embeddings (input) | Attention scores (every layer) |
    | Translation invariance | No | Yes ("the cat sat" has same internal relations regardless of position) |
    | Length extrapolation | Poor (unseen positions) | Better (distances are seen during training) |
    | Modern usage | Mostly obsolete | Standard in all modern LLMs |

    **Why relative is better:** In language, the relationship between tokens typically depends on their **distance** (the word 3 positions back), not their **absolute position** (the 47th word). Relative encodings directly model this.

---

### What is Mixture of Experts routing? Explain load balancing and expert collapse - Google, DeepSeek Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `MoE`, `Routing`, `Load Balancing`, `Expert Collapse` | **Asked by:** Google, DeepSeek, Mistral

??? success "View Answer"

    **Router (gating network)** decides which expert(s) each token uses:

    $$g(x) = \text{softmax}(W_g x + \text{noise})$$
    $$\text{TopK}(g(x)) \rightarrow \text{select } k \text{ experts with highest scores}$$

    **Expert collapse:** A failure mode where the router learns to send most tokens to the same few experts, leaving others undertrained:

    ```
    Healthy routing:                    Expert collapse:
    Expert 1: ████████ (15%)           Expert 1: ████████████████ (60%)
    Expert 2: ████████ (14%)           Expert 2: ██ (5%)
    Expert 3: ███████ (13%)            Expert 3: █ (2%)
    Expert 4: ████████ (12%)           Expert 4: ████████ (25%)
    Expert 5: ███████ (12%)            Expert 5: █ (3%)
    Expert 6: ████████ (12%)           Expert 6: █ (2%)
    Expert 7: ███████ (11%)            Expert 7: █ (2%)
    Expert 8: ████████ (11%)           Expert 8: █ (1%)
    ```

    **Load balancing loss** (auxiliary loss to prevent collapse):

    $$\mathcal{L}_{balance} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot p_i$$

    Where:

    - $f_i$ = fraction of tokens routed to expert $i$ in the batch
    - $p_i$ = average router probability assigned to expert $i$
    - $N$ = number of experts
    - $\alpha$ = balancing coefficient (typically 0.01)

    This loss is minimized when $f_i = p_i = 1/N$ for all experts (uniform routing).

    **Advanced routing strategies:**

    | Strategy | Key idea | Used in |
    |----------|---------|---------|
    | Token choice | Each token picks top-k experts | Mixtral, GShard |
    | Expert choice | Each expert picks top-k tokens | Expert Choice routing |
    | Hash routing | Deterministic assignment by hash | Hash layers |
    | Aux-loss free | Use sigmoid instead of softmax, no aux loss | DeepSeek-V3 |

    DeepSeek-V3 notably uses **auxiliary-loss-free load balancing** with per-expert bias terms adjusted during training, avoiding the quality trade-off from the auxiliary loss.

---

### How does the Transformer decoder generate text during inference? Walk through the full autoregressive process - OpenAI, Amazon Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Autoregressive`, `Inference`, `Generation`, `Decoding` | **Asked by:** OpenAI, Amazon, Google

??? success "View Answer"

    **Autoregressive generation** produces one token at a time, using each predicted token as input for the next step:

    ```
    Prompt: "The cat"

    Step 1: Input: [The, cat]
            Model output: probability distribution over vocab
            Sample/Argmax → "sat"
            Append to sequence

    Step 2: Input: [The, cat, sat]  (or just "sat" with KV cache)
            Model output: probability distribution
            → "on"

    Step 3: Input: [The, cat, sat, on]
            → "the"

    Step 4: Input: [The, cat, sat, on, the]
            → "mat"

    ...continue until <EOS> or max length
    ```

    **Decoding strategies:**

    | Strategy | Formula | Property |
    |----------|---------|----------|
    | Greedy | $\arg\max P(w)$ | Deterministic, can be repetitive |
    | Temperature sampling | $P'(w) = \frac{\exp(z_w / T)}{\sum \exp(z_i / T)}$ | $T<1$: sharper, $T>1$: more random |
    | Top-k | Sample from top $k$ tokens | Limits to $k$ most likely |
    | Top-p (nucleus) | Sample from smallest set where $\sum P \geq p$ | Adaptive number of candidates |
    | Beam search | Track top $B$ sequences | Better for translation, worse for open-ended gen |

    **With KV cache (efficient inference):**

    ```
    Step 1: Process full prompt [The, cat] → cache K1,V1 and K2,V2
            Predict → "sat"

    Step 2: Only process new token [sat]
            Use cached K1,V1, K2,V2 for attention
            Cache K3,V3
            Predict → "on"

    Step 3: Only process [on]
            Use cached K1-K3, V1-V3
            Cache K4,V4
            Predict → "the"
    ```

    Each step only requires computing Q, K, V for the **new token** — the KV cache makes this $O(1)$ per token (instead of reprocessing the entire sequence).

---

### What are attention sinks? Why do LLMs heavily attend to the first token? - Meta, Anthropic Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Attention Sinks`, `First Token`, `Streaming`, `Inference` | **Asked by:** Meta, Anthropic, Research Labs

??? success "View Answer"

    **Attention sinks** (Xiao et al., 2023) is the phenomenon where Transformer LLMs disproportionately attend to the **first token** (or BOS token), regardless of its semantic relevance.

    **Why it happens:**

    Softmax requires attention weights to sum to 1. When no tokens are particularly relevant to the query, the model needs somewhere to "dump" the excess attention weight. The first token becomes a **learned "sink"** because:

    1. It's always present (every sequence starts with it)
    2. Its KV representation becomes optimized during training to absorb "unused" attention
    3. This acts like a "no-op" — attending to the sink doesn't corrupt the representation

    ```
    Typical attention pattern in Layer 15, Head 3:
    Token:    [BOS]  The   cat   sat   on   the   mat
    Weight:   0.35   0.05  0.15  0.20  0.05 0.10  0.10
              ^^^^
              Attention sink — high weight despite no semantic relevance
    ```

    **Practical impact for streaming/long-context inference:**

    If you use a sliding window KV cache and evict the first token, quality degrades catastrophically. The solution:

    ```
    Naive sliding window (breaks!):
    Keep tokens: [tok_100, tok_101, ..., tok_200]
    ← First token evicted → quality crashes

    StreamingLLM fix:
    Keep tokens: [BOS, tok_100, tok_101, ..., tok_200]
    ← Always keep the first few "sink" tokens → quality preserved
    ```

    **StreamingLLM** (Xiao et al., 2023) shows that keeping just the first 4 attention sink tokens + a sliding window enables **infinite length generation** with constant memory and stable quality.

---

### How do Transformers handle code generation differently from natural language? - OpenAI, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Code Generation`, `Codex`, `Tokenization`, `Structured` | **Asked by:** OpenAI, Google, Meta, Amazon

??? success "View Answer"

    Code generation uses the same Transformer architecture but requires special considerations:

    **1. Tokenization differences:**

    ```
    Natural language: "The cat sat" → ["The", " cat", " sat"] (3 tokens)

    Code: "def hello_world():" → ["def", " hello", "_world", "():", ...] (varies)

    Problem: Standard BPE can split identifiers badly:
    "getUserName" → ["get", "User", "Name"] (ok)
    "XMLParser"   → ["XML", "Parser"] (ok)
    "x_coord_3d"  → ["x", "_coord", "_", "3", "d"] (fragmented)
    ```

    Code-specific tokenizers use larger vocabularies and include common code patterns (indentation, brackets, keywords) as single tokens.

    **2. Indentation sensitivity (Python):**

    ```python
    # Indentation is semantically meaningful
    if True:
        print("in block")    # ← 4 spaces = part of if-block
    print("outside")          # ← no indent = outside if-block
    ```

    Models must learn that whitespace **changes program semantics**, unlike natural language.

    **3. Long-range structural dependencies:**

    ```python
    class MyClass:                    # Line 1
        def __init__(self, x):        # Line 2 - must match class indent
            self.x = x                # Line 3 - must reference self
        ...
        # 200 lines later
        def method(self):             # Must know self.x exists from line 3
            return self.x + 1
    ```

    **4. Infilling (Fill-in-the-Middle):**

    Code models often support FIM — predicting code that goes between a prefix and suffix:

    ```
    Prefix: "def add(a, b):\n"
    Suffix: "\n    return result"
    Model fills: "    result = a + b"
    ```

    This requires special training with `<PRE>`, `<SUF>`, `<MID>` tokens.

    **Notable code models:** Codex (OpenAI), CodeLlama (Meta), StarCoder (BigCode), DeepSeek-Coder, Claude.

---

### What is the role of the output projection $W^O$ in multi-head attention? Can it be removed? - Research, DeepMind Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Output Projection`, `Multi-Head Attention`, `Linear Map` | **Asked by:** Research Labs, DeepMind, Google

??? success "View Answer"

    After concatenating all head outputs, the **output projection** $W^O \in \mathbb{R}^{hd_v \times d_{model}}$ maps the concatenated multi-head result back to $d_{model}$ dimensions:

    $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

    **Why it exists:**

    1. **Mixes head outputs:** Each head operates in its own subspace. $W^O$ allows the model to learn **how to combine information** from different heads — which head's output matters most for each feature dimension.

    2. **Dimension matching:** Ensures the attention output has the same dimension ($d_{model}$) as the input, enabling the residual connection.

    3. **Added expressiveness:** Without $W^O$, the output is simply a concatenation of independent subspace projections. $W^O$ allows cross-head interaction.

    **Can it be removed?**

    - If $d_v = d_{model} / h$ (standard case), the concatenation already has dimension $d_{model}$, so $W^O$ could theoretically be an identity matrix
    - Experiments show removing $W^O$ causes a noticeable (but not catastrophic) quality drop
    - In practice, $W^O$ adds relatively few parameters ($d_{model}^2$) and is always included

---

### What is DeepNorm and why was it introduced for very deep Transformers? - Microsoft, Research Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `DeepNorm`, `Deep Transformers`, `Initialization`, `Stability` | **Asked by:** Microsoft, Research Labs, Google

??? success "View Answer"

    **DeepNorm** (Wang et al., 2022, Microsoft) is a normalization method that enables training **very deep** Transformers (up to 1000 layers) by combining a modified residual connection with specific initialization.

    **The problem:** Standard Post-Norm Transformers diverge beyond ~100 layers. Pre-Norm is stable but performs worse. Neither scales well to extreme depths.

    **DeepNorm formula:**

    $$x_{l+1} = \text{LayerNorm}(\alpha \cdot x_l + \text{Sublayer}(x_l))$$

    Where $\alpha > 1$ is a constant that **upweights the residual** connection, and sublayer weights are initialized with a smaller scale $\beta < 1$.

    For a Transformer with $N$ encoder and $M$ decoder layers:

    - Encoder: $\alpha = (2N)^{1/4}$, $\beta = (8N)^{-1/4}$
    - Decoder: $\alpha = (2M)^{1/4}$, $\beta = (8M)^{-1/4}$

    **Intuition:**

    - $\alpha > 1$ makes the residual connection stronger → gradient flows more easily
    - $\beta < 1$ makes sublayer outputs smaller → prevents destabilizing the residual stream
    - As depth increases, $\alpha$ grows and $\beta$ shrinks, automatically compensating

    **Results:** Successfully trained a 1000-layer Transformer (2500 attention/FFN sublayers) — previously impossible. The resulting model showed consistent improvements from adding more layers.

---

### Explain beam search vs sampling for Transformer text generation. When would you use each? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Beam Search`, `Sampling`, `Decoding`, `Generation` | **Asked by:** Amazon, Google, OpenAI

??? success "View Answer"

    **Beam search** maintains $B$ (beam width) candidate sequences and expands the most promising ones:

    ```
    Beam search (B=2):
    Step 1: "The" → P=0.3, "A" → P=0.2
    Step 2: "The cat" → P=0.12, "The dog" → P=0.09
    Step 3: "The cat sat" → P=0.05, "The cat is" → P=0.04
    → Final: highest cumulative probability sequence
    ```

    **Sampling** randomly draws from the probability distribution:

    ```
    Sampling (temperature=0.8, top_p=0.9):
    Step 1: Sample from top-90% mass → "The" (drawn)
    Step 2: Sample → "happy" (drawn)
    Step 3: Sample → "penguin" (drawn)
    → Output: "The happy penguin" (creative, diverse)
    ```

    | Aspect | Beam Search | Sampling (top-p/top-k) |
    |--------|------------|----------------------|
    | Determinism | Deterministic | Stochastic |
    | Diversity | Low (tends toward generic/high-frequency) | High |
    | Quality for factual tasks | Better | Risk of hallucination |
    | Creative writing | Boring/repetitive | Natural/creative |
    | Repetition | Can get stuck in loops | Less repetition |
    | Speed | Slower (B parallel sequences) | Faster (1 sequence) |

    **When to use each:**

    - **Beam search:** Machine translation, summarization, structured output, code completion (need precision)
    - **Sampling:** Creative writing, conversational AI, brainstorming (need diversity)
    - **Modern LLM chatbots:** Almost always use sampling with temperature + top-p. Beam search is rarely used for open-ended generation in modern systems.

---

### What is the relationship between Transformers and graph neural networks? - Research, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `GNN`, `Graph`, `Attention`, `Connection` | **Asked by:** Research Labs, Google, DeepMind

??? success "View Answer"

    Self-attention can be viewed as **message passing on a fully connected graph** where each token is a node.

    **Standard Graph Attention Network (GAT):**

    $$h_i' = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} W_v h_j$$

    Where $\mathcal{N}(i)$ is the neighborhood of node $i$ and $\alpha_{ij}$ are attention-based weights.

    **Transformer self-attention:**

    $$h_i' = \sum_{j=1}^{N} \alpha_{ij} W_V h_j$$

    Where $\alpha_{ij} = \text{softmax}_j(q_i^T k_j / \sqrt{d_k})$ — attention over ALL nodes (fully connected graph).

    ```
    GNN (sparse graph):           Transformer (complete graph):
    ○─○─○                         ○═══○═══○
    │ │                           ║ ╲ ║ ╱ ║
    ○─○                           ○═══○═══○
    Each node attends to           Every node attends to
    its neighbors only             every other node
    ```

    | Aspect | GNN | Transformer |
    |--------|-----|-------------|
    | Graph structure | Explicit (adjacency matrix) | Implicit (fully connected or masked) |
    | Attention scope | Neighbors only | All tokens |
    | Position encoding | Often none (permutation invariant) | Required (sinusoidal, RoPE) |
    | Inductive bias | Local structure matters | No structural bias |

    **Implications:**

    - Transformers are a special case of GNNs on complete graphs
    - Sparse attention patterns (Longformer, BigBird) make Transformers more like traditional GNNs
    - **Graph Transformers** explicitly combine both: use full attention but incorporate graph structure via edge features or structural encodings

---

### What is the impact of vocabulary size on Transformer performance and efficiency? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Vocabulary`, `Embedding`, `Tokenization`, `Efficiency` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    Vocabulary size affects multiple aspects of Transformer models:

    **Embedding layer parameters:** $V \times d_{model}$

    | Model | Vocab Size | d_model | Embedding Params |
    |-------|-----------|---------|-----------------|
    | GPT-2 | 50,257 | 1,600 | 80M |
    | LLaMA | 32,000 | 4,096 | 131M |
    | LLaMA 3 | 128,256 | 4,096 | 525M |
    | Gemma | 256,000 | 3,072 | 786M |

    **Trade-offs of vocabulary size:**

    | Larger Vocabulary | Smaller Vocabulary |
    |------------------|--------------------|
    | Fewer tokens per text (more efficient inference) | More tokens per text |
    | Better representation of rare words | Rare words split into many sub-tokens |
    | Larger embedding table (more parameters) | Smaller embedding table |
    | Better multilingual coverage | Poor on non-English languages |
    | Higher logit computation cost ($d_{model} \times V$ at output) | Lower logit cost |

    **Key insight — fertility (tokens per word):**

    ```
    English "artificial intelligence" with different vocab sizes:
    V=32K:  ["art", "ific", "ial", " intelligence"]     → 4 tokens
    V=128K: ["artificial", " intelligence"]              → 2 tokens

    2x fewer tokens = 2x faster inference, 2x longer effective context!
    ```

    **Modern trend:** Vocabulary sizes are increasing (LLaMA 3: 128K, Gemini: 256K) because the benefits of shorter sequences (faster inference, longer context) outweigh the slightly larger embedding layer.

---

### What is the connection between attention and kernel methods? Explain linear attention - Research, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Linear Attention`, `Kernel`, `Performer`, `Efficiency` | **Asked by:** Research Labs, Google, DeepMind

??? success "View Answer"

    Standard softmax attention can be viewed through a **kernel** lens, and this perspective enables linear-complexity approximations.

    **Standard attention as a kernel:**

    $$\text{Attn}(Q, K, V)_i = \frac{\sum_j \exp(q_i^T k_j / \sqrt{d}) \cdot v_j}{\sum_j \exp(q_i^T k_j / \sqrt{d})} = \frac{\sum_j \kappa(q_i, k_j) v_j}{\sum_j \kappa(q_i, k_j)}$$

    Where $\kappa(q, k) = \exp(q^T k / \sqrt{d})$ is the softmax kernel.

    **Linear attention** replaces this kernel with a decomposable feature map $\phi$:

    $$\kappa(q, k) \approx \phi(q)^T \phi(k)$$

    This allows rewriting attention as:

    $$\text{Attn}(Q, K, V)_i = \frac{\phi(q_i)^T \sum_j \phi(k_j) v_j^T}{\phi(q_i)^T \sum_j \phi(k_j)}$$

    The key: $\sum_j \phi(k_j) v_j^T$ can be **precomputed once** and shared across all queries!

    **Complexity comparison:**

    ```
    Standard attention:
    Q(N×d) × K^T(d×N) = S(N×N) × V(N×d) = O(N²d)
                           ^^^^
                           N×N matrix (bottleneck)

    Linear attention:
    K^T(d×N) × V(N×d) = KV(d×d)  ← precompute once: O(Nd²)
    Q(N×d) × KV(d×d) = O(Nd²)    ← for all queries
    Total: O(Nd²) instead of O(N²d)
    ```

    When $d \ll N$ (which is true for long sequences), this is a significant speedup.

    **Examples:** Performer (random features), Linear Transformers (ELU+1 feature map), cosFormer.

    **Limitation:** Linear attention approximations generally perform worse than exact attention for language modeling, which is why Flash Attention (exact but IO-efficient) is preferred in practice.

---

### How do modern LLMs handle long context windows (128K+ tokens)? What techniques enable this? - Anthropic, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Long Context`, `RoPE Scaling`, `YaRN`, `Ring Attention` | **Asked by:** Anthropic, Google, Meta

??? success "View Answer"

    Extending Transformer context windows from 4K to 128K+ tokens requires multiple complementary techniques:

    **1. RoPE scaling (position interpolation):**

    Instead of extrapolating to unseen positions, scale existing positions to fit within the training range:

    $$\theta'_i = \theta_i / s \quad \text{where } s = \text{target\_len} / \text{training\_len}$$

    Variants:
    - **Linear interpolation** (Meta): Simple division
    - **NTK-aware scaling**: Adjusts the base frequency of RoPE
    - **YaRN** (Yet Another RoPE extensioN): Combines NTK scaling with attention temperature scaling, achieves best extrapolation

    **2. Continued pre-training on long data:**

    Train for a small number of steps on long documents after initial training (e.g., LLaMA 2 Long: additional training on 32K+ documents).

    **3. Flash Attention:** Makes $O(N^2)$ attention feasible by eliminating memory bottleneck.

    **4. Ring Attention:** Distributes the sequence across multiple GPUs in a ring topology:

    ```
    GPU 1: tokens 1-32K      ←→  GPU 2: tokens 32K-64K
       ↑                              ↑
       └──── GPU 4: tokens 96K-128K ←→ GPU 3: tokens 64K-96K

    Each GPU holds its portion's KV cache.
    Attention blocks circulate around the ring.
    Total memory: O(N/P) per GPU for P GPUs.
    ```

    **5. Architectural choices:**

    - Sliding window attention (some layers local, some global)
    - Sparse attention patterns for very long sequences
    - Efficient KV cache management (GQA, MLA, quantized KV cache)

    **Current state-of-the-art context lengths:**

    | Model | Context Length |
    |-------|--------------|
    | GPT-4 Turbo | 128K |
    | Claude 3.5 | 200K |
    | Gemini 1.5 Pro | 1M-2M |
    | Command R+ | 128K |

---

### What is weight tying in Transformers? When and why is it used? - Google, Research Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Weight Tying`, `Embedding`, `Parameter Sharing` | **Asked by:** Google, Research Labs, Meta

??? success "View Answer"

    **Weight tying** shares the same weight matrix between the input embedding layer and the output (pre-softmax) projection:

    $$\text{Input:} \quad e = E[token\_id] \quad (E \in \mathbb{R}^{V \times d_{model}})$$

    $$\text{Output:} \quad \text{logits} = h \cdot E^T \quad (\text{reusing same } E)$$

    Instead of having separate input embedding ($E$) and output projection ($W_{out}$), we set $W_{out} = E^T$.

    **Benefits:**

    1. **Parameter reduction:** For LLaMA with $V = 32000$, $d = 4096$: saves 131M parameters
    2. **Regularization:** Forces the model to use a consistent representation space for input and output
    3. **Geometric consistency:** Tokens with similar meanings should be close in embedding space — tying ensures input similarity translates to output similarity

    **When it's used:**

    | Model | Weight Tying |
    |-------|-------------|
    | BERT | Yes |
    | GPT-2 | Yes |
    | T5 | Yes |
    | LLaMA 1 | No |
    | LLaMA 3 | No |
    | GPT-3+ | No (generally) |

    **Modern trend:** Larger models tend **not** to tie weights because:

    - The embedding parameter saving is proportionally small ($V \times d$ vs billions of parameters)
    - Untied embeddings allow the input and output spaces to specialize independently
    - Avoids the softmax bottleneck limitation imposed by shared representations

---

### What is the "Attention Is All You Need" warmup learning rate schedule mathematically and why does it work? - Google, DeepMind Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Learning Rate Schedule`, `Warmup`, `Noam`, `Optimization` | **Asked by:** Google, DeepMind, Research Labs

??? success "View Answer"

    The original Transformer uses the "Noam" schedule (named after the first author Noam Shazeer):

    $$lr(\text{step}) = d_{model}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup\_steps}^{-1.5})$$

    **Two phases:**

    During warmup ($\text{step} < \text{warmup\_steps}$):

    $$lr = d_{model}^{-0.5} \cdot \text{step} \cdot \text{warmup\_steps}^{-1.5}$$

    Linear increase from 0 to peak.

    After warmup ($\text{step} \geq \text{warmup\_steps}$):

    $$lr = d_{model}^{-0.5} \cdot \text{step}^{-0.5}$$

    Decay proportional to $1/\sqrt{\text{step}}$.

    **Why $d_{model}^{-0.5}$?** The peak learning rate scales inversely with model size — larger models use smaller learning rates, which is necessary because gradient magnitudes scale with dimension.

    ```python
    def noam_schedule(step, d_model=512, warmup_steps=4000):
        """Original Transformer learning rate schedule."""
        step = max(step, 1)  # Avoid division by zero
        return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

    # Peak LR at step=warmup_steps:
    # For d_model=512, warmup=4000: peak ≈ 0.00070
    ```

    **Modern replacement — cosine with warmup** (used by nearly all current LLMs) is simpler and often performs better, with explicit control over peak and minimum learning rates.

---

### What happens inside a Transformer when it processes a prompt? Trace a single forward pass - OpenAI, Anthropic Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Forward Pass`, `Inference`, `Step-by-Step`, `Internals` | **Asked by:** OpenAI, Anthropic, Google

??? success "View Answer"

    Let's trace the forward pass for a decoder-only model (like GPT/LLaMA) on the input "The cat":

    ```
    INPUT: "The cat"

    ┌─ Step 1: TOKENIZATION ─────────────────────────────────┐
    │ "The cat" → token IDs: [464, 3797]                     │
    └────────────────────────────────┬────────────────────────┘
                                     ↓
    ┌─ Step 2: EMBEDDING ────────────────────────────────────┐
    │ Token embedding: E[464] → [0.12, -0.34, ...]  (d=4096)│
    │                  E[3797] → [-0.56, 0.78, ...]          │
    │ + Positional encoding (RoPE applied later in attention) │
    │ → X ∈ ℝ^(2 × 4096)                                    │
    └────────────────────────────────┬────────────────────────┘
                                     ↓
    ┌─ Step 3: TRANSFORMER LAYERS (×32) ─────────────────────┐
    │                                                         │
    │  For each layer l = 1 to 32:                           │
    │                                                         │
    │  3a. RMSNorm(X)                                        │
    │                                                         │
    │  3b. Multi-Head Self-Attention:                         │
    │      Q = X·W_Q, K = X·W_K, V = X·W_V                  │
    │      Apply RoPE to Q, K                                │
    │      Apply causal mask (token 1 can't see token 2)     │
    │      Attn = softmax(QK^T/√d_k) · V                    │
    │      X = X + Attn·W_O          (residual connection)   │
    │                                                         │
    │  3c. RMSNorm(X)                                        │
    │                                                         │
    │  3d. Feed-Forward (SwiGLU):                            │
    │      X = X + W2·(Swish(W1·X) ⊙ W3·X)  (residual)     │
    │                                                         │
    └────────────────────────────────┬────────────────────────┘
                                     ↓
    ┌─ Step 4: FINAL NORM + OUTPUT ──────────────────────────┐
    │ X = RMSNorm(X)                                         │
    │ logits = X · E^T    (project to vocabulary, ℝ^(2×32K)) │
    │ Take logits for LAST position (token 2 = "cat")        │
    │ probs = softmax(logits[-1])                            │
    │ next_token = sample(probs)  → e.g., "sat"             │
    └────────────────────────────────────────────────────────┘
    ```

    **Key details:**

    - Only the **last position's** logits matter for next-token prediction
    - The causal mask ensures each position can only attend to itself and earlier positions
    - RoPE is applied inside the attention computation, not at the input
    - The residual stream carries information from embedding to output, with each layer making additive updates

---

### What is quantization in the context of LLM deployment? Compare INT8, INT4, and GPTQ - Meta, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Quantization`, `INT8`, `INT4`, `GPTQ`, `Deployment` | **Asked by:** Meta, Google, Amazon

??? success "View Answer"

    **Quantization** reduces the precision of model weights (and optionally activations) to decrease memory usage and increase inference speed.

    ```
    Original (FP16):    16 bits per weight → 140 GB for 70B model
    INT8 quantization:   8 bits per weight →  70 GB
    INT4 quantization:   4 bits per weight →  35 GB
    ```

    **Types of quantization:**

    | Method | Weights | Activations | Calibration | Quality |
    |--------|---------|-------------|-------------|---------|
    | **INT8 (LLM.int8())** | 8-bit | FP16 | Data-free | Minimal loss |
    | **GPTQ** | 4-bit | FP16 | Requires calibration data | Good |
    | **AWQ** | 4-bit | FP16 | Activation-aware | Better than GPTQ |
    | **GGUF (llama.cpp)** | 2-8 bit mixed | FP32/FP16 | Data-free | Varies |
    | **QLoRA** | 4-bit (NF4) | BF16 | None (for training) | Excellent |

    **GPTQ** (Frantar et al., 2023):

    - Post-training quantization using second-order information (approximate Hessian)
    - Quantizes weights one layer at a time, using calibration data to minimize output error
    - Compensates for quantization error in remaining weights (Optimal Brain Quantization)

    **Key insight — not all weights are equal:**

    ```
    Weight distribution:
    │     ╱╲
    │    ╱  ╲       Most weights are small
    │   ╱    ╲      and can be quantized
    │  ╱      ╲     aggressively
    │ ╱        ╲
    ─┴──●─●──●──●── A few "outlier" weights are
         ↑↑↑↑       very large and must stay
       outliers     high-precision (mixed precision)
    ```

    **LLM.int8()** keeps outlier features (>6σ) in FP16 and quantizes the rest to INT8 — a mixed-precision approach that maintains quality even for very large models.

---

### What is Differential Transformer and how does it improve attention? - Microsoft, Research Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Differential Attention`, `Noise Reduction`, `Modern` | **Asked by:** Microsoft, Research Labs

??? success "View Answer"

    **Differential Transformer** (Ye et al., 2024, Microsoft) modifies the attention mechanism to compute the **difference** between two softmax attention maps, effectively canceling out noise:

    $$\text{DiffAttn}(X) = \left(\text{softmax}\left(\frac{Q_1 K_1^T}{\sqrt{d}}\right) - \lambda \cdot \text{softmax}\left(\frac{Q_2 K_2^T}{\sqrt{d}}\right)\right) V$$

    Where $Q_1, Q_2, K_1, K_2$ are two sets of query/key projections and $\lambda$ is a learnable scalar.

    **Intuition:**

    ```
    Standard attention:        Differential attention:
    ┌─────────────────┐       ┌─────────────────┐
    │ Signal + Noise  │       │ (Signal + Noise) │
    │ ████░░░░░░░░░░  │       │  - λ · Noise     │
    │ (noisy weights) │       │ = Clean Signal    │
    └─────────────────┘       └─────────────────┘
    ```

    The second attention map acts as a "noise estimator" — by subtracting it, the model amplifies true signal and suppresses irrelevant attention patterns.

    **Benefits:**

    1. **Reduced attention noise:** Less attention to irrelevant tokens
    2. **Better long-context performance:** Cleaner attention maps degrade less with sequence length
    3. **Fewer hallucinations:** More focused attention leads to more faithful outputs
    4. **Fewer attention heads needed:** Each differential head is more expressive

    This is one of the newer architectural innovations gaining attention in the research community.

---

### How do Transformers compare to CNNs for computer vision? What is Vision Transformer (ViT)? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `ViT`, `Vision Transformer`, `CNN`, `Computer Vision` | **Asked by:** Google, Meta, OpenAI

??? success "View Answer"

    **Vision Transformer (ViT)** (Dosovitskiy et al., 2020, Google) applies the standard Transformer encoder to image patches:

    ```
    Image (224×224×3)
         │
         ▼
    Split into 16×16 patches → 196 patches (14×14 grid)
         │
         ▼
    Flatten each patch: 16×16×3 = 768 dims
         │
         ▼
    Linear projection → Patch embeddings (768 or 1024 dims)
         │
         ▼
    Prepend [CLS] token + Add positional embeddings
         │
         ▼
    Standard Transformer Encoder (12-24 layers)
         │
         ▼
    [CLS] token output → Classification head
    ```

    | Aspect | CNN (ResNet) | ViT |
    |--------|-------------|-----|
    | Inductive bias | Strong (locality, translation equivariance) | Minimal (must learn spatial relations) |
    | Data efficiency | Better with small datasets | Needs large datasets (JFT-300M) or strong augmentation |
    | Scalability | Diminishing returns beyond ~1B params | Scales well to 22B+ params |
    | Global context | Limited to receptive field | Full image from layer 1 |
    | Computational pattern | Local convolutions | Global attention |
    | Pre-training | ImageNet | ImageNet, JFT, or self-supervised (MAE, DINO) |

    **Key finding:** ViT underperforms CNNs on small datasets (ImageNet alone) because it lacks the spatial inductive bias. But when pre-trained on large datasets (300M+ images), ViT significantly outperforms CNNs, showing that **scale can replace inductive bias**.

    **Modern hybrids** combine both: early CNN layers for local features, then Transformer layers for global reasoning (e.g., CoAtNet, EfficientFormer).

---

### What is the relationship between temperature, top-k, and top-p during text generation? How do they interact? - OpenAI, Anthropic Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Temperature`, `Top-K`, `Top-P`, `Sampling`, `Generation` | **Asked by:** OpenAI, Anthropic, Google

??? success "View Answer"

    These three parameters control the **randomness and diversity** of generated text:

    **Temperature ($T$):** Scales logits before softmax.

    $$P(w) = \frac{\exp(z_w / T)}{\sum_i \exp(z_i / T)}$$

    - $T \rightarrow 0$: Greedy (deterministic, always picks highest probability)
    - $T = 1$: Standard softmax (original distribution)
    - $T > 1$: Flatter distribution (more random)

    **Top-k:** After temperature scaling, keep only the $k$ highest-probability tokens, renormalize.

    **Top-p (nucleus sampling):** Keep the smallest set of tokens whose cumulative probability $\geq p$, renormalize.

    ```
    Original logits: [cat: 5.0, dog: 3.0, fish: 2.0, xyz: 0.1, ...]

    After temperature=0.5:
    Softmax → [cat: 0.85, dog: 0.10, fish: 0.04, xyz: 0.00, ...]

    After top-k=3:
    → [cat: 0.86, dog: 0.10, fish: 0.04]  (renormalized)

    After top-p=0.9:
    → [cat: 0.90, dog: 0.10]  (cumsum: 0.85+0.10=0.95 ≥ 0.9, keep 2)
    ```

    **Typical settings:**

    | Use case | Temperature | Top-p | Top-k |
    |----------|-----------|-------|-------|
    | Factual Q&A | 0.0-0.3 | 1.0 | — |
    | Conversational | 0.7-0.9 | 0.9-0.95 | — |
    | Creative writing | 0.9-1.2 | 0.95-1.0 | — |
    | Code generation | 0.0-0.2 | 0.95 | — |

    **Key insight:** Temperature controls the shape of the distribution; top-k and top-p control the truncation. They are applied sequentially: temperature first, then top-k or top-p.

---

### What is the computational difference between the prefill and decode phases in LLM inference? - Anthropic, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Prefill`, `Decode`, `Inference`, `Latency`, `Throughput` | **Asked by:** Anthropic, Meta, Google

??? success "View Answer"

    LLM inference has two distinct phases with very different computational profiles:

    **Prefill phase** (processing the prompt):

    - Processes ALL prompt tokens in parallel (like training)
    - **Compute-bound:** Large matrix multiplications (batch of tokens × model weights)
    - Populates the KV cache for all prompt tokens
    - Time proportional to prompt length

    **Decode phase** (generating output):

    - Generates ONE token at a time
    - **Memory-bound:** Must load entire model weights from memory for each single token
    - GPU compute is vastly underutilized (computing with 1 token instead of thousands)
    - Time proportional to output length

    ```
    Prompt: "Explain quantum computing in 3 sentences" (8 tokens)
    Output: "Quantum computing uses..." (50 tokens)

    ┌─ PREFILL ───────────────────────────────┐
    │ Process 8 tokens in parallel             │
    │ Time: ~50ms                              │
    │ GPU utilization: HIGH (compute-bound)    │
    │ Arithmetic intensity: HIGH               │
    │ Fills KV cache for 8 positions           │
    └──────────────────────────────────────────┘

    ┌─ DECODE ────────────────────────────────┐
    │ Generate 50 tokens sequentially          │
    │ Time: ~2000ms (40ms per token)           │
    │ GPU utilization: LOW (memory-bound)      │
    │ Arithmetic intensity: LOW                │
    │ Each step: 1 new token, full model load  │
    └──────────────────────────────────────────┘
    ```

    **Key metrics:**

    | Metric | Prefill | Decode |
    |--------|---------|--------|
    | Bottleneck | Compute (FLOPS) | Memory bandwidth |
    | Tokens per step | All prompt tokens | 1 token |
    | GPU utilization | 50-80% | 5-15% |
    | Batching helps? | Yes (more compute) | Yes (amortize memory loads) |
    | Optimization | Tensor parallelism, Flash Attention | Batching, speculative decoding, quantization |

    **Time to First Token (TTFT)** = prefill time. **Time per Output Token (TPOT)** = decode time per token. These are the two key latency metrics for LLM serving.
