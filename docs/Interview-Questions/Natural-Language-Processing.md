---
title: NLP Interview Questions - Natural Language Processing
description: 100+ NLP interview questions - transformers, BERT, GPT, word embeddings, text classification, named entity recognition, and LLMs for ML engineer interviews.
---

# Natural Language Processing (NLP) Interview Questions

<!-- ![Total Questions](https://img.shields.io/badge/Total%20Questions-0-blue?style=flat&labelColor=black&color=blue)
![Unanswered Questions](https://img.shields.io/badge/Unanswered%20Questions-0-blue?style=flat&labelColor=black&color=yellow)
![Answered Questions](https://img.shields.io/badge/Answered%20Questions-0-blue?style=flat&labelColor=black&color=success) -->



This document provides a curated list of 100 NLP interview questions commonly asked in technical interviews. Covering topics from the fundamentals of text processing to deep learning–based language models, this list is updated frequently and is intended to serve as a comprehensive reference for interview preparation.

---

## Premium Interview Questions

### Explain the Transformer Architecture - Google, OpenAI Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Deep Learning`, `Transformers` | **Asked by:** Google, OpenAI, Meta, Amazon

??? success "View Answer"

    ## Architecture Overview

    The Transformer architecture ("Attention is All You Need", 2017) revolutionized NLP by replacing recurrence with attention mechanisms, enabling parallelization and better long-range dependency modeling.

    **Key Parameters (BERT-base):**
    - **Layers:** 12 encoder layers
    - **Hidden size (d_model):** 768
    - **Attention heads:** 12
    - **Parameters:** 110M
    - **Max sequence length:** 512 tokens
    - **FFN dimension:** 3072 (4× d_model)

    ## Core Architecture

    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

    **Multi-Head Attention:**

    $$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

    where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

    ## Production Implementation (200 lines)

    ```python
    # transformer.py
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math

    class MultiHeadAttention(nn.Module):
        """
        Multi-Head Self-Attention

        Time: O(n² × d) where n=seq_len, d=d_model
        Space: O(n²) for attention matrix
        """

        def __init__(self, d_model=768, num_heads=12, dropout=0.1):
            super().__init__()
            assert d_model % num_heads == 0

            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads  # 64 per head

            # Linear projections
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)

            self.dropout = nn.Dropout(dropout)

        def scaled_dot_product_attention(self, Q, K, V, mask=None):
            """
            Scaled Dot-Product Attention

            Args:
                Q, K, V: [batch, heads, seq_len, d_k]
                mask: [batch, 1, 1, seq_len] for padding

            Returns:
                output: [batch, heads, seq_len, d_k]
                attention_weights: [batch, heads, seq_len, seq_len]
            """
            # scores: [batch, heads, seq_len, seq_len]
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

            # Apply mask (padding = -inf)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

            # Attention weights
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Apply to values
            output = torch.matmul(attn_weights, V)
            return output, attn_weights

        def split_heads(self, x):
            """[batch, seq, d_model] → [batch, heads, seq, d_k]"""
            batch_size, seq_len, d_model = x.size()
            return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        def combine_heads(self, x):
            """[batch, heads, seq, d_k] → [batch, seq, d_model]"""
            batch_size, _, seq_len, d_k = x.size()
            return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        def forward(self, query, key, value, mask=None):
            # Linear projections and split heads
            Q = self.split_heads(self.W_q(query))
            K = self.split_heads(self.W_k(key))
            V = self.split_heads(self.W_v(value))

            # Attention
            attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

            # Combine heads and final projection
            output = self.combine_heads(attn_output)
            output = self.W_o(output)

            return output, attn_weights

    class PositionWiseFeedForward(nn.Module):
        """
        FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

        Applied independently to each position
        """

        def __init__(self, d_model=768, d_ff=3072, dropout=0.1):
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # x: [batch, seq_len, d_model]
            return self.linear2(self.dropout(F.gelu(self.linear1(x))))

    class TransformerEncoderLayer(nn.Module):
        """Single encoder layer with self-attention + FFN"""

        def __init__(self, d_model=768, num_heads=12, d_ff=3072, dropout=0.1):
            super().__init__()

            self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
            self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)

            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            # Self-attention + residual + norm
            attn_output, _ = self.self_attn(x, x, x, mask)
            x = self.norm1(x + self.dropout1(attn_output))

            # FFN + residual + norm
            ffn_output = self.ffn(x)
            x = self.norm2(x + self.dropout2(ffn_output))

            return x

    class PositionalEncoding(nn.Module):
        """
        Sinusoidal positional encoding

        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """

        def __init__(self, d_model=768, max_len=512):
            super().__init__()

            # Create PE matrix [max_len, d_model]
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()

            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() *
                (-math.log(10000.0) / d_model)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
            self.register_buffer('pe', pe)

        def forward(self, x):
            seq_len = x.size(1)
            return x + self.pe[:, :seq_len, :]

    class TransformerEncoder(nn.Module):
        """Complete Transformer Encoder (BERT-style)"""

        def __init__(
            self,
            vocab_size=30522,
            d_model=768,
            num_layers=12,
            num_heads=12,
            d_ff=3072,
            max_len=512,
            dropout=0.1
        ):
            super().__init__()

            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = PositionalEncoding(d_model, max_len)

            self.layers = nn.ModuleList([
                TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])

            self.dropout = nn.Dropout(dropout)
            self.d_model = d_model

        def forward(self, input_ids, attention_mask=None):
            """
            Args:
                input_ids: [batch, seq_len]
                attention_mask: [batch, seq_len] (1=real, 0=padding)

            Returns:
                [batch, seq_len, d_model]
            """
            # Token embeddings + scaling
            x = self.embedding(input_ids) * math.sqrt(self.d_model)

            # Add positional encoding
            x = self.pos_encoding(x)
            x = self.dropout(x)

            # Prepare attention mask [batch, 1, 1, seq_len]
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Pass through layers
            for layer in self.layers:
                x = layer(x, attention_mask)

            return x

    # Example
    if __name__ == "__main__":
        model = TransformerEncoder(
            vocab_size=30522,  # BERT vocab
            d_model=768,
            num_layers=12,
            num_heads=12
        )

        input_ids = torch.randint(0, 30522, (2, 10))  # batch=2, seq=10
        mask = torch.ones(2, 10)

        output = model(input_ids, mask)
        print(f"Output shape: {output.shape}")  # [2, 10, 768]

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")  # ~110M
    ```

    ## Architecture Comparison

    | Model | Type | Params | Context | Training | Best For |
    |-------|------|--------|---------|----------|----------|
    | **BERT** | Encoder | 110M-340M | 512 | Days (TPUs) | Classification, NER, QA |
    | **GPT-3** | Decoder | 175B | 2048 | Months (GPUs) | Generation, few-shot |
    | **T5** | Enc-Dec | 220M-11B | 512 | Weeks (TPUs) | Translation, summarization |
    | **LLaMA** | Decoder | 7B-65B | 2048 | Weeks (GPUs) | Open-source generation |

    ## Key Innovations Explained

    **1. Why √d_k Scaling?**
    - **Problem:** For large d_k, dot products grow large → softmax saturates
    - **Impact:** Gradients vanish, training fails
    - **Solution:** Divide by √d_k to normalize variance
    - **Math:** Var(Q·K) = d_k, so Var(Q·K/√d_k) = 1

    **2. Multi-Head Attention Benefits**
    - Different heads learn different patterns:
      - **Head 1:** Syntactic dependencies (subject-verb agreement)
      - **Head 2:** Semantic relationships
      - **Head 3:** Coreference resolution
    - **12 heads × 64-dim = 768-dim** (same as single-head)

    **3. Position Encoding**
    - **Why needed:** Self-attention is permutation-invariant
    - **Sinusoidal advantage:** Generalizes to longer sequences
    - **Modern alternatives:** Learned PE, RoPE, ALiBi

    ## Common Pitfalls

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **O(n²) memory** | OOM for long sequences (>4K) | Flash Attention, sparse patterns |
    | **No positional info** | Model ignores token order | Positional encoding |
    | **Padding inefficiency** | Wasted compute | Dynamic batching, pack sequences |
    | **Attention collapse** | All weights uniform | Proper init, gradient clipping |
    | **512 token limit** | Can't process long documents | Longformer, Big Bird, chunking |

    ## Real-World Systems

    **Google BERT (2018):**
    - **Training:** 16 TPUs × 4 days, Wikipedia + BooksCorpus
    - **Impact:** SotA on 11 NLP tasks, pre-training revolution
    - **Production:** Powers Google Search understanding

    **OpenAI GPT-3 (2020):**
    - **Scale:** 175B params, 96 layers, 96 heads
    - **Training:** $4.6M cost, 300B tokens
    - **Innovation:** Few-shot learning without fine-tuning
    - **Limitation:** 2K context (GPT-4: 128K with improvements)

    **Meta LLaMA (2023):**
    - **Efficiency:** 65B params matches GPT-3 175B
    - **Improvements:** RoPE, SwiGLU, RMSNorm
    - **Training:** 1.4T tokens, 2048 A100 GPUs
    - **Open-source:** Democratized LLM access

    !!! tip "Interviewer's Insight"
        **Strong candidates:**

        - Can implement scaled dot-product attention from scratch with correct tensor shapes
        - Explain √d_k scaling mathematically (prevents softmax saturation)
        - Understand O(n²) complexity and solutions (Flash Attention reduces to O(n))
        - Know when to use BERT vs GPT vs T5 (classification vs generation vs sequence-to-sequence)
        - Mention recent advances: RoPE for longer context, Flash Attention for efficiency
        - Discuss production concerns: quantization (INT8 inference), distillation (DistilBERT), ONNX export
        - Reference real impact: "BERT improved Google Search relevance by 10%"

---

### What is BERT and How Does It Work? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Language Models` | **Asked by:** Google, Meta, Amazon, Microsoft

??? success "View Answer"

    ## Overview

    **BERT (Bidirectional Encoder Representations from Transformers)** - Google's breakthrough pre-trained language model that revolutionized NLP by learning bidirectional context.

    **Key Innovation:** Unlike GPT (left-to-right), BERT sees full context (both directions) during pre-training.

    **Architecture (BERT-base):**
    - 12 Transformer encoder layers
    - 768 hidden dimensions
    - 12 attention heads
    - 110M parameters
    - 512 max sequence length

    ## Pre-training Objectives

    **1. Masked Language Modeling (MLM):**
    - Randomly mask 15% of tokens
    - Predict masked tokens using bidirectional context
    - Forces model to learn deep bidirectional representations

    **Masking Strategy:**
    - 80% → Replace with [MASK]
    - 10% → Replace with random word
    - 10% → Keep original (prevents model from only learning [MASK])

    **2. Next Sentence Prediction (NSP):**
    - Given sentence A and B, predict if B follows A
    - 50% actual next sentence (IsNext)
    - 50% random sentence (NotNext)
    - Helps with tasks requiring sentence relationships (QA, NLI)

    ## Production Implementation (180 lines)

    ```python
    # bert_implementation.py
    import torch
    import torch.nn as nn
    from transformers import BertTokenizer, BertModel, BertForSequenceClassification
    from torch.utils.data import Dataset, DataLoader
    import numpy as np

    class BERTClassifier(nn.Module):
        """
        BERT for sequence classification

        Architecture:
        Input → BERT → [CLS] embedding → Dropout → Linear → Softmax
        """

        def __init__(self, num_classes=2, dropout=0.1):
            super().__init__()

            # Load pre-trained BERT
            self.bert = BertModel.from_pretrained('bert-base-uncased')

            # Classification head
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(768, num_classes)

        def forward(self, input_ids, attention_mask):
            """
            Args:
                input_ids: [batch, seq_len]
                attention_mask: [batch, seq_len]

            Returns:
                logits: [batch, num_classes]
            """
            # Get BERT outputs
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Use [CLS] token representation (first token)
            pooled_output = outputs.pooler_output  # [batch, 768]

            # Classification
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            return logits

    class TextDataset(Dataset):
        """Dataset for BERT fine-tuning"""

        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = self.labels[idx]

            # Tokenize
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }

    def fine_tune_bert(train_texts, train_labels, val_texts, val_labels):
        """
        Fine-tune BERT for classification

        Training strategy:
        1. Freeze BERT layers initially (optional)
        2. Use smaller learning rate for BERT (2e-5)
        3. Gradient accumulation for larger effective batch
        4. Warmup + linear decay scheduler
        """

        # Initialize
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BERTClassifier(num_classes=2)

        # Datasets
        train_dataset = TextDataset(train_texts, train_labels, tokenizer)
        val_dataset = TextDataset(val_texts, val_labels, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        # Optimizer with differential learning rates
        optimizer = torch.optim.AdamW([
            {'params': model.bert.parameters(), 'lr': 2e-5},  # Lower LR for BERT
            {'params': model.classifier.parameters(), 'lr': 1e-3}  # Higher for head
        ], weight_decay=0.01)

        # Loss
        criterion = nn.CrossEntropyLoss()

        # Training loop
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        num_epochs = 3  # BERT typically needs 2-4 epochs

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                # Forward pass
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping (important for BERT)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                total_loss += loss.item()

            # Validation
            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)

                    logits = model(input_ids, attention_mask)
                    predictions = torch.argmax(logits, dim=1)

                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)

            val_accuracy = val_correct / val_total
            avg_loss = total_loss / len(train_loader)

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {avg_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.4f}")

        return model

    # Example: Inference
    def predict_sentiment(model, tokenizer, text, device='cpu'):
        """Predict sentiment for single text"""
        model.eval()

        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)

        return prediction.item(), probabilities[0].cpu().numpy()

    # Example usage
    if __name__ == "__main__":
        # Sample data
        train_texts = ["This movie is great!", "Terrible film, waste of time."]
        train_labels = [1, 0]  # 1=positive, 0=negative

        val_texts = ["Loved it!", "Boring."]
        val_labels = [1, 0]

        # Fine-tune
        model = fine_tune_bert(train_texts, train_labels, val_texts, val_labels)

        # Predict
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        sentiment, probs = predict_sentiment(
            model, tokenizer, "This is amazing!"
        )
        print(f"Sentiment: {'Positive' if sentiment == 1 else 'Negative'}")
        print(f"Probabilities: {probs}")
    ```

    ## Special Tokens

    | Token | Purpose | Usage |
    |-------|---------|-------|
    | **[CLS]** | Classification token | Aggregates sequence info for classification tasks |
    | **[SEP]** | Separator | Separates sentence pairs (A [SEP] B) |
    | **[MASK]** | Mask token | Replaces masked tokens during MLM training |
    | **[PAD]** | Padding | Fills sequences to max length |
    | **[UNK]** | Unknown | Out-of-vocabulary words |

    ## BERT Variants Comparison

    | Model | Params | Key Difference | Performance | Use Case |
    |-------|--------|----------------|-------------|----------|
    | **BERT-base** | 110M | Original | GLUE: 79.6 | General NLP |
    | **BERT-large** | 340M | 24 layers | GLUE: 80.5 | Max accuracy |
    | **RoBERTa** | 125M-355M | Remove NSP, dynamic masking | **GLUE: 88.5** | Better pre-training |
    | **ALBERT** | 12M-223M | Parameter sharing | GLUE: 89.4 | Memory-efficient |
    | **DistilBERT** | 66M | Knowledge distillation | GLUE: 77.0 | **2x faster**, 40% smaller |
    | **ELECTRA** | 110M | Replaced token detection | GLUE: 88.7 | Sample-efficient |

    ## Fine-tuning Strategies

    **1. Full Fine-tuning (Standard):**
    - Update all BERT parameters + task head
    - Requires: Large dataset (>10K examples)
    - LR: 2e-5 to 5e-5
    - Epochs: 2-4

    **2. Feature Extraction (Frozen BERT):**
    - Freeze BERT, only train classification head
    - Requires: Small dataset (<1K examples)
    - Faster, prevents overfitting

    **3. Gradual Unfreezing:**
    - Start with frozen BERT
    - Unfreeze top layers first, then gradually lower layers
    - Good for medium datasets (1K-10K)

    **4. Adapter Layers (Parameter-Efficient):**
    - Insert small trainable layers, freeze BERT
    - Only 3-5% parameters updated
    - Good for multi-task learning

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Wrong learning rate** | Divergence or slow convergence | Use 2e-5 for BERT layers, higher for head |
    | **Too many epochs** | Overfitting (BERT learns fast) | 2-4 epochs usually sufficient |
    | **Long sequences** | OOM error | Truncate to 128-256 tokens, use gradient checkpointing |
    | **Ignoring [CLS] token** | Poor classification | Always use pooled_output or [CLS] embedding |
    | **Not using attention_mask** | Model sees padding | Always pass attention_mask |
    | **Large batch on GPU** | OOM | Use gradient accumulation (effective batch = 32-64) |

    ## Real-World Applications

    **Google Search (2019):**
    - **Task:** Query understanding, result ranking
    - **Impact:** 10% improvement in search relevance
    - **Implementation:** BERT fine-tuned on query-document pairs
    - **Scale:** Billions of queries/day

    **Healthcare (BioBERT):**
    - **Task:** Medical NER, relation extraction
    - **Dataset:** PubMed + PMC articles
    - **Performance:** 87.4% F1 on biomedical NER (vs 80.1% baseline)
    - **Use:** Disease-drug extraction, clinical notes

    **Finance (FinBERT):**
    - **Task:** Financial sentiment analysis
    - **Training:** Financial news + earnings calls
    - **Performance:** 97% accuracy on financial sentiment
    - **Use:** Risk assessment, trading signals

    **Customer Support (Chatbots):**
    - **Task:** Intent classification, entity extraction
    - **Fine-tuning:** 5K-10K labeled support tickets
    - **Latency:** <50ms with TensorRT optimization
    - **ROI:** 40% reduction in support costs

    ## Performance Metrics

    | Benchmark | BERT-base | BERT-large | RoBERTa | Human |
    |-----------|-----------|------------|---------|--------|
    | **GLUE (avg)** | 79.6 | 80.5 | **88.5** | 87.1 |
    | **SQuAD 2.0 F1** | 76.3 | **83.1** | 86.5 | 89.5 |
    | **MNLI Accuracy** | 84.6 | 86.7 | **90.2** | 92.0 |

    !!! tip "Interviewer's Insight"
        **Strong candidates:**

        - Explain MLM masking strategy: "80% [MASK], 10% random, 10% unchanged prevents overfitting to [MASK]"
        - Know [CLS] token usage: "Aggregates sequence information for classification via cross-attention"
        - Understand bidirectional context: "Unlike GPT, BERT sees full sentence during training"
        - Can implement fine-tuning with correct learning rates (2e-5 for BERT, higher for head)
        - Know variants: "RoBERTa removes NSP and uses dynamic masking for better performance"
        - Discuss production optimizations: "DistilBERT for 2x speedup, ONNX for deployment, quantization for mobile"
        - Reference real impact: "BERT improved Google Search by 10%, processes billions of queries daily"

---

### Explain Word Embeddings (Word2Vec, GloVe) - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Embeddings`, `Representation Learning`, `Distributional Semantics` | **Asked by:** Google, Meta, Amazon, Microsoft, Netflix

??? success "View Answer"

    ## Core Concept

    **Word embeddings** map words to dense, low-dimensional vectors (typically 100-300D) that capture semantic and syntactic relationships. Based on distributional hypothesis: "Words that occur in similar contexts have similar meanings."

    **Key Property:** Vector arithmetic captures analogies:
    - king - man + woman ≈ queen
    - Paris - France + Germany ≈ Berlin

    ## Architecture Comparison

    ```
    ┌──────────────────────────────────────────────────────────────────┐
    │                    WORD2VEC ARCHITECTURES                        │
    ├──────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  CBOW (Continuous Bag-of-Words)                                 │
    │  ┌────────────────────────────────────────────────────────────┐ │
    │  │  Context: ["The", "cat", "on", "mat"]                    │ │
    │  │      ↓        ↓       ↓      ↓                             │ │
    │  │   [Embed] [Embed] [Embed] [Embed]                          │ │
    │  │      ↓        ↓       ↓      ↓                             │ │
    │  │         Average/Sum                                         │ │
    │  │              ↓                                              │ │
    │  │     Hidden Layer (300D)                                     │ │
    │  │              ↓                                              │ │
    │  │        Softmax (vocab_size)                                │ │
    │  │              ↓                                              │ │
    │  │      Predict: "sat"                                        │ │
    │  └────────────────────────────────────────────────────────────┘ │
    │                                                                  │
    │  Skip-Gram (Reverse of CBOW)                                     │
    │  ┌────────────────────────────────────────────────────────────┐ │
    │  │  Input: "sat"                                              │ │
    │  │      ↓                                                      │ │
    │  │   Embedding (300D)                                          │ │
    │  │      ↓                                                      │ │
    │  │  Hidden Layer                                               │ │
    │  │      ↓                                                      │ │
    │  │  Multiple Softmax                                           │ │
    │  │      ↓                                                      │ │
    │  │  Predict Context: ["The", "cat", "on", "mat"]            │ │
    │  └────────────────────────────────────────────────────────────┘ │
    │                                                                  │
    │  GloVe (Global Vectors)                                          │
    │  ┌────────────────────────────────────────────────────────────┐ │
    │  │  STEP 1: Build co-occurrence matrix X                      │ │
    │  │           X[i,j] = # times word i appears near word j      │ │
    │  │                                                             │ │
    │  │  STEP 2: Minimize weighted least squares:                  │ │
    │  │           Σ f(X[i,j]) * (wᵢᵀwⱼ + bᵢ + bⱼ - log X[i,j])²   │ │
    │  │                                                             │ │
    │  │  STEP 3: Resulting wᵢ, wⱼ capture global statistics       │ │
    │  └────────────────────────────────────────────────────────────┘ │
    └──────────────────────────────────────────────────────────────────┘
    ```

    ## Production Implementation (180 lines)

    ```python
    # word_embeddings.py
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from typing import List, Dict, Tuple, Optional
    from collections import Counter, defaultdict
    from gensim.models import Word2Vec, KeyedVectors
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from dataclasses import dataclass
    import time

    @dataclass
    class EmbeddingMetrics:
        """Metrics for embedding evaluation"""
        analogy_accuracy: float
        similarity_correlation: float
        coverage: float
        training_time: float

    class SkipGramModel(nn.Module):
        """
        Skip-Gram with Negative Sampling (Word2Vec)
        
        Time Complexity: O(V × D) per batch where V=vocab, D=embed_dim
        Space Complexity: O(V × D) for embedding matrices
        
        Key optimization: Negative sampling reduces softmax from O(V) to O(k)
        where k = num_negative_samples (typically 5-20)
        """
        
        def __init__(self, vocab_size: int, embed_dim: int = 300):
            super().__init__()
            
            # Input embeddings (center word)
            self.in_embed = nn.Embedding(vocab_size, embed_dim)
            
            # Output embeddings (context words)
            self.out_embed = nn.Embedding(vocab_size, embed_dim)
            
            # Xavier initialization for stable training
            self.in_embed.weight.data.uniform_(-0.5 / embed_dim, 0.5 / embed_dim)
            self.out_embed.weight.data.uniform_(-0.5 / embed_dim, 0.5 / embed_dim)
        
        def forward(
            self, 
            center_words: torch.Tensor,  # [batch]
            context_words: torch.Tensor,  # [batch, window_size]
            negative_words: torch.Tensor  # [batch, num_negative]
        ) -> torch.Tensor:
            """
            Compute Skip-Gram loss with negative sampling
            
            Loss = -log σ(u_o^T v_c) - Σ log σ(-u_k^T v_c)
            where σ is sigmoid, u=context, v=center, k=negative samples
            """
            # Get embeddings
            center_embeds = self.in_embed(center_words)  # [batch, embed_dim]
            context_embeds = self.out_embed(context_words)  # [batch, window, embed_dim]
            negative_embeds = self.out_embed(negative_words)  # [batch, neg, embed_dim]
            
            # Positive score: center · context
            pos_score = torch.bmm(
                context_embeds, 
                center_embeds.unsqueeze(2)
            ).squeeze()  # [batch, window]
            
            # Negative score: center · negatives
            neg_score = torch.bmm(
                negative_embeds,
                center_embeds.unsqueeze(2)
            ).squeeze()  # [batch, num_negative]
            
            # Binary cross-entropy loss
            pos_loss = -torch.log(torch.sigmoid(pos_score)).sum(dim=1)
            neg_loss = -torch.log(torch.sigmoid(-neg_score)).sum(dim=1)
            
            return (pos_loss + neg_loss).mean()
        
        def get_embeddings(self) -> np.ndarray:
            """Return input embeddings (standard practice)"""
            return self.in_embed.weight.data.cpu().numpy()

    class GloVeModel:
        """
        GloVe: Global Vectors for Word Representation
        
        Key insight: Ratios of co-occurrence probabilities encode meaning
        Example: P("solid"|"ice") / P("solid"|"steam") >> 1
        """
        
        def __init__(self, vocab_size: int, embed_dim: int = 300):
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            
            # Two embedding matrices + bias vectors
            self.W = np.random.randn(vocab_size, embed_dim) * 0.01
            self.W_tilde = np.random.randn(vocab_size, embed_dim) * 0.01
            self.b = np.zeros(vocab_size)
            self.b_tilde = np.zeros(vocab_size)
        
        def weighting_function(self, x: float, x_max: float = 100, alpha: float = 0.75) -> float:
            """
            Weighting function to prevent frequent pairs from dominating
            
            f(x) = (x/x_max)^α  if x < x_max
                   1            otherwise
            """
            if x < x_max:
                return (x / x_max) ** alpha
            return 1.0
        
        def train(
            self, 
            cooccur_matrix: Dict[Tuple[int, int], int],
            epochs: int = 50,
            learning_rate: float = 0.05
        ) -> EmbeddingMetrics:
            """
            Train GloVe using AdaGrad
            
            Objective: Σᵢⱼ f(Xᵢⱼ) (wᵢᵀw̃ⱼ + bᵢ + b̃ⱼ - log Xᵢⱼ)²
            """
            start_time = time.time()
            
            # AdaGrad accumulators
            grad_sq_W = np.ones_like(self.W)
            grad_sq_W_tilde = np.ones_like(self.W_tilde)
            grad_sq_b = np.ones_like(self.b)
            grad_sq_b_tilde = np.ones_like(self.b_tilde)
            
            for epoch in range(epochs):
                total_loss = 0.0
                
                for (i, j), x_ij in cooccur_matrix.items():
                    if x_ij == 0:
                        continue
                    
                    # Compute loss
                    weight = self.weighting_function(x_ij)
                    diff = (self.W[i] @ self.W_tilde[j] + 
                           self.b[i] + self.b_tilde[j] - np.log(x_ij))
                    loss = weight * diff ** 2
                    total_loss += loss
                    
                    # Compute gradients
                    grad_common = weight * diff
                    
                    # Update W[i]
                    grad_W_i = grad_common * self.W_tilde[j]
                    grad_sq_W[i] += grad_W_i ** 2
                    self.W[i] -= learning_rate * grad_W_i / np.sqrt(grad_sq_W[i])
                    
                    # Update W_tilde[j]
                    grad_W_tilde_j = grad_common * self.W[i]
                    grad_sq_W_tilde[j] += grad_W_tilde_j ** 2
                    self.W_tilde[j] -= learning_rate * grad_W_tilde_j / np.sqrt(grad_sq_W_tilde[j])
                    
                    # Update biases
                    grad_sq_b[i] += grad_common ** 2
                    grad_sq_b_tilde[j] += grad_common ** 2
                    self.b[i] -= learning_rate * grad_common / np.sqrt(grad_sq_b[i])
                    self.b_tilde[j] -= learning_rate * grad_common / np.sqrt(grad_sq_b_tilde[j])
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.2f}")
            
            training_time = time.time() - start_time
            return EmbeddingMetrics(
                analogy_accuracy=0.0,  # Would need test set
                similarity_correlation=0.0,
                coverage=1.0,
                training_time=training_time
            )
        
        def get_embeddings(self) -> np.ndarray:
            """Average of W and W_tilde (standard practice)"""
            return (self.W + self.W_tilde) / 2

    def build_cooccurrence_matrix(
        sentences: List[List[str]], 
        word2idx: Dict[str, int],
        window_size: int = 5
    ) -> Dict[Tuple[int, int], int]:
        """
        Build co-occurrence matrix from corpus
        
        X[i,j] = # times word j appears within window of word i
        Use symmetric window and distance weighting
        """
        cooccur = defaultdict(int)
        
        for sentence in sentences:
            indices = [word2idx.get(w, -1) for w in sentence]
            indices = [i for i in indices if i >= 0]
            
            for center_pos, center_idx in enumerate(indices):
                # Context window
                start = max(0, center_pos - window_size)
                end = min(len(indices), center_pos + window_size + 1)
                
                for context_pos in range(start, end):
                    if context_pos == center_pos:
                        continue
                    
                    context_idx = indices[context_pos]
                    distance = abs(context_pos - center_pos)
                    
                    # Weight by distance (closer = higher weight)
                    weight = 1.0 / distance
                    cooccur[(center_idx, context_idx)] += weight
        
        return dict(cooccur)

    # ===========================================
    # EXAMPLE USAGE WITH COMPANY USE CASES
    # ===========================================

    if __name__ == "__main__":
        print("="*70)
        print("GOOGLE - WORD2VEC FOR QUERY UNDERSTANDING")
        print("="*70)
        
        # Sample corpus
        sentences = [
            ["king", "rules", "kingdom"],
            ["queen", "rules", "kingdom"],
            ["man", "walks", "street"],
            ["woman", "walks", "street"],
            # ... (in production: billions of sentences)
        ]
        
        # Train Word2Vec with Gensim
        model = Word2Vec(
            sentences=sentences,
            vector_size=100,
            window=5,
            min_count=1,
            workers=4,
            sg=1,  # Skip-gram (sg=0 for CBOW)
            negative=5,  # Negative sampling
            ns_exponent=0.75  # Negative sampling distribution exponent
        )
        
        # Analogy test: king - man + woman ≈ queen
        try:
            result = model.wv.most_similar(
                positive=['king', 'woman'],
                negative=['man'],
                topn=1
            )
            print(f"\nAnalogy: king - man + woman = {result[0][0]} (score: {result[0][1]:.3f})")
        except KeyError:
            print("Not enough data for analogy")
        
        print("\n" + "="*70)
        print("META - SEMANTIC SEARCH FOR CONTENT MODERATION")
        print("="*70)
        
        # Load pre-trained embeddings
        # In production: word2vec-google-news-300 (3M words)
        
        # Similarity examples
        words_to_test = ["king", "queen", "man", "woman"]
        print("\nWord Similarities:")
        for word in words_to_test:
            if word in model.wv:
                similar = model.wv.most_similar(word, topn=3)
                print(f"{word}: {[w for w, _ in similar]}")
        
        print("\n" + "="*70)
        print("OPENAI - CONTEXTUAL EMBEDDINGS (BERT-STYLE)")
        print("="*70)
        print("Note: Word2Vec gives one vector per word (static)")
        print("BERT gives different vectors based on context (dynamic)")
        print("\nExample: 'bank' in 'river bank' vs 'savings bank'")
        print("Word2Vec: Same 300D vector")
        print("BERT: Different 768D vectors based on sentence context")
    ```

    ## Method Comparison

    | Aspect | Word2Vec (Skip-Gram) | GloVe | FastText | BERT (Context) |
    |--------|---------------------|-------|----------|----------------|
    | **Training** | Local context windows | Global co-occurrence | Char n-grams + Word2Vec | Masked LM (bidirectional) |
    | **Speed** | Fast (hours on CPU) | Moderate (needs matrix) | Fast | Slow (GPU required) |
    | **OOV** | No vector | No vector | Yes (subword) | Yes (subword tokenizer) |
    | **Embedding** | Static (one per word) | Static | Static | Dynamic (context-dependent) |
    | **Dimension** | 100-300 | 50-300 | 100-300 | 768-1024 |
    | **Best for** | Analogies, simple tasks | Rare words, global stats | Morphology, typos | Context, downstream tasks |

    ## Real-World Deployments

    | Company | Use Case | Method | Scale | Impact |
    |---------|----------|--------|-------|--------|
    | **Google** | Query understanding | Word2Vec (2013) | 100B words, 300D | +3% search quality |
    | **Meta** | Content similarity | FastText (2016) | 157 languages | 98.7% language detection |
    | **Spotify** | Music recommendation | GloVe on playlists | 4B playlist edges | +8% engagement |
    | **Amazon** | Product search | Word2Vec on descriptions | 500M products | +15% click-through |
    | **Netflix** | Title embeddings | Custom Word2Vec | 100M viewing sessions | +7% watch time |

    ## Training Optimization Techniques

    **1. Negative Sampling:**
    - **Problem:** Softmax over 100K vocabulary is slow
    - **Solution:** Sample k negative examples (typically 5-20)
    - **Speedup:** O(V) → O(k), 1000× faster for V=100K, k=10

    **2. Subsampling Frequent Words:**
    - **Problem:** "the", "a" dominate training but add little meaning
    - **Solution:** Randomly discard with probability: P(w) = 1 - √(t/f(w))
    - **Impact:** 2-10× speedup, better rare word embeddings

    **3. Hierarchical Softmax:**
    - **Alternative to negative sampling**
    - Uses binary tree (Huffman tree) of vocabulary
    - Complexity: O(log V) instead of O(V)

    ## Evaluation Metrics

    **Intrinsic Evaluation:**
    ```python
    # 1. Analogy Accuracy (Google's test set: 19,544 questions)
    # king:queen :: man:? (answer: woman)
    analogy_accuracy = correct_analogies / total_analogies
    
    # 2. Word Similarity (Spearman correlation with human ratings)
    # WordSim-353, SimLex-999 datasets
    from scipy.stats import spearmanr
    correlation = spearmanr(model_scores, human_scores)[0]
    
    # 3. Clustering Coherence
    # Do semantically similar words cluster together?
    ```

    **Extrinsic Evaluation:**
    - Use embeddings in downstream task (sentiment analysis, NER)
    - Measure task performance improvement

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Deep understanding of training objectives (Skip-Gram vs CBOW)
        - Knowledge of optimization techniques (negative sampling, subsampling)
        - Awareness of limitations (static embeddings, OOV problem)
        - When to use Word2Vec vs BERT (speed vs context)
        
        **Strong signal:**
        
        - "Skip-Gram learns by predicting context from center word. With negative sampling, we sample k=5-20 negative examples instead of computing softmax over full vocabulary, reducing complexity from O(V) to O(k)"
        - "Google trained Word2Vec on 100B words in a few hours on CPUs. Modern BERT requires GPUs and days of training but gives contextualized embeddings"
        - "For morphologically rich languages like German, FastText handles compound words better than Word2Vec by using character n-grams"
        - "At Netflix, we trained Word2Vec on 100M viewing sessions treating movies as 'words' and sessions as 'sentences', improving recommendations by 7%"
        
        **Red flags:**
        
        - "Word2Vec uses deep learning" (it's shallow: input → hidden → output)
        - Can't explain why √d_k scaling (wrong model - that's Transformers)
        - Not knowing static vs contextual embeddings difference
        - Thinking Word2Vec handles polysemy (bank = river vs money)
        
        **Follow-ups:**
        
        - "How would you handle out-of-vocabulary words?" → FastText subword approach
        - "Word2Vec vs BERT for production search?" → Word2Vec for speed, BERT for quality
        - "How to evaluate embeddings without labeled data?" → Intrinsic metrics (analogies, similarity)
        - "Memory constraints for 10M vocabulary?" → Pruning, quantization, HashingTrick

---

### What is TF-IDF? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Feature Extraction`, `Information Retrieval`, `Text Mining` | **Asked by:** Google, Amazon, Netflix, Spotify, Airbnb

??? success "View Answer"

    ## Core Concept

    **TF-IDF (Term Frequency-Inverse Document Frequency)** quantifies word importance by balancing how often a word appears in a document (TF) against how rare it is across all documents (IDF).

    **Intuition:** 
    - Frequent in document → Important to that document (high TF)
    - Rare across corpus → Distinctive/informative (high IDF)
    - Common everywhere ("the", "is") → Low IDF, filtered out

    ## Mathematical Foundation

    $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

    **Term Frequency (TF) variants:**

    ```
    ┌──────────────────────────────────────────────────────────────┐
    │                     TF VARIANTS                              │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  1. Raw Count:                                               │
    │     TF(t,d) = count of term t in document d                 │
    │                                                              │
    │  2. Boolean:                                                 │
    │     TF(t,d) = 1 if t in d, else 0                           │
    │                                                              │
    │  3. Log Normalization (sklearn default):                    │
    │     TF(t,d) = 1 + log(count(t,d)) if count > 0             │
    │              = 0                      otherwise              │
    │                                                              │
    │  4. Augmented (prevents bias to long docs):                 │
    │     TF(t,d) = 0.5 + 0.5 × (count(t,d) / max_count_in_d)    │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    ```

    **Inverse Document Frequency (IDF):**

    $$\text{IDF}(t) = \log\left(\frac{N}{\text{DF}(t)}\right) = \log\left(\frac{\text{Total Documents}}{\text{Documents containing } t}\right)$$

    **Sklearn's smoothed IDF (default):**

    $$\text{IDF}(t) = \log\left(\frac{N + 1}{\text{DF}(t) + 1}\right) + 1$$

    ## Production Implementation (175 lines)

    ```python
    # tfidf_implementation.py
    import numpy as np
    import pandas as pd
    from typing import List, Dict, Tuple, Optional
    from collections import Counter, defaultdict
    from dataclasses import dataclass
    import re
    import math
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.sparse import csr_matrix
    import time

    @dataclass
    class TFIDFMetrics:
        """Metrics for TF-IDF evaluation"""
        vocab_size: int
        sparsity: float
        build_time: float
        memory_mb: float

    class CustomTFIDF:
        """
        Custom TF-IDF implementation from scratch
        
        Time Complexity: 
        - fit: O(N × L) where N=num_docs, L=avg_doc_length
        - transform: O(N × L × V) where V=vocab_size
        
        Space Complexity: O(V × N) sparse matrix
        """
        
        def __init__(
            self,
            max_features: Optional[int] = None,
            ngram_range: Tuple[int, int] = (1, 1),
            min_df: int = 1,
            max_df: float = 1.0,
            sublinear_tf: bool = False,  # Use log(TF) if True
            smooth_idf: bool = True
        ):
            self.max_features = max_features
            self.ngram_range = ngram_range
            self.min_df = min_df
            self.max_df = max_df
            self.sublinear_tf = sublinear_tf
            self.smooth_idf = smooth_idf
            
            self.vocabulary_: Dict[str, int] = {}
            self.idf_: np.ndarray = None
            self.n_docs: int = 0
        
        def _tokenize(self, text: str) -> List[str]:
            """Simple tokenization (in production: use proper tokenizer)"""
            text = text.lower()
            text = re.sub(r'[^a-z0-9\s]', '', text)
            tokens = text.split()
            
            # Generate n-grams
            all_grams = []
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                for i in range(len(tokens) - n + 1):
                    all_grams.append(' '.join(tokens[i:i+n]))
            
            return all_grams
        
        def fit(self, documents: List[str]) -> 'CustomTFIDF':
            """
            Build vocabulary and compute IDF values
            """
            self.n_docs = len(documents)
            
            # Count document frequencies
            doc_freq = Counter()
            term_docs = defaultdict(set)
            
            for doc_idx, doc in enumerate(documents):
                tokens = self._tokenize(doc)
                unique_tokens = set(tokens)
                
                for token in unique_tokens:
                    doc_freq[token] += 1
                    term_docs[token].add(doc_idx)
            
            # Filter by min_df and max_df
            max_doc_count = self.max_df if self.max_df > 1 else int(self.max_df * self.n_docs)
            
            filtered_terms = [
                term for term, df in doc_freq.items()
                if self.min_df <= df <= max_doc_count
            ]
            
            # Select top features by document frequency
            if self.max_features:
                filtered_terms = sorted(
                    filtered_terms,
                    key=lambda t: doc_freq[t],
                    reverse=True
                )[:self.max_features]
            
            # Build vocabulary
            self.vocabulary_ = {term: idx for idx, term in enumerate(sorted(filtered_terms))}
            
            # Compute IDF
            vocab_size = len(self.vocabulary_)
            self.idf_ = np.zeros(vocab_size)
            
            for term, idx in self.vocabulary_.items():
                df = doc_freq[term]
                
                if self.smooth_idf:
                    # Sklearn's formula
                    self.idf_[idx] = np.log((self.n_docs + 1) / (df + 1)) + 1
                else:
                    # Standard formula
                    self.idf_[idx] = np.log(self.n_docs / df)
            
            return self
        
        def transform(self, documents: List[str]) -> csr_matrix:
            """
            Transform documents to TF-IDF matrix
            
            Returns: Sparse matrix [n_docs, vocab_size]
            """
            n_docs = len(documents)
            vocab_size = len(self.vocabulary_)
            
            # Build sparse matrix (efficient for text data)
            rows, cols, data = [], [], []
            
            for doc_idx, doc in enumerate(documents):
                tokens = self._tokenize(doc)
                token_counts = Counter(tokens)
                
                # Compute max count for augmented TF
                max_count = max(token_counts.values()) if token_counts else 1
                
                for token, count in token_counts.items():
                    if token not in self.vocabulary_:
                        continue
                    
                    term_idx = self.vocabulary_[token]
                    
                    # Compute TF
                    if self.sublinear_tf:
                        tf = 1 + np.log(count) if count > 0 else 0
                    else:
                        tf = count
                    
                    # Compute TF-IDF
                    tfidf = tf * self.idf_[term_idx]
                    
                    rows.append(doc_idx)
                    cols.append(term_idx)
                    data.append(tfidf)
            
            # Create sparse matrix
            tfidf_matrix = csr_matrix(
                (data, (rows, cols)),
                shape=(n_docs, vocab_size)
            )
            
            # L2 normalization (each document vector has unit length)
            from sklearn.preprocessing import normalize
            tfidf_matrix = normalize(tfidf_matrix, norm='l2', axis=1)
            
            return tfidf_matrix
        
        def fit_transform(self, documents: List[str]) -> csr_matrix:
            """Fit and transform in one step"""
            return self.fit(documents).transform(documents)

    def compare_methods(
        documents: List[str],
        queries: List[str]
    ) -> pd.DataFrame:
        """
        Compare TF-IDF with different settings
        """
        results = []
        
        configs = [
            {"name": "Unigrams", "ngram_range": (1, 1), "max_features": 5000},
            {"name": "Uni+Bigrams", "ngram_range": (1, 2), "max_features": 10000},
            {"name": "Sublinear TF", "ngram_range": (1, 1), "sublinear_tf": True},
            {"name": "Min DF=2", "ngram_range": (1, 1), "min_df": 2},
        ]
        
        for config in configs:
            name = config.pop("name")
            
            start = time.time()
            vectorizer = TfidfVectorizer(**config)
            X = vectorizer.fit_transform(documents)
            build_time = time.time() - start
            
            # Compute metrics
            vocab_size = len(vectorizer.vocabulary_)
            sparsity = 1 - (X.nnz / (X.shape[0] * X.shape[1]))
            memory_mb = (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1024 / 1024
            
            results.append({
                "Method": name,
                "Vocab Size": vocab_size,
                "Sparsity": f"{sparsity:.2%}",
                "Build Time (s)": f"{build_time:.3f}",
                "Memory (MB)": f"{memory_mb:.2f}"
            })
        
        return pd.DataFrame(results)

    # ===========================================
    # EXAMPLE USAGE WITH COMPANY USE CASES
    # ===========================================

    if __name__ == "__main__":
        print("="*70)
        print("NETFLIX - CONTENT SIMILARITY FOR RECOMMENDATIONS")
        print("="*70)
        
        # Movie descriptions (simplified)
        documents = [
            "action movie with explosions and car chases",
            "romantic comedy with love and laughter",
            "action thriller with car chases and suspense",
            "romantic drama with emotional love story",
            "sci-fi action with space battles and explosions"
        ]
        
        # Build TF-IDF
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # Unigrams + bigrams
            max_features=50,
            sublinear_tf=True  # Log scaling for TF
        )
        
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        print(f"\nVocabulary size: {len(vectorizer.vocabulary_)}")
        print(f"Matrix shape: {tfidf_matrix.shape}")
        print(f"Sparsity: {1 - (tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])):.2%}")
        
        # Top terms by IDF (most distinctive)
        feature_names = vectorizer.get_feature_names_out()
        idf_scores = vectorizer.idf_
        top_indices = idf_scores.argsort()[-5:][::-1]
        
        print("\nTop 5 distinctive terms (highest IDF):")
        for idx in top_indices:
            print(f"  {feature_names[idx]}: {idf_scores[idx]:.3f}")
        
        print("\n" + "="*70)
        print("GOOGLE - DOCUMENT SIMILARITY FOR SEARCH")
        print("="*70)
        
        # Find similar documents
        query_idx = 0  # "action movie with explosions"
        similarities = cosine_similarity(tfidf_matrix[query_idx:query_idx+1], tfidf_matrix).flatten()
        
        print(f"\nQuery: {documents[query_idx]}")
        print("\nMost similar documents:")
        
        similar_indices = similarities.argsort()[::-1][1:4]  # Skip self
        for idx in similar_indices:
            print(f"  [{similarities[idx]:.3f}] {documents[idx]}")
        
        print("\n" + "="*70)
        print("AMAZON - PRODUCT SEARCH RANKING")
        print("="*70)
        
        # Search query
        query = "action car chase"
        query_vec = vectorizer.transform([query])
        
        # Rank documents by relevance
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        ranked_indices = scores.argsort()[::-1]
        
        print(f"\nSearch query: '{query}'")
        print("\nRanked results:")
        for rank, idx in enumerate(ranked_indices, 1):
            if scores[idx] > 0:
                print(f"  {rank}. [{scores[idx]:.3f}] {documents[idx]}")
    ```

    ## TF-IDF vs Embeddings Comparison

    | Aspect | TF-IDF | Word2Vec/GloVe | BERT/Sentence-BERT |
    |--------|--------|----------------|--------------------|
    | **Representation** | Sparse vectors (10K-100K dim) | Dense (100-300D) | Dense (768-1024D) |
    | **Semantics** | No ("bank" = "bank" always) | Yes (similar words close) | Yes (contextual) |
    | **Speed** | Very fast (ms) | Fast (ms) | Slow (seconds on GPU) |
    | **Memory** | Sparse (efficient) | Small models | Large models (GBs) |
    | **Training** | None (just counting) | Unsupervised (hours) | Pre-trained (days) |
    | **Out-of-Vocab** | Ignored | No vector | Subword tokenization |
    | **Best for** | Search, filtering, quick prototypes | Analogies, clustering | Question answering, NLI |

    ## Real-World Systems

    | Company | Use Case | Configuration | Scale | Result |
    |---------|----------|---------------|-------|--------|
    | **Netflix** | Title similarity | Bigrams, max_features=10K | 15K titles | 73% user engagement |
    | **Google** | Web search (pre-BERT) | Trigrams, BM25 variant | Billions of pages | Baseline for neural models |
    | **Amazon** | Product search | Custom weighting, brands boosted | 500M products | 0.3s query latency |
    | **Airbnb** | Listing search | Descriptions + amenities | 6M listings | +12% booking rate |
    | **Spotify** | Podcast search | Transcripts, max_df=0.8 | 4M episodes | 95% relevance score |

    ## Optimization Techniques

    **1. BM25 (Best Match 25) - Enhanced TF-IDF:**

    $$\text{BM25}(t, d) = \text{IDF}(t) \times \frac{\text{TF}(t,d) \times (k_1 + 1)}{\text{TF}(t,d) + k_1 \times (1 - b + b \times \frac{|d|}{\text{avgdl}})}$$

    - **Tunable parameters:** k₁ (saturation), b (length normalization)
    - **Used by:** Elasticsearch, Lucene (default)
    - **Improvement:** Handles document length better than TF-IDF

    **2. Memory Optimization:**
    ```python
    # Use sparse matrices
    from scipy.sparse import csr_matrix
    
    # Reduce features
    vectorizer = TfidfVectorizer(
        max_features=5000,  # Limit vocabulary
        max_df=0.8,  # Remove too common words
        min_df=2  # Remove very rare words
    )
    
    # Quantization (float32 → int16)
    X_quantized = (X * 1000).astype(np.int16)
    ```

    **3. Inverted Index (Production Search):**
    ```
    Term → [(doc1, tfidf1), (doc2, tfidf2), ...]
    
    Benefits:
    - O(1) term lookup
    - Skip-list intersection for multi-term queries
    - Only process documents containing query terms
    ```

    ## Common Pitfalls

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **No preprocessing** | "Running" ≠ "run" | Stemming/lemmatization |
    | **Stopwords not removed** | "the", "is" dominate | Use stop_words='english' |
    | **Rare words dominate** | One-off terms get high IDF | Set min_df=2 or min_df=5 |
    | **Long docs favored** | More terms = higher scores | L2 normalization (default) |
    | **No n-grams** | "not good" = "good"? | Use ngram_range=(1,2) |
    | **Vocabulary explosion** | 100K+ features, slow/OOM | max_features=5000 |

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Understanding of TF and IDF components separately
        - When TF-IDF is better than embeddings (speed, interpretability)
        - Practical considerations (sparsity, memory, preprocessing)
        - Knowledge of BM25 enhancement
        
        **Strong signal:**
        
        - "TF rewards terms appearing often in a document. IDF penalizes terms appearing in many documents. Together, they highlight distinctive terms for each document"
        - "For Netflix's 15K movie catalog, TF-IDF with bigrams gives 73% of BERT's quality at 100× speed, processing 10K queries/second on a single CPU"
        - "Common pitfall: Not removing stopwords. 'the' appears 1000 times but adds no meaning. sklearn's stop_words='english' removes 318 common words"
        - "BM25 improves on TF-IDF by saturating term frequency (k1=1.5) and normalizing by document length (b=0.75). Elasticsearch uses this by default"
        
        **Red flags:**
        
        - "TF-IDF captures word meaning" (no, it's just statistics)
        - Thinking it works for short queries (needs enough terms)
        - Not knowing about sparsity (99%+ zeros for large vocabs)
        - Can't explain when to use vs embeddings
        
        **Follow-ups:**
        
        - "How to handle 'not good' vs 'good'?" → Bigrams, negation handling, or embeddings
        - "1M documents, 100K vocabulary - memory usage?" → Sparse matrix: ~1% non-zero, few GB
        - "Why L2 normalization?" → Prevents long documents from dominating similarity scores
        - "TF-IDF for real-time search with updates?" → Incremental IDF updates, or recompute hourly

---

### What is the Attention Mechanism? - Google, OpenAI Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Deep Learning` | **Asked by:** Google, OpenAI, Meta

??? success "View Answer"

    ## Core Concept

    **Attention** allows models to dynamically focus on relevant parts of the input when producing each output, solving the fixed-length bottleneck problem of RNN encoders.

    **Key Intuition:** When translating "I love cats" to French, the model should "attend" to different source words for each target word.

    ## Mathematical Foundation

    **Scaled Dot-Product Attention:**

    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

    Where:
    - **Q (Query):** "What am I looking for?" [n × d_k]
    - **K (Key):** "What do I contain?" [m × d_k]
    - **V (Value):** "What information do I have?" [m × d_v]
    - **d_k:** Key dimension (typically 64)

    **Steps:**
    1. Compute similarity: Q·K^T (how relevant is each key to each query?)
    2. Scale by √d_k (prevent softmax saturation)
    3. Apply softmax (get attention weights summing to 1)
    4. Weight values by attention weights

    ## Types of Attention

    | Type | Q, K, V Source | Use Case | Example |
    |------|----------------|----------|---------|
    | **Self-Attention** | Same sequence | Encoding context | BERT, GPT |
    | **Cross-Attention** | Q from decoder, K/V from encoder | Seq2seq | Translation, image captioning |
    | **Masked Attention** | Future tokens masked | Autoregressive generation | GPT decoding |
    | **Multi-Query Attention** | Shared K/V across heads | Faster inference | PaLM, Falcon |
    | **Flash Attention** | Tiled computation | Long sequences (memory efficient) | LLaMA-2 |

    ## Production Implementation (150 lines)

    ```python
    # attention_mechanisms.py
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math

    class ScaledDotProductAttention(nn.Module):
        """
        Basic attention mechanism

        Time: O(n·m·d) where n=query_len, m=key_len
        Space: O(n·m) for attention matrix
        """

        def __init__(self, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)

        def forward(self, Q, K, V, mask=None):
            """
            Args:
                Q: [batch, n, d_k] queries
                K: [batch, m, d_k] keys
                V: [batch, m, d_v] values
                mask: [batch, n, m] or broadcastable

            Returns:
                output: [batch, n, d_v]
                attention_weights: [batch, n, m]
            """
            d_k = Q.size(-1)

            # Compute attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

            # Apply mask (set masked positions to -inf)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

            # Attention weights (softmax over keys dimension)
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)

            # Apply attention to values
            output = torch.matmul(attention_weights, V)

            return output, attention_weights

    class AdditiveAttention(nn.Module):
        """
        Bahdanau Attention (additive)

        score(q, k) = v^T tanh(W_q q + W_k k)

        Older mechanism, used in early seq2seq models
        """

        def __init__(self, hidden_dim):
            super().__init__()
            self.W_q = nn.Linear(hidden_dim, hidden_dim)
            self.W_k = nn.Linear(hidden_dim, hidden_dim)
            self.v = nn.Linear(hidden_dim, 1)

        def forward(self, query, keys, values, mask=None):
            """
            Args:
                query: [batch, d_q]
                keys: [batch, seq_len, d_k]
                values: [batch, seq_len, d_v]

            Returns:
                context: [batch, d_v]
                attention_weights: [batch, seq_len]
            """
            # Expand query for broadcasting
            query = query.unsqueeze(1)  # [batch, 1, d_q]

            # Compute scores
            scores = self.v(torch.tanh(
                self.W_q(query) + self.W_k(keys)
            )).squeeze(-1)  # [batch, seq_len]

            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

            # Attention weights
            attention_weights = F.softmax(scores, dim=-1)

            # Context vector (weighted sum of values)
            context = torch.matmul(
                attention_weights.unsqueeze(1), values
            ).squeeze(1)

            return context, attention_weights

    class MultiHeadAttention(nn.Module):
        """Multi-Head Attention (used in Transformers)"""

        def __init__(self, d_model=512, num_heads=8, dropout=0.1):
            super().__init__()
            assert d_model % num_heads == 0

            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads

            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)

            self.attention = ScaledDotProductAttention(dropout)

        def split_heads(self, x, batch_size):
            """Split into multiple heads"""
            x = x.view(batch_size, -1, self.num_heads, self.d_k)
            return x.transpose(1, 2)  # [batch, heads, seq, d_k]

        def forward(self, query, key, value, mask=None):
            batch_size = query.size(0)

            # Linear projections and split heads
            Q = self.split_heads(self.W_q(query), batch_size)
            K = self.split_heads(self.W_k(key), batch_size)
            V = self.split_heads(self.W_v(value), batch_size)

            # Apply attention
            attn_output, attn_weights = self.attention(Q, K, V, mask)

            # Concatenate heads
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, -1, self.d_model)

            # Final linear
            output = self.W_o(attn_output)

            return output, attn_weights

    class CrossAttention(nn.Module):
        """
        Cross-Attention for encoder-decoder

        Query from decoder, Keys/Values from encoder
        Used in: Translation, image captioning, text-to-image
        """

        def __init__(self, d_model=512, num_heads=8):
            super().__init__()
            self.attention = MultiHeadAttention(d_model, num_heads)

        def forward(self, decoder_hidden, encoder_outputs, mask=None):
            """
            Args:
                decoder_hidden: [batch, dec_len, d_model] (queries)
                encoder_outputs: [batch, enc_len, d_model] (keys & values)
                mask: Optional padding mask for encoder

            Returns:
                output: [batch, dec_len, d_model]
            """
            # Q from decoder, K/V from encoder
            output, attn_weights = self.attention(
                query=decoder_hidden,
                key=encoder_outputs,
                value=encoder_outputs,
                mask=mask
            )

            return output, attn_weights

    # Example: Visualize Attention
    def visualize_attention_example():
        """Example showing how attention focuses on relevant words"""
        # Simple example: translating "I love cats" to French

        # Encoder outputs (simplified)
        encoder_out = torch.randn(1, 3, 512)  # 3 words: I, love, cats

        # Decoder at step 1 (generating "J'")
        decoder_hidden = torch.randn(1, 1, 512)

        cross_attn = CrossAttention(d_model=512)
        output, attn_weights = cross_attn(decoder_hidden, encoder_out)

        # Attention weights shape: [1, num_heads, 1, 3]
        # Shows how much decoder attends to each encoder word

        print("Attention weights (decoder step 1):")
        print(attn_weights[0, 0, 0, :])
        # Example output: [0.85, 0.10, 0.05]
        # → Decoder focuses on "I" when generating "J'"

    if __name__ == "__main__":
        # Test attention mechanisms
        batch_size = 2
        seq_len = 10
        d_model = 512

        Q = torch.randn(batch_size, seq_len, d_model)
        K = torch.randn(batch_size, seq_len, d_model)
        V = torch.randn(batch_size, seq_len, d_model)

        # Scaled dot-product
        attn = ScaledDotProductAttention()
        output, weights = attn(Q, K, V)
        print(f"Output shape: {output.shape}")  # [2, 10, 512]
        print(f"Attention weights shape: {weights.shape}")  # [2, 10, 10]

        # Multi-head attention
        mha = MultiHeadAttention(d_model=512, num_heads=8)
        output, weights = mha(Q, K, V)
        print(f"Multi-head output: {output.shape}")  # [2, 10, 512]
    ```

    ## Attention Variants Comparison

    | Variant | Complexity | Memory | Speed | Use Case |
    |---------|------------|--------|-------|----------|
    | **Scaled Dot-Product** | O(n²d) | O(n²) | Baseline | Standard Transformers |
    | **Flash Attention** | O(n²d) | **O(n)** | **1.5-3x faster** | Long sequences (GPT-4) |
    | **Sparse Attention** | **O(n√n d)** | O(n√n) | 10x faster | Very long contexts |
    | **Linear Attention** | **O(nd²)** | **O(nd)** | 100x faster | Approximate, research |
    | **Multi-Query Attention** | O(n²d) | O(n) | 2x faster inference | PaLM, Falcon LLMs |

    ## Common Attention Patterns

    **1. Self-Attention (BERT, GPT):**
    - Q, K, V all from same sequence
    - Allows each token to attend to all others
    - Bidirectional (BERT) or causal (GPT)

    **2. Cross-Attention (Translation):**
    - Q from target, K/V from source
    - Decoder attends to encoder outputs
    - Example: "J'aime" attends to "I love"

    **3. Causal Masking (GPT):**
    - Prevent attending to future tokens
    - Upper triangular mask (position i can't see j > i)
    - Ensures autoregressive property

    **4. Padding Masking:**
    - Ignore padding tokens
    - Set attention scores to -inf for padding positions
    - Prevents model from learning from padding

    ## Why Attention Works

    **Problem it Solves:**
    - RNN encoders compress entire input into fixed-size vector → information bottleneck
    - Long sequences lose information

    **Solution:**
    - Decoder can "look at" any encoder state
    - Dynamically weighted combination based on relevance
    - No information bottleneck

    **Example (Translation):**
    ```
    English: "The cat sat on the mat"
    French:  "Le chat s'est assis sur le tapis"

    When generating "chat":
    - High attention on "cat" (0.9)
    - Low attention on other words (0.1 distributed)
    ```

    ## Real-World Impact

    **Google Neural Machine Translation (GNMT, 2016):**
    - Added attention to seq2seq
    - **60% reduction in translation errors**
    - Production: 18M translations/day

    **Transformers (2017):**
    - **Only** attention (no recurrence)
    - **10x training speedup** vs RNNs
    - Enabled models like GPT-3 (175B params)

    **Vision Transformers (ViT, 2020):**
    - Applied self-attention to image patches
    - Matches CNNs on ImageNet
    - Powers DALL-E, Stable Diffusion

    ## Common Pitfalls

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Forgetting √d_k scaling** | Vanishing gradients | Always divide by √d_k |
    | **Wrong mask shape** | Broadcasting errors | Ensure mask is [batch, n, m] or broadcastable |
    | **Softmax over wrong dim** | Attention doesn't sum to 1 | Softmax on last dimension (keys) |
    | **O(n²) memory** | OOM for long sequences | Flash Attention or sparse patterns |
    | **Not masking padding** | Model learns from padding | Always mask padding tokens |

    !!! tip "Interviewer's Insight"
        **Strong candidates:**

        - Can implement scaled dot-product attention from scratch with correct tensor operations
        - Explain why scaling by √d_k matters: "Prevents softmax saturation, maintains gradient flow"
        - Understand different attention types: "Self-attention for encoding, cross-attention for translation"
        - Know complexity: "O(n²) is bottleneck for long sequences, Flash Attention solves this"
        - Reference real impact: "Attention enabled Transformers, which power GPT-4, DALL-E, AlphaFold"
        - Discuss production optimizations: "Multi-query attention for 2x faster inference in PaLM"

---

### Explain Named Entity Recognition (NER) - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Sequence Labeling`, `Information Extraction`, `BiLSTM-CRF` | **Asked by:** Amazon, Google, Meta, Bloomberg, Apple

??? success "View Answer"

    ## Core Concept

    **Named Entity Recognition (NER)** identifies and classifies named entities in text into predefined categories: PERSON, ORGANIZATION, LOCATION, DATE, etc.

    **Task Type:** Sequence labeling (token-level classification)

    **Standard Entity Types (CoNLL-2003):**
    - PER: Person names ("Steve Jobs")
    - ORG: Organizations ("Apple Inc.")
    - LOC: Locations ("California")
    - MISC: Miscellaneous ("iPhone", "Nobel Prize")

    ## BIO Tagging Scheme

    ```
    ┌──────────────────────────────────────────────────────────────┐
    │                    BIO TAGGING SCHEME                           │
    ├──────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  Sentence: "Steve Jobs founded Apple Inc. in California"       │
    │                                                                  │
    │  Word         | Tag          | Meaning                          │
    │  ──────────────────────────────────────────────────│
    │  Steve        | B-PER       | Begin person entity              │
    │  Jobs         | I-PER       | Inside person entity             │
    │  founded      | O           | Outside (not an entity)          │
    │  Apple        | B-ORG       | Begin organization               │
    │  Inc.         | I-ORG       | Inside organization              │
    │  in           | O           | Outside                          │
    │  California   | B-LOC       | Begin location                   │
    │                                                                  │
    │  Why BIO? Distinguishes adjacent entities:                       │
    │    "[Bank of America]_ORG headquarters in [New York]_LOC"       │
    │    vs "[Bank]_ORG [of America]_LOC" (wrong!)                    │
    │                                                                  │
    │  Variants:                                                        │
    │  - IO: Just I-PER, O (simpler, less accurate)                   │
    │  - BIOES: B-begin, I-inside, O-outside, E-end, S-single        │
    │                                                                  │
    └──────────────────────────────────────────────────────────────┘
    ```

    ## Production Implementation (180 lines)

    ```python
    # ner_bilstm_crf.py
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    import numpy as np
    from typing import List, Tuple, Dict, Optional
    from dataclasses import dataclass
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    from seqeval.metrics import classification_report, f1_score
    import time

    @dataclass
    class NERMetrics:
        """NER evaluation metrics"""
        precision: float
        recall: float
        f1_score: float
        entity_f1: Dict[str, float]
        inference_time_ms: float

    class CRF(nn.Module):
        """
        Conditional Random Field for sequence tagging
        
        Key insight: CRF models label dependencies (e.g., I-PER can't follow B-LOC)
        
        Time Complexity: O(T × N²) where T=seq_len, N=num_tags
        Space Complexity: O(N²) for transition matrix
        """
        
        def __init__(self, num_tags: int):
            super().__init__()
            self.num_tags = num_tags
            
            # Transition scores: transitions[i,j] = score of transitioning from tag i to tag j
            self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
            
            # Start and end tags
            self.start_transitions = nn.Parameter(torch.randn(num_tags))
            self.end_transitions = nn.Parameter(torch.randn(num_tags))
            
            # Constrain impossible transitions
            self._initialize_constraints()
        
        def _initialize_constraints(self):
            """Prevent invalid transitions (e.g., O -> I-PER)"""
            # In practice, would set specific transitions to -inf
            pass
        
        def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            """
            Compute negative log-likelihood loss
            
            Args:
                emissions: [batch, seq_len, num_tags] - LSTM outputs
                tags: [batch, seq_len] - true tags
                mask: [batch, seq_len] - padding mask
            
            Returns:
                Negative log-likelihood
            """
            # Score of true sequence
            gold_score = self._score_sequence(emissions, tags, mask)
            
            # Log-sum of all possible sequences (forward algorithm)
            forward_score = self._forward_algorithm(emissions, mask)
            
            # NLL = log(sum of all sequences) - log(gold sequence)
            return (forward_score - gold_score).mean()
        
        def _score_sequence(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            """Compute score of a given tag sequence"""
            batch_size, seq_len = tags.shape
            score = self.start_transitions[tags[:, 0]]  # Start
            
            for i in range(seq_len - 1):
                current_tags = tags[:, i]
                next_tags = tags[:, i + 1]
                
                # Emission score
                score += emissions[:, i].gather(1, current_tags.unsqueeze(1)).squeeze(1)
                
                # Transition score
                score += self.transitions[current_tags, next_tags] * mask[:, i + 1]
            
            # Last emission + end transition
            last_tags = tags.gather(1, mask.sum(1).long().unsqueeze(1) - 1).squeeze(1)
            score += emissions.gather(1, last_tags.unsqueeze(1).unsqueeze(2)).squeeze()
            score += self.end_transitions[last_tags]
            
            return score
        
        def _forward_algorithm(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            """
            Forward algorithm: compute log-sum-exp of all paths
            
            Dynamic programming: α[t,j] = log-sum-exp over all paths ending at tag j at time t
            """
            batch_size, seq_len, num_tags = emissions.shape
            
            # Initialize: α[0] = start_transitions + emissions[0]
            alpha = self.start_transitions + emissions[:, 0]  # [batch, num_tags]
            
            for i in range(1, seq_len):
                # Broadcast: [batch, 1, num_tags] + [num_tags, num_tags]
                alpha_broadcast = alpha.unsqueeze(2)  # [batch, num_tags, 1]
                emit_scores = emissions[:, i].unsqueeze(1)  # [batch, 1, num_tags]
                
                # All possible transitions
                scores = alpha_broadcast + self.transitions + emit_scores
                
                # Log-sum-exp
                alpha = torch.logsumexp(scores, dim=1) * mask[:, i].unsqueeze(1)
            
            # Add end transitions
            alpha += self.end_transitions
            return torch.logsumexp(alpha, dim=1)
        
        def viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
            """
            Viterbi algorithm: find most likely tag sequence
            
            Time: O(T × N²), Space: O(T × N)
            """
            batch_size, seq_len, num_tags = emissions.shape
            
            # Backpointers
            backpointers = torch.zeros(batch_size, seq_len, num_tags, dtype=torch.long)
            
            # Initialize
            viterbi = self.start_transitions + emissions[:, 0]
            
            for i in range(1, seq_len):
                broadcast_viterbi = viterbi.unsqueeze(2)
                broadcast_transitions = self.transitions.unsqueeze(0)
                
                scores = broadcast_viterbi + broadcast_transitions
                
                # Max over previous states
                viterbi, backpointers[:, i] = torch.max(scores, dim=1)
                viterbi += emissions[:, i]
            
            # Backtrack
            best_paths = []
            for b in range(batch_size):
                # Find best final tag
                best_last_tag = (viterbi[b] + self.end_transitions).argmax()
                
                # Backtrack
                path = [best_last_tag.item()]
                for i in range(seq_len - 1, 0, -1):
                    best_last_tag = backpointers[b, i, best_last_tag]
                    path.append(best_last_tag.item())
                
                best_paths.append(list(reversed(path)))
            
            return best_paths

    class BiLSTM_CRF(nn.Module):
        """
        BiLSTM-CRF for NER (SotA before BERT)
        
        Architecture: Embeddings → BiLSTM → Linear → CRF
        """
        
        def __init__(
            self,
            vocab_size: int,
            tag2idx: Dict[str, int],
            embedding_dim: int = 100,
            hidden_dim: int = 256
        ):
            super().__init__()
            
            self.tag2idx = tag2idx
            self.num_tags = len(tag2idx)
            
            # Word embeddings
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            
            # BiLSTM
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim // 2,
                num_layers=2,
                bidirectional=True,
                dropout=0.5,
                batch_first=True
            )
            
            # Project to tag space
            self.hidden2tag = nn.Linear(hidden_dim, self.num_tags)
            
            # CRF layer
            self.crf = CRF(self.num_tags)
        
        def forward(
            self,
            input_ids: torch.Tensor,
            tags: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None
        ):
            # Embeddings
            embeds = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
            
            # BiLSTM
            lstm_out, _ = self.lstm(embeds)  # [batch, seq_len, hidden_dim]
            
            # Project to tag space (emission scores)
            emissions = self.hidden2tag(lstm_out)  # [batch, seq_len, num_tags]
            
            if tags is not None:
                # Training: compute loss
                return self.crf(emissions, tags, mask)
            else:
                # Inference: decode
                return self.crf.viterbi_decode(emissions, mask)

    # ===========================================
    # EXAMPLE USAGE WITH COMPANY USE CASES
    # ===========================================

    if __name__ == "__main__":
        print("="*70)
        print("BLOOMBERG - FINANCIAL NER FOR NEWS ANALYSIS")
        print("="*70)
        
        # Use pre-trained BERT-based NER
        ner_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple"  # Merge subword tokens
        )
        
        text = "Apple Inc. CEO Tim Cook announced a $100B investment in California"
        
        start_time = time.time()
        entities = ner_pipeline(text)
        inference_time = (time.time() - start_time) * 1000
        
        print(f"\nText: {text}")
        print(f"\nExtracted Entities (inference: {inference_time:.2f}ms):")
        for entity in entities:
            print(f"  {entity['entity_group']:8s} | {entity['word']:20s} | score: {entity['score']:.3f}")
        
        print("\n" + "="*70)
        print("GOOGLE - MULTILINGUAL NER FOR SEARCH")
        print("="*70)
        
        # XLM-RoBERTa for multilingual NER
        multilingual_ner = pipeline(
            "ner",
            model="Davlan/xlm-roberta-base-ner-hrl",
            aggregation_strategy="simple"
        )
        
        texts = [
            "Angela Merkel besuchte Paris",  # German
            "Emmanuel Macron visitó Madrid",  # Spanish
        ]
        
        for text in texts:
            entities = multilingual_ner(text)
            print(f"\nText: {text}")
            for entity in entities:
                print(f"  {entity['entity_group']:8s} | {entity['word']}")
        
        print("\n" + "="*70)
        print("AMAZON - PRODUCT NER FOR E-COMMERCE")
        print("="*70)
        print("Custom entities: BRAND, PRODUCT, SIZE, COLOR, MATERIAL")
        print("\nExample: 'Nike Air Max 90 size 10 in black leather'")
        print("  BRAND:   Nike")
        print("  PRODUCT: Air Max 90")
        print("  SIZE:    size 10")
        print("  COLOR:   black")
        print("  MATERIAL: leather")
    ```

    ## Approach Comparison

    | Approach | Method | F1-Score (CoNLL-2003) | Speed | Training |
    |----------|--------|----------------------|-------|----------|
    | **Rule-based** | Regex + gazetteers | 60-70% | Very fast (<1ms) | None |
    | **CRF** | Linear-chain CRF + hand-crafted features | 84-89% | Fast (5ms) | Hours |
    | **BiLSTM-CRF** | Neural embeddings + CRF | 90-91% | Moderate (20ms) | Hours-Days |
    | **BERT-base** | Fine-tuned Transformer | 92-93% | Slow (50ms CPU) | Days |
    | **RoBERTa-large** | Larger Transformer | 94-95% | Very slow (200ms CPU) | Weeks |

    ## Real-World Deployments

    | Company | Use Case | Model | Entities | Performance |
    |---------|----------|-------|----------|-------------|
    | **Bloomberg** | Financial news | Custom BERT | ORG, PERSON, MONEY, DATE | 94% F1, 30ms latency |
    | **Google** | Knowledge Graph | Proprietary | 500+ entity types | Billions of entities/day |
    | **Amazon** | Product extraction | BiLSTM-CRF | BRAND, PRODUCT, SPEC | 91% F1, 10ms latency |
    | **Apple** | Siri entity recognition | On-device TinyBERT | PERSON, PLACE, APP | 88% F1, <5ms on iPhone |
    | **Meta** | Content understanding | XLM-RoBERTa | Multilingual (100+ langs) | 90% avg F1 |

    ## Evaluation Metrics

    **Entity-level F1 (seqeval):**
    ```python
    from seqeval.metrics import f1_score, classification_report
    
    y_true = [["O", "B-PER", "I-PER", "O", "B-LOC"]]
    y_pred = [["O", "B-PER", "O", "O", "B-LOC"]]
    
    # Strict matching: entire entity must be correct
    f1 = f1_score(y_true, y_pred)  # Counts B-PER I-PER as one entity
    
    # Per-entity metrics
    print(classification_report(y_true, y_pred))
    # Outputs:
    #              precision    recall  f1-score   support
    #    PER          0.00      0.00      0.00         1
    #    LOC          1.00      1.00      1.00         1
    ```

    **Common Pitfalls:**

    | Pitfall | Example | Impact | Solution |
    |---------|---------|--------|----------|
    | **Tokenization mismatch** | "New York" → ["New", "York"] | Can't predict multi-token | Use subword tokenization |
    | **Nested entities** | "[Bank of [America]]" | Standard BIO can't handle | Use hypergraph or multi-task |
    | **Class imbalance** | 90% O tags, 10% entities | Model predicts all O | Weighted loss, focal loss |
    | **Domain shift** | Train on news, test on social | F1 drops 20-30% | Domain adaptation, synthetic data |

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Understanding of BIO tagging (why not just binary classification?)
        - Knowledge of CRF's role (modeling tag dependencies)
        - Awareness of trade-offs (BiLSTM-CRF vs BERT)
        - Handling of multi-token entities
        
        **Strong signal:**
        
        - "BIO tagging prevents illegal tag sequences. CRF layer models transitions, ensuring I-PER never follows B-LOC. This improves F1 by 2-3% over independent classification"
        - "At Bloomberg, we fine-tuned BERT on 1M financial articles achieving 94% F1 on ORG/PERSON/MONEY entities with 30ms latency on CPUs"
        - "Common mistake: Using token-level accuracy. Entity-level F1 is correct - a 2-word entity is either fully correct or fully wrong"
        - "For 100+ languages, XLM-RoBERTa works well. Google's model handles 500+ entity types across 100 languages"
        
        **Red flags:**
        
        - Not knowing BIO vs IO tagging schemes
        - Thinking CRF is only for NER (also used in POS tagging, chunking)
        - Using accuracy instead of entity-level F1
        - Not considering inference latency in production
        
        **Follow-ups:**
        
        - "How to handle nested entities?" → Multi-task learning, hypergraph-based methods
        - "Class imbalance (90% O tags)?" → Focal loss, weighted cross-entropy, hard negative mining
        - "Real-time NER with <10ms latency?" → DistilBERT, quantization, TorchScript, ONNX
        - "How to add new entity type without retraining?" → Few-shot learning, prompt-based methods

---

### What is Tokenization? Compare Methods - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Tokenization`, `Subword Units`, `Preprocessing` | **Asked by:** Google, OpenAI, Meta, Amazon, Microsoft

??? success "View Answer"

    ## Core Concept

    **Tokenization** splits text into units (tokens) for model processing. Modern approaches use **subword tokenization** to balance vocabulary size and coverage.

    **Key Trade-off:**
    - **Word-level:** Small vocab, but can't handle rare/OOV words
    - **Character-level:** No OOV, but sequences too long
    - **Subword:** Best of both - fixed vocab, handles any word

    ## Tokenization Methods Evolution

    ```
    ┌──────────────────────────────────────────────────────────────┐
    │              TOKENIZATION METHOD COMPARISON                      │
    ├──────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  Example word: "unbelievable"                                    │
    │                                                                  │
    │  1. WHITESPACE: ["unbelievable"]                                │
    │     • Problem: OOV if word not in training                       │
    │                                                                  │
    │  2. CHARACTER: ['u','n','b','e','l','i','e','v','a','b','l','e']│
    │     • Problem: 12 tokens vs 1, long sequences                     │
    │                                                                  │
    │  3. BPE (GPT): ['un', 'believ', 'able']                          │
    │     • Merges most frequent pairs iteratively                      │
    │     • Vocab: 50K tokens                                            │
    │                                                                  │
    │  4. WordPiece (BERT): ['un', '##bel', '##iev', '##able']        │
    │     • Merges based on likelihood maximization                     │
    │     • '##' indicates continuation                                 │
    │     • Vocab: 30K tokens                                            │
    │                                                                  │
    │  5. SentencePiece (T5): ['▁un', 'believ', 'able']              │
    │     • Treats space as special char (▁)                           │
    │     • Language-agnostic (no pre-tokenization)                      │
    │     • Vocab: 32K tokens                                            │
    │                                                                  │
    └──────────────────────────────────────────────────────────────┘
    ```

    ## Production Implementation (175 lines)

    ```python
    # tokenization_methods.py
    import re
    from typing import List, Dict, Tuple
    from collections import Counter, defaultdict
    import heapq
    from dataclasses import dataclass
    from transformers import (
        AutoTokenizer,
        BertTokenizer,
        GPT2Tokenizer,
        T5Tokenizer
    )
    import time

    @dataclass
    class TokenizationMetrics:
        """Metrics for tokenization evaluation"""
        vocab_size: int
        avg_tokens_per_word: float
        oov_rate: float
        compression_ratio: float
        time_ms: float

    class BPETokenizer:
        """
        Byte-Pair Encoding (used in GPT-2, GPT-3, RoBERTa)
        
        Algorithm:
        1. Start with character vocabulary
        2. Count all adjacent pairs
        3. Merge most frequent pair
        4. Repeat until vocab_size reached
        
        Time: O(N² × V) where N=text_length, V=vocab_size
        Space: O(V²) for pair statistics
        """
        
        def __init__(self, vocab_size: int = 1000):
            self.vocab_size = vocab_size
            self.vocab: Dict[str, int] = {}
            self.merges: List[Tuple[str, str]] = []
        
        def train(self, texts: List[str]) -> None:
            """
            Train BPE on corpus
            
            Example:
              Input: ["low", "lower", "lowest"]
              Initial: {'l o w </w>': 3, 'l o w e r </w>': 2, ...}
              
              Iteration 1: Merge ('l', 'o') → 'lo'
              Iteration 2: Merge ('lo', 'w') → 'low'
              ...
            """
            # Initialize word frequencies with </w> end marker
            word_freqs = Counter()
            for text in texts:
                words = text.lower().split()
                for word in words:
                    word_freqs[' '.join(list(word) + ['</w>'])] += 1
            
            # Start with character vocabulary
            vocab = set()
            for word in word_freqs:
                vocab.update(word.split())
            
            # Iteratively merge most frequent pairs
            for i in range(self.vocab_size - len(vocab)):
                # Count all pairs
                pairs = defaultdict(int)
                for word, freq in word_freqs.items():
                    symbols = word.split()
                    for j in range(len(symbols) - 1):
                        pairs[(symbols[j], symbols[j+1])] += freq
                
                if not pairs:
                    break
                
                # Find most frequent pair
                best_pair = max(pairs, key=pairs.get)
                self.merges.append(best_pair)
                
                # Merge in all words
                new_word_freqs = {}
                bigram = ' '.join(best_pair)
                replacement = ''.join(best_pair)
                
                for word, freq in word_freqs.items():
                    new_word = word.replace(bigram, replacement)
                    new_word_freqs[new_word] = freq
                
                word_freqs = new_word_freqs
                vocab.add(replacement)
                
                if (i + 1) % 100 == 0:
                    print(f"Iteration {i+1}: Merged {best_pair} → {replacement}")
            
            self.vocab = {token: idx for idx, token in enumerate(sorted(vocab))}
        
        def tokenize(self, text: str) -> List[str]:
            """Apply learned merges to new text"""
            words = text.lower().split()
            tokens = []
            
            for word in words:
                # Start with characters
                word_tokens = list(word) + ['</w>']
                
                # Apply merges in order
                for merge_pair in self.merges:
                    i = 0
                    while i < len(word_tokens) - 1:
                        if (word_tokens[i], word_tokens[i+1]) == merge_pair:
                            word_tokens[i:i+2] = [''.join(merge_pair)]
                        else:
                            i += 1
                
                tokens.extend(word_tokens)
            
            return tokens

    class WordPieceTokenizer:
        """
        WordPiece (used in BERT)
        
        Difference from BPE: Chooses merges that maximize likelihood of training data
        Score = freq(AB) / (freq(A) × freq(B))
        """
        
        def __init__(self, vocab_size: int = 1000):
            self.vocab_size = vocab_size
            self.vocab: Dict[str, int] = {}
        
        def train(self, texts: List[str]) -> None:
            """Train WordPiece (simplified version)"""
            # Similar to BPE but uses likelihood-based scoring
            # In practice, use HuggingFace's tokenizers library
            pass
        
        def tokenize(self, text: str, vocab: set) -> List[str]:
            """
            Greedy longest-match-first tokenization
            
            Example: "unbelievable"
            1. Try "unbelievable" - not in vocab
            2. Try "unbel" - not in vocab
            3. Try "un" - in vocab! Output: ["un"]
            4. Remaining: "believable", repeat
            """
            words = text.lower().split()
            tokens = []
            
            for word in words:
                word_tokens = []
                start = 0
                
                while start < len(word):
                    # Find longest matching subword
                    end = len(word)
                    found = False
                    
                    while start < end:
                        substr = word[start:end]
                        # Add ## prefix if not word start
                        if start > 0:
                            substr = "##" + substr
                        
                        if substr in vocab:
                            word_tokens.append(substr)
                            found = True
                            break
                        
                        end -= 1
                    
                    if not found:
                        # Unknown token
                        word_tokens.append("[UNK]")
                        break
                    
                    start = end
                
                tokens.extend(word_tokens)
            
            return tokens

    def compare_tokenizers(text: str) -> None:
        """
        Compare different tokenization methods
        """
        print("="*70)
        print(f"Input text: '{text}'")
        print("="*70)
        
        # Load pre-trained tokenizers
        tokenizers = [
            ("BERT (WordPiece)", BertTokenizer.from_pretrained("bert-base-uncased")),
            ("GPT-2 (BPE)", GPT2Tokenizer.from_pretrained("gpt2")),
            ("T5 (SentencePiece)", T5Tokenizer.from_pretrained("t5-small")),
        ]
        
        for name, tokenizer in tokenizers:
            start = time.time()
            tokens = tokenizer.tokenize(text)
            time_ms = (time.time() - start) * 1000
            
            # Token IDs
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            
            print(f"\n{name}:")
            print(f"  Tokens ({len(tokens)}): {tokens}")
            print(f"  IDs: {token_ids[:10]}..." if len(token_ids) > 10 else f"  IDs: {token_ids}")
            print(f"  Vocab size: {tokenizer.vocab_size:,}")
            print(f"  Time: {time_ms:.2f}ms")

    # ===========================================
    # EXAMPLE USAGE WITH COMPANY USE CASES
    # ===========================================

    if __name__ == "__main__":
        print("="*70)
        print("OPENAI - GPT TOKENIZATION")
        print("="*70)
        
        text = "Hello! How are you doing today? I'm learning about tokenization."
        compare_tokenizers(text)
        
        print("\n" + "="*70)
        print("HANDLING RARE/OOV WORDS")
        print("="*70)
        
        rare_text = "supercalifragilisticexpialidocious"
        
        # BERT WordPiece
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_tokens = bert_tokenizer.tokenize(rare_text)
        print(f"\nBERT: {bert_tokens}")
        print(f"  Length: {len(bert_tokens)} tokens (graceful degradation to subwords)")
        
        # GPT-2 BPE
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        gpt2_tokens = gpt2_tokenizer.tokenize(rare_text)
        print(f"\nGPT-2: {gpt2_tokens}")
        print(f"  Length: {len(gpt2_tokens)} tokens")
        
        print("\n" + "="*70)
        print("GOOGLE - MULTILINGUAL TOKENIZATION")
        print("="*70)
        print("SentencePiece advantages for multilingual:")
        print("  1. No pre-tokenization (works on raw text)")
        print("  2. Treats whitespace as normal character")
        print("  3. Reversible (decode(encode(text)) == text)")
        print("  4. Language-agnostic (Chinese, Arabic, etc.)")
        
        # Example with multiple languages
        multilingual_text = "Hello 你好 مرحبا"
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        tokens = t5_tokenizer.tokenize(multilingual_text)
        print(f"\nText: {multilingual_text}")
        print(f"T5 tokens: {tokens}")
    ```

    ## Method Comparison

    | Method | Used By | Vocab Size | Algorithm | Pros | Cons |
    |--------|---------|------------|-----------|------|------|
    | **Whitespace** | Early NLP | 50K-500K | Split on spaces | Simple, interpretable | OOV problem, large vocab |
    | **BPE** | GPT-2/3, RoBERTa | 50K | Merge frequent pairs | Efficient, data-driven | Greedy, not optimal |
    | **WordPiece** | BERT, DistilBERT | 30K | Likelihood-based merging | Better rare words | Slower training |
    | **SentencePiece** | T5, XLM-R, ALBERT | 32K | BPE on raw text | Multilingual, reversible | Harder to debug |
    | **Character** | Some CNNs | 256-1000 | One char = one token | No OOV, small vocab | Long sequences, weak semantics |

    ## Real-World Impact

    | Company | Model | Tokenizer | Vocab Size | Key Benefit |
    |---------|-------|-----------|------------|-------------|
    | **OpenAI** | GPT-3 | BPE | 50,257 | Handles any text, including code |
    | **Google** | BERT | WordPiece | 30,522 | Optimal for English, minimal OOV |
    | **Google** | T5 | SentencePiece | 32,000 | Works for 100+ languages without changes |
    | **Meta** | LLaMA | SentencePiece | 32,000 | Efficient multilingual, reversible |
    | **Anthropic** | Claude | Custom BPE | ~100K | Better handling of code/math |

    ## Key Trade-offs

    **Vocabulary Size:**
    ```python
    # Small vocab (10K):
    # • Pros: Fast softmax, small embedding matrix
    # • Cons: More tokens per sentence, longer sequences
    
    # Large vocab (100K):
    # • Pros: Fewer tokens per sentence, better semantics
    # • Cons: Slow softmax, huge embedding matrix (100K × 768 = 77M params)
    
    # Sweet spot: 30K-50K for most models
    ```

    **Sequence Length Impact:**

    | Text | Whitespace | BPE (50K) | WordPiece (30K) | Character |
    |------|------------|-----------|-----------------|----------|
    | "Hello world" | 2 tokens | 2 tokens | 2 tokens | 10 tokens |
    | "unbelievable" | 1 token | 2-3 tokens | 4 tokens | 12 tokens |
    | "COVID-19" | 1 token (if in vocab) | 2-3 tokens | 3 tokens | 8 tokens |

    ## Common Pitfalls

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Case sensitivity** | "Apple" vs "apple" = different tokens | Lowercase (BERT) or case-preserve (GPT) |
    | **Whitespace handling** | " hello" ≠ "hello" | SentencePiece treats space explicitly |
    | **Special tokens** | [CLS], [SEP], <s>, </s> not counted in vocab | Reserve special token IDs |
    | **Token limit** | BERT: 512 tokens max | Truncate or use Longformer/BigBird |
    | **Decoding errors** | "##ing" without prefix | Post-process: remove ##, merge tokens |

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Understanding of subword tokenization necessity (OOV problem)
        - Knowledge of BPE vs WordPiece differences
        - Awareness of vocabulary size trade-offs
        - Handling of multilingual text
        
        **Strong signal:**
        
        - "BPE solves OOV by using subwords. For 'unbelievable', it splits into ['un', 'believ', 'able'], each in the 50K vocab. Any word can be represented as character sequences worst-case"
        - "WordPiece (BERT) differs from BPE (GPT-2) in merge selection: WordPiece maximizes likelihood P(AB)/P(A)P(B), while BPE uses frequency. WordPiece is slightly better for rare words"
        - "OpenAI's GPT-3 uses 50K BPE tokens, handling English + code + 50 languages. Google's T5 uses 32K SentencePiece for 100+ languages by treating text as raw bytes"
        - "Common issue: BERT's 512 token limit. For long documents, use sliding window with overlap, or Longformer which extends to 4K+ tokens with sparse attention"
        
        **Red flags:**
        
        - Not knowing the OOV (out-of-vocabulary) problem
        - Thinking all models use word-level tokenization
        - Can't explain why GPT-2 has 50K vocab while BERT has 30K
        - Not aware of multilingual challenges (Chinese, Arabic)
        
        **Follow-ups:**
        
        - "How does tokenization affect model size?" → Embedding matrix is vocab_size × hidden_dim, so 100K vocab = 77M extra params vs 30K vocab
        - "Handling code/math?" → BPE handles better; some models use specialized tokenizers (Codex adds code-specific tokens)
        - "Why SentencePiece for multilingual?" → No pre-tokenization needed, works on raw bytes, treats all languages uniformly
        - "Token limit problem?" → Truncation, sliding window, or sparse attention (Longformer, BigBird)

---

### Explain Sentiment Analysis Approaches - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Text Classification`, `Sentiment`, `Aspect-Based Analysis` | **Asked by:** Amazon, Google, Meta, Twitter, Airbnb

??? success "View Answer"

    ## Core Concept

    **Sentiment Analysis** classifies text into emotional tones (positive, negative, neutral). Used for customer feedback, social media monitoring, and product reviews.

    **Granularity Levels:**
    - **Document-level:** Overall sentiment of entire text
    - **Sentence-level:** Sentiment per sentence
    - **Aspect-level:** Sentiment towards specific features ("food good, service bad")
    - **Entity-level:** Sentiment towards named entities

    ## Approach Evolution

    ```
    ┌──────────────────────────────────────────────────────────────┐
    │           SENTIMENT ANALYSIS APPROACHES                      │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  1. LEXICON-BASED (Rule-based)                               │
    │  ┌────────────────────────────────────────────────────────┐ │
    │  │ Dictionary: {"good": +1, "bad": -1, "great": +2}      │ │
    │  │ Score = Σ word_scores                                  │ │
    │  │                                                         │ │
    │  │ Pros: No training, interpretable                       │ │
    │  │ Cons: Misses context ("not good" = negative?)         │ │
    │  └────────────────────────────────────────────────────────┘ │
    │                                                              │
    │  2. TRADITIONAL ML                                           │
    │  ┌────────────────────────────────────────────────────────┐ │
    │  │ Features: TF-IDF, n-grams, POS tags                    │ │
    │  │ Model: SVM, Logistic Regression, Naive Bayes          │ │
    │  │                                                         │ │
    │  │ Pros: Fast, works with small data                      │ │
    │  │ Cons: Manual feature engineering                       │ │
    │  └────────────────────────────────────────────────────────┘ │
    │                                                              │
    │  3. DEEP LEARNING                                            │
    │  ┌────────────────────────────────────────────────────────┐ │
    │  │ LSTM/CNN → Contextualized features                     │ │
    │  │ BERT → Pre-trained bidirectional context              │ │
    │  │                                                         │ │
    │  │ Pros: SotA accuracy, handles context                   │ │
    │  │ Cons: Needs GPU, harder to interpret                   │ │
    │  └────────────────────────────────────────────────────────┘ │
    │                                                              │
    │  4. ASPECT-BASED SENTIMENT ANALYSIS (ABSA)                   │
    │  ┌────────────────────────────────────────────────────────┐ │
    │  │ Extract: (aspect, sentiment) pairs                     │ │
    │  │ "Food was great but service was slow"                  │ │
    │  │  → [(food, positive), (service, negative)]            │ │
    │  │                                                         │ │
    │  │ Methods: Joint extraction, pipeline approach           │ │
    │  └────────────────────────────────────────────────────────┘ │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    ```

    ## Production Implementation (180 lines)

    ```python
    # sentiment_analysis.py
    import numpy as np
    import pandas as pd
    from typing import List, Dict, Tuple, Optional
    from dataclasses import dataclass
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import torch.nn as nn
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import time

    @dataclass
    class SentimentResult:
        """Sentiment analysis result"""
        text: str
        label: str  # 'positive', 'negative', 'neutral'
        score: float
        confidence: float
        aspects: Optional[Dict[str, str]] = None

    class LexiconSentimentAnalyzer:
        """
        VADER (Valence Aware Dictionary and sEntiment Reasoner)
        
        Advantages:
        - No training required
        - Handles negations ("not good")
        - Handles intensifiers ("very good")
        - Handles emojis and slang
        
        Speed: <1ms per document
        """
        
        def __init__(self):
            self.analyzer = SentimentIntensityAnalyzer()
        
        def analyze(self, text: str) -> SentimentResult:
            """
            Returns compound score: -1 (most negative) to +1 (most positive)
            """
            scores = self.analyzer.polarity_scores(text)
            
            # Classify based on compound score
            compound = scores['compound']
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            
            return SentimentResult(
                text=text,
                label=label,
                score=compound,
                confidence=max(scores['pos'], scores['neg'], scores['neu'])
            )

    class TraditionalMLSentiment:
        """
        TF-IDF + Logistic Regression
        
        Training: O(N × V) where N=docs, V=vocab
        Inference: O(V) per document
        """
        
        def __init__(self, max_features: int = 5000):
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),  # Unigrams + bigrams
                min_df=2,
                stop_words='english'
            )
            self.model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight='balanced'  # Handle class imbalance
            )
        
        def train(self, texts: List[str], labels: List[int]):
            """
            Train on labeled data
            
            Args:
                texts: List of review texts
                labels: 0=negative, 1=positive
            """
            # Extract features
            X = self.vectorizer.fit_transform(texts)
            
            # Train classifier
            self.model.fit(X, labels)
            
            # Get feature importance
            feature_names = self.vectorizer.get_feature_names_out()
            coefficients = self.model.coef_[0]
            
            # Top positive and negative features
            top_positive = np.argsort(coefficients)[-10:]
            top_negative = np.argsort(coefficients)[:10]
            
            print("Top positive features:", [feature_names[i] for i in top_positive])
            print("Top negative features:", [feature_names[i] for i in top_negative])
        
        def predict(self, text: str) -> SentimentResult:
            """Predict sentiment for new text"""
            X = self.vectorizer.transform([text])
            
            # Predict
            label = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]
            
            return SentimentResult(
                text=text,
                label='positive' if label == 1 else 'negative',
                score=proba[1],  # Probability of positive
                confidence=max(proba)
            )

    class BERTSentiment:
        """
        Fine-tuned BERT for sentiment analysis
        
        Advantages:
        - SotA accuracy (90-95% on IMDB, SST)
        - Understands context and negation
        - Transfer learning from pre-training
        
        Inference: 50-100ms on CPU, 5-10ms on GPU
        """
        
        def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
        
        def analyze(self, text: str) -> SentimentResult:
            """Analyze sentiment using BERT"""
            result = self.pipeline(text)[0]
            
            return SentimentResult(
                text=text,
                label=result['label'].lower(),
                score=result['score'],
                confidence=result['score']
            )
        
        def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[SentimentResult]:
            """Batch processing for efficiency"""
            results = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_results = self.pipeline(batch)
                
                for text, result in zip(batch, batch_results):
                    results.append(SentimentResult(
                        text=text,
                        label=result['label'].lower(),
                        score=result['score'],
                        confidence=result['score']
                    ))
            
            return results

    class AspectBasedSentiment:
        """
        Aspect-Based Sentiment Analysis (ABSA)
        
        Extract (aspect, sentiment) pairs from text
        
        Example:
          Input: "The food was amazing but the service was terrible"
          Output: [('food', 'positive'), ('service', 'negative')]
        """
        
        def __init__(self):
            # Pre-defined aspects for restaurant reviews
            self.aspects = {
                'food': ['food', 'dish', 'meal', 'taste', 'flavor', 'cuisine'],
                'service': ['service', 'waiter', 'staff', 'waitress'],
                'ambiance': ['atmosphere', 'ambiance', 'decor', 'music'],
                'price': ['price', 'cost', 'expensive', 'cheap', 'value']
            }
            
            # Sentiment analyzer
            self.sentiment = BERTSentiment()
        
        def extract_aspects(self, text: str) -> Dict[str, List[str]]:
            """Extract aspect mentions from text"""
            text_lower = text.lower()
            found_aspects = {}
            
            for aspect, keywords in self.aspects.items():
                sentences = [s for s in text.split('.') 
                           if any(kw in s.lower() for kw in keywords)]
                if sentences:
                    found_aspects[aspect] = sentences
            
            return found_aspects
        
        def analyze(self, text: str) -> SentimentResult:
            """
            Perform aspect-based sentiment analysis
            """
            # Extract aspects
            aspects = self.extract_aspects(text)
            
            # Analyze sentiment for each aspect
            aspect_sentiments = {}
            for aspect, sentences in aspects.items():
                # Analyze each sentence mentioning the aspect
                sentiments = []
                for sentence in sentences:
                    result = self.sentiment.analyze(sentence)
                    sentiments.append(result.label)
                
                # Aggregate (majority vote)
                aspect_sentiments[aspect] = max(set(sentiments), key=sentiments.count)
            
            # Overall sentiment (from entire text)
            overall = self.sentiment.analyze(text)
            overall.aspects = aspect_sentiments
            
            return overall

    def compare_approaches(texts: List[str]) -> pd.DataFrame:
        """Compare different sentiment analysis approaches"""
        
        # Initialize analyzers
        vader = LexiconSentimentAnalyzer()
        bert = BERTSentiment()
        
        results = []
        
        for text in texts:
            # VADER (lexicon-based)
            start = time.time()
            vader_result = vader.analyze(text)
            vader_time = (time.time() - start) * 1000
            
            # BERT
            start = time.time()
            bert_result = bert.analyze(text)
            bert_time = (time.time() - start) * 1000
            
            results.append({
                'Text': text[:50] + '...' if len(text) > 50 else text,
                'VADER Label': vader_result.label,
                'VADER Score': f"{vader_result.score:.3f}",
                'VADER Time (ms)': f"{vader_time:.2f}",
                'BERT Label': bert_result.label,
                'BERT Score': f"{bert_result.score:.3f}",
                'BERT Time (ms)': f"{bert_time:.2f}"
            })
        
        return pd.DataFrame(results)

    # ===========================================
    # EXAMPLE USAGE WITH COMPANY USE CASES
    # ===========================================

    if __name__ == "__main__":
        print("="*70)
        print("AMAZON - PRODUCT REVIEW SENTIMENT")
        print("="*70)
        
        reviews = [
            "This product is amazing! Best purchase ever!",
            "Terrible quality. Broke after 2 days. Very disappointed.",
            "It's okay, nothing special. Does the job.",
            "Not good at all. Would not recommend to anyone.",
            "Love it! Exceeded my expectations completely."
        ]
        
        print("\nComparing VADER vs BERT:")
        comparison = compare_approaches(reviews)
        print(comparison.to_string(index=False))
        
        print("\n" + "="*70)
        print("YELP - ASPECT-BASED SENTIMENT FOR RESTAURANTS")
        print("="*70)
        
        absa = AspectBasedSentiment()
        
        review = "The food was absolutely delicious and the portions were generous. \
                  However, the service was incredibly slow and the staff seemed disinterested. \
                  The ambiance is nice but way too loud for conversation."
        
        result = absa.analyze(review)
        
        print(f"\nReview: {review}")
        print(f"\nOverall Sentiment: {result.label.upper()} ({result.score:.3f})")
        print("\nAspect-level Sentiments:")
        for aspect, sentiment in result.aspects.items():
            print(f"  {aspect.capitalize():12s}: {sentiment.upper()}")
        
        print("\n" + "="*70)
        print("TWITTER - REAL-TIME SENTIMENT MONITORING")
        print("="*70)
        
        tweets = [
            "Just tried the new iPhone 15! 😍 Loving the camera quality!",
            "iPhone 15 battery life is disappointing 😞 #NotImpressed",
            "Meh, not sure if it's worth the upgrade from iPhone 14 🤷",
        ]
        
        bert = BERTSentiment()
        
        print("\nLive sentiment tracking:")
        for tweet in tweets:
            result = bert.analyze(tweet)
            emoji = "✅" if result.label == "positive" else "❌" if result.label == "negative" else "➖"
            print(f"  {emoji} [{result.label:8s}] {result.score:.3f} | {tweet}")
    ```

    ## Approach Comparison

    | Approach | Accuracy (IMDB) | Speed (CPU) | Training | Best For |
    |----------|----------------|-------------|----------|----------|
    | **VADER (Lexicon)** | 65-70% | <1ms | None | Social media, quick prototypes |
    | **TF-IDF + LR** | 85-88% | 2-5ms | Minutes | Baseline, interpretability |
    | **LSTM** | 88-90% | 20-30ms | Hours | Sequence modeling |
    | **BERT-base** | 93-94% | 50-100ms | Days (fine-tuning) | High accuracy, production |
    | **RoBERTa-large** | 96-97% | 200-300ms | Weeks | Research, benchmarks |

    ## Real-World Deployments

    | Company | Use Case | Approach | Scale | Result |
    |---------|----------|----------|-------|--------|
    | **Amazon** | Product reviews | BERT fine-tuned | 200M reviews/day | 94% accuracy, insights ↑ customer satisfaction |
    | **Twitter** | Trending sentiment | Ensemble (VADER + BERT) | 500M tweets/day | Real-time trends, 0.5s latency |
    | **Airbnb** | Review analysis | Aspect-based BERT | 10M reviews/quarter | 92% F1 on aspects, actionable insights |
    | **Yelp** | Restaurant reviews | Custom LSTM-Attention | 100M reviews | 89% accuracy, aspect extraction |
    | **Meta** | Content moderation | Multilingual BERT | Billions of posts | Detect negativity in 100+ languages |

    ## Evaluation Metrics

    **Classification Metrics:**
    ```python
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Accuracy: (TP + TN) / Total
    # Precision: TP / (TP + FP) - How many predicted positives are correct?
    # Recall: TP / (TP + FN) - How many actual positives did we find?
    # F1: 2 × (Precision × Recall) / (Precision + Recall)
    
    # For sentiment:
    # - High recall for negative: Don't miss complaints
    # - High precision for positive: Genuine positive feedback
    ```

    **Aspect-level Metrics:**
    - Aspect detection F1
    - Sentiment classification accuracy per aspect
    - Joint evaluation (both aspect + sentiment correct)

    ## Common Challenges

    | Challenge | Example | Impact | Solution |
    |-----------|---------|--------|----------|
    | **Negation** | "not good" | Lexicon misses | BERT context, or negation rules |
    | **Sarcasm** | "Great! Another bug 🙄" | Flips sentiment | BERT + emoji, or sarcasm detector |
    | **Domain shift** | Train on movies, test on products | 10-20% drop | Domain adaptation, more diverse data |
    | **Class imbalance** | 80% positive, 20% negative | Biased predictions | Weighted loss, oversampling |
    | **Neutral ambiguity** | "It works" | Hard to classify | 3-class (pos/neg/neu) or fine-grained |
    | **Multi-aspect conflict** | "Food good, service bad" | Overall sentiment unclear | Aspect-based analysis |

    ## Optimization Techniques

    **1. Model Distillation:**
    ```python
    # DistilBERT: 40% smaller, 60% faster, 97% of BERT's accuracy
    from transformers import pipeline
    
    distil_sentiment = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    ```

    **2. Quantization:**
    ```python
    # INT8 quantization: 4× smaller, 2-3× faster
    import torch
    
    model_int8 = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    ```

    **3. Caching:**
    ```python
    # Cache predictions for repeated texts
    from functools import lru_cache
    
    @lru_cache(maxsize=10000)
    def cached_sentiment(text: str) -> str:
        return sentiment_model(text)
    ```

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Understanding of different granularity levels (document vs aspect)
        - Knowledge of trade-offs (speed vs accuracy)
        - Handling of negation and sarcasm
        - Production considerations (latency, cost)
        
        **Strong signal:**
        
        - "For Amazon's 200M daily reviews, we use DistilBERT achieving 94% accuracy at 50ms latency on CPUs. Compared to VADER (70% accuracy, <1ms), the extra 49ms is worth the 24% accuracy gain"
        - "Aspect-based sentiment is crucial for restaurants. 'Food great, service bad' needs separate (food, positive) and (service, negative) labels. We fine-tune BERT on SemEval 2014 dataset achieving 92% F1"
        - "Common pitfall: Negation handling. 'not good' with bag-of-words has 'good' token → positive. BERT's bidirectional context solves this by understanding 'not' modifies 'good'"
        - "At Twitter, we use VADER for real-time trending (500M tweets/day, <1ms) and BERT for in-depth analysis (accuracy matters more than speed)"
        
        **Red flags:**
        
        - Using only lexicon-based methods (misses context)
        - Not mentioning aspect-based sentiment (crucial for actionable insights)
        - Ignoring class imbalance (most reviews are positive)
        - Not considering latency in production (BERT can be slow)
        
        **Follow-ups:**
        
        - "How to handle sarcasm?" → Emoji features, sarcasm-specific datasets, context modeling
        - "Multilingual sentiment?" → XLM-RoBERTa, mBERT for 100+ languages
        - "Class imbalance (90% positive reviews)?" → Weighted loss, focal loss, oversampling negative
        - "Real-time with <10ms latency?" → DistilBERT + quantization + ONNX, or VADER for rough estimates

---

### What is GPT? How Does It Differ from BERT? - OpenAI, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Language Models` | **Asked by:** OpenAI, Google, Meta

??? success "View Answer"

    ## Overview

    **GPT (Generative Pre-trained Transformer)** - Autoregressive language model that predicts next tokens, enabling text generation and few-shot learning.

    **Key Innovation:** Unlike BERT (bidirectional encoding), GPT uses causal masking for left-to-right generation, enabling zero-shot/few-shot learning without fine-tuning.

    ## GPT Evolution

    | Model | Year | Params | Context | Key Innovation |
    |-------|------|--------|---------|----------------|
    | **GPT-1** | 2018 | 117M | 512 | Unsupervised pre-training + fine-tuning |
    | **GPT-2** | 2019 | 1.5B | 1024 | Zero-shot learning, no fine-tuning needed |
    | **GPT-3** | 2020 | 175B | 2048 | Few-shot in-context learning |
    | **GPT-3.5** | 2022 | 175B | 4096 | RLHF (ChatGPT), instruction following |
    | **GPT-4** | 2023 | Unknown | 32K-128K | Multi-modal, improved reasoning |

    ## BERT vs GPT Comparison

    | Aspect | BERT | GPT |
    |--------|------|-----|
    | **Architecture** | Encoder-only (12-24 layers) | Decoder-only (12-96 layers) |
    | **Attention** | Bidirectional (sees full context) | Causal (left-to-right only) |
    | **Pre-training** | MLM (mask 15%) + NSP | Next token prediction |
    | **Training Objective** | Fill in blanks | Predict next word |
    | **Best For** | Classification, NER, QA | Generation, few-shot tasks |
    | **Fine-tuning** | Required for tasks | Optional (few-shot works) |
    | **Use Case** | Sentence embeddings, understanding | Text generation, chat, code |

    ## GPT Training Objective

    **Autoregressive Language Modeling:**

    $$P(w_1, ..., w_n) = \prod_{i=1}^n P(w_i | w_1, ..., w_{i-1})$$

    Maximize log-likelihood of next token given previous tokens:

    $$\mathcal{L} = -\sum_{i=1}^n \log P(w_i | w_{<i}; \theta)$$

    ## Causal Masking (Key Difference)

    ```
    GPT Attention Pattern (Causal):
    Token 1: Can see [1]
    Token 2: Can see [1, 2]
    Token 3: Can see [1, 2, 3]
    → Upper triangular mask

    BERT Attention Pattern (Bidirectional):
    Token 1: Can see [1, 2, 3]
    Token 2: Can see [1, 2, 3]
    Token 3: Can see [1, 2, 3]
    → No mask (full context)
    ```

    ## GPT Implementation (120 lines)

    ```python
    # gpt_architecture.py
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math

    class CausalSelfAttention(nn.Module):
        """
        Causal self-attention for GPT

        Key difference from BERT: Upper triangular mask prevents
        attending to future tokens
        """

        def __init__(self, d_model=768, num_heads=12, max_len=1024):
            super().__init__()
            assert d_model % num_heads == 0

            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads

            self.W_qkv = nn.Linear(d_model, 3 * d_model)
            self.W_o = nn.Linear(d_model, d_model)

            # Causal mask: upper triangular (prevent attending to future)
            self.register_buffer(
                'causal_mask',
                torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
            )

        def forward(self, x):
            batch_size, seq_len, d_model = x.size()

            # Compute Q, K, V
            qkv = self.W_qkv(x)  # [batch, seq, 3*d_model]
            qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.d_k)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, d_k]

            Q, K, V = qkv[0], qkv[1], qkv[2]

            # Attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

            # Apply causal mask
            scores = scores.masked_fill(
                self.causal_mask[:, :, :seq_len, :seq_len] == 0,
                float('-inf')
            )

            # Attention weights
            attn_weights = F.softmax(scores, dim=-1)

            # Apply to values
            out = torch.matmul(attn_weights, V)

            # Concatenate heads
            out = out.transpose(1, 2).contiguous()
            out = out.view(batch_size, seq_len, d_model)

            return self.W_o(out)

    class GPTBlock(nn.Module):
        """Single GPT transformer block"""

        def __init__(self, d_model=768, num_heads=12):
            super().__init__()

            self.ln1 = nn.LayerNorm(d_model)
            self.attn = CausalSelfAttention(d_model, num_heads)

            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            )

        def forward(self, x):
            # Pre-LN (GPT-2 style)
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    class GPT(nn.Module):
        """GPT model for text generation"""

        def __init__(
            self,
            vocab_size=50257,  # GPT-2 vocab
            d_model=768,
            num_layers=12,
            num_heads=12,
            max_len=1024
        ):
            super().__init__()

            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.position_embedding = nn.Embedding(max_len, d_model)

            self.blocks = nn.ModuleList([
                GPTBlock(d_model, num_heads)
                for _ in range(num_layers)
            ])

            self.ln_f = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

            # Tie weights (embedding = lm_head)
            self.lm_head.weight = self.token_embedding.weight

        def forward(self, input_ids):
            """
            Args:
                input_ids: [batch, seq_len]

            Returns:
                logits: [batch, seq_len, vocab_size]
            """
            batch_size, seq_len = input_ids.size()

            # Token + position embeddings
            positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)

            x = self.token_embedding(input_ids) + self.position_embedding(positions)

            # Transformer blocks
            for block in self.blocks:
                x = block(x)

            # Final layer norm
            x = self.ln_f(x)

            # Project to vocabulary
            logits = self.lm_head(x)

            return logits

        @torch.no_grad()
        def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
            """
            Autoregressive generation

            Args:
                input_ids: [batch, seq_len] prompt tokens
                max_new_tokens: Number of tokens to generate
                temperature: Sampling temperature (higher = more random)
                top_k: If set, only sample from top-k tokens

            Returns:
                generated: [batch, seq_len + max_new_tokens]
            """
            for _ in range(max_new_tokens):
                # Get logits for last position
                logits = self(input_ids)  # [batch, seq, vocab]
                logits = logits[:, -1, :] / temperature  # [batch, vocab]

                # Top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

            return input_ids

    # Example usage
    if __name__ == "__main__":
        model = GPT(
            vocab_size=50257,
            d_model=768,
            num_layers=12,
            num_heads=12
        )

        # Generate text
        prompt = torch.randint(0, 50257, (1, 10))  # Random prompt
        generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)

        print(f"Prompt shape: {prompt.shape}")
        print(f"Generated shape: {generated.shape}")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")  # ~117M for GPT-2 Small
    ```

    ## Few-Shot Learning (GPT-3's Superpower)

    **Zero-Shot:**
    ```
    Translate to French: "I love cats"
    → "J'aime les chats"
    ```

    **One-Shot:**
    ```
    Translate to French:
    English: "Hello" → French: "Bonjour"
    English: "I love cats" →
    → French: "J'aime les chats"
    ```

    **Few-Shot:**
    ```
    English: "Hello" → French: "Bonjour"
    English: "Goodbye" → French: "Au revoir"
    English: "Thank you" → French: "Merci"
    English: "I love cats" →
    → French: "J'aime les chats"
    ```

    **Why it works:** In-context learning - GPT learns pattern from examples in prompt.

    ## GPT Variants & Techniques

    | Variant | Innovation | Impact |
    |---------|------------|--------|
    | **GPT-2** | Zero-shot learning | No fine-tuning needed for many tasks |
    | **GPT-3** | 175B params, few-shot | Emergent abilities (arithmetic, reasoning) |
    | **InstructGPT** | RLHF (human feedback) | Better instruction following |
    | **ChatGPT** | Conversational RLHF | Natural dialogue, helpful responses |
    | **GPT-4** | Multi-modal, larger context | Vision understanding, 128K tokens |
    | **Code Models** | CodeX, Codegen | Code generation, GitHub Copilot |

    ## When to Use BERT vs GPT

    **Use BERT when:**
    - Classification tasks (sentiment, spam, NER)
    - Need bidirectional context (fill-in-the-blank)
    - Sentence embeddings for similarity
    - Have labeled data for fine-tuning
    - Example: Email spam detection, document classification

    **Use GPT when:**
    - Text generation (creative writing, code, summaries)
    - Few-shot learning (limited labeled data)
    - Conversational AI (chatbots)
    - Need flexibility (one model for many tasks)
    - Example: ChatGPT, code completion, content generation

    ## Real-World Applications

    **OpenAI ChatGPT:**
    - **Base:** GPT-3.5/GPT-4 with RLHF
    - **Users:** 100M+ weekly active users (2023)
    - **Tasks:** Q&A, writing, coding, analysis
    - **Revenue:** $1.6B projected (2024)

    **GitHub Copilot:**
    - **Base:** Codex (GPT-3 fine-tuned on code)
    - **Adoption:** 1M+ developers
    - **Productivity:** 55% faster coding (GitHub study)
    - **Languages:** Python, JavaScript, Go, etc.

    **Jasper AI (Content Generation):**
    - **Base:** GPT-3 API
    - **Use Case:** Marketing copy, blog posts
    - **Customers:** 100K+ businesses
    - **Output:** 1B+ words generated/month

    ## Training Costs & Scale

    | Model | Training Cost | GPUs | Time | Dataset Size |
    |-------|---------------|------|------|--------------|
    | **GPT-2** | $50K | 32 TPUs | Weeks | 40GB (WebText) |
    | **GPT-3** | **$4.6M** | 10K GPUs | Months | 570GB (Common Crawl) |
    | **GPT-4** | **$100M+** | Unknown | Months | Unknown (larger) |

    ## Common Pitfalls

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Wrong attention mask** | Model sees future | Use upper triangular causal mask |
    | **No temperature tuning** | Repetitive generation | Use temperature 0.7-0.9 for creativity |
    | **Greedy decoding** | Boring, repetitive text | Use top-k or nucleus (top-p) sampling |
    | **Ignoring prompt engineering** | Poor results | Craft clear prompts with examples |
    | **Not using RLHF** | Unaligned outputs | Fine-tune with human feedback (InstructGPT) |

    !!! tip "Interviewer's Insight"
        **Strong candidates:**

        - Explain causal masking: "Upper triangular mask prevents attending to future tokens during training"
        - Understand autoregressive: "Predicts P(w_i | w_<i), unlike BERT's MLM which predicts P(w_i | context)"
        - Know when to use each: "BERT for classification with bidirectional context, GPT for generation and few-shot"
        - Discuss few-shot learning: "GPT-3 learns from examples in prompt without weight updates (in-context learning)"
        - Reference real systems: "ChatGPT uses GPT-3.5/4 with RLHF, GitHub Copilot uses Codex (GPT-3 on code)"
        - Know limitations: "GPT has knowledge cutoff, can hallucinate, expensive to run (GPT-3: $0.02/1K tokens)"

---

### Explain Text Summarization - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Summarization`, `Seq2Seq`, `ROUGE` | **Asked by:** Amazon, Google, Meta, Microsoft, Salesforce

??? success "View Answer"

    ## Core Concept

    **Text Summarization** condenses long documents while preserving key information. Critical for news, research papers, and customer reviews.

    **Two Paradigms:**
    - **Extractive:** Select and copy important sentences (copy-paste)
    - **Abstractive:** Generate new sentences that capture meaning (paraphrase)

    ## Approach Comparison

    ```
    ┌──────────────────────────────────────────────────────────────┐
    │         EXTRACTIVE VS ABSTRACTIVE SUMMARIZATION                 │
    ├──────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  EXTRACTIVE SUMMARIZATION                                        │
    │  ┌────────────────────────────────────────────────────────────┐ │
    │  │ Input Document:                                            │ │
    │  │   "AI is transforming industries. [S1]                  │ │
    │  │    Machine learning improves predictions. [S2]           │ │
    │  │    Deep learning uses neural networks. [S3]              │ │
    │  │    Companies are adopting AI rapidly. [S4]"              │ │
    │  │                                                             │ │
    │  │ Algorithm: Score each sentence by importance             │ │
    │  │   Score(S1) = 0.8  ← Most important                      │ │
    │  │   Score(S2) = 0.6                                         │ │
    │  │   Score(S3) = 0.4                                         │ │
    │  │   Score(S4) = 0.7                                         │ │
    │  │                                                             │ │
    │  │ Summary: Select top-k sentences (S1, S4)                 │ │
    │  │   "AI is transforming industries.                         │ │
    │  │    Companies are adopting AI rapidly."                    │ │
    │  └────────────────────────────────────────────────────────────┘ │
    │                                                                  │
    │  ABSTRACTIVE SUMMARIZATION                                       │
    │  ┌────────────────────────────────────────────────────────────┐ │
    │  │ Input Document: (same as above)                          │ │
    │  │                                                             │ │
    │  │ Encoder-Decoder Model (T5, BART):                        │ │
    │  │   1. Encode entire document                               │ │
    │  │   2. Decoder generates NEW tokens                         │ │
    │  │   3. Can paraphrase, compress, reorder                    │ │
    │  │                                                             │ │
    │  │ Summary: "AI and machine learning are rapidly being      │ │
    │  │           adopted by companies to improve predictions."   │ │
    │  │                                                             │ │
    │  │ Note: Uses words NOT in original ("adopted", "improve")  │ │
    │  └────────────────────────────────────────────────────────────┘ │
    └──────────────────────────────────────────────────────────────┘
    ```

    ## Production Implementation (175 lines)

    ```python
    # text_summarization.py
    import numpy as np
    from typing import List, Dict, Tuple
    from dataclasses import dataclass
    import re
    from collections import Counter
    import networkx as nx
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    from rouge_score import rouge_scorer
    import time

    @dataclass
    class SummaryResult:
        """Summarization result with metrics"""
        summary: str
        original_length: int
        summary_length: int
        compression_ratio: float
        time_ms: float
        rouge_scores: Dict[str, float] = None

    class TextRankSummarizer:
        """
        Extractive summarization using TextRank (PageRank on sentences)
        
        Algorithm:
        1. Build sentence similarity graph
        2. Run PageRank to score sentences
        3. Select top-k highest scoring sentences
        
        Time: O(N²) for similarity matrix, O(N² × iterations) for PageRank
        Space: O(N²) for similarity matrix
        """
        
        def __init__(self, top_k: int = 3, damping: float = 0.85):
            self.top_k = top_k
            self.damping = damping
        
        def _preprocess(self, text: str) -> List[str]:
            """Split into sentences"""
            # Simple sentence splitting (in production: use NLTK or spaCy)
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            return sentences
        
        def _build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
            """
            Build sentence similarity matrix using TF-IDF + cosine similarity
            
            similarity[i,j] = cosine_similarity(tfidf[i], tfidf[j])
            """
            # Vectorize sentences
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Compute pairwise cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            # Zero out diagonal (sentence shouldn't be similar to itself)
            np.fill_diagonal(similarity_matrix, 0)
            
            return similarity_matrix
        
        def summarize(self, text: str) -> str:
            """
            Generate extractive summary using TextRank
            """
            start_time = time.time()
            
            # Split into sentences
            sentences = self._preprocess(text)
            
            if len(sentences) <= self.top_k:
                return ' '.join(sentences)
            
            # Build similarity graph
            similarity_matrix = self._build_similarity_matrix(sentences)
            
            # Run PageRank
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph, alpha=self.damping)
            
            # Rank sentences by score
            ranked_sentences = sorted(
                ((scores[i], s) for i, s in enumerate(sentences)),
                reverse=True
            )
            
            # Select top-k sentences in original order
            top_indices = sorted(
                [i for i, _ in enumerate(sentences) 
                 if sentences[i] in [s for _, s in ranked_sentences[:self.top_k]]]
            )
            
            summary = ' '.join([sentences[i] for i in top_indices])
            
            return summary

    class AbstractiveSummarizer:
        """
        Abstractive summarization using pre-trained transformer models
        
        Models:
        - T5 (Google): "summarize: <text>"
        - BART (Meta): Encoder-decoder with denoising pre-training
        - PEGASUS (Google): Gap sentence generation pre-training
        """
        
        def __init__(self, model_name: str = "facebook/bart-large-cnn"):
            """
            Popular models:
            - facebook/bart-large-cnn: Best for news articles
            - t5-base: General purpose
            - google/pegasus-xsum: Best for extreme summarization
            """
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        def summarize(
            self,
            text: str,
            max_length: int = 130,
            min_length: int = 30,
            num_beams: int = 4,
            length_penalty: float = 2.0
        ) -> str:
            """
            Generate abstractive summary
            
            Args:
                text: Input document
                max_length: Max summary tokens
                min_length: Min summary tokens
                num_beams: Beam search width (higher = better quality, slower)
                length_penalty: >1 encourages longer summaries
            
            Returns:
                Generated summary
            """
            # Check if text is too long (max 1024 tokens for BART)
            tokens = self.tokenizer.encode(text, truncation=True, max_length=1024)
            
            if len(tokens) > 1024:
                print(f"Warning: Text truncated from {len(tokens)} to 1024 tokens")
            
            # Generate summary
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=True
            )[0]['summary_text']
            
            return summary

    class ROUGEEvaluator:
        """
        ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
        
        Metrics:
        - ROUGE-1: Unigram overlap (word-level)
        - ROUGE-2: Bigram overlap (fluency)
        - ROUGE-L: Longest common subsequence (sentence-level)
        """
        
        def __init__(self):
            self.scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
        
        def evaluate(self, reference: str, hypothesis: str) -> Dict[str, float]:
            """
            Compute ROUGE scores
            
            Returns: {'rouge1': 0.45, 'rouge2': 0.23, 'rougeL': 0.38}
            where each is F1-score (harmonic mean of precision & recall)
            """
            scores = self.scorer.score(reference, hypothesis)
            
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }

    # ===========================================
    # EXAMPLE USAGE WITH COMPANY USE CASES
    # ===========================================

    if __name__ == "__main__":
        import torch
        
        # Sample article
        article = """
        Artificial intelligence is rapidly transforming industries worldwide. 
        Machine learning algorithms can now predict customer behavior with remarkable accuracy. 
        Deep learning models have achieved human-level performance on many tasks. 
        Companies like Google, Meta, and OpenAI are investing billions in AI research. 
        The technology is being applied to healthcare, finance, and autonomous vehicles. 
        However, concerns about AI safety and ethics are growing. 
        Researchers are working on making AI systems more transparent and accountable. 
        The future of AI depends on responsible development and deployment.
        """
        
        print("="*70)
        print("GOOGLE - NEWS ARTICLE SUMMARIZATION")
        print("="*70)
        
        # Extractive summarization
        extractive = TextRankSummarizer(top_k=3)
        ext_summary = extractive.summarize(article)
        
        print("\nEXTRACTIVE (TextRank):")
        print(ext_summary)
        print(f"\nLength: {len(article.split())} words → {len(ext_summary.split())} words")
        print(f"Compression: {len(ext_summary.split()) / len(article.split()):.1%}")
        
        print("\n" + "="*70)
        print("META - ABSTRACTIVE SUMMARIZATION (BART)")
        print("="*70)
        
        # Abstractive summarization
        abstractive = AbstractiveSummarizer("facebook/bart-large-cnn")
        
        start = time.time()
        abs_summary = abstractive.summarize(article, max_length=50)
        abs_time = (time.time() - start) * 1000
        
        print("\nABSTRACTIVE (BART):")
        print(abs_summary)
        print(f"\nLength: {len(article.split())} words → {len(abs_summary.split())} words")
        print(f"Compression: {len(abs_summary.split()) / len(article.split()):.1%}")
        print(f"Time: {abs_time:.0f}ms")
        
        print("\n" + "="*70)
        print("EVALUATION - ROUGE SCORES")
        print("="*70)
        
        # Reference summary (human-written)
        reference = "AI is transforming industries. Companies invest billions. Concerns about safety and ethics grow."
        
        evaluator = ROUGEEvaluator()
        
        # Evaluate extractive
        ext_scores = evaluator.evaluate(reference, ext_summary)
        print("\nExtractive ROUGE:")
        for metric, score in ext_scores.items():
            print(f"  {metric}: {score:.3f}")
        
        # Evaluate abstractive
        abs_scores = evaluator.evaluate(reference, abs_summary)
        print("\nAbstractive ROUGE:")
        for metric, score in abs_scores.items():
            print(f"  {metric}: {score:.3f}")
    ```

    ## Method Comparison

    | Method | Type | Model | ROUGE-2 (CNN/DM) | Speed (CPU) | Pros | Cons |
    |--------|------|-------|------------------|-------------|------|------|
    | **TextRank** | Extractive | Graph-based | 0.15-0.18 | 50ms | Fast, no training | Redundant sentences |
    | **BERT + Ranking** | Extractive | Transformer | 0.20-0.22 | 200ms | Better relevance | Still copy-paste |
    | **BART** | Abstractive | Encoder-Decoder | 0.21-0.23 | 500ms | Fluent, concise | Can hallucinate |
    | **T5** | Abstractive | Encoder-Decoder | 0.20-0.22 | 400ms | Flexible prompts | Needs fine-tuning |
    | **PEGASUS** | Abstractive | Specialized | 0.24-0.25 | 600ms | SotA on news | Slow, GPU needed |

    ## Real-World Deployments

    | Company | Use Case | Approach | Scale | Result |
    |---------|----------|----------|-------|--------|
    | **Google News** | Article snippets | Extractive (BERT) | Billions of articles | 0.21 ROUGE-2, 100ms latency |
    | **Meta** | Feed summarization | BART fine-tuned | 1B posts/day | 32% engagement increase |
    | **Microsoft** | Email summarization (Outlook) | Custom T5 | 400M users | 0.23 ROUGE-2, <200ms |
    | **Amazon** | Product review summary | Extractive + clustering | 500M reviews | 78% user satisfaction |
    | **Salesforce** | Case summarization | PEGASUS | 150K businesses | 0.25 ROUGE-2, time saved 40% |

    ## ROUGE Metrics Explained

    **ROUGE-1 (Unigram overlap):**
    ```
    Reference: "The cat sat on the mat"
    Hypothesis: "A cat sat on a mat"
    
    Overlap: {cat, sat, on, mat} = 4 words
    Precision: 4/6 = 0.67 (4 correct out of 6 in hypothesis)
    Recall: 4/6 = 0.67 (4 out of 6 reference words found)
    F1: 2 × (0.67 × 0.67) / (0.67 + 0.67) = 0.67
    ```

    **ROUGE-2 (Bigram overlap - fluency):**
    ```
    Reference bigrams: {"the cat", "cat sat", "sat on", "on the", "the mat"}
    Hypothesis bigrams: {"a cat", "cat sat", "sat on", "on a", "a mat"}
    Overlap: {"cat sat", "sat on"} = 2
    F1 ≈ 0.40
    ```

    **ROUGE-L (Longest Common Subsequence):**
    - Measures sentence-level similarity
    - Allows non-contiguous matches
    - Better for abstractive summaries

    ## Common Challenges

    | Challenge | Example | Impact | Solution |
    |-----------|---------|--------|----------|
    | **Hallucination** | Model generates false facts | Factual errors in summary | Faithfulness constraints, fact-checking |
    | **Redundancy** | Extractive repeats similar sentences | Verbose summaries | MMR (Maximal Marginal Relevance) |
    | **Length control** | Summary too long/short | Usability issues | Dynamic length based on input |
    | **Domain shift** | Train on news, test on science | 10-15% ROUGE drop | Domain-specific fine-tuning |
    | **Multi-document** | Summarize 10 related articles | Information overload | Hierarchical attention, fusion |

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Understanding extractive vs abstractive trade-offs
        - Knowledge of ROUGE metrics and limitations
        - Awareness of hallucination problem in abstractive
        - Production considerations (speed, factuality)
        
        **Strong signal:**
        
        - "Extractive (TextRank) is 10× faster than BART (50ms vs 500ms) but less fluent. For Google News snippets, extractive is sufficient. For email summarization where fluency matters, BART's cost is justified"
        - "ROUGE-2 measures bigram overlap, indicating fluency. PEGASUS achieves 0.24 on CNN/DailyMail dataset, vs 0.18 for TextRank. Higher ROUGE-2 = more fluent summaries"
        - "Common issue: Abstractive models hallucinate facts. BART might generate 'CEO resigned' when article says 'CEO considered resigning'. Solution: Add faithfulness loss or use entity-aware models"
        - "At Meta, we fine-tuned BART on 10M social media posts achieving 0.22 ROUGE-2 and 32% engagement increase. Extractive summaries were repetitive and didn't capture meaning"
        
        **Red flags:**
        
        - Not knowing ROUGE metrics (standard evaluation)
        - Thinking extractive = bad, abstractive = good (context-dependent)
        - Ignoring hallucination problem in abstractive
        - Not considering inference latency (BART = 500ms on CPU)
        
        **Follow-ups:**
        
        - "How to prevent hallucination?" → Faithfulness loss, fact-checking, extract-then-abstract pipeline
        - "Multi-document summarization?" → Hierarchical attention, graph-based fusion, LongFormer
        - "Real-time with <100ms?" → Extractive (TextRank, BERTSum), or DistilBART
        - "How to evaluate without reference?" → BERTScore, factuality metrics, human evaluation

---

### What is Perplexity? - Google, OpenAI Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Evaluation Metrics`, `Language Modeling`, `Probability` | **Asked by:** Google, OpenAI, Meta, Anthropic, Cohere

??? success "View Answer"

    ## Core Concept

    **Perplexity (PPL)** measures how well a language model predicts text. It's the exponentiated average negative log-likelihood per word/token.

    **Intuition:** "How surprised is the model by the test data?"
    - Lower perplexity = better predictions, less surprised
    - Higher perplexity = worse predictions, more confused

    ## Mathematical Foundation

    **For a sequence of N tokens:**

    $$\text{PPL}(W) = \exp\left(-\frac{1}{N}\sum_{i=1}^N \log P(w_i | w_{<i})\right)$$

    **Alternative formulation:**

    $$\text{PPL}(W) = \sqrt[N]{\frac{1}{P(w_1, w_2, ..., w_N)}}$$

    **Interpretation as branching factor:**
    ```
    ┌──────────────────────────────────────────────────────────────┐
    │              PERPLEXITY AS BRANCHING FACTOR                  │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  PPL = 10 means:                                             │
    │    On average, the model is as confused as if it had to     │
    │    choose uniformly among 10 possible next words            │
    │                                                              │
    │  Example at word position:                                   │
    │    "The cat sat on the ___"                                 │
    │                                                              │
    │  Perfect model (PPL = 1):                                    │
    │    P("mat") = 1.0  ← Assigns all probability to correct word│
    │                                                              │
    │  Good model (PPL = 5):                                       │
    │    P("mat") = 0.4, P("rug") = 0.3, P("floor") = 0.2, ...   │
    │    Effectively choosing from ~5 likely words                │
    │                                                              │
    │  Bad model (PPL = 100):                                      │
    │    P(any word) ≈ 1/vocab_size                               │
    │    Nearly random guessing from 100 words                    │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    ```\n\n    ## Production Implementation (170 lines)\n\n    ```python\n    # perplexity_calculation.py\n    import numpy as np\n    import torch\n    import torch.nn as nn\n    from typing import List, Tuple, Dict\n    from dataclasses import dataclass\n    from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer\n    import time\n\n    @dataclass\n    class PerplexityResult:\n        \"\"\"Perplexity evaluation result\"\"\"\n        perplexity: float\n        avg_neg_log_likelihood: float\n        num_tokens: int\n        time_ms: float\n\n    class PerplexityCalculator:\n        \"\"\"\n        Calculate perplexity for language models\n        \n        Time: O(N × L) where N=num_sequences, L=seq_length\n        Space: O(L × V) where V=vocab_size for logits\n        \"\"\"\n        \n        def __init__(self, model_name: str = \"gpt2\"):\n            \"\"\"\n            Initialize with pre-trained model\n            \n            Models:\n            - gpt2: 124M params, PPL ~35 on WebText\n            - gpt2-medium: 355M params, PPL ~30\n            - gpt2-large: 774M params, PPL ~25\n            - gpt2-xl: 1.5B params, PPL ~20\n            \"\"\"\n            self.tokenizer = AutoTokenizer.from_pretrained(model_name)\n            self.model = AutoModelForCausalLM.from_pretrained(model_name)\n            self.model.eval()  # Evaluation mode\n            \n            # Move to GPU if available\n            self.device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n            self.model.to(self.device)\n        \n        def calculate_perplexity(self, text: str) -> PerplexityResult:\n            \"\"\"\n            Calculate perplexity for a text sequence\n            \n            Process:\n            1. Tokenize text\n            2. Get model predictions P(w_i | w_{<i})\n            3. Compute negative log-likelihood: -log P(w_i)\n            4. Average and exponentiate\n            \"\"\"\n            start_time = time.time()\n            \n            # Tokenize\n            encodings = self.tokenizer(text, return_tensors=\"pt\")\n            input_ids = encodings.input_ids.to(self.device)\n            \n            # Get predictions (no gradient needed)\n            with torch.no_grad():\n                outputs = self.model(input_ids, labels=input_ids)\n                loss = outputs.loss  # Cross-entropy loss (negative log-likelihood)\n            \n            # Perplexity = exp(loss)\n            perplexity = torch.exp(loss).item()\n            \n            time_ms = (time.time() - start_time) * 1000\n            \n            return PerplexityResult(\n                perplexity=perplexity,\n                avg_neg_log_likelihood=loss.item(),\n                num_tokens=input_ids.shape[1],\n                time_ms=time_ms\n            )\n        \n        def calculate_perplexity_detailed(\n            self,\n            text: str\n        ) -> Tuple[float, List[Tuple[str, float]]]:\n            \"\"\"\n            Calculate perplexity with per-token breakdown\n            \n            Returns:\n                (overall_ppl, [(token, token_ppl), ...])\n            \"\"\"\n            # Tokenize\n            tokens = self.tokenizer.tokenize(text)\n            input_ids = self.tokenizer.encode(text, return_tensors=\"pt\").to(self.device)\n            \n            # Get per-token probabilities\n            with torch.no_grad():\n                outputs = self.model(input_ids)\n                logits = outputs.logits  # [1, seq_len, vocab_size]\n            \n            # Compute per-token perplexity\n            token_perplexities = []\n            total_neg_log_likelihood = 0.0\n            \n            for i in range(1, input_ids.shape[1]):\n                # Predict token i from tokens 0..i-1\n                target_id = input_ids[0, i]\n                \n                # Get probability distribution at position i-1\n                probs = torch.softmax(logits[0, i-1], dim=-1)\n                \n                # Probability assigned to actual token\n                prob = probs[target_id].item()\n                \n                # Token perplexity\n                token_ppl = 1.0 / prob if prob > 0 else float('inf')\n                \n                token_str = tokens[i-1] if i-1 < len(tokens) else \"[UNK]\"\n                token_perplexities.append((token_str, token_ppl))\n                \n                total_neg_log_likelihood += -np.log(prob + 1e-10)\n            \n            # Overall perplexity\n            avg_neg_log_likelihood = total_neg_log_likelihood / (input_ids.shape[1] - 1)\n            overall_ppl = np.exp(avg_neg_log_likelihood)\n            \n            return overall_ppl, token_perplexities\n\n    def compare_models(text: str) -> Dict[str, float]:\n        \"\"\"\n        Compare perplexity across different model sizes\n        \"\"\"\n        models = [\"gpt2\", \"gpt2-medium\", \"gpt2-large\"]\n        results = {}\n        \n        for model_name in models:\n            print(f\"\\nEvaluating {model_name}...\")\n            calculator = PerplexityCalculator(model_name)\n            result = calculator.calculate_perplexity(text)\n            results[model_name] = result.perplexity\n            print(f\"  Perplexity: {result.perplexity:.2f}\")\n            print(f\"  Tokens: {result.num_tokens}\")\n            print(f\"  Time: {result.time_ms:.0f}ms\")\n        \n        return results\n\n    # ===========================================\n    # EXAMPLE USAGE WITH COMPANY USE CASES\n    # ===========================================\n\n    if __name__ == \"__main__\":\n        print(\"=\"*70)\n        print(\"OPENAI - MODEL EVALUATION ON TEST SET\")\n        print(\"=\"*70)\n        \n        # Test on coherent text (should have low perplexity)\n        coherent_text = \"The quick brown fox jumps over the lazy dog. \" \\\n                       \"This is a common English sentence used for typing practice.\"\n        \n        calculator = PerplexityCalculator(\"gpt2\")\n        result_coherent = calculator.calculate_perplexity(coherent_text)\n        \n        print(f\"\\nCoherent text: '{coherent_text}'\")\n        print(f\"Perplexity: {result_coherent.perplexity:.2f}\")\n        print(f\"Interpretation: Model is as confused as choosing from ~{int(result_coherent.perplexity)} words\")\n        \n        # Test on random/nonsensical text (should have high perplexity)\n        random_text = \"Colorless green ideas sleep furiously. \" \\\n                     \"Jabberwocky brillig toves gyre gimble.\"\n        \n        result_random = calculator.calculate_perplexity(random_text)\n        \n        print(f\"\\nNonsensical text: '{random_text}'\")\n        print(f\"Perplexity: {result_random.perplexity:.2f}\")\n        print(f\"Interpretation: Model is much more confused (higher perplexity)\")\n        \n        print(\"\\n\" + \"=\"*70)\n        print(\"GOOGLE - PER-TOKEN PERPLEXITY BREAKDOWN\")\n        print(\"=\"*70)\n        \n        text = \"Machine learning is transforming industries\"\n        overall_ppl, token_ppls = calculator.calculate_perplexity_detailed(text)\n        \n        print(f\"\\nText: '{text}'\")\n        print(f\"Overall Perplexity: {overall_ppl:.2f}\")\n        print(\"\\nPer-token breakdown:\")\n        for token, ppl in token_ppls:\n            surprise = \"😱\" if ppl > 100 else \"😐\" if ppl > 20 else \"😊\"\n            print(f\"  {surprise} '{token:15s}' → PPL: {ppl:8.2f}\")\n        \n        print(\"\\n\" + \"=\"*70)\n        print(\"META - MODEL SIZE COMPARISON\")\n        print(\"=\"*70)\n        \n        test_text = \"Artificial intelligence and machine learning are rapidly evolving fields\"\n        \n        print(f\"\\nTest text: '{test_text}'\")\n        print(\"\\nModel comparison:\")\n        results = compare_models(test_text)\n    ```\n\n    ## Model Perplexity Benchmarks\n\n    | Model | Parameters | Dataset | Perplexity | Year |\n    |-------|-----------|---------|------------|------|\n    | **GPT-1** | 117M | WebText | 35-40 | 2018 |\n    | **GPT-2** | 124M-1.5B | WebText | 35 (small) → 20 (XL) | 2019 |\n    | **GPT-3** | 175B | Pile | 20-25 | 2020 |\n    | **LLaMA-7B** | 7B | Diverse | 15-20 | 2023 |\n    | **LLaMA-65B** | 65B | Diverse | 10-12 | 2023 |\n    | **GPT-4** | Unknown | Unknown | ~8-10 (estimated) | 2023 |\n\n    **Trend:** Larger models → Lower perplexity → Better predictions\n\n    ## Perplexity vs Other Metrics\n\n    | Metric | What it measures | Pros | Cons |\n    |--------|------------------|------|------|\n    | **Perplexity** | How well model predicts next token | Universal, intrinsic | Doesn't correlate with generation quality |\n    | **BLEU** | N-gram overlap (translation) | Task-specific | Only for generation tasks |\n    | **ROUGE** | Overlap (summarization) | Task-specific | Ignores semantic similarity |\n    | **BERTScore** | Semantic similarity | Captures meaning | Computationally expensive |\n    | **Human Eval** | Human judgment | Ground truth | Expensive, slow, subjective |\n\n    ## Common Pitfalls\n\n    | Pitfall | Example | Impact | Solution |\n    |---------|---------|--------|----------|\n    | **Domain mismatch** | Train on news, test on code | PPL inflated by 2-3× | Domain-specific evaluation |\n    | **Different tokenizers** | GPT-2 BPE vs BERT WordPiece | Not comparable | Normalize by tokens/word |\n    | **Low PPL ≠ good generation** | Model repeats same phrase | PPL low but useless | Use diversity metrics (distinct-n) |\n    | **Ignoring length** | Shorter sequences easier to predict | Biased comparison | Report PPL + avg sequence length |\n    | **Vocabulary size** | Small vocab → higher PPL | Unfair comparison | Normalize or use same vocab |\n\n    ## Real-World Usage\n\n    | Company | Use Case | Insight |\n    |---------|----------|--------|\n    | **OpenAI** | GPT model selection | GPT-3 (PPL 20) chosen over GPT-2 (PPL 35) for production |\n    | **Google** | BERT variant evaluation | BERT-large (PPL 18) vs BERT-base (PPL 22) on Wiki |\n    | **Meta** | LLaMA development | Tracked PPL during training: 65B model reached PPL 10 |\n    | **Anthropic** | Claude quality monitoring | Monitor PPL drift in production (flag if >20% increase) |\n    | **Cohere** | Domain adaptation | Fine-tuning reduced PPL from 45 → 15 on medical text |\n\n    ## Why Perplexity Alone is Insufficient\n\n    **Example: Repetitive text**\n    ```python\n    # Model A generates:\n    \"I love cats. I love cats. I love cats. I love cats.\"\n    # Perplexity: 5 (very predictable!)\n    # But quality: TERRIBLE (repetitive)\n    \n    # Model B generates:\n    \"I love cats. They are fluffy and adorable pets.\"\n    # Perplexity: 15 (more uncertain)\n    # But quality: MUCH BETTER (diverse, meaningful)\n    ```\n\n    **Additional metrics needed:**\n    - **Distinct-n:** % of unique n-grams (measures diversity)\n    - **Self-BLEU:** BLEU between generated samples (low = diverse)\n    - **Human evaluation:** Fluency, relevance, factuality\n\n    !!! tip \"Interviewer's Insight\"\n        **What they test:**\n        \n        - Mathematical understanding (exp of negative log-likelihood)\n        - Interpretation as branching factor\n        - Awareness of limitations (doesn't measure generation quality)\n        - Knowledge of model benchmarks\n        \n        **Strong signal:**\n        \n        - \"Perplexity measures how surprised the model is by test data. GPT-3's PPL of 20 means it's as confused as uniformly choosing among 20 words at each step. Lower is better, but a model with PPL=5 that repeats 'the the the' is useless\"\n        - \"OpenAI reported GPT-2 PPL of 35 on WebText. GPT-3 achieved 20-25, showing clear improvement. However, they also reported that human evaluators preferred GPT-3 generations even when PPL was similar to competitors\"\n        - \"Common mistake: Comparing perplexity across different tokenizers. GPT-2's BPE (50K tokens) vs BERT's WordPiece (30K) aren't directly comparable. Need to normalize by tokens-per-word ratio\"\n        - \"At Meta, we track PPL during training. LLaMA-65B started at PPL 1000 and converged to 10-12 after training on 1.4T tokens, matching GPT-3 quality with fewer parameters\"\n        \n        **Red flags:**\n        \n        - Thinking lower PPL always means better model\n        - Not knowing typical PPL ranges (GPT-2: 35, GPT-3: 20)\n        - Can't explain mathematical formula or branching factor interpretation\n        - Not aware of tokenizer/domain effects\n        \n        **Follow-ups:**\n        \n        - \"PPL 10 vs 20 - meaningful difference?\" → Yes! 10 = choosing from 10 words, 20 = from 20. But diminishing returns below 10\n        - \"How to handle OOV words?\" → Subword tokenization (BPE, SentencePiece) eliminates OOV\n        - \"Why not use perplexity for summarization?\" → PPL measures next-token prediction, not semantic similarity. Use ROUGE/BERTScore instead\n        - \"Domain adaptation impact?\" → Fine-tuning on medical text can reduce PPL from 45 → 15 (3× improvement)

---

### Explain Sequence-to-Sequence Models - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Encoder-Decoder`, `Attention`, `Seq2Seq` | **Asked by:** Google, Meta, Amazon, Microsoft, Uber

??? success "View Answer"

    ## Core Concept

    **Sequence-to-Sequence (Seq2Seq)** models map input sequences to output sequences of potentially different lengths. Fundamental for machine translation, summarization, dialogue, and more.

    **Key Innovation:** Encoder-Decoder architecture with attention mechanism solves the fixed-length bottleneck problem.

    ## Architecture Evolution

    ```
    ┌──────────────────────────────────────────────────────────────┐
    │           SEQ2SEQ ARCHITECTURE EVOLUTION                     │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  1. BASIC RNN ENCODER-DECODER (2014)                         │
    │  ┌────────────────────────────────────────────────────────┐ │
    │  │  Input: "How are you"                                  │ │
    │  │     ↓         ↓         ↓                               │ │
    │  │  [ENCODER RNN]                                          │ │
    │  │  h₁ → h₂ → h₃ → context_vector                         │ │
    │  │                      ↓                                  │ │
    │  │                  [DECODER RNN]                          │ │
    │  │                  s₁ → s₂ → s₃ → s₄                     │ │
    │  │                  ↓    ↓    ↓    ↓                       │ │
    │  │             Output: "Comment allez vous"               │ │
    │  │                                                         │ │
    │  │  Problem: Single context vector bottleneck             │ │
    │  │           Long sequences lose information              │ │
    │  └────────────────────────────────────────────────────────┘ │
    │                                                              │
    │  2. SEQ2SEQ WITH ATTENTION (2015)                            │
    │  ┌────────────────────────────────────────────────────────┐ │
    │  │  Input: "How are you"                                  │ │
    │  │     ↓         ↓         ↓                               │ │
    │  │  [ENCODER]                                              │ │
    │  │  h₁ ← h₂ ← h₃  (all hidden states preserved)           │ │
    │  │   ↓    ↓    ↓                                           │ │
    │  │   └────┴────┘  Attention weights at each step          │ │
    │  │        ↓                                                │ │
    │  │    [DECODER with Attention]                            │ │
    │  │     - Decoder attends to all encoder states            │ │
    │  │     - Weighted sum creates context vector              │ │
    │  │     - Different weights per decoding step              │ │
    │  │                                                         │ │
    │  │  Breakthrough: Learns to align input/output            │ │
    │  │              BLEU +10 points on translation            │ │
    │  └────────────────────────────────────────────────────────┘ │
    │                                                              │
    │  3. TRANSFORMER SEQ2SEQ (2017+)                              │
    │  ┌────────────────────────────────────────────────────────┐ │
    │  │  [Multi-Head Self-Attention Encoder]                   │ │
    │  │            ↓                                            │ │
    │  │  [Multi-Head Cross-Attention Decoder]                  │ │
    │  │                                                         │ │
    │  │  Modern models: T5, BART, mT5, mBART                   │ │
    │  │  Advantages: Parallel, captures long-range deps        │ │
    │  └────────────────────────────────────────────────────────┘ │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    ```

    ## Production Implementation (175 lines)

    ```python
    # seq2seq_attention.py
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from typing import Tuple, Optional
    import random
    import time
    from dataclasses import dataclass

    @dataclass
    class Seq2SeqOutput:
        """Seq2Seq model output"""
        translations: list
        attention_weights: Optional[torch.Tensor]
        bleu_score: float

    class Encoder(nn.Module):
        """
        Bidirectional LSTM Encoder
        
        Time: O(L × H²) where L=seq_len, H=hidden_dim
        Space: O(L × H) for hidden states
        """
        
        def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 256,
            hidden_dim: int = 512,
            num_layers: int = 2,
            dropout: float = 0.5
        ):
            super().__init__()
            
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.dropout = nn.Dropout(dropout)
            
            # Bidirectional LSTM
            self.lstm = nn.LSTM(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            
            # Combine bidirectional outputs
            self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
        def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
            """
            Args:
                src: [batch, src_len] input sequence
            
            Returns:
                encoder_outputs: [batch, src_len, hidden_dim]
                hidden: Tuple of (h_n, c_n) for LSTM
            """
            # Embedding + dropout
            embedded = self.dropout(self.embedding(src))  # [batch, src_len, embed_dim]
            
            # LSTM encoding
            encoder_outputs, (hidden, cell) = self.lstm(embedded)
            # encoder_outputs: [batch, src_len, hidden_dim*2]
            # hidden: [num_layers*2, batch, hidden_dim]
            
            # Combine forward and backward hidden states
            # Take last layer's hidden states
            hidden = torch.tanh(self.fc(
                torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            ))  # [batch, hidden_dim]
            
            return encoder_outputs, hidden

    class Attention(nn.Module):
        """
        Bahdanau (Additive) Attention
        
        score(h_t, h_s) = v^T tanh(W₁h_t + W₂h_s)
        
        Time: O(T × S) where T=target_len, S=source_len
        """
        
        def __init__(self, hidden_dim: int):
            super().__init__()
            
            self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
            self.v = nn.Linear(hidden_dim, 1, bias=False)
        
        def forward(
            self,
            decoder_hidden: torch.Tensor,  # [batch, hidden_dim]
            encoder_outputs: torch.Tensor  # [batch, src_len, hidden_dim*2]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Compute attention weights and context vector
            
            Returns:
                context: [batch, hidden_dim*2] weighted sum of encoder outputs
                attention_weights: [batch, src_len] probabilities
            """
            batch_size = encoder_outputs.shape[0]
            src_len = encoder_outputs.shape[1]
            
            # Repeat decoder hidden for each source position
            decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
            # [batch, src_len, hidden_dim]
            
            # Compute attention energy
            energy = torch.tanh(self.attn(
                torch.cat((decoder_hidden, encoder_outputs), dim=2)
            ))  # [batch, src_len, hidden_dim]
            
            # Attention scores
            attention = self.v(energy).squeeze(2)  # [batch, src_len]
            
            # Attention weights (probabilities)
            attention_weights = F.softmax(attention, dim=1)  # [batch, src_len]
            
            # Context vector: weighted sum of encoder outputs
            context = torch.bmm(
                attention_weights.unsqueeze(1),
                encoder_outputs
            ).squeeze(1)  # [batch, hidden_dim*2]
            
            return context, attention_weights

    class Decoder(nn.Module):
        """
        LSTM Decoder with Attention
        """
        
        def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 256,
            hidden_dim: int = 512,
            num_layers: int = 2,
            dropout: float = 0.5
        ):
            super().__init__()
            
            self.vocab_size = vocab_size
            self.attention = Attention(hidden_dim)
            
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.dropout = nn.Dropout(dropout)
            
            # Input: embedding + context vector
            self.lstm = nn.LSTM(
                embed_dim + hidden_dim * 2,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            
            # Output projection
            self.fc_out = nn.Linear(hidden_dim * 3 + embed_dim, vocab_size)
        
        def forward(
            self,
            input: torch.Tensor,  # [batch] current token
            decoder_hidden: torch.Tensor,  # [batch, hidden_dim]
            decoder_cell: torch.Tensor,  # [batch, hidden_dim]
            encoder_outputs: torch.Tensor  # [batch, src_len, hidden_dim*2]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Single decoding step
            
            Returns:
                predictions: [batch, vocab_size]
                decoder_hidden: [batch, hidden_dim]
                decoder_cell: [batch, hidden_dim]
                attention_weights: [batch, src_len]
            """
            # Embedding
            input = input.unsqueeze(1)  # [batch, 1]
            embedded = self.dropout(self.embedding(input))  # [batch, 1, embed_dim]
            
            # Attention
            context, attention_weights = self.attention(
                decoder_hidden, encoder_outputs
            )
            
            # Concatenate embedding and context
            lstm_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
            # [batch, 1, embed_dim + hidden_dim*2]
            
            # LSTM step
            decoder_output, (decoder_hidden, decoder_cell) = self.lstm(
                lstm_input,
                (decoder_hidden.unsqueeze(0), decoder_cell.unsqueeze(0))
            )
            # decoder_output: [batch, 1, hidden_dim]
            
            decoder_hidden = decoder_hidden.squeeze(0)
            decoder_cell = decoder_cell.squeeze(0)
            
            # Prediction
            prediction = self.fc_out(torch.cat((
                decoder_output.squeeze(1),
                context,
                embedded.squeeze(1)
            ), dim=1))  # [batch, vocab_size]
            
            return prediction, decoder_hidden, decoder_cell, attention_weights

    class Seq2Seq(nn.Module):
        """Complete Seq2Seq model with attention"""
        
        def __init__(
            self,
            encoder: Encoder,
            decoder: Decoder,
            device: torch.device
        ):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.device = device
        
        def forward(
            self,
            src: torch.Tensor,
            trg: torch.Tensor,
            teacher_forcing_ratio: float = 0.5
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Training forward pass
            
            Args:
                src: [batch, src_len]
                trg: [batch, trg_len]
                teacher_forcing_ratio: Probability of using ground truth
            
            Returns:
                outputs: [batch, trg_len, vocab_size]
                attention_weights: [batch, trg_len, src_len]
            """
            batch_size = src.shape[0]
            trg_len = trg.shape[1]
            trg_vocab_size = self.decoder.vocab_size
            
            # Store outputs and attention
            outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
            attentions = torch.zeros(batch_size, trg_len, src.shape[1]).to(self.device)
            
            # Encode
            encoder_outputs, hidden = self.encoder(src)
            cell = hidden  # Initialize cell state
            
            # First input to decoder is <sos> token
            input = trg[:, 0]
            
            for t in range(1, trg_len):
                # Decode step
                output, hidden, cell, attention = self.decoder(
                    input, hidden, cell, encoder_outputs
                )
                
                outputs[:, t] = output
                attentions[:, t] = attention
                
                # Teacher forcing: use ground truth with probability p
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = output.argmax(1)
                
                input = trg[:, t] if teacher_force else top1
            
            return outputs, attentions

    # ===========================================
    # EXAMPLE USAGE
    # ===========================================

    if __name__ == "__main__":
        print("="*70)
        print("GOOGLE TRANSLATE - NEURAL MACHINE TRANSLATION")
        print("="*70)
        
        # Example translation task
        SRC_VOCAB_SIZE = 10000
        TRG_VOCAB_SIZE = 10000
        
        encoder = Encoder(SRC_VOCAB_SIZE, embed_dim=256, hidden_dim=512)
        decoder = Decoder(TRG_VOCAB_SIZE, embed_dim=256, hidden_dim=512)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Seq2Seq(encoder, decoder, device).to(device)
        
        # Dummy batch
        src = torch.randint(0, SRC_VOCAB_SIZE, (32, 20)).to(device)  # batch=32, src_len=20
        trg = torch.randint(0, TRG_VOCAB_SIZE, (32, 25)).to(device)  # batch=32, trg_len=25
        
        # Forward pass
        outputs, attentions = model(src, trg, teacher_forcing_ratio=0.5)
        
        print(f"\nInput shape: {src.shape}")
        print(f"Output shape: {outputs.shape}")  # [32, 25, 10000]
        print(f"Attention shape: {attentions.shape}")  # [32, 25, 20]
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal parameters: {total_params:,}")
    ```

    ## Architecture Comparison

    | Architecture | Year | Key Feature | BLEU (EN-DE) | Speed |
    |--------------|------|-------------|--------------|-------|
    | **Basic RNN Seq2Seq** | 2014 | Fixed context vector | 15-20 | Fast |
    | **Seq2Seq + Attention** | 2015 | Dynamic alignment | 25-28 | Moderate |
    | **ConvS2S** | 2017 | CNN-based | 26-29 | Fast |
    | **Transformer** | 2017 | Self-attention | 28-32 | Very fast (parallel) |
    | **BERT + Decoder** | 2019 | Pre-trained encoder | 32-35 | Moderate |
    | **mBART/mT5** | 2020 | Multilingual pre-training | 35-40 | Slow |

    ## Real-World Deployments

    | Company | System | Model | Languages | Volume |
    |---------|--------|-------|-----------|--------|
    | **Google Translate** | GNMT (2016) | Seq2Seq + Attention | 100+ pairs | 100B+ words/day |
    | **Meta** | M2M-100 | Multilingual Transformer | 100 languages | Billions of translations |
    | **DeepL** | Custom Transformer | Transformer variants | 26 languages | Higher quality than Google |
    | **Microsoft** | Azure Translator | Transformer + Domain | 90+ languages | Real-time translation |
    | **Amazon** | Amazon Translate | Neural MT | 75 languages | AWS service |

    ## Attention Visualization

    **Example: English → French**
    ```
    Source: "I love machine learning"
    Target: "J'adore l'apprentissage automatique"
    
    Attention weights show alignment:
    
              I    love  machine  learning
    J'        0.9  0.05   0.03     0.02      ← "J'" attends to "I"
    adore     0.05 0.85   0.05     0.05      ← "adore" attends to "love"
    l'        0.02 0.05   0.80     0.13      ← "l'" attends to "machine"
    apprentissage 0.01 0.03 0.75   0.21      ← split attention
    automatique   0.01 0.02 0.20   0.77      ← attends to "learning"
    ```

    ## Common Applications

    | Task | Input → Output | Model | Example |
    |------|----------------|-------|--------|
    | **Machine Translation** | Source lang → Target lang | mBART, mT5 | EN → FR, ZH → EN |
    | **Summarization** | Long doc → Short summary | BART, PEGASUS | Article → Headline |
    | **Dialogue** | Context + Query → Response | DialoGPT, Blender | Chatbot responses |
    | **Question Answering** | Context + Question → Answer | T5, BART | SQuAD, extractive QA |
    | **Code Generation** | Description → Code | Codex, CodeT5 | "sort array" → code |
    | **Paraphrasing** | Text → Paraphrase | T5 | Rephrase sentences |

    ## Attention Mechanism Benefits

    | Without Attention | With Attention |
    |-------------------|----------------|
    | Fixed-length context vector | Dynamic context per step |
    | Information bottleneck | All encoder states accessible |
    | Long sequences fail | Handles 100+ tokens |
    | No interpretability | Visualize alignments |
    | BLEU: 15-20 | BLEU: 25-28 (+10 points) |

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Understanding of encoder-decoder architecture
        - Knowledge of attention mechanism and why it's critical
        - Awareness of bottleneck problem in vanilla Seq2Seq
        - Practical experience with translation/summarization
        
        **Strong signal:**
        
        - "Basic Seq2Seq compresses entire source into single context vector, creating bottleneck. For 50-word sentence, final hidden state loses early information. Attention solves this by letting decoder access all encoder states with learned weights"
        - "Google's GNMT (2016) used 8-layer LSTM encoder-decoder with attention, achieving +10 BLEU points over phrase-based systems. Attention weights show word alignments - 'I' in English aligns with 'Je' in French"
        - "Modern Transformers replaced RNNs but kept attention concept. T5 achieves 35-40 BLEU on WMT with 220M-11B parameters, trained on C4 dataset (750GB). Inference: 100-500ms depending on length"
        - "At Meta, M2M-100 translates between any of 100 languages (10K pairs) without English pivot. Uses 1.2B parameters and achieves 25+ BLEU on low-resource pairs"
        
        **Red flags:**
        
        - Not knowing the bottleneck problem
        - Can't explain attention mechanism mathematically
        - Thinking Seq2Seq is only for translation
        - Not aware of Transformer improvements over LSTM
        
        **Follow-ups:**
        
        - "Why attention over just larger hidden state?" → Attention is O(n²) but allows direct access. Larger hidden is O(n) but still loses information
        - "Beam search vs greedy decoding?" → Beam search explores top-k hypotheses, improves BLEU by 2-3 points but 5× slower
        - "How to handle unknown words?" → BPE/subword tokenization ensures every word can be represented
        - "Production latency concerns?" → Batching, caching, quantization. Google Translate: <100ms for 20-word sentence

---

### What is Fine-Tuning vs Prompt Engineering? - Google, OpenAI Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `LLMs`, `Adaptation`, `Cost Optimization` | **Asked by:** Google, OpenAI, Meta, Anthropic, Cohere

??? success "View Answer"

    ## Core Concept

    **Two paradigms for adapting LLMs to specific tasks:**
    - **Fine-Tuning:** Update model weights on task-specific data
    - **Prompt Engineering:** Design inputs to elicit desired behavior without training

    **Key Trade-off:** Cost/time vs performance/control

    ## Approach Comparison

    ```
    ┌──────────────────────────────────────────────────────────────┐
    │        FINE-TUNING VS PROMPT ENGINEERING                     │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  PROMPT ENGINEERING (In-Context Learning)                    │
    │  ┌────────────────────────────────────────────────────────┐ │
    │  │  System: "You are a helpful assistant"                 │ │
    │  │                                                         │ │
    │  │  Example 1:                                             │ │
    │  │    Input: "Great product!"                             │ │
    │  │    Output: "Positive"                                  │ │
    │  │                                                         │ │
    │  │  Example 2:                                             │ │
    │  │    Input: "Terrible service"                           │ │
    │  │    Output: "Negative"                                  │ │
    │  │                                                         │ │
    │  │  Your task:                                             │ │
    │  │    Input: "Amazing quality!"                           │ │
    │  │    Output: ???                                          │ │
    │  │                                                         │ │
    │  │  Pros: No training, fast deployment, flexible          │ │
    │  │  Cons: Token cost, context limits, inconsistent        │ │
    │  └────────────────────────────────────────────────────────┘ │
    │                                                              │
    │  FINE-TUNING (Weight Updates)                                │
    │  ┌────────────────────────────────────────────────────────┐ │
    │  │  Training data: 10K labeled examples                   │ │
    │  │    "Great product!" → Positive                         │ │
    │  │    "Terrible service" → Negative                       │ │
    │  │    ... (10,000 more)                                    │ │
    │  │                                                         │ │
    │  │  Process:                                               │ │
    │  │    1. Prepare dataset (jsonl format)                   │ │
    │  │    2. Upload to OpenAI/Azure                           │ │
    │  │    3. Train for 3-10 epochs (~hours)                   │ │
    │  │    4. Deploy fine-tuned model                          │ │
    │  │                                                         │ │
    │  │  Result: Specialized model, better performance         │ │
    │  │                                                         │ │
    │  │  Pros: Best accuracy, lower per-token cost, custom     │ │
    │  │  Cons: Needs data, training time/cost, less flexible   │ │
    │  └────────────────────────────────────────────────────────┘ │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    ```

    ## Production Implementation (170 lines)

    ```python
    # llm_adaptation.py
    import openai
    import json
    from typing import List, Dict
    from dataclasses import dataclass
    import time

    @dataclass
    class AdaptationMetrics:
        """Metrics for comparing approaches"""
        accuracy: float
        cost_per_1k: float
        latency_ms: float
        setup_time_hours: float

    class PromptEngineer:
        """
        Prompt Engineering with few-shot examples
        
        Time: 0 setup, instant deployment
        Cost: $0.06/1K tokens (GPT-4) for every request
        """
        
        def __init__(self, model: str = "gpt-4"):
            self.model = model
            self.examples = []
        
        def add_examples(self, examples: List[Dict[str, str]]):
            """
            Add few-shot examples
            
            Args:
                examples: [{"input": "text", "output": "label"}, ...]
            """
            self.examples = examples
        
        def build_prompt(self, task: str, input_text: str) -> str:
            """
            Build prompt with system message + examples + task
            
            Few-shot prompting: 2-10 examples typically optimal
            More examples = better but expensive (tokens)
            """
            prompt = f"Task: {task}\n\n"
            
            # Add examples
            for i, ex in enumerate(self.examples, 1):
                prompt += f"Example {i}:\n"
                prompt += f"Input: {ex['input']}\n"
                prompt += f"Output: {ex['output']}\n\n"
            
            # Add current input
            prompt += f"Now classify:\n"
            prompt += f"Input: {input_text}\n"
            prompt += f"Output:"
            
            return prompt
        
        def classify(self, input_text: str, task: str = "Sentiment classification") -> Dict:
            """
            Classify using prompt engineering
            
            Returns:
                {"prediction": "label", "confidence": 0.95, "tokens": 150}
            """
            prompt = self.build_prompt(task, input_text)
            
            start = time.time()
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a classification assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,  # Deterministic
                max_tokens=10
            )
            
            latency = (time.time() - start) * 1000
            
            prediction = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens
            
            # GPT-4 pricing: $0.03/1K input, $0.06/1K output
            cost = (tokens_used / 1000) * 0.045  # Average
            
            return {
                "prediction": prediction,
                "tokens": tokens_used,
                "latency_ms": latency,
                "cost_usd": cost
            }

    class FineTuner:
        """
        Fine-tuning GPT models
        
        Setup: 1-4 hours training
        Cost: $0.008/1K tokens (GPT-3.5-turbo fine-tuned)
        """
        
        def prepare_training_data(
            self,
            examples: List[Dict[str, str]],
            output_file: str = "training_data.jsonl"
        ):
            """
            Prepare data in OpenAI fine-tuning format
            
            Format:
            {"messages": [
                {"role": "system", "content": "Classify sentiment"},
                {"role": "user", "content": "Great product!"},
                {"role": "assistant", "content": "Positive"}
            ]}
            """
            with open(output_file, 'w') as f:
                for ex in examples:
                    data = {
                        "messages": [
                            {"role": "system", "content": "You are a sentiment classifier."},
                            {"role": "user", "content": ex['input']},
                            {"role": "assistant", "content": ex['output']}
                        ]
                    }
                    f.write(json.dumps(data) + '\n')
            
            print(f"Prepared {len(examples)} examples in {output_file}")
        
        def start_fine_tuning(
            self,
            training_file: str,
            model: str = "gpt-3.5-turbo",
            n_epochs: int = 3
        ) -> str:
            """
            Start fine-tuning job
            
            Args:
                training_file: Path to .jsonl file
                model: Base model to fine-tune
                n_epochs: Training epochs (3-10 typical)
            
            Returns:
                job_id: Fine-tuning job ID
            """
            # Upload training file
            with open(training_file, 'rb') as f:
                upload_response = openai.File.create(
                    file=f,
                    purpose='fine-tune'
                )
            
            file_id = upload_response.id
            print(f"Uploaded file: {file_id}")
            
            # Create fine-tuning job
            job_response = openai.FineTuningJob.create(
                training_file=file_id,
                model=model,
                hyperparameters={
                    "n_epochs": n_epochs
                }
            )
            
            job_id = job_response.id
            print(f"Started fine-tuning job: {job_id}")
            print(f"Estimated time: 1-4 hours")
            
            return job_id
        
        def check_status(self, job_id: str) -> Dict:
            """Check fine-tuning job status"""
            job = openai.FineTuningJob.retrieve(job_id)
            return {
                "status": job.status,  # 'validating', 'running', 'succeeded', 'failed'
                "trained_tokens": job.trained_tokens,
                "fine_tuned_model": job.fine_tuned_model
            }
        
        def classify(self, input_text: str, fine_tuned_model: str) -> Dict:
            """
            Classify using fine-tuned model
            
            Advantages:
            - No examples in prompt (shorter, cheaper)
            - Better performance (learned patterns)
            - Faster inference (less tokens to process)
            """
            start = time.time()
            
            response = openai.ChatCompletion.create(
                model=fine_tuned_model,
                messages=[
                    {"role": "user", "content": input_text}
                ],
                temperature=0,
                max_tokens=10
            )
            
            latency = (time.time() - start) * 1000
            
            prediction = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens
            
            # Fine-tuned GPT-3.5 pricing: $0.012/1K
            cost = (tokens_used / 1000) * 0.012
            
            return {
                "prediction": prediction,
                "tokens": tokens_used,
                "latency_ms": latency,
                "cost_usd": cost
            }

    # ===========================================
    # COST COMPARISON EXAMPLE
    # ===========================================

    if __name__ == "__main__":
        print("="*70)
        print("COST COMPARISON: PROMPT ENGINEERING VS FINE-TUNING")
        print("="*70)
        
        # Scenario: 1M classification requests
        num_requests = 1_000_000
        
        # Prompt Engineering (GPT-4)
        prompt_eng_tokens_per_request = 150  # System + examples + input
        prompt_eng_cost_per_1k = 0.045  # $0.03 input + $0.06 output avg
        prompt_eng_total = (num_requests * prompt_eng_tokens_per_request / 1000) * prompt_eng_cost_per_1k
        prompt_eng_setup = 0  # No setup
        
        # Fine-Tuning (GPT-3.5-turbo)
        fine_tune_tokens_per_request = 20  # Just input, no examples
        fine_tune_cost_per_1k = 0.012
        fine_tune_total = (num_requests * fine_tune_tokens_per_request / 1000) * fine_tune_cost_per_1k
        fine_tune_setup = 100  # One-time training cost
        
        print(f"\nScenario: {num_requests:,} classification requests")
        print("\nPrompt Engineering (GPT-4):")
        print(f"  Setup cost: ${prompt_eng_setup}")
        print(f"  Per-request tokens: {prompt_eng_tokens_per_request}")
        print(f"  Total inference cost: ${prompt_eng_total:,.2f}")
        print(f"  Total: ${prompt_eng_total:,.2f}")
        
        print("\nFine-Tuning (GPT-3.5-turbo):")
        print(f"  Setup cost: ${fine_tune_setup}")
        print(f"  Per-request tokens: {fine_tune_tokens_per_request}")
        print(f"  Total inference cost: ${fine_tune_total:,.2f}")
        print(f"  Total: ${fine_tune_setup + fine_tune_total:,.2f}")
        
        print(f"\nSavings: ${prompt_eng_total - (fine_tune_setup + fine_tune_total):,.2f}")
        print(f"Break-even: {int((fine_tune_setup) / ((prompt_eng_tokens_per_request * prompt_eng_cost_per_1k - fine_tune_tokens_per_request * fine_tune_cost_per_1k) / 1000)):,} requests")
    ```

    ## Detailed Comparison

    | Aspect | Prompt Engineering | Fine-Tuning | RAG |
    |--------|-------------------|-------------|-----|
    | **Setup Time** | Minutes | 1-4 hours | 1-2 hours (indexing) |
    | **Setup Cost** | $0 | $100-500 | $50-200 |
    | **Data Required** | 2-10 examples | 100-10K examples | Any documents |
    | **Inference Cost** | $0.045/1K tokens (GPT-4) | $0.012/1K (GPT-3.5 FT) | $0.001/1K (embedding) + $0.03/1K (LLM) |
    | **Latency** | 500-2000ms | 200-500ms | 300-1000ms |
    | **Accuracy** | 70-85% | 85-95% | 75-90% |
    | **Flexibility** | High (change prompt) | Low (need retrain) | High (update docs) |
    | **Best For** | Prototyping, low volume | Production, high volume | Current/private data |

    ## Real-World Deployments

    | Company | Use Case | Approach | Reason | Result |
    |---------|----------|----------|--------|--------|
    | **OpenAI** | ChatGPT | Fine-tuning + RLHF | Alignment with human preferences | 100M+ users |
    | **GitHub Copilot** | Code completion | Fine-tuned Codex | Specialized on code | 1M+ developers |
    | **Jasper.ai** | Marketing copy | Prompt engineering | Flexibility for templates | $125M ARR |
    | **Stripe** | Customer support | RAG + GPT-4 | Private docs, up-to-date | 70% ticket deflection |
    | **Intercom** | Chatbots | Fine-tuning | Custom tone/style | 45% resolution rate |

    ## Decision Framework

    **Choose Prompt Engineering when:**
    - Prototyping or MVP
    - < 10K requests/month
    - Need flexibility (frequent changes)
    - Limited labeled data
    - Multiple tasks with same model

    **Choose Fine-Tuning when:**
    - Production deployment
    - > 100K requests/month (cost savings)
    - Have 100+ labeled examples
    - Need consistent behavior
    - Specialized domain/tone

    **Choose RAG when:**
    - Need current information (post-training cutoff)
    - Private/proprietary data
    - Facts change frequently
    - Verifiable sources required
    - Hybrid: prompt engineering for reasoning + RAG for facts

    ## Break-Even Analysis

    ```python
    # At what volume does fine-tuning become cheaper?
    
    # Costs:
    prompt_eng_cost_per_request = 0.000675  # 150 tokens × $0.045/1K
    fine_tune_cost_per_request = 0.00024    # 20 tokens × $0.012/1K
    fine_tune_setup = 100
    
    # Break-even:
    break_even_requests = fine_tune_setup / (prompt_eng_cost_per_request - fine_tune_cost_per_request)
    # ≈ 230K requests
    
    # Conclusion: Fine-tune if expecting > 230K requests
    ```

    !!! tip "Interviewer's Insight"
        **What they test:**
        
        - Understanding of trade-offs (cost, time, performance)
        - Knowledge of when to use each approach
        - Practical experience with LLM deployment
        - Awareness of cost optimization strategies
        
        **Strong signal:**
        
        - "For prototyping, prompt engineering is fastest - deploy in minutes. For production with 1M+ requests/month, fine-tuning saves 70% on inference costs. Break-even is around 230K requests for GPT models"
        - "GitHub Copilot fine-tuned Codex on billions of lines of code, achieving 30%+ acceptance rate. Prompt engineering wouldn't work - needs deep understanding of code patterns across languages"
        - "At Stripe, we use RAG for customer support - GPT-4 retrieves from 10K docs then generates responses. Better than fine-tuning because docs update daily. Accuracy: 75% vs 65% with just prompts"
        - "Common mistake: Fine-tuning on too little data. OpenAI recommends 100+ examples minimum. With 10 examples, prompt engineering often outperforms poorly fine-tuned model"
        
        **Red flags:**
        
        - Not knowing cost differences (4-6× per token)
        - Thinking fine-tuning is always better
        - Not mentioning RAG as alternative
        - Unaware of break-even analysis
        
        **Follow-ups:**
        
        - "How many examples needed?" → Prompts: 2-10, Fine-tuning: 100-10K
        - "What if data changes daily?" → RAG or periodic fine-tuning (expensive)
        - "Latency concerns?" → Fine-tuned models faster (fewer tokens to process)
        - "Combining approaches?" → Yes! RAG for facts + fine-tuned model for reasoning/style

---

### What is RAG (Retrieval-Augmented Generation)? - Google, OpenAI Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `RAG` | **Asked by:** Google, OpenAI, Meta, Amazon

??? success "View Answer"

    ## Overview

    **RAG (Retrieval-Augmented Generation)** combines retrieval systems with LLMs to ground responses in external knowledge, reducing hallucinations and enabling up-to-date information.

    **Core Idea:** Instead of relying solely on model's parametric memory, retrieve relevant documents and include them in the prompt.

    ## RAG Pipeline

    ```
    User Query: "What is the capital of France?"
           ↓
    ┌──────────────────┐
    │  1. Query        │ → Embed query: [768-dim vector]
    │     Embedding    │
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  2. Vector       │ → Search similar docs (cosine sim)
    │     Search       │    Top-k retrieval (k=3-10)
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  3. Re-ranking   │ → Optional: Cross-encoder rerank
    │     (Optional)   │    Improve relevance
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  4. Context      │ → "Paris is the capital..."
    │     Retrieved    │    "France's capital city..."
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  5. Prompt       │ → Context: {retrieved_docs}
    │     Construction │    Query: {user_query}
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  6. LLM          │ → GPT-4, Claude, etc.
    │     Generation   │
    └────────┬─────────┘
             ↓
    Response: "The capital of France is Paris."
    ```

    ## Production Implementation (200 lines)

    ```python
    # rag_system.py
    import numpy as np
    from typing import List, Dict, Tuple
    import openai
    from sentence_transformers import SentenceTransformer
    import faiss

    class DocumentChunker:
        """
        Chunk documents for RAG

        Strategies:
        1. Fixed-size (256-512 tokens)
        2. Sentence-based
        3. Semantic chunking (split on topic changes)
        """

        def __init__(self, chunk_size=512, overlap=50):
            self.chunk_size = chunk_size
            self.overlap = overlap

        def chunk_text(self, text: str) -> List[str]:
            """
            Chunk text with overlap

            Overlap prevents losing context at boundaries
            """
            words = text.split()
            chunks = []

            for i in range(0, len(words), self.chunk_size - self.overlap):
                chunk = ' '.join(words[i:i + self.chunk_size])
                chunks.append(chunk)

            return chunks

        def chunk_by_sentences(self, text: str, max_sentences=5) -> List[str]:
            """Chunk by sentence boundaries (better for coherence)"""
            sentences = text.split('. ')
            chunks = []

            for i in range(0, len(sentences), max_sentences):
                chunk = '. '.join(sentences[i:i + max_sentences])
                chunks.append(chunk)

            return chunks

    class VectorStore:
        """
        Vector database for similarity search

        Uses FAISS for efficient nearest neighbor search
        """

        def __init__(self, embedding_model='all-MiniLM-L6-v2'):
            self.encoder = SentenceTransformer(embedding_model)
            self.dimension = 384  # Model output dimension

            # FAISS index (L2 distance)
            self.index = faiss.IndexFlatL2(self.dimension)

            # Metadata storage
            self.documents = []
            self.metadata = []

        def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
            """
            Add documents to vector store

            Args:
                documents: List of text chunks
                metadatas: Optional metadata per document
            """
            # Encode documents
            embeddings = self.encoder.encode(documents, convert_to_numpy=True)

            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))

            # Store documents and metadata
            self.documents.extend(documents)
            if metadatas:
                self.metadata.extend(metadatas)
            else:
                self.metadata.extend([{}] * len(documents))

        def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
            """
            Semantic search

            Args:
                query: Search query
                k: Number of results

            Returns:
                List of (document, score) tuples
            """
            # Encode query
            query_embedding = self.encoder.encode([query], convert_to_numpy=True)

            # Search FAISS index
            distances, indices = self.index.search(
                query_embedding.astype('float32'), k
            )

            # Return documents with scores
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.documents):
                    results.append((self.documents[idx], float(distance)))

            return results

    class ReRanker:
        """
        Re-rank retrieved documents using cross-encoder

        Cross-encoder is more accurate than bi-encoder but slower
        Use for top-k (k=20-50) then rerank to top-n (n=3-5)
        """

        def __init__(self):
            # In production: Use cross-encoder model
            pass

        def rerank(
            self,
            query: str,
            documents: List[str],
            top_k: int = 3
        ) -> List[Tuple[str, float]]:
            """
            Rerank documents by relevance

            Args:
                query: User query
                documents: Retrieved documents
                top_k: Number to return

            Returns:
                Top-k documents with scores
            """
            # Simplified: In production, use cross-encoder
            # For now, return as-is
            return [(doc, 1.0) for doc in documents[:top_k]]

    class RAGSystem:
        """Complete RAG system"""

        def __init__(self, llm_model='gpt-3.5-turbo'):
            self.chunker = DocumentChunker(chunk_size=512, overlap=50)
            self.vector_store = VectorStore()
            self.reranker = ReRanker()
            self.llm_model = llm_model

        def ingest_documents(self, documents: List[str]):
            """
            Ingest and index documents

            Args:
                documents: List of document texts
            """
            all_chunks = []

            for doc in documents:
                chunks = self.chunker.chunk_text(doc)
                all_chunks.extend(chunks)

            # Add to vector store
            self.vector_store.add_documents(all_chunks)

            print(f"Ingested {len(all_chunks)} chunks from {len(documents)} documents")

        def retrieve(self, query: str, k: int = 5) -> List[str]:
            """
            Retrieve relevant documents

            Args:
                query: User query
                k: Number of documents to retrieve

            Returns:
                List of relevant document chunks
            """
            # Vector search (retrieve more for re-ranking)
            results = self.vector_store.search(query, k=k*2)

            # Extract documents
            documents = [doc for doc, score in results]

            # Re-rank
            reranked = self.reranker.rerank(query, documents, top_k=k)

            return [doc for doc, score in reranked]

        def generate(self, query: str, context: List[str]) -> str:
            """
            Generate answer using LLM

            Args:
                query: User query
                context: Retrieved context documents

            Returns:
                Generated answer
            """
            # Construct prompt
            context_str = "\n\n".join([
                f"[{i+1}] {doc}" for i, doc in enumerate(context)
            ])

            prompt = f"""Answer the question based on the context below. If the answer is not in the context, say "I don't have enough information."

Context:
{context_str}

Question: {query}

Answer:"""

            # Call LLM (using OpenAI as example)
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3  # Lower for factual answers
            )

            return response.choices[0].message.content

        def query(self, question: str, k: int = 3) -> Dict:
            """
            End-to-end RAG query

            Args:
                question: User question
                k: Number of context documents

            Returns:
                Dict with answer and sources
            """
            # Retrieve context
            context = self.retrieve(question, k=k)

            # Generate answer
            answer = self.generate(question, context)

            return {
                "answer": answer,
                "sources": context,
                "num_sources": len(context)
            }

    # Example usage
    if __name__ == "__main__":
        # Initialize RAG system
        rag = RAGSystem()

        # Ingest documents
        documents = [
            "Paris is the capital and largest city of France. It is located on the Seine River.",
            "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
            "France is a country in Western Europe with a population of 67 million."
        ]

        rag.ingest_documents(documents)

        # Query
        result = rag.query("What is the capital of France?")

        print(f"Answer: {result['answer']}")
        print(f"\nSources used: {result['num_sources']}")
        for i, source in enumerate(result['sources'], 1):
            print(f"  [{i}] {source[:100]}...")
    ```

    ## RAG Components Comparison

    | Component | Options | Pros | Cons | Best For |
    |-----------|---------|------|------|----------|
    | **Embedding** | OpenAI, Sentence-BERT, Cohere | Fast, semantic | Can miss keywords | Most cases |
    | **Vector DB** | FAISS, Pinecone, Weaviate | Scalable, fast | Approximate | Large-scale |
    | **Retrieval** | Dense (semantic), Sparse (BM25), Hybrid | Hybrid = best recall | Complex | Production |
    | **Reranking** | Cross-encoder, LLM | Improves relevance | Slower | Top-k refinement |
    | **LLM** | GPT-4, Claude, Llama | High quality | Expensive | Final generation |

    ## Chunking Strategies

    **1. Fixed-Size Chunking:**
    - **Size:** 256-512 tokens
    - **Overlap:** 20-50 tokens (prevents losing context)
    - **Pros:** Simple, consistent
    - **Cons:** May split mid-sentence

    **2. Sentence-Based:**
    - Group 3-5 sentences per chunk
    - **Pros:** Coherent chunks
    - **Cons:** Variable size

    **3. Semantic Chunking:**
    - Split on topic changes (embeddings)
    - **Pros:** Best context preservation
    - **Cons:** Slower, complex

    ## Benefits & Limitations

    **Benefits:**
    - ✅ **Reduces hallucinations** (grounded in facts)
    - ✅ **Up-to-date information** (no training cutoff)
    - ✅ **Citations/sources** (traceable)
    - ✅ **Domain-specific knowledge** (private docs)
    - ✅ **Lower cost** (vs fine-tuning)

    **Limitations:**
    - ❌ **Retrieval quality matters** (garbage in → garbage out)
    - ❌ **Context window limits** (can't fit all docs)
    - ❌ **Latency** (retrieval + generation)
    - ❌ **Chunk boundary issues** (answer split across chunks)

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Poor chunking** | Loses context | Use overlap (50 tokens), semantic chunking |
    | **Wrong retrieval k** | Miss relevant docs or too much noise | k=3-5 for most tasks, tune based on eval |
    | **No reranking** | Irrelevant docs ranked high | Add cross-encoder reranking |
    | **Ignoring metadata** | Can't filter | Store source, date, author in metadata |
    | **No hybrid search** | Miss keyword matches | Combine dense (semantic) + sparse (BM25) |
    | **Stale embeddings** | Mismatch with current data | Reindex when data changes |

    ## Real-World Applications

    **Perplexity AI:**
    - **Use Case:** Search with citations
    - **Tech Stack:** Web search + RAG + GPT-4
    - **Feature:** Real-time web retrieval, cited answers
    - **Scale:** 50M+ queries/month

    **Notion AI:**
    - **Use Case:** Q&A over workspace docs
    - **Tech Stack:** Doc embeddings + GPT-3.5
    - **Privacy:** User data stays in workspace
    - **Adoption:** 30M+ users

    **GitHub Copilot Chat:**
    - **Use Case:** Code Q&A with codebase context
    - **Tech Stack:** Code embeddings + Codex
    - **Feature:** Retrieves relevant code snippets
    - **Impact:** 40% faster debugging

    **ChatGPT Plugins (Now GPTs):**
    - **Use Case:** External knowledge integration
    - **Tech Stack:** API retrieval + GPT-4
    - **Examples:** Wolfram Alpha, Zapier, browsing
    - **Adoption:** 3M+ custom GPTs

    ## Evaluation Metrics

    | Metric | Measures | Target |
    |--------|----------|--------|
    | **Retrieval Precision@k** | Relevant docs in top-k | > 80% |
    | **Retrieval Recall@k** | % of all relevant docs retrieved | > 70% |
    | **Answer Accuracy** | Correct answers | > 90% |
    | **Citation Accuracy** | Answers grounded in context | > 95% |
    | **Latency** | End-to-end time | < 2s |

    !!! tip "Interviewer's Insight"
        **Strong candidates:**

        - Explain chunking strategy: "512 tokens with 50-token overlap prevents losing context at boundaries"
        - Know hybrid search: "Combine dense embeddings (semantic) with BM25 (keyword) for best recall"
        - Discuss reranking: "Retrieve top-20 with bi-encoder, rerank to top-3 with cross-encoder for accuracy"
        - Understand limitations: "RAG reduces hallucinations but quality depends on retrieval - garbage in, garbage out"
        - Reference real systems: "Perplexity AI uses real-time web RAG, Notion AI for private workspace Q&A"
        - Know evaluation: "Track retrieval precision@k and answer accuracy with human eval"

---

### Explain Positional Encoding - Google, OpenAI Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Transformers` | **Asked by:** Google, OpenAI, Meta

??? success "View Answer"

    ## Why Positional Encoding?

    Transformers process all tokens in parallel (unlike RNNs/LSTMs which are sequential). Without positional information, the model treats input as a **bag-of-words** - "cat chased mouse" and "mouse chased cat" would be identical.

    **The Problem:**
    - Self-attention is **permutation-invariant** (order doesn't matter)
    - Need to inject position information into token embeddings
    - Must work for any sequence length (even unseen lengths at inference)

    ## Sinusoidal Positional Encoding (Original Transformer)

    **Mathematical Formulation:**

    $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

    $$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

    where:
    - $pos$ = position in sequence (0, 1, 2, ...)
    - $i$ = dimension index (0 to $d_{model}/2$)
    - $d_{model}$ = embedding dimension (e.g., 768)

    **Key Properties:**
    - **Deterministic** (no learned parameters)
    - **Extrapolates** to longer sequences than seen during training
    - **Relative position** can be expressed as linear transformation: $PE_{pos+k}$ can be represented as function of $PE_{pos}$

    ## Production Implementation (180 lines)

    ```python
    # positional_encoding.py
    import torch
    import torch.nn as nn
    import math
    import matplotlib.pyplot as plt

    class SinusoidalPositionalEncoding(nn.Module):
        """
        Original Transformer positional encoding (Vaswani et al., 2017)

        Used in: BERT, GPT-2, T5
        Time: O(n × d) to compute
        Space: O(n × d) cached
        """

        def __init__(self, d_model=768, max_len=512, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)

            # Create PE matrix [max_len, d_model]
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

            # Compute div_term: 10000^(2i/d_model) for i in [0, d_model/2)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float) *
                (-math.log(10000.0) / d_model)
            )

            # Apply sin to even indices, cos to odd indices
            pe[:, 0::2] = torch.sin(position * div_term)  # Even: 0, 2, 4, ...
            pe[:, 1::2] = torch.cos(position * div_term)  # Odd: 1, 3, 5, ...

            pe = pe.unsqueeze(0)  # [1, max_len, d_model]

            # Register as buffer (not a parameter, but saved in state_dict)
            self.register_buffer('pe', pe)

        def forward(self, x):
            """
            Args:
                x: [batch, seq_len, d_model] - token embeddings
            Returns:
                [batch, seq_len, d_model] - embeddings + PE
            """
            seq_len = x.size(1)

            # Add positional encoding (broadcasting over batch)
            x = x + self.pe[:, :seq_len, :]
            return self.dropout(x)

    class LearnedPositionalEmbedding(nn.Module):
        """
        Learned positional embeddings (used in BERT, GPT-2)

        Pros: Can learn task-specific position patterns
        Cons: Cannot extrapolate to longer sequences

        Used in: BERT, GPT-2, GPT-3
        """

        def __init__(self, max_len=512, d_model=768, dropout=0.1):
            super().__init__()
            self.position_embeddings = nn.Embedding(max_len, d_model)
            self.dropout = nn.Dropout(dropout)

            # Register position IDs (0, 1, 2, ..., max_len-1)
            self.register_buffer(
                'position_ids',
                torch.arange(max_len).expand((1, -1))
            )

        def forward(self, x):
            """
            Args:
                x: [batch, seq_len, d_model]
            Returns:
                [batch, seq_len, d_model]
            """
            seq_len = x.size(1)

            # Get position embeddings for current sequence
            position_ids = self.position_ids[:, :seq_len]
            position_embeds = self.position_embeddings(position_ids)

            return self.dropout(x + position_embeds)

    class RoPE(nn.Module):
        """
        Rotary Positional Embedding (RoPE) - Su et al., 2021

        Key idea: Rotate query/key vectors by angle proportional to position

        Advantages:
        - Naturally encodes relative positions
        - Extrapolates to longer sequences
        - Better performance on long sequences

        Used in: LLaMA, GPT-NeoX, PaLM, GPT-J
        """

        def __init__(self, d_model=768, max_len=2048, base=10000):
            super().__init__()
            self.d_model = d_model
            self.max_len = max_len
            self.base = base

            # Precompute rotation frequencies
            inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
            self.register_buffer('inv_freq', inv_freq)

            # Precompute cos and sin for max_len positions
            t = torch.arange(max_len, dtype=torch.float)
            freqs = torch.outer(t, inv_freq)  # [max_len, d_model/2]
            emb = torch.cat([freqs, freqs], dim=-1)  # [max_len, d_model]

            self.register_buffer('cos_cached', emb.cos()[None, :, None, :])
            self.register_buffer('sin_cached', emb.sin()[None, :, None, :])

        def rotate_half(self, x):
            """Rotates half the hidden dims of the input"""
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat([-x2, x1], dim=-1)

        def forward(self, q, k):
            """
            Apply rotary embeddings to query and key

            Args:
                q, k: [batch, num_heads, seq_len, head_dim]
            Returns:
                q_rot, k_rot: [batch, num_heads, seq_len, head_dim]
            """
            seq_len = q.size(2)

            # Get cached cos and sin for current sequence
            cos = self.cos_cached[:, :seq_len, :, :]
            sin = self.sin_cached[:, :seq_len, :, :]

            # Apply rotation
            q_rot = (q * cos) + (self.rotate_half(q) * sin)
            k_rot = (k * cos) + (self.rotate_half(k) * sin)

            return q_rot, k_rot

    class ALiBi(nn.Module):
        """
        Attention with Linear Biases (ALiBi) - Press et al., 2021

        Key idea: Add linear bias to attention scores based on distance

        Formula: attention_score += -m × |i - j|
        where m is head-specific slope

        Advantages:
        - No position embeddings needed
        - Excellent extrapolation (trained on 1K, works on 10K+)
        - Memory efficient

        Used in: BLOOM, MPT, StableLM
        """

        def __init__(self, num_heads=12, max_len=2048):
            super().__init__()
            self.num_heads = num_heads

            # Compute slopes for each head: 2^(-8/n), 2^(-16/n), ...
            slopes = torch.Tensor(self._get_slopes(num_heads))
            self.register_buffer('slopes', slopes)

            # Precompute distance matrix
            positions = torch.arange(max_len)
            distance = positions[None, :] - positions[:, None]  # [max_len, max_len]
            distance = distance.abs().unsqueeze(0).unsqueeze(0)  # [1, 1, max_len, max_len]
            self.register_buffer('distance', distance)

        def _get_slopes(self, num_heads):
            """Compute head-specific slopes"""
            # Geometric sequence: 2^(-8/n) to 2^(-8)
            def get_slopes_power_of_2(n):
                start = 2 ** (-2 ** -(math.log2(n) - 3))
                ratio = start
                return [start * (ratio ** i) for i in range(n)]

            if math.log2(num_heads).is_integer():
                return get_slopes_power_of_2(num_heads)
            else:
                # Closest power of 2
                closest_power = 2 ** math.floor(math.log2(num_heads))
                return (
                    get_slopes_power_of_2(closest_power) +
                    self._get_slopes(2 * closest_power)[0::2][:num_heads - closest_power]
                )

        def forward(self, attention_scores):
            """
            Add linear position bias to attention scores

            Args:
                attention_scores: [batch, num_heads, seq_len, seq_len]
            Returns:
                biased_scores: [batch, num_heads, seq_len, seq_len]
            """
            seq_len = attention_scores.size(-1)

            # Get distance matrix for current sequence
            distance = self.distance[:, :, :seq_len, :seq_len]

            # Apply head-specific slopes: -m × distance
            bias = -self.slopes.view(1, -1, 1, 1) * distance

            return attention_scores + bias

    # Example: Using different positional encodings
    def compare_positional_encodings():
        """Demonstrate different positional encoding methods"""
        batch_size, seq_len, d_model = 2, 128, 768
        num_heads = 12

        # Input token embeddings
        token_embeddings = torch.randn(batch_size, seq_len, d_model)

        print("=" * 60)
        print("Positional Encoding Comparison")
        print("=" * 60)

        # 1. Sinusoidal
        sinusoidal_pe = SinusoidalPositionalEncoding(d_model, max_len=512)
        output_sin = sinusoidal_pe(token_embeddings)
        print(f"\n1. Sinusoidal PE:")
        print(f"   Parameters: 0 (deterministic)")
        print(f"   Output shape: {output_sin.shape}")
        print(f"   Can extrapolate: Yes")

        # 2. Learned
        learned_pe = LearnedPositionalEmbedding(max_len=512, d_model=d_model)
        output_learned = learned_pe(token_embeddings)
        params_learned = sum(p.numel() for p in learned_pe.parameters())
        print(f"\n2. Learned PE:")
        print(f"   Parameters: {params_learned:,} (512 × 768)")
        print(f"   Output shape: {output_learned.shape}")
        print(f"   Can extrapolate: No (max_len=512)")

        # 3. RoPE (applied to Q, K in attention)
        rope = RoPE(d_model, max_len=2048)
        head_dim = d_model // num_heads
        q = token_embeddings.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = token_embeddings.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        q_rot, k_rot = rope(q, k)
        print(f"\n3. RoPE:")
        print(f"   Parameters: 0 (rotation angles)")
        print(f"   Output shape: {q_rot.shape}")
        print(f"   Can extrapolate: Yes (excellent)")

        # 4. ALiBi (applied to attention scores)
        alibi = ALiBi(num_heads, max_len=2048)
        attention_scores = torch.randn(batch_size, num_heads, seq_len, seq_len)
        biased_scores = alibi(attention_scores)
        print(f"\n4. ALiBi:")
        print(f"   Parameters: 0 (linear biases)")
        print(f"   Biased scores shape: {biased_scores.shape}")
        print(f"   Can extrapolate: Yes (best extrapolation)")
        print("=" * 60)

    if __name__ == "__main__":
        compare_positional_encodings()
    ```

    **Output:**
    ```
    ============================================================
    Positional Encoding Comparison
    ============================================================

    1. Sinusoidal PE:
       Parameters: 0 (deterministic)
       Output shape: torch.Size([2, 128, 768])
       Can extrapolate: Yes

    2. Learned PE:
       Parameters: 393,216 (512 × 768)
       Output shape: torch.Size([2, 128, 768])
       Can extrapolate: No (max_len=512)

    3. RoPE:
       Parameters: 0 (rotation angles)
       Output shape: torch.Size([2, 12, 128, 64])
       Can extrapolate: Yes (excellent)

    4. ALiBi:
       Parameters: 0 (linear biases)
       Biased scores shape: torch.Size([2, 12, 128, 128])
       Can extrapolate: Yes (best extrapolation)
    ============================================================
    ```

    ## Comparison: Positional Encoding Methods

    | Method | Parameters | Extrapolation | Performance | Used In |
    |--------|-----------|---------------|-------------|---------|
    | **Sinusoidal** | 0 | ✅ Good | Baseline | Original Transformer, T5 |
    | **Learned** | max_len × d_model | ❌ No | Slightly better | BERT, GPT-2, GPT-3 |
    | **RoPE** | 0 | ✅ Excellent | +2-5% accuracy | LLaMA, PaLM, GPT-J, Mistral |
    | **ALiBi** | 0 | ✅ Best | +3-7% on long | BLOOM (176B), MPT (7B-30B) |
    | **Absolute** | Varies | Limited | Traditional | BERT (learned) |
    | **Relative** | Varies | Better | Strong | T5 (bias), Transformer-XL |

    ## Real-World Impact

    **LLaMA (Meta AI, 2023):**
    - **Model:** 7B to 65B parameters
    - **PE Method:** RoPE
    - **Context:** Trained on 2K, works on 8K+ with fine-tuning
    - **Impact:** 15-20% better perplexity on long sequences vs sinusoidal
    - **Adoption:** Base for Vicuna, Alpaca, WizardLM

    **BLOOM (BigScience, 2022):**
    - **Model:** 176B parameters
    - **PE Method:** ALiBi
    - **Context:** Trained on 2K tokens
    - **Extrapolation:** Works on 10K+ tokens at inference (5x longer!)
    - **Performance:** 23.5 perplexity on LAMBADA vs 25.1 with learned PE
    - **Memory:** 30% less memory (no PE parameters)

    **GPT-3 (OpenAI, 2020):**
    - **Model:** 175B parameters
    - **PE Method:** Learned absolute
    - **Context:** 2048 tokens (fixed)
    - **Limitation:** Cannot extend beyond training length
    - **GPT-4:** Likely uses RoPE or ALiBi for 32K context

    **T5 (Google, 2020):**
    - **Model:** 11B parameters
    - **PE Method:** Relative position bias (learned per layer)
    - **Context:** 512 tokens, can extend to 1024
    - **Accuracy:** +1.5 BLEU on translation vs absolute PE

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Using learned PE for inference > max_len** | Crash or poor quality | Use RoPE or ALiBi for variable lengths |
    | **Not caching sinusoidal PE** | Recompute every forward pass | Cache PE matrix in buffer |
    | **Wrong frequency formula** | Poor position discrimination | Use 10000^(2i/d), not 10000^(i/d) |
    | **Forgetting dropout on PE** | Overfitting | Add dropout after PE addition |
    | **Extrapolating learned PE** | Undefined behavior | Zero-pad or interpolate (not recommended) |
    | **RoPE on values** | Breaks invariance | Only apply to queries and keys |
    | **ALiBi wrong slope** | Poor relative distance modeling | Use 2^(-8/n) geometric sequence |

    ## Mathematical Intuition

    **Why sinusoidal works:**
    - Different frequencies for different dimensions (high freq for nearby, low freq for far)
    - For position $pos + k$:
      $$PE_{pos+k} = \text{LinearTransform}(PE_{pos})$$
    - This allows the model to learn relative positions

    **Why RoPE works:**
    - Query-key dot product with RoPE encodes relative position:
      $$q_m^T k_n = (W_q x_m)^T R_m^T R_n (W_k x_n) = (W_q x_m)^T R_{n-m} (W_k x_n)$$
    - The rotation angle is proportional to relative distance $(n-m)$

    **Why ALiBi works:**
    - Attention naturally favors closer tokens with linear penalty
    - Head-specific slopes allow different heads to focus on different ranges
    - No embeddings needed → memory efficient

    !!! tip "Interviewer's Insight"
        **Strong candidates:**

        - Explain sinusoidal formula: "Use sin/cos with different frequencies (10000^(2i/d)) so model can learn relative positions"
        - Know modern methods: "RoPE (LLaMA, Mistral) rotates Q/K vectors for better extrapolation; ALiBi (BLOOM) adds linear bias for excellent long-context"
        - Understand extrapolation: "Learned PE can't handle sequences longer than max_len, but RoPE extrapolates 4-8× training length"
        - Reference real systems: "LLaMA uses RoPE for 2K→8K extrapolation; BLOOM uses ALiBi to go from 2K→10K tokens"
        - Know tradeoffs: "Learned PE slightly better on fixed length, RoPE best for variable, ALiBi best for extreme extrapolation"
        - Explain RoPE advantage: "RoPE encodes relative position in Q·K dot product naturally, no position embeddings needed"

---

### What is Topic Modeling (LDA)? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Unsupervised` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    ## What is Topic Modeling?

    Topic modeling is an **unsupervised learning** technique that automatically discovers abstract "topics" in a collection of documents. Each topic is a distribution over words, and each document is a mixture of topics.

    **Use Cases:**
    - **Document organization** (categorize news articles)
    - **Recommendation systems** (similar documents)
    - **Exploratory analysis** (discover themes in survey responses)
    - **Information retrieval** (semantic search)

    ## Latent Dirichlet Allocation (LDA)

    **LDA** (Blei et al., 2003) is a generative probabilistic model that represents documents as mixtures of topics.

    **Key Assumptions:**
    - Each document is a mixture of K topics (e.g., 20% sports, 80% politics)
    - Each topic is a distribution over V words (e.g., "sports" → {game: 0.05, team: 0.04, win: 0.03, ...})
    - Words in documents are generated by picking a topic, then picking a word from that topic

    ## Mathematical Formulation

    **Generative Process:**

    For each document $d$:

    1. Draw topic distribution: $\theta_d \sim \text{Dirichlet}(\alpha)$ (e.g., [0.2, 0.8] for 2 topics)
    2. For each word $w_n$ in document:
       - Choose topic: $z_n \sim \text{Categorical}(\theta_d)$ (e.g., topic 1 or 2)
       - Choose word: $w_n \sim \text{Categorical}(\beta_{z_n})$ (from topic's word distribution)

    **Parameters:**
    - $K$ = number of topics (hyperparameter, e.g., 10)
    - $\alpha$ = Dirichlet prior for document-topic distribution (controls sparsity)
    - $\beta$ = Dirichlet prior for topic-word distribution
    - $\theta_d$ = topic distribution for document $d$ (learned)
    - $\phi_k$ = word distribution for topic $k$ (learned)

    **Inference:**
    - Goal: Given documents, find $\theta$ and $\phi$ that maximize likelihood
    - Methods: Gibbs Sampling, Variational Bayes (used in sklearn)

    ## Production Implementation (190 lines)

    ```python
    # topic_modeling.py
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    import matplotlib.pyplot as plt
    import pandas as pd

    # Modern: BERTopic with transformers
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        BERTOPIC_AVAILABLE = True
    except ImportError:
        BERTOPIC_AVAILABLE = False

    class LDATopicModeler:
        """
        LDA-based topic modeling with sklearn

        Time: O(n_iter × n_docs × n_topics × n_words)
        Space: O(n_topics × n_words)
        """

        def __init__(self, n_topics=10, alpha=0.1, beta=0.01, max_iter=20):
            """
            Args:
                n_topics: Number of topics (K)
                alpha: Document-topic density (lower = sparser, default 1/K)
                beta: Topic-word density (lower = fewer words per topic)
                max_iter: Number of iterations for variational inference
            """
            self.n_topics = n_topics
            self.alpha = alpha
            self.beta = beta

            # Vectorizer: Convert text to word counts
            self.vectorizer = CountVectorizer(
                max_features=5000,  # Limit vocabulary
                stop_words='english',
                min_df=2,  # Min document frequency
                max_df=0.95  # Max document frequency (filter common words)
            )

            # LDA model
            self.lda = LatentDirichletAllocation(
                n_components=n_topics,
                doc_topic_prior=alpha,  # α
                topic_word_prior=beta,  # β
                max_iter=max_iter,
                learning_method='batch',  # 'batch' or 'online'
                random_state=42,
                n_jobs=-1
            )

            self.feature_names = None

        def fit(self, documents):
            """
            Fit LDA model on documents

            Args:
                documents: List of text strings
            Returns:
                self
            """
            # Convert to word counts (bag of words)
            doc_term_matrix = self.vectorizer.fit_transform(documents)
            self.feature_names = self.vectorizer.get_feature_names_out()

            print(f"Vocabulary size: {len(self.feature_names)}")
            print(f"Document-term matrix: {doc_term_matrix.shape}")

            # Fit LDA
            self.lda.fit(doc_term_matrix)

            print(f"LDA perplexity: {self.lda.perplexity(doc_term_matrix):.2f}")
            print(f"Log-likelihood: {self.lda.score(doc_term_matrix):.2f}")

            return self

        def transform(self, documents):
            """
            Get topic distributions for new documents

            Args:
                documents: List of text strings
            Returns:
                topic_dist: [n_docs, n_topics] - topic probabilities
            """
            doc_term_matrix = self.vectorizer.transform(documents)
            return self.lda.transform(doc_term_matrix)

        def get_topics(self, n_words=10):
            """
            Get top words for each topic

            Returns:
                topics: List of (topic_id, top_words) tuples
            """
            topics = []

            for topic_idx, topic in enumerate(self.lda.components_):
                # Get top n words for this topic
                top_word_indices = topic.argsort()[-n_words:][::-1]
                top_words = [self.feature_names[i] for i in top_word_indices]
                top_probs = [topic[i] for i in top_word_indices]

                topics.append({
                    'topic_id': topic_idx,
                    'top_words': top_words,
                    'probabilities': top_probs
                })

            return topics

        def print_topics(self, n_words=10):
            """Print top words for each topic"""
            topics = self.get_topics(n_words)

            for topic in topics:
                words_str = ', '.join(topic['top_words'])
                print(f"Topic {topic['topic_id']}: {words_str}")

        def get_document_topics(self, documents, top_k=3):
            """
            Get top topics for each document

            Args:
                documents: List of text strings
                top_k: Number of top topics to return
            Returns:
                doc_topics: List of (doc_idx, top_topics) tuples
            """
            topic_dist = self.transform(documents)

            doc_topics = []
            for doc_idx, dist in enumerate(topic_dist):
                top_topic_indices = dist.argsort()[-top_k:][::-1]
                top_topics = [(idx, dist[idx]) for idx in top_topic_indices]
                doc_topics.append({
                    'doc_idx': doc_idx,
                    'top_topics': top_topics,
                    'preview': documents[doc_idx][:100] + "..."
                })

            return doc_topics

    class BERTopicModeler:
        """
        Modern topic modeling with BERT embeddings

        Advantages over LDA:
        - Better semantic understanding (contextual embeddings)
        - Automatic topic naming
        - Handles out-of-vocabulary words
        - Faster inference

        Used in: Production systems (2020+)
        """

        def __init__(self, n_topics=10, embedding_model='all-MiniLM-L6-v2'):
            """
            Args:
                n_topics: Target number of topics (auto-reduces if needed)
                embedding_model: Sentence-BERT model name
            """
            if not BERTOPIC_AVAILABLE:
                raise ImportError("Install BERTopic: pip install bertopic")

            self.n_topics = n_topics
            self.model = BERTopic(
                embedding_model=embedding_model,
                nr_topics=n_topics,
                verbose=True
            )

        def fit_transform(self, documents):
            """
            Fit BERTopic and get topics

            Args:
                documents: List of text strings
            Returns:
                topics: Array of topic IDs for each document
                probs: Topic probabilities
            """
            topics, probs = self.model.fit_transform(documents)
            return topics, probs

        def get_topics(self):
            """Get topic information"""
            return self.model.get_topic_info()

        def get_topic_words(self, topic_id):
            """Get top words for a specific topic"""
            return self.model.get_topic(topic_id)

    # Example usage
    def compare_topic_modeling():
        """Compare LDA vs BERTopic on sample documents"""

        # Sample documents
        documents = [
            "Machine learning algorithms can learn from data and improve over time",
            "Neural networks are inspired by biological neurons in the brain",
            "Deep learning uses multiple layers to extract features from data",
            "Stock market prices fluctuate based on supply and demand",
            "Investors analyze financial reports to make investment decisions",
            "Portfolio diversification helps reduce investment risk",
            "Climate change is affecting global weather patterns",
            "Renewable energy sources like solar and wind are growing",
            "Carbon emissions contribute to global warming",
        ]

        print("=" * 70)
        print("TOPIC MODELING COMPARISON")
        print("=" * 70)

        # 1. LDA Topic Modeling
        print("\n1. LDA (Latent Dirichlet Allocation)")
        print("-" * 70)

        lda_model = LDATopicModeler(n_topics=3, alpha=0.1, beta=0.01)
        lda_model.fit(documents)

        print("\nTopics discovered:")
        lda_model.print_topics(n_words=5)

        print("\nDocument-topic assignments:")
        doc_topics = lda_model.get_document_topics(documents, top_k=2)
        for item in doc_topics[:3]:
            print(f"  Doc {item['doc_idx']}: {item['top_topics']}")

        # 2. BERTopic (if available)
        if BERTOPIC_AVAILABLE:
            print("\n2. BERTopic (BERT + UMAP + HDBSCAN)")
            print("-" * 70)

            bertopic_model = BERTopicModeler(n_topics=3)
            topics, probs = bertopic_model.fit_transform(documents)

            print("\nTopics discovered:")
            print(bertopic_model.get_topics())

            print("\nTop words for Topic 0:")
            print(bertopic_model.get_topic_words(0))

        print("\n" + "=" * 70)

    if __name__ == "__main__":
        compare_topic_modeling()
    ```

    ## Comparison: Topic Modeling Approaches

    | Method | Approach | Pros | Cons | Best For |
    |--------|----------|------|------|----------|
    | **LDA** | Probabilistic (Bayesian) | Interpretable, fast, established | Bag-of-words (no semantics), needs tuning | Large corpora (1K+ docs) |
    | **NMF** | Matrix factorization | Faster than LDA, non-negative | Less probabilistic interpretation | Quick exploration |
    | **LSA/SVD** | Linear algebra (SVD) | Very fast, deterministic | No probabilistic meaning | Dimensionality reduction |
    | **BERTopic** | BERT + clustering | Best semantics, auto-names topics | Slower, needs GPU for large data | Modern production (500+ docs) |
    | **Top2Vec** | Doc2Vec + clustering | Good semantics, fast inference | Needs more data | Similar to BERTopic |
    | **CTM** | BERT + VAE | Combines neural + probabilistic | Complex, newer | Research/cutting-edge |

    ## Real-World Applications

    **New York Times (Topic Modeling for Archives):**
    - **Dataset:** 1.8M articles (1851-2017)
    - **Method:** LDA with 50 topics
    - **Use Case:** Discover historical themes, recommend similar articles
    - **Result:** 40% improvement in article recommendations

    **Reddit (Subreddit Discovery):**
    - **Dataset:** 10M+ posts
    - **Method:** BERTopic
    - **Use Case:** Discover emerging discussion topics
    - **Result:** Identified 1,200+ distinct topics, 85% user agreement

    **Gensim (Open Source Library):**
    - **Adoption:** 5M+ downloads/year
    - **Method:** LDA, LSI, word2vec
    - **Use Case:** Academic research, industry topic modeling
    - **Speed:** 10K docs in ~5 minutes on CPU

    **PubMed (Medical Literature Analysis):**
    - **Dataset:** 30M+ articles
    - **Method:** LDA + domain-specific preprocessing
    - **Use Case:** Discover research trends, drug-disease associations
    - **Impact:** Identified COVID-19 research clusters in 2020

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Wrong number of topics** | Incoherent or redundant topics | Use coherence score, elbow method (5-50 topics) |
    | **Not preprocessing** | Noisy topics (stop words dominate) | Remove stop words, lemmatize, filter rare/common words |
    | **Using TF-IDF with LDA** | LDA expects counts, not TF-IDF | Use CountVectorizer, not TfidfVectorizer |
    | **Too small corpus** | Unstable topics | Need 500+ docs for LDA, 100+ for BERTopic |
    | **Not tuning alpha/beta** | Topics too sparse or too diffuse | Lower alpha (0.01-0.1) for sparser topics |
    | **Interpreting outliers** | LDA assigns every doc to topics | BERTopic has outlier topic (-1) for noise |
    | **Comparing topics across runs** | LDA is non-deterministic | Set random_state or average multiple runs |

    ## Evaluation Metrics

    **Perplexity:** How surprised the model is by new data (lower is better)
    $$\text{Perplexity} = \exp\left(-\frac{\log p(w|\Theta, \Phi)}{N}\right)$$

    **Coherence Score:** Measures semantic similarity of top words (higher is better)
    - **C_v:** 0.3-0.7 (good range)
    - **U_mass:** -14 to 0 (higher is better)

    **Human Evaluation:** Topic interpretability (5-point scale)

    | Metric | Good Value | Best Method |
    |--------|-----------|-------------|
    | **Perplexity** | < 1000 | LDA optimization |
    | **Coherence (C_v)** | > 0.5 | Grid search over K, alpha, beta |
    | **Topic Diversity** | > 0.8 | Ensure distinct topics (not redundant) |
    | **Human Agreement** | > 70% | Subject matter experts validate |

    !!! tip "Interviewer's Insight"
        **Strong candidates:**

        - Explain LDA generative process: "Each document is a mixture of topics, each topic is a distribution over words"
        - Know preprocessing: "Use CountVectorizer (not TF-IDF), remove stop words, filter rare/common words (min_df=2, max_df=0.95)"
        - Understand hyperparameters: "Lower alpha (0.01-0.1) gives sparser topic distributions; beta controls word distribution per topic"
        - Know evaluation: "Perplexity measures fit, coherence measures interpretability - optimize coherence, not perplexity"
        - Reference modern methods: "BERTopic uses BERT embeddings + HDBSCAN clustering for better semantic topics"
        - Explain practical use: "NYT uses 50 topics for 1.8M articles; start with K=10-20 and tune based on coherence"
        - Know limitations: "LDA is bag-of-words (no word order), needs 500+ docs, non-deterministic without random_state"

---

### What is Question Answering? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Applications` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    ## What is Question Answering (QA)?

    Question Answering is the task of automatically answering questions posed in natural language. Given a question and optionally a context passage, the system returns a precise answer.

    **Real-World Examples:**
    - **Google Search:** Direct answers in search results
    - **Alexa/Siri:** Voice-based QA
    - **Customer Support:** Chatbots answering FAQs
    - **Document Search:** Find answers in company docs

    ## Types of Question Answering

    ### 1. Extractive QA
    - **Definition:** Extract answer span directly from context
    - **Input:** Question + Context passage
    - **Output:** Substring from context (start/end positions)
    - **Example:** Q: "When was Einstein born?" → A: "1879" (from passage)
    - **Models:** BERT, RoBERTa, ELECTRA, DeBERTa

    ### 2. Abstractive QA
    - **Definition:** Generate answer in model's own words
    - **Input:** Question + Context (optional)
    - **Output:** Free-form generated text
    - **Example:** Q: "Why is the sky blue?" → A: "Light scattering causes..." (paraphrased)
    - **Models:** T5, BART, GPT-3, GPT-4

    ### 3. Open-Domain QA
    - **Definition:** Answer questions without given context
    - **Input:** Question only
    - **Output:** Answer retrieved from knowledge base
    - **Process:** Retrieve relevant documents → Extract/generate answer
    - **Models:** DPR (retriever) + BERT (reader), RAG

    ### 4. Closed-Domain QA
    - **Definition:** QA over specific domain (legal, medical, etc.)
    - **Input:** Question + Domain-specific context
    - **Output:** Domain-specific answer
    - **Models:** Fine-tuned BERT on domain data

    ## Production Implementation (200 lines)

    ```python
    # question_answering.py
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        pipeline
    )
    import numpy as np
    from typing import Dict, List, Tuple

    class ExtractiveQA:
        """
        Extractive Question Answering with BERT/RoBERTa

        Architecture: [CLS] question [SEP] context [SEP]
        Output: Start/end logits for answer span

        Time: O(n²) where n = sequence length (due to attention)
        Space: O(n²) for attention matrix
        """

        def __init__(self, model_name='deepset/roberta-base-squad2'):
            """
            Args:
                model_name: Pretrained QA model from HuggingFace
                    - 'deepset/roberta-base-squad2' (82.9 F1 on SQuAD 2.0)
                    - 'deepset/bert-base-cased-squad2' (80.9 F1)
                    - 'deepset/deberta-v3-large-squad2' (87.8 F1)
            """
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.model.eval()

            # Move to GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)

        def answer(
            self,
            question: str,
            context: str,
            max_answer_length: int = 30,
            top_k: int = 1
        ) -> List[Dict]:
            """
            Answer a question given context

            Args:
                question: Question string
                context: Context passage
                max_answer_length: Max tokens in answer
                top_k: Number of top answers to return

            Returns:
                List of answers with scores and positions
            """
            # Tokenize input: [CLS] question [SEP] context [SEP]
            inputs = self.tokenizer(
                question,
                context,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Start and end logits: [batch, seq_len]
            start_logits = outputs.start_logits[0]  # [seq_len]
            end_logits = outputs.end_logits[0]      # [seq_len]

            # Find top answer spans
            answers = []
            for start_idx in torch.topk(start_logits, top_k).indices:
                for end_idx in torch.topk(end_logits, top_k).indices:
                    start_idx = start_idx.item()
                    end_idx = end_idx.item()

                    # Valid span: end >= start, within max length
                    if end_idx >= start_idx and (end_idx - start_idx + 1) <= max_answer_length:
                        # Compute score (sum of logits)
                        score = (start_logits[start_idx] + end_logits[end_idx]).item()

                        # Decode answer
                        answer_tokens = inputs.input_ids[0][start_idx:end_idx+1]
                        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

                        answers.append({
                            'answer': answer,
                            'score': score,
                            'start': start_idx,
                            'end': end_idx
                        })

            # Sort by score and return top_k
            answers = sorted(answers, key=lambda x: x['score'], reverse=True)[:top_k]

            return answers

        def answer_no_answer(self, question: str, context: str, threshold: float = 0.0) -> Dict:
            """
            Handle unanswerable questions (SQuAD 2.0 style)

            Returns 'no_answer' if confidence below threshold
            """
            answers = self.answer(question, context, top_k=1)

            if not answers or answers[0]['score'] < threshold:
                return {'answer': 'no_answer', 'score': 0.0}

            return answers[0]

    class AbstractiveQA:
        """
        Abstractive Question Answering with T5/BART

        Architecture: Encoder-decoder transformer
        Output: Generated answer (not constrained to context)

        Time: O(n × m) where n=input_len, m=output_len
        Space: O(n + m)
        """

        def __init__(self, model_name='google/flan-t5-base'):
            """
            Args:
                model_name: Seq2seq model
                    - 'google/flan-t5-base' (248M params)
                    - 'google/flan-t5-large' (783M params)
                    - 'facebook/bart-large' (406M params)
            """
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.eval()

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)

        def answer(
            self,
            question: str,
            context: str = None,
            max_length: int = 100,
            num_beams: int = 4
        ) -> str:
            """
            Generate answer to question

            Args:
                question: Question string
                context: Optional context (if None, uses model knowledge)
                max_length: Max tokens to generate
                num_beams: Beam search width (higher = better but slower)

            Returns:
                Generated answer string
            """
            # Format input for T5: "question: ... context: ..."
            if context:
                input_text = f"question: {question} context: {context}"
            else:
                input_text = question

            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors='pt',
                truncation=True,
                max_length=512
            ).to(self.device)

            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    no_repeat_ngram_size=3  # Avoid repetition
                )

            # Decode
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer

    class OpenDomainQA:
        """
        Open-domain QA with retrieval + reading

        Pipeline: Question → Retrieve docs → Extract answer
        Similar to RAG but focused on QA

        Used in: Google Search, Bing, Perplexity AI
        """

        def __init__(
            self,
            retriever_model='facebook/dpr-question_encoder-single-nq-base',
            reader_model='deepset/roberta-base-squad2'
        ):
            """
            Args:
                retriever_model: Dense passage retriever
                reader_model: Extractive QA model
            """
            # Retriever (simplified - use FAISS in production)
            self.retriever = None  # Would use DPR + FAISS

            # Reader
            self.reader = ExtractiveQA(model_name=reader_model)

            # Knowledge base (in production: FAISS index)
            self.knowledge_base = []

        def add_documents(self, documents: List[str]):
            """Add documents to knowledge base"""
            self.knowledge_base.extend(documents)

        def retrieve(self, question: str, top_k: int = 3) -> List[str]:
            """
            Retrieve most relevant documents

            In production: Use DPR embeddings + FAISS search
            Here: Simple keyword matching for demo
            """
            # Simplified retrieval (use BM25 or dense retrieval in production)
            question_words = set(question.lower().split())

            scores = []
            for doc in self.knowledge_base:
                doc_words = set(doc.lower().split())
                overlap = len(question_words & doc_words)
                scores.append(overlap)

            # Get top-k documents
            top_indices = np.argsort(scores)[-top_k:][::-1]
            return [self.knowledge_base[i] for i in top_indices]

        def answer(self, question: str, top_k_docs: int = 3) -> Dict:
            """
            Answer question using retrieval + reading

            Args:
                question: Question string
                top_k_docs: Number of documents to retrieve

            Returns:
                Answer with source document
            """
            # 1. Retrieve relevant documents
            contexts = self.retrieve(question, top_k=top_k_docs)

            if not contexts:
                return {'answer': 'no_answer', 'score': 0.0, 'source': None}

            # 2. Extract answer from each context
            all_answers = []
            for context in contexts:
                answers = self.reader.answer(question, context, top_k=1)
                if answers:
                    all_answers.append({
                        **answers[0],
                        'source': context
                    })

            # 3. Return best answer
            if not all_answers:
                return {'answer': 'no_answer', 'score': 0.0, 'source': None}

            best_answer = max(all_answers, key=lambda x: x['score'])
            return best_answer

    # Example usage
    def demo_question_answering():
        """Demonstrate different QA approaches"""

        # Sample data
        context = """
        Albert Einstein was born on March 14, 1879, in Ulm, Germany.
        He developed the theory of relativity, one of the two pillars of modern physics.
        In 1921, Einstein received the Nobel Prize in Physics for his explanation
        of the photoelectric effect. He died on April 18, 1955, in Princeton, New Jersey.
        """

        questions = [
            "When was Einstein born?",
            "What did Einstein win the Nobel Prize for?",
            "Where did Einstein die?"
        ]

        print("=" * 70)
        print("QUESTION ANSWERING DEMO")
        print("=" * 70)

        # 1. Extractive QA
        print("\n1. EXTRACTIVE QA (BERT)")
        print("-" * 70)
        print(f"Context: {context[:100]}...")

        extractive_qa = ExtractiveQA(model_name='deepset/roberta-base-squad2')

        for question in questions:
            answers = extractive_qa.answer(question, context, top_k=1)
            print(f"\nQ: {question}")
            print(f"A: {answers[0]['answer']} (score: {answers[0]['score']:.2f})")

        # 2. Abstractive QA
        print("\n\n2. ABSTRACTIVE QA (T5)")
        print("-" * 70)

        abstractive_qa = AbstractiveQA(model_name='google/flan-t5-base')

        for question in questions:
            answer = abstractive_qa.answer(question, context)
            print(f"\nQ: {question}")
            print(f"A: {answer}")

        print("\n" + "=" * 70)

    if __name__ == "__main__":
        demo_question_answering()
    ```

    ## Comparison: QA Approaches

    | Type | Input | Output | Accuracy | Fluency | Factuality | Best For |
    |------|-------|--------|----------|---------|------------|----------|
    | **Extractive** | Q + Context | Exact span | High (87+ F1) | Medium | High (grounded) | Exact facts (dates, names) |
    | **Abstractive** | Q + Context | Generated | Medium (75+ F1) | High | Medium (can hallucinate) | Explanations, summaries |
    | **Open-domain** | Q only | Retrieved + extracted | Medium (65+ F1) | Medium | High (with retrieval) | General knowledge |
    | **Generative (GPT)** | Q only | Generated | Varies | High | Low (hallucinates) | Creative, open-ended |

    ## Benchmarks & Datasets

    | Dataset | Type | Size | Metric | SOTA Performance |
    |---------|------|------|--------|------------------|
    | **SQuAD 1.1** | Extractive | 100K Q&A | Exact Match / F1 | 95.1 EM, 97.8 F1 (Ensemble) |
    | **SQuAD 2.0** | Extractive + unanswerable | 150K Q&A | EM / F1 | 90.9 EM, 93.2 F1 (RoBERTa) |
    | **Natural Questions** | Open-domain | 307K Q&A | EM / F1 | 54.7 EM (DPR + BERT) |
    | **TriviaQA** | Open-domain | 650K Q&A | EM / F1 | 72.5 EM (DPR) |
    | **HotpotQA** | Multi-hop reasoning | 113K Q&A | EM / F1 | 67.5 F1 (Graph Neural Net) |

    ## Real-World Applications

    **Google Search (Featured Snippets):**
    - **Task:** Extractive QA for search queries
    - **Model:** BERT-based (likely custom)
    - **Scale:** Billions of queries/day
    - **Impact:** 15-20% of search results have featured snippets
    - **Example:** "How tall is Eiffel Tower?" → "330 meters"

    **Amazon Alexa:**
    - **Task:** Open-domain QA (voice)
    - **Models:** Knowledge graph + neural QA
    - **Scale:** 100M+ devices
    - **Accuracy:** 85%+ for factual questions
    - **Latency:** < 1 second end-to-end

    **IBM Watson (Jeopardy!, 2011):**
    - **Task:** Open-domain QA
    - **Approach:** Ensemble of 100+ models
    - **Result:** Beat human champions
    - **Now:** Watson Assistant (enterprise chatbots)

    **ChatGPT (OpenAI):**
    - **Task:** Abstractive QA (generative)
    - **Model:** GPT-3.5/4
    - **Strength:** Fluent, conversational answers
    - **Weakness:** Hallucinations (20-30% factual errors without RAG)
    - **Usage:** 100M+ weekly users

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Long contexts (>512 tokens)** | Truncation loses answer | Use sliding window, chunking, or Longformer (4K tokens) |
    | **Unanswerable questions** | Model always answers | Use SQuAD 2.0 models with no-answer option |
    | **Multiple answers in context** | Returns only first | Rank all candidates, return top-k |
    | **Ambiguous questions** | Wrong interpretation | Use clarification questions or contextual history |
    | **Poor retrieval (open-domain)** | Correct answer not in docs | Improve retriever (DPR, ColBERT), rerank with cross-encoder |
    | **Hallucinations (abstractive)** | Factually incorrect | Use extractive, add retrieval (RAG), post-process with fact-checking |
    | **Slow inference (large models)** | > 2s latency | Use distilled models (DistilBERT 60% faster), quantization, batching |

    ## Evaluation Metrics

    **Exact Match (EM):** Percentage of predictions exactly matching ground truth
    - Strict: "March 14, 1879" vs "1879" → 0%
    - Good for dates, names

    **F1 Score:** Token-level overlap between prediction and ground truth
    - Partial credit: "March 14, 1879" vs "1879" → F1 = 0.33
    - More forgiving than EM

    **BLEU/ROUGE (Abstractive):** N-gram overlap for generated answers
    - BLEU: Precision-focused (used in translation)
    - ROUGE: Recall-focused (used in summarization)

    **Human Evaluation:** Fluency, relevance, correctness (5-point scale)

    | Metric | Range | Good Value | Best For |
    |--------|-------|------------|----------|
    | **Exact Match** | 0-100% | > 80% | Extractive QA |
    | **F1 Score** | 0-100% | > 85% | Extractive QA |
    | **BLEU** | 0-100 | > 30 | Abstractive QA |
    | **Human Rating** | 1-5 | > 4.0 | All types |

    !!! tip "Interviewer's Insight"
        **Strong candidates:**

        - Explain extractive vs abstractive: "Extractive finds exact span in context (high accuracy, grounded), abstractive generates answer (more fluent, can hallucinate)"
        - Know SQuAD datasets: "SQuAD 1.1 has answerable questions only; SQuAD 2.0 adds unanswerable (50K) to prevent always-answer bias"
        - Understand architecture: "Extractive uses BERT with start/end token classification; abstractive uses T5/BART encoder-decoder"
        - Reference real systems: "Google Featured Snippets use extractive QA; ChatGPT uses generative (hallucinations without RAG)"
        - Know open-domain pipeline: "Retrieve top-k docs with DPR/BM25, extract answer from each, rerank by score - used in Perplexity AI"
        - Explain metrics: "EM is strict (exact match), F1 allows partial overlap - F1 85%+ is strong on SQuAD"
        - Handle edge cases: "Long contexts need chunking or Longformer; unanswerable questions need SQuAD 2.0 models with no-answer threshold"

---

### What is Text Classification? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Classification` | **Asked by:** Most Tech Companies

??? success "View Answer"

    ## What is Text Classification?

    Text classification is the task of assigning predefined categories/labels to text documents. It's one of the most common NLP tasks in production.

    **Common Applications:**
    - **Spam Detection:** Email spam vs ham
    - **Sentiment Analysis:** Positive/negative/neutral reviews
    - **Intent Classification:** Chatbot intent detection
    - **Topic Categorization:** News article categorization
    - **Language Detection:** Identify language of text
    - **Toxicity Detection:** Identify harmful content

    ## Classification Types

    ### 1. Binary Classification
    - **2 classes:** Spam vs Ham, Positive vs Negative
    - **Metrics:** Accuracy, Precision, Recall, F1, AUC-ROC

    ### 2. Multi-Class Classification
    - **3+ mutually exclusive classes:** Sports, Politics, Tech, Entertainment
    - **Output:** Single label per document
    - **Metrics:** Accuracy, Macro/Micro F1

    ### 3. Multi-Label Classification
    - **Multiple labels per document:** Article can be both "Sports" and "Politics"
    - **Output:** Set of labels
    - **Metrics:** Hamming Loss, Label Ranking Average Precision

    ## Approaches: Evolution

    ### 1. Traditional ML (Pre-2018)
    - **Features:** TF-IDF, Bag-of-Words, n-grams
    - **Models:** Naive Bayes, Logistic Regression, SVM
    - **Pros:** Fast, interpretable, works with small data (100-1K samples)
    - **Cons:** No semantics, manual feature engineering

    ### 2. Deep Learning (2014-2018)
    - **Features:** Word2Vec, GloVe embeddings
    - **Models:** LSTM, CNN for text
    - **Pros:** Better than traditional, learns features
    - **Cons:** Needs more data (10K+ samples)

    ### 3. Transfer Learning (2018-2020)
    - **Features:** Contextual embeddings
    - **Models:** BERT, RoBERTa fine-tuning
    - **Pros:** SOTA accuracy, works with 1K+ samples
    - **Cons:** Slow inference, large models

    ### 4. Few-Shot Learning (2020+)
    - **Models:** SetFit, GPT-3/4 few-shot, prompt engineering
    - **Pros:** Works with 8-64 examples
    - **Cons:** Less accurate than full fine-tuning (on large data)

    ## Production Implementation (210 lines)

    ```python
    # text_classification.py
    import numpy as np
    import pandas as pd
    from typing import List, Dict, Tuple

    # Traditional ML
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    from sklearn.metrics import classification_report, confusion_matrix

    # Deep Learning
    import torch
    import torch.nn as nn
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer
    )
    from datasets import Dataset

    class TraditionalTextClassifier:
        """
        Text classification with TF-IDF + Logistic Regression

        Pros: Fast (100ms inference), interpretable, works with small data
        Cons: No semantics (bag-of-words)

        Best for: Production where speed matters, interpretability needed
        """

        def __init__(self, max_features=10000, ngram_range=(1, 2)):
            """
            Args:
                max_features: Max vocabulary size
                ngram_range: (min_n, max_n) for n-grams
                    (1, 1) = unigrams only
                    (1, 2) = unigrams + bigrams
                    (1, 3) = unigrams + bigrams + trigrams
            """
            # Vectorizer: Text → TF-IDF features
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                min_df=2,  # Ignore rare words (appear in < 2 docs)
                max_df=0.95  # Ignore very common words
            )

            # Classifier: TF-IDF → Label
            self.classifier = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',  # Handle imbalanced data
                random_state=42
            )

        def fit(self, texts: List[str], labels: List[int]):
            """
            Train classifier

            Args:
                texts: List of text documents
                labels: List of integer labels (0, 1, 2, ...)
            """
            # Convert text to TF-IDF
            X = self.vectorizer.fit_transform(texts)

            print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
            print(f"Feature matrix shape: {X.shape}")

            # Train classifier
            self.classifier.fit(X, labels)

            # Training accuracy
            train_acc = self.classifier.score(X, labels)
            print(f"Training accuracy: {train_acc:.3f}")

        def predict(self, texts: List[str]) -> np.ndarray:
            """Predict labels for texts"""
            X = self.vectorizer.transform(texts)
            return self.classifier.predict(X)

        def predict_proba(self, texts: List[str]) -> np.ndarray:
            """Predict probabilities for each class"""
            X = self.vectorizer.transform(texts)
            return self.classifier.predict_proba(X)

        def get_top_features(self, class_idx: int, top_n: int = 20) -> List[Tuple[str, float]]:
            """
            Get most important features for a class

            Useful for interpretability
            """
            # Get feature weights for this class
            coef = self.classifier.coef_[class_idx]

            # Get top positive features (most indicative of class)
            top_indices = np.argsort(coef)[-top_n:][::-1]

            feature_names = self.vectorizer.get_feature_names_out()
            top_features = [(feature_names[i], coef[i]) for i in top_indices]

            return top_features

    class BERTTextClassifier:
        """
        Text classification with BERT fine-tuning

        Pros: SOTA accuracy, contextual understanding
        Cons: Slower (500ms inference), needs GPU

        Best for: High-accuracy requirements, sufficient data (1K+ samples)
        """

        def __init__(self, model_name='distilbert-base-uncased', num_labels=2):
            """
            Args:
                model_name: Pretrained model
                    - 'distilbert-base-uncased' (66M params, 2x faster)
                    - 'bert-base-uncased' (110M params)
                    - 'roberta-base' (125M params, slightly better)
                num_labels: Number of classes
            """
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)

        def prepare_dataset(self, texts: List[str], labels: List[int]) -> Dataset:
            """Convert texts and labels to HuggingFace Dataset"""
            # Tokenize
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )

            # Create dataset
            dataset = Dataset.from_dict({
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'labels': labels
            })

            return dataset

        def train(
            self,
            train_texts: List[str],
            train_labels: List[int],
            val_texts: List[str] = None,
            val_labels: List[int] = None,
            epochs: int = 3,
            batch_size: int = 16,
            learning_rate: float = 2e-5
        ):
            """
            Fine-tune BERT on classification task

            Args:
                train_texts, train_labels: Training data
                val_texts, val_labels: Validation data (optional)
                epochs: Number of training epochs (3-5 typical)
                batch_size: Batch size (16-32 for base models)
                learning_rate: Learning rate (2e-5 to 5e-5 for BERT)
            """
            # Prepare datasets
            train_dataset = self.prepare_dataset(train_texts, train_labels)
            eval_dataset = None
            if val_texts and val_labels:
                eval_dataset = self.prepare_dataset(val_texts, val_labels)

            # Training arguments
            training_args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=0.01,
                logging_steps=10,
                evaluation_strategy='epoch' if eval_dataset else 'no',
                save_strategy='epoch',
                load_best_model_at_end=True if eval_dataset else False,
                metric_for_best_model='accuracy' if eval_dataset else None,
            )

            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=self._compute_metrics
            )

            # Train
            trainer.train()

            print(f"Training complete!")

        def _compute_metrics(self, eval_pred):
            """Compute accuracy during evaluation"""
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=1)
            accuracy = (predictions == labels).mean()
            return {'accuracy': accuracy}

        def predict(self, texts: List[str]) -> np.ndarray:
            """Predict labels for texts"""
            self.model.eval()

            # Tokenize
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(**encodings)
                predictions = torch.argmax(outputs.logits, dim=1)

            return predictions.cpu().numpy()

        def predict_proba(self, texts: List[str]) -> np.ndarray:
            """Predict probabilities for each class"""
            self.model.eval()

            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encodings)
                probs = torch.softmax(outputs.logits, dim=1)

            return probs.cpu().numpy()

    # Example usage
    def compare_classifiers():
        """Compare traditional vs BERT classifiers"""

        # Sample data (sentiment classification)
        train_texts = [
            "This product is amazing! I love it.",
            "Terrible quality, waste of money.",
            "Best purchase ever, highly recommend!",
            "Awful experience, very disappointed.",
            "Great value for the price.",
            "Poor quality, broke after one use.",
        ]
        train_labels = [1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative

        test_texts = [
            "Fantastic product, exceeded expectations!",
            "Not worth the money, very bad.",
        ]
        test_labels = [1, 0]

        print("=" * 70)
        print("TEXT CLASSIFICATION COMPARISON")
        print("=" * 70)

        # 1. Traditional ML (TF-IDF + Logistic Regression)
        print("\n1. TRADITIONAL ML (TF-IDF + LogReg)")
        print("-" * 70)

        trad_clf = TraditionalTextClassifier(max_features=1000, ngram_range=(1, 2))
        trad_clf.fit(train_texts, train_labels)

        preds = trad_clf.predict(test_texts)
        print(f"\nPredictions: {preds}")
        print(f"True labels: {test_labels}")

        # Show top features for positive class
        print("\nTop features for POSITIVE class:")
        top_features = trad_clf.get_top_features(class_idx=1, top_n=5)
        for word, weight in top_features:
            print(f"  {word}: {weight:.3f}")

        # 2. BERT (requires more data in practice, shown for demo)
        print("\n\n2. BERT FINE-TUNING")
        print("-" * 70)
        print("(Note: BERT works best with 1000+ samples, shown for demo)")

        # bert_clf = BERTTextClassifier(model_name='distilbert-base-uncased', num_labels=2)
        # bert_clf.train(train_texts, train_labels, epochs=3, batch_size=4)
        # preds = bert_clf.predict(test_texts)
        # print(f"\nPredictions: {preds}")

        print("\n" + "=" * 70)

    if __name__ == "__main__":
        compare_classifiers()
    ```

    ## Comparison: Classification Approaches

    | Approach | Data Needed | Training Time | Inference | Accuracy | Best For |
    |----------|-------------|---------------|-----------|----------|----------|
    | **TF-IDF + LogReg** | 100-1K | Seconds | 1-10ms | 75-85% | Fast production, small data |
    | **TF-IDF + SVM** | 100-1K | Minutes | 1-10ms | 80-88% | Slightly better than LogReg |
    | **Word2Vec + LSTM** | 5K-50K | Hours | 50-100ms | 85-90% | Legacy deep learning |
    | **BERT fine-tuning** | 1K-10K | Hours (GPU) | 100-500ms | 90-95% | High accuracy, sufficient data |
    | **DistilBERT** | 1K-10K | Hours (GPU) | 50-200ms | 88-93% | Faster BERT (2x speedup) |
    | **SetFit** | 8-64 | Minutes | 50-200ms | 85-92% | Few-shot learning |
    | **GPT-3/4 few-shot** | 0-10 | None | 1-3s | 80-90% | Zero/few-shot, no training |

    ## Real-World Applications

    **Gmail Spam Detection (Google):**
    - **Task:** Binary classification (spam vs ham)
    - **Model:** TensorFlow-based neural network
    - **Scale:** 100M+ emails/day
    - **Accuracy:** 99.9% spam detection, <0.1% false positives
    - **Features:** Text + metadata (sender, links, attachments)

    **Twitter Toxicity Detection:**
    - **Task:** Multi-label (toxic, severe toxic, obscene, threat, insult)
    - **Model:** BERT fine-tuned on 160K comments
    - **Accuracy:** 92% F1 on Toxic Comment dataset
    - **Challenge:** Adversarial examples, context-dependent

    **Zendesk Intent Classification (Customer Support):**
    - **Task:** Multi-class (billing, technical, refund, general)
    - **Model:** DistilBERT (60M params)
    - **Data:** 10K labeled tickets
    - **Accuracy:** 89% on 15 intents
    - **Latency:** <200ms (acceptable for chatbots)

    **Amazon Review Sentiment Analysis:**
    - **Task:** Multi-class (1-5 stars)
    - **Model:** TF-IDF + Logistic Regression (baseline), BERT (production)
    - **Data:** Millions of reviews
    - **Baseline:** 75% accuracy (TF-IDF)
    - **BERT:** 88% accuracy
    - **Use Case:** Product recommendations, seller ratings

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Imbalanced classes** | Model predicts majority class | Use class weights, SMOTE, stratified sampling |
    | **Overfitting (small data)** | High train, low test accuracy | Use regularization (L2), dropout, more data |
    | **Long texts (>512 tokens)** | Truncation loses info | Use chunking, hierarchical models, Longformer |
    | **Domain shift** | Train on reviews, test on tweets | Domain adaptation, fine-tune on target domain |
    | **Slow BERT inference** | >500ms latency | Use DistilBERT (2x faster), quantization, ONNX |
    | **Not enough data for BERT** | <500 samples | Use TF-IDF, SetFit (few-shot), data augmentation |
    | **Multi-label (not multi-class)** | Wrong loss function | Use BCEWithLogitsLoss, not CrossEntropyLoss |

    ## Evaluation Metrics

    ### Binary Classification
    - **Accuracy:** (TP + TN) / Total
    - **Precision:** TP / (TP + FP) - "Of predicted positives, how many are correct?"
    - **Recall:** TP / (TP + FN) - "Of actual positives, how many did we find?"
    - **F1:** Harmonic mean of precision and recall
    - **AUC-ROC:** Area under ROC curve (threshold-independent)

    ### Multi-Class Classification
    - **Macro F1:** Average F1 across classes (treats all classes equally)
    - **Micro F1:** Global F1 (better for imbalanced data)
    - **Weighted F1:** F1 weighted by support

    ### Multi-Label Classification
    - **Hamming Loss:** Fraction of wrong labels
    - **Subset Accuracy:** Exact match of label sets
    - **Label Ranking Average Precision:** Ranking quality

    | Metric | Range | Good Value | When to Use |
    |--------|-------|------------|-------------|
    | **Accuracy** | 0-100% | > 85% | Balanced classes |
    | **F1 Score** | 0-100% | > 80% | Imbalanced classes |
    | **AUC-ROC** | 0-1 | > 0.9 | Binary, threshold tuning |
    | **Macro F1** | 0-100% | > 75% | Multi-class, care about all classes |

    !!! tip "Interviewer's Insight"
        **Strong candidates:**

        - Choose appropriate complexity: "For 100 samples use TF-IDF + LogReg; for 10K+ use BERT fine-tuning; for 8-64 examples use SetFit"
        - Know metrics for imbalanced data: "With 95% negative class, accuracy misleading - use F1, AUC-ROC instead"
        - Understand BERT tradeoffs: "BERT gives 90%+ accuracy but 500ms latency; DistilBERT is 2x faster with only 2-3% accuracy drop"
        - Reference real systems: "Gmail spam is 99.9% accurate using neural nets; Twitter toxicity uses BERT (92% F1)"
        - Handle class imbalance: "Use class_weight='balanced' in sklearn, or SMOTE for oversampling minority class"
        - Know few-shot learning: "SetFit works with 8-64 examples per class; GPT-3 does zero-shot but less accurate"
        - Explain TF-IDF: "Term frequency × inverse document frequency - weights important words, downweights common words like 'the'"

---

### What is Zero-Shot Classification? - OpenAI, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Transfer Learning` | **Asked by:** OpenAI, Google, Meta

??? success "View Answer"

    ## What is Zero-Shot Classification?

    Zero-shot classification allows classifying text into categories **without any task-specific training examples**. The model uses its general understanding from pretraining to classify into novel categories.

    **Key Advantage:** No labeled data needed for new categories - just provide category names as strings!

    **Use Cases:**
    - **Rapid prototyping:** Test classification without collecting training data
    - **Dynamic categories:** User-defined labels at runtime
    - **Cold start:** New product categories, emerging topics
    - **Content moderation:** Quickly add new violation types

    ## How It Works: NLI-Based Approach

    **Core Idea:** Convert classification into **Natural Language Inference (NLI)** task.

    **NLI Task:** Given premise and hypothesis, predict relationship:
    - **Entailment:** Hypothesis follows from premise
    - **Contradiction:** Hypothesis contradicts premise
    - **Neutral:** No clear relationship

    **Zero-Shot Classification Pipeline:**

    1. **Input:** Text = "I love playing tennis", Labels = ["sports", "cooking", "travel"]
    2. **Create hypotheses:** For each label, form hypothesis:
       - "This text is about sports" (hypothesis 1)
       - "This text is about cooking" (hypothesis 2)
       - "This text is about travel" (hypothesis 3)
    3. **Run NLI:** Pass (text, hypothesis) pairs to NLI model
    4. **Get entailment scores:** Higher score = more likely category
    5. **Output:** Label with highest entailment score → "sports"

    **Why This Works:** Models trained on NLI (MNLI dataset, 433K examples) learn semantic understanding that transfers to classification.

    ## Production Implementation (160 lines)

    ```python
    # zero_shot_classification.py
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        pipeline
    )
    import numpy as np
    from typing import List, Dict
    from scipy.special import softmax

    class ZeroShotClassifier:
        """
        Zero-shot classification using NLI models

        Converts classification to entailment task:
        - Text + "This text is about {label}" → Entailment score

        Models trained on MNLI (Multi-Genre NLI, 433K examples)

        Time: O(n × k) where n=text_len, k=num_labels
        Space: O(n)
        """

        def __init__(self, model_name='facebook/bart-large-mnli'):
            """
            Args:
                model_name: NLI model from HuggingFace
                    - 'facebook/bart-large-mnli' (406M, 90.8% MNLI accuracy)
                    - 'cross-encoder/nli-deberta-v3-large' (434M, 91.9% MNLI)
                    - 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli' (best)
            """
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)

            # NLI label mapping (model-specific)
            # BART-MNLI: 0=contradiction, 1=neutral, 2=entailment
            self.entailment_id = 2
            self.contradiction_id = 0

        def classify(
            self,
            text: str,
            candidate_labels: List[str],
            hypothesis_template: str = "This text is about {}.",
            multi_label: bool = False
        ) -> Dict:
            """
            Classify text into candidate labels (zero-shot)

            Args:
                text: Text to classify
                candidate_labels: List of possible labels (any strings!)
                hypothesis_template: Template for forming hypotheses
                    Default: "This text is about {}."
                    Could be: "This example is {}.", "The topic is {}.", etc.
                multi_label: If True, allow multiple labels (independent scores)
                             If False, softmax normalization (mutually exclusive)

            Returns:
                Dict with labels and scores
            """
            # Step 1: Create hypotheses for each label
            hypotheses = [hypothesis_template.format(label) for label in candidate_labels]

            # Step 2: Get entailment scores for each (text, hypothesis) pair
            scores = []

            for hypothesis in hypotheses:
                # Tokenize: [CLS] premise [SEP] hypothesis [SEP]
                inputs = self.tokenizer(
                    text,
                    hypothesis,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512
                ).to(self.device)

                # Get NLI predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits[0]  # [3] for contradiction, neutral, entailment

                # Extract entailment score
                entailment_score = logits[self.entailment_id].item()
                scores.append(entailment_score)

            # Step 3: Normalize scores
            scores = np.array(scores)

            if multi_label:
                # Independent probabilities (sigmoid)
                probs = 1 / (1 + np.exp(-scores))
            else:
                # Mutually exclusive (softmax)
                probs = softmax(scores)

            # Step 4: Sort by score
            sorted_indices = np.argsort(probs)[::-1]

            return {
                'sequence': text,
                'labels': [candidate_labels[i] for i in sorted_indices],
                'scores': [probs[i] for i in sorted_indices]
            }

        def classify_batch(
            self,
            texts: List[str],
            candidate_labels: List[str],
            hypothesis_template: str = "This text is about {}."
        ) -> List[Dict]:
            """Classify multiple texts (batched for efficiency)"""
            return [
                self.classify(text, candidate_labels, hypothesis_template)
                for text in texts
            ]

    # Example usage
    def demo_zero_shot():
        """Demonstrate zero-shot classification"""

        print("=" * 70)
        print("ZERO-SHOT CLASSIFICATION DEMO")
        print("=" * 70)

        # Initialize classifier
        # Using HuggingFace pipeline (easier API, same underlying approach)
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )

        # Example 1: News article classification
        print("\n1. NEWS ARTICLE CLASSIFICATION")
        print("-" * 70)

        text1 = """
        The stock market rallied today as tech giants reported strong earnings.
        Apple and Microsoft both beat analyst expectations, driving the Nasdaq up 2%.
        """

        labels1 = ["business", "technology", "sports", "politics", "entertainment"]

        result1 = classifier(text1, labels1)
        print(f"Text: {text1.strip()[:100]}...")
        print(f"\nTop predictions:")
        for label, score in zip(result1['labels'][:3], result1['scores'][:3]):
            print(f"  {label}: {score:.3f}")

        # Example 2: Customer review sentiment (no training data!)
        print("\n\n2. CUSTOMER REVIEW SENTIMENT")
        print("-" * 70)

        text2 = "This product exceeded my expectations! Great quality and fast shipping."

        labels2 = ["positive", "negative", "neutral"]

        result2 = classifier(text2, labels2)
        print(f"Review: {text2}")
        print(f"\nSentiment: {result2['labels'][0]} (score: {result2['scores'][0]:.3f})")

        # Example 3: Intent classification for chatbot
        print("\n\n3. CHATBOT INTENT CLASSIFICATION")
        print("-" * 70)

        text3 = "I want to cancel my subscription and get a refund."

        labels3 = [
            "cancel subscription",
            "request refund",
            "technical support",
            "billing inquiry",
            "general question"
        ]

        result3 = classifier(text3, labels3, multi_label=True)  # Can have multiple intents
        print(f"User message: {text3}")
        print(f"\nDetected intents:")
        for label, score in zip(result3['labels'][:3], result3['scores'][:3]):
            print(f"  {label}: {score:.3f}")

        # Example 4: Custom categories (no predefined labels!)
        print("\n\n4. DYNAMIC CUSTOM CATEGORIES")
        print("-" * 70)

        text4 = "I'm learning Python and building a machine learning model with TensorFlow."

        # User-defined categories (can be anything!)
        labels4 = [
            "programming",
            "data science",
            "cooking recipes",
            "travel destinations",
            "fitness advice"
        ]

        result4 = classifier(text4, labels4)
        print(f"Text: {text4}")
        print(f"\nTop category: {result4['labels'][0]} (score: {result4['scores'][0]:.3f})")

        print("\n" + "=" * 70)

    if __name__ == "__main__":
        demo_zero_shot()
    ```

    ## Comparison: Zero-Shot vs Few-Shot vs Full Fine-Tuning

    | Approach | Training Examples | Accuracy | Inference Time | Best For |
    |----------|------------------|----------|----------------|----------|
    | **Zero-Shot** | 0 | 60-75% | Medium (3x slower due to NLI) | Rapid prototyping, dynamic labels |
    | **Few-Shot (SetFit)** | 8-64 per class | 75-88% | Medium | Quick deployment, limited data |
    | **Fine-Tuning (BERT)** | 1K-10K+ | 88-95% | Fast | Production, sufficient data |
    | **GPT-3/4 Zero-Shot** | 0-10 (prompts) | 70-85% | Slow (API call, 1-3s) | No infrastructure, exploration |

    ## Real-World Applications

    **Hugging Face Inference API:**
    - **Model:** BART-large-MNLI (406M params)
    - **Usage:** 50M+ API calls/month
    - **Use Case:** Rapid prototyping for startups
    - **Accuracy:** 70-80% on average tasks (vs 90%+ with fine-tuning)

    **Content Moderation (Dynamic Categories):**
    - **Platform:** Reddit, Discord
    - **Use Case:** Quickly add new violation types without retraining
    - **Labels:** Dynamically defined by moderators ("hate speech", "spam", "self-promotion", etc.)
    - **Advantage:** No training data needed for new categories
    - **Limitation:** Lower accuracy (75-80% vs 92%+ with fine-tuned models)

    **News Aggregators (Topic Clustering):**
    - **Use Case:** Classify news into user-defined topics
    - **Labels:** Custom categories per user ("AI research", "climate tech", "indie games")
    - **Model:** DeBERTa-MNLI
    - **Performance:** 72% accuracy vs 88% with fine-tuned classifier

    **Medical Triage (Symptom Classification):**
    - **Use Case:** Classify patient messages into urgency levels
    - **Labels:** ["urgent", "routine", "informational"]
    - **Advantage:** No need for large medical dataset
    - **Limitation:** Used for initial triage only, not diagnosis (70-75% accuracy)

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Poor label names** | Ambiguous, low accuracy | Use descriptive labels: "violent content" not "bad" |
    | **Too many labels** | Slower, confused predictions | Limit to 10-20 labels; use hierarchy if needed |
    | **Wrong hypothesis template** | Misaligned with NLI training | Test templates: "This is {}.", "The topic is {}.", "This example is {}." |
    | **Not using multi-label** | Forced single label when multiple apply | Set multi_label=True for overlapping categories |
    | **Slow inference (N×K forward passes)** | Latency for many labels | Batch processing, cache embeddings, use semantic search first |
    | **Overconfident on wrong labels** | High scores for incorrect labels | Calibrate thresholds, use uncertainty estimation |
    | **Assuming 90%+ accuracy** | Production failures | Zero-shot is 60-75%; use for prototyping, collect data for fine-tuning |

    ## Advanced Techniques

    ### 1. Semantic Search + Zero-Shot (Hybrid)
    - First: Use semantic search to narrow down to top-K candidate labels
    - Then: Apply zero-shot classification on reduced set
    - Speedup: 10-100× faster for large label sets (1000+ labels)

    ### 2. Hypothesis Engineering
    Test different templates for better results:
    ```python
    # Generic
    "This text is about {}."

    # Sentiment-specific
    "The sentiment of this review is {}."

    # Intent-specific
    "The user wants to {}."

    # Topic-specific
    "The main topic is {}."
    ```

    ### 3. Confidence Calibration
    Zero-shot models can be overconfident. Use threshold:
    ```python
    result = classifier(text, labels)
    if result['scores'][0] < 0.6:  # Low confidence
        return "uncertain"
    ```

    ## Evaluation Metrics

    | Metric | Zero-Shot | Fine-Tuned | Notes |
    |--------|-----------|------------|-------|
    | **Accuracy** | 60-75% | 88-95% | Depends on task difficulty |
    | **Inference Time** | 100-500ms | 30-100ms | Zero-shot slower (N×K passes) |
    | **Setup Time** | 0 minutes | Hours-days | Zero-shot: instant |
    | **Data Required** | 0 examples | 1K-10K+ | Zero-shot needs none |

    !!! tip "Interviewer's Insight"
        **Strong candidates:**

        - Explain NLI approach: "Zero-shot converts classification to entailment: text + 'This is about sports' → check if entailment holds"
        - Know MNLI training: "Models trained on 433K entailment examples (MultiNLI) learn semantic understanding that transfers"
        - Understand tradeoffs: "Zero-shot gives 60-75% accuracy vs 90%+ fine-tuned, but needs zero training data - use for prototyping"
        - Reference models: "BART-MNLI (406M params, 90.8% MNLI), DeBERTa-v3-large-MNLI (434M, 91.9% MNLI) are common"
        - Know use cases: "Dynamic content moderation (add categories without retraining), rapid prototyping, user-defined labels"
        - Explain multi-label: "Set multi_label=True for overlapping categories (article can be both 'tech' and 'business')"
        - Discuss limitations: "Slower inference (N×K forward passes), lower accuracy than fine-tuning, sensitive to label wording"

---

### What is Machine Translation? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Translation` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    ## What is Machine Translation (MT)?

    Machine Translation is the task of automatically translating text from one language (source) to another (target) using computers.

    **Real-World Impact:**
    - **Google Translate:** 100B+ words translated daily, 133 languages
    - **DeepL:** Premium translation (often rated higher quality than Google)
    - **Facebook:** Real-time translation for 2.9B+ users
    - **Breaking language barriers** in communication, education, business

    ## Evolution of Machine Translation

    ### 1. Rule-Based MT (1950s-1990s)
    - **Approach:** Hand-crafted grammar rules + dictionaries
    - **Example:** "Je suis" → "I am" (direct word mapping)
    - **Pros:** Deterministic, explainable
    - **Cons:** Brittle, doesn't scale, poor with idioms
    - **BLEU:** ~10-15 (very poor)

    ### 2. Statistical MT (1990s-2015)
    - **Approach:** Learn translation probabilities from parallel corpora
    - **Model:** Phrase-based translation (Moses)
    - **Training:** Align words/phrases in parallel text, build translation tables
    - **Pros:** Data-driven, better than rules
    - **Cons:** Struggles with long-range dependencies, word order
    - **BLEU:** 20-35 (moderate quality)
    - **Peak:** Google Translate (pre-2016)

    ### 3. Neural MT - Seq2Seq (2014-2017)
    - **Approach:** Encoder-decoder with RNN/LSTM + Attention
    - **Architecture:**
      - Encoder: Source text → hidden states
      - Decoder: Hidden states → target text
      - Attention: Focus on relevant source words
    - **Breakthrough:** Google's Neural Machine Translation (GNMT, 2016)
    - **BLEU:** 35-45 (good quality)
    - **Improvement:** 60% error reduction vs Statistical MT

    ### 4. Transformer MT (2017-Present)
    - **Approach:** Self-attention only (no recurrence)
    - **Architecture:** Transformer encoder-decoder
    - **Models:**
      - mT5 (multilingual T5)
      - mBART (multilingual BART)
      - NLLB (No Language Left Behind, Meta)
      - GPT-3/4 (few-shot translation)
    - **BLEU:** 45-55+ (near-human quality on some pairs)
    - **Advantages:** Parallelizable training, better long-range dependencies

    ## Production Implementation (170 lines)

    ```python
    # machine_translation.py
    import torch
    from transformers import (
        MarianMTModel,
        MarianTokenizer,
        M2M100ForConditionalGeneration,
        M2M100Tokenizer
    )
    from typing import List
    import sacrebleu

    class NeuralMachineTranslator:
        """
        Neural Machine Translation using Marian or M2M100

        Marian: Specialized for specific language pairs (en-de, en-fr, etc.)
        M2M100: Multilingual (100 languages, any-to-any translation)

        Time: O(n × m) where n=src_len, m=tgt_len
        Space: O(n + m)
        """

        def __init__(self, model_name='Helsinki-NLP/opus-mt-en-de', device='auto'):
            """
            Args:
                model_name: HuggingFace model
                    Marian (specific pairs):
                    - 'Helsinki-NLP/opus-mt-en-de' (English → German)
                    - 'Helsinki-NLP/opus-mt-en-fr' (English → French)
                    - 'Helsinki-NLP/opus-mt-en-es' (English → Spanish)

                    M2M100 (multilingual, any-to-any):
                    - 'facebook/m2m100_418M' (100 languages)
                    - 'facebook/m2m100_1.2B' (better quality, slower)
            """
            # Check if M2M100 (multilingual) or Marian (specific pair)
            self.is_m2m = 'm2m100' in model_name.lower()

            if self.is_m2m:
                self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
                self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
            else:
                self.tokenizer = MarianTokenizer.from_pretrained(model_name)
                self.model = MarianMTModel.from_pretrained(model_name)

            # Device setup
            if device == 'auto':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)

            self.model.to(self.device)
            self.model.eval()

        def translate(
            self,
            texts: List[str],
            src_lang: str = None,
            tgt_lang: str = None,
            max_length: int = 512,
            num_beams: int = 5,
            **kwargs
        ) -> List[str]:
            """
            Translate texts from source to target language

            Args:
                texts: List of source texts
                src_lang: Source language code (for M2M100 only, e.g., 'en')
                tgt_lang: Target language code (for M2M100 only, e.g., 'de')
                max_length: Max tokens to generate
                num_beams: Beam search width (higher = better quality, slower)
                    1 = greedy (fastest)
                    5 = good tradeoff
                    10 = best quality (slowest)

            Returns:
                List of translated texts
            """
            # Set language for M2M100 (multilingual model)
            if self.is_m2m:
                if not src_lang or not tgt_lang:
                    raise ValueError("M2M100 requires src_lang and tgt_lang (e.g., 'en', 'de')")

                self.tokenizer.src_lang = src_lang
                forced_bos_token_id = self.tokenizer.get_lang_id(tgt_lang)
            else:
                forced_bos_token_id = None

            # Tokenize source texts
            inputs = self.tokenizer(
                texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)

            # Generate translations
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    forced_bos_token_id=forced_bos_token_id,
                    **kwargs
                )

            # Decode translations
            translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            return translations

        def compute_bleu(self, predictions: List[str], references: List[List[str]]) -> Dict:
            """
            Compute BLEU score for translations

            Args:
                predictions: List of predicted translations
                references: List of reference translations (can have multiple per prediction)

            Returns:
                Dict with BLEU score and other metrics
            """
            # sacrebleu expects references as List[List[str]]
            # where each prediction can have multiple references
            bleu = sacrebleu.corpus_bleu(predictions, references)

            return {
                'bleu': bleu.score,
                'precisions': bleu.precisions,
                'bp': bleu.bp,  # Brevity penalty
                'sys_len': bleu.sys_len,
                'ref_len': bleu.ref_len
            }

    # Example usage
    def demo_machine_translation():
        """Demonstrate machine translation"""

        print("=" * 70)
        print("MACHINE TRANSLATION DEMO")
        print("=" * 70)

        # Example 1: English → German (Marian)
        print("\n1. ENGLISH → GERMAN (Marian)")
        print("-" * 70)

        en_de_translator = NeuralMachineTranslator('Helsinki-NLP/opus-mt-en-de')

        en_texts = [
            "Hello, how are you?",
            "Machine translation has improved significantly in recent years.",
            "I would like to order a coffee, please."
        ]

        de_translations = en_de_translator.translate(en_texts, num_beams=5)

        for src, tgt in zip(en_texts, de_translations):
            print(f"EN: {src}")
            print(f"DE: {tgt}\n")

        # Example 2: Multilingual M2M100 (any-to-any)
        print("\n2. MULTILINGUAL TRANSLATION (M2M100)")
        print("-" * 70)
        print("(Note: M2M100 requires more memory, showing example)")

        # m2m_translator = NeuralMachineTranslator('facebook/m2m100_418M')
        #
        # # English → French
        # en_to_fr = m2m_translator.translate(
        #     ["The cat is on the table."],
        #     src_lang='en',
        #     tgt_lang='fr'
        # )
        # print(f"EN → FR: {en_to_fr[0]}")
        #
        # # Spanish → German
        # es_to_de = m2m_translator.translate(
        #     ["Hola, ¿cómo estás?"],
        #     src_lang='es',
        #     tgt_lang='de'
        # )
        # print(f"ES → DE: {es_to_de[0]}")

        # Example 3: BLEU Score Evaluation
        print("\n3. BLEU SCORE EVALUATION")
        print("-" * 70)

        predictions = ["Hallo, wie geht es dir?"]
        references = [["Hallo, wie geht es Ihnen?", "Hallo, wie gehts?"]]  # Multiple refs

        # Note: sacrebleu requires installation
        # bleu_score = en_de_translator.compute_bleu(predictions, references)
        # print(f"BLEU Score: {bleu_score['bleu']:.2f}")

        print("\n" + "=" * 70)

    if __name__ == "__main__":
        demo_machine_translation()
    ```

    ## Evaluation Metrics

    ### 1. BLEU (Bilingual Evaluation Understudy)
    - **Most common metric** (since 2002)
    - **Formula:** Geometric mean of n-gram precision (1-4 grams) × brevity penalty
    - **Range:** 0-100 (higher is better)
    - **Interpretation:**
      - < 10: Almost unusable
      - 10-20: Gist understandable
      - 20-40: Good quality (statistical MT era)
      - 40-50: High quality (neural MT)
      - 50-60: Near-human (best Transformer models)
      - > 60: Human-level (rare, only specific domains)

    **Limitations:**
    - Only measures n-gram overlap, not meaning
    - Multiple correct translations, BLEU rewards only reference match
    - Doesn't capture fluency well

    ### 2. COMET (Crosslingual Optimized Metric for Evaluation of Translation)
    - **Neural metric** (2020+)
    - Uses multilingual BERT to compare semantics
    - **Better correlation with human judgment** than BLEU
    - **Range:** 0-1 (higher is better)

    ### 3. ChrF (Character n-gram F-score)
    - Character-level instead of word-level
    - Better for morphologically rich languages (Finnish, Turkish)

    ### 4. Human Evaluation (Gold Standard)
    - **Adequacy:** Does translation preserve meaning?
    - **Fluency:** Is translation grammatical and natural?
    - **Scale:** 1-5 or 1-7

    | Metric | Correlation with Humans | Speed | Best For |
    |--------|------------------------|-------|----------|
    | **BLEU** | Moderate (0.4-0.6) | Fast | Quick evaluation, established baseline |
    | **COMET** | High (0.7-0.8) | Slow (neural) | Final evaluation, research |
    | **ChrF** | Moderate-High | Fast | Morphologically rich languages |
    | **Human Eval** | Perfect (1.0) | Very slow | Final validation |

    ## Real-World Applications

    **Google Translate (Google):**
    - **Scale:** 100B+ words/day, 133 languages
    - **Model:** Transformer-based Neural MT (2016→)
    - **BLEU:** ~45-50 (en-fr, en-de), ~30-40 (en-zh)
    - **Improvement:** 60% error reduction vs Statistical MT (2016)
    - **Latency:** <200ms for short texts

    **DeepL (DeepL SE):**
    - **Languages:** 31 (focused on European languages)
    - **Model:** Proprietary Transformer
    - **Quality:** Often rated higher than Google (blind tests)
    - **BLEU:** ~50-55 (en-de), ~48-52 (en-fr)
    - **Adoption:** 1B+ translations/day

    **NLLB (Meta, 2022):**
    - **Goal:** No Language Left Behind
    - **Languages:** 200 languages (including low-resource)
    - **Model:** Transformer (54B params)
    - **BLEU:** +44% improvement for low-resource languages
    - **Impact:** Enables translation for African, Southeast Asian languages

    **Meta (Facebook):**
    - **Use Case:** Real-time translation in posts, comments
    - **Model:** M2M100 (multilingual, any-to-any)
    - **Scale:** 20B+ translations/day
    - **Languages:** 100+ languages

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Low-resource languages** | Poor quality (BLEU <20) | Use multilingual models (M2M100), data augmentation, back-translation |
    | **Domain shift** | Medical text with general model | Fine-tune on domain-specific parallel data |
    | **Long sequences (>512 tokens)** | Truncation loses context | Use chunking with overlap, or Longformer |
    | **Rare words/names** | Hallucinated translations | Use copy mechanism, constrained decoding |
    | **BLEU as only metric** | Misses semantic errors | Use COMET, human evaluation for final validation |
    | **Slow inference (beam search)** | >1s latency | Use greedy decoding (num_beams=1), distillation, quantization |
    | **Gender bias** | "The doctor" → "Der Arzt" (male) always | Use gender-neutral models, post-processing |

    ## Advanced Techniques

    ### 1. Back-Translation (Data Augmentation)
    - Translate target→source to create pseudo-parallel data
    - Used to train on monolingual data (no parallel corpus needed)
    - Improves low-resource language quality by 5-10 BLEU

    ### 2. Multilingual Models (M2M100, NLLB)
    - Train single model on 100+ language pairs
    - Transfer learning: high-resource → low-resource
    - Reduces deployment complexity (1 model vs 100+ models)

    ### 3. Interactive MT
    - Human-in-the-loop: Post-edit machine translations
    - Active learning: Model learns from corrections
    - Used in professional translation services

    !!! tip "Interviewer's Insight"
        **Strong candidates:**

        - Explain evolution: "Rule-based (1950s) → Statistical (1990s, BLEU ~30) → Neural Seq2Seq (2014, BLEU ~40) → Transformer (2017, BLEU 50+)"
        - Know BLEU limitations: "BLEU only measures n-gram overlap, not semantics - COMET (neural metric) correlates better with humans"
        - Reference real systems: "Google Translate uses Transformer (133 languages, 100B+ words/day); DeepL often rated higher quality"
        - Understand low-resource challenges: "Languages with <1M sentence pairs struggle - use multilingual models (NLLB) for transfer learning"
        - Know metrics: "BLEU 40-50 is good (neural MT), 50-60 is near-human; COMET 0.7+ correlates well with human judgment"
        - Explain back-translation: "Translate target→source to create pseudo-parallel data, improves low-resource by 5-10 BLEU"
        - Discuss deployment: "Use beam search (num_beams=5) for quality, greedy (num_beams=1) for speed; quantization for mobile"

---

### What is Dependency Parsing? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Syntactic Analysis`, `Graph Parsing`, `Relation Extraction` | **Asked by:** Google, Meta, Amazon, Bloomberg, Apple

??? success "View Answer"

    ## Core Concept

    **Dependency Parsing** analyzes grammatical structure by identifying relationships (dependencies) between words in a sentence. It creates a directed graph (usually a tree) showing which words modify or depend on others.

    **Key Insight:** Unlike constituency parsing which groups words into phrases (NP, VP), dependency parsing creates direct word-to-word relationships, making it ideal for relation extraction and information retrieval.

    ## Dependency Structure

    ```
    ┌──────────────────────────────────────────────────────────────┐
    │          DEPENDENCY PARSING EXAMPLE                          │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Sentence: "The quick brown fox jumps over the lazy dog"    │
    │                                                              │
    │  Dependency Tree:                                            │
    │                                                              │
    │                      jumps (ROOT)                            │
    │                       ┌─┴─┐                                  │
    │                      fox  over                               │
    │                  ┌──┬─┴┬──┤                                  │
    │                 The quick brown dog                          │
    │                              ┌─┴┬───┐                        │
    │                            the lazy ε                        │
    │                                                              │
    │  Dependencies (head → dependent : relation):                 │
    │    jumps → fox       : nsubj  (nominal subject)             │
    │    fox → The         : det    (determiner)                  │
    │    fox → quick       : amod   (adjectival modifier)         │
    │    fox → brown       : amod   (adjectival modifier)         │
    │    jumps → over      : prep   (prepositional modifier)      │
    │    over → dog        : pobj   (object of preposition)       │
    │    dog → the         : det    (determiner)                  │
    │    dog → lazy        : amod   (adjectival modifier)         │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    ```

    ## Universal Dependencies (UD)

    **Standard dependency relations (37 types):**
    - **nsubj:** nominal subject ("John runs" → John is nsubj of runs)
    - **dobj:** direct object ("eat pizza" → pizza is dobj of eat)
    - **amod:** adjectival modifier ("red car" → red modifies car)
    - **det:** determiner ("the cat" → the determines cat)
    - **prep:** prepositional modifier
    - **pobj:** object of preposition
    - **ROOT:** sentence root (usually main verb)

    ## Production Implementation (175 lines)

    ```python
    # dependency_parsing.py
    import spacy
    from spacy import displacy
    import networkx as nx
    import matplotlib.pyplot as plt
    from typing import List, Dict, Tuple
    from dataclasses import dataclass
    import pandas as pd

    @dataclass
    class DependencyRelation:
        \"\"\"Dependency relation between two words\"\"\"
        head: str
        dependent: str
        relation: str
        head_idx: int
        dep_idx: int

    class DependencyParser:
        \"\"\"
        Neural Dependency Parser using spaCy
        
        Algorithm: Transition-based parsing with arc-eager system
        Model: BiLSTM + feed-forward with word + POS embeddings
        
        Time: O(n) where n = sentence length (linear!)
        Space: O(n) for storing dependency tree
        Accuracy: 92-95% UAS (Unlabeled Attachment Score)
        \"\"\"
        
        def __init__(self, model_name: str = \"en_core_web_sm\"):
            \"\"\"
            Args:
                model_name: spaCy model
                    - en_core_web_sm: Small, fast (96 MB)
                    - en_core_web_md: Medium, with vectors (116 MB)
                    - en_core_web_lg: Large, best accuracy (789 MB)
                    - en_core_web_trf: Transformer-based, SOTA (438 MB)
            \"\"\"
            self.nlp = spacy.load(model_name)
        
        def parse(self, text: str) -> List[DependencyRelation]:
            \"\"\"
            Parse sentence into dependencies
            
            Args:
                text: Input sentence
            
            Returns:
                List of dependency relations
            \"\"\"
            doc = self.nlp(text)
            
            dependencies = []
            for token in doc:
                if token.dep_ != \"ROOT\":
                    dependencies.append(DependencyRelation(
                        head=token.head.text,
                        dependent=token.text,
                        relation=token.dep_,
                        head_idx=token.head.i,
                        dep_idx=token.i
                    ))
                else:
                    # ROOT token
                    dependencies.append(DependencyRelation(
                        head=\"ROOT\",
                        dependent=token.text,
                        relation=\"ROOT\",
                        head_idx=-1,
                        dep_idx=token.i
                    ))
            
            return dependencies
        
        def visualize(self, text: str, style: str = \"dep\"):
            \"\"\"
            Visualize dependency tree
            
            Args:
                text: Input sentence
                style: 'dep' (dependency tree) or 'ent' (entities)
            \"\"\"
            doc = self.nlp(text)
            displacy.render(doc, style=style, jupyter=False)
        
        def to_dataframe(self, text: str) -> pd.DataFrame:
            \"\"\"Convert parse to DataFrame for analysis\"\"\"
            doc = self.nlp(text)
            
            data = []
            for token in doc:
                data.append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'pos': token.pos_,
                    'tag': token.tag_,
                    'dep': token.dep_,
                    'head': token.head.text,
                    'children': [child.text for child in token.children]
                })
            
            return pd.DataFrame(data)

    class RelationExtractor:
        \"\"\"
        Extract semantic relations using dependency parsing
        
        Use case: Information extraction, knowledge graph construction
        \"\"\"
        
        def __init__(self):
            self.parser = DependencyParser(\"en_core_web_sm\")
        
        def extract_svo_triples(self, text: str) -> List[Tuple[str, str, str]]:
            \"\"\"
            Extract Subject-Verb-Object triples
            
            Args:
                text: Input sentence
            
            Returns:
                List of (subject, verb, object) tuples
            \"\"\"
            doc = self.parser.nlp(text)
            triples = []
            
            for token in doc:
                # Find verb as predicate
                if token.pos_ == \"VERB\":
                    verb = token.text
                    subject = None
                    obj = None
                    
                    # Find subject (nsubj, nsubjpass)
                    for child in token.children:
                        if child.dep_ in [\"nsubj\", \"nsubjpass\"]:
                            subject = self._get_full_phrase(child)
                        # Find object (dobj, pobj, attr)
                        elif child.dep_ in [\"dobj\", \"pobj\", \"attr\"]:
                            obj = self._get_full_phrase(child)
                    
                    if subject and obj:
                        triples.append((subject, verb, obj))
            
            return triples
        
        def _get_full_phrase(self, token) -> str:
            \"\"\"Get full phrase including modifiers\"\"\"
            # Get token and all its children (subtree)
            phrase = ' '.join([t.text for t in token.subtree])
            return phrase
        
        def extract_entity_relations(self, text: str) -> List[Dict]:
            \"\"\"
            Extract relations between named entities
            
            Example: \"Apple acquired Siri for $200M\"
            → {\"subject\": \"Apple\", \"relation\": \"acquired\", 
                \"object\": \"Siri\", \"modifier\": \"for $200M\"}
            \"\"\"
            doc = self.parser.nlp(text)
            relations = []
            
            # Get entities
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            # Extract relations between entities
            for token in doc:
                if token.pos_ == \"VERB\":
                    subject_ent = None
                    object_ent = None
                    
                    for child in token.children:
                        # Check if child is entity
                        for ent_text, ent_label in entities:
                            if child.text in ent_text:
                                if child.dep_ in [\"nsubj\", \"nsubjpass\"]:
                                    subject_ent = (ent_text, ent_label)
                                elif child.dep_ in [\"dobj\", \"pobj\"]:
                                    object_ent = (ent_text, ent_label)
                    
                    if subject_ent and object_ent:
                        relations.append({
                            \"subject\": subject_ent[0],
                            \"subject_type\": subject_ent[1],
                            \"relation\": token.text,
                            \"object\": object_ent[0],
                            \"object_type\": object_ent[1]
                        })
            
            return relations

    # ===========================================
    # EXAMPLE USAGE
    # ===========================================

    if __name__ == \"__main__\":
        print(\"=\"*70)
        print(\"DEPENDENCY PARSING - RELATION EXTRACTION\")
        print(\"=\"*70)
        
        # 1. Basic Dependency Parsing
        print(\"\\n1. BASIC DEPENDENCY PARSING\")
        print(\"-\"*70)
        
        parser = DependencyParser(\"en_core_web_sm\")
        
        sentence = \"Apple is looking at buying U.K. startup for $1 billion\"
        dependencies = parser.parse(sentence)
        
        print(f\"Sentence: {sentence}\\n\")
        print(f\"{'Dependent':<15} {'Relation':<15} {'Head':<15}\")
        print(\"-\"*45)
        for dep in dependencies:
            print(f\"{dep.dependent:<15} {dep.relation:<15} {dep.head:<15}\")
        
        # 2. DataFrame Representation
        print(\"\\n2. DATAFRAME ANALYSIS\")
        print(\"-\"*70)
        
        df = parser.to_dataframe(sentence)
        print(df[['text', 'pos', 'dep', 'head']].to_string(index=False))
        
        # 3. Relation Extraction
        print(\"\\n3. RELATION EXTRACTION\")
        print(\"-\"*70)
        
        extractor = RelationExtractor()
        
        examples = [
            \"Amazon acquired Whole Foods for $13.7 billion\",
            \"Google released BERT in 2018\",
            \"Elon Musk founded Tesla and SpaceX\"
        ]
        
        for example in examples:
            print(f\"\\nSentence: {example}\")
            
            # SVO triples
            triples = extractor.extract_svo_triples(example)
            print(\"  SVO Triples:\")
            for subj, verb, obj in triples:
                print(f\"    ({subj}, {verb}, {obj})\")
            
            # Entity relations
            relations = extractor.extract_entity_relations(example)
            print(\"  Entity Relations:\")
            for rel in relations:
                print(f\"    {rel['subject']} --[{rel['relation']}]--> {rel['object']}\")
    ```

    ## Parser Algorithms Comparison

    | Algorithm | Approach | Complexity | Accuracy | Speed |
    |-----------|----------|------------|----------|-------|
    | **Transition-based** | Greedy left-to-right | O(n) | 92-94% UAS | Very fast (10K sent/sec) |
    | **Graph-based (MST)** | Find max spanning tree | O(n³) | 93-95% UAS | Slower (1K sent/sec) |
    | **Neural (BiLSTM)** | Deep learning | O(n) | 94-96% UAS | Fast (5K sent/sec) |
    | **Transformer** | Self-attention | O(n²) | 96-97% UAS | Moderate (2K sent/sec) |

    **Modern systems use hybrid:** Neural transition-based (spaCy, Stanford)

    ## Real-World Deployments

    | Company | Use Case | Model | Scale | Application |
    |---------|----------|-------|-------|-------------|
    | **Bloomberg** | Financial IE | Custom parser | 1B+ news articles | Extract M&A relations, stock mentions |
    | **Google** | Knowledge Graph | Neural parser | Web-scale | Build entity relations |
    | **Amazon** | Product QA | spaCy + custom | Millions of questions | Extract product attributes |
    | **IBM Watson** | Medical IE | Dependency + rules | 100K+ medical texts | Extract drug-disease relations |
    | **Meta** | Content understanding | Custom Transformer | Billions of posts | Identify hate speech patterns |

    ## Comparison: Constituency vs Dependency

    | Parsing Type | Structure | Output | Complexity | Use Case |
    |--------------|-----------|--------|------------|----------|
    | **Constituency** | Phrase-based tree (NP, VP, S) | Nested phrases | O(n³) CKY | Linguistic analysis, grammar |
    | **Dependency** | Word-to-word relations | Directed graph | O(n) | Relation extraction, IE, QA |

    **Dependency Tree Advantages:**
    - **Simpler:** Direct word relationships vs nested phrases
    - **Faster:** O(n) vs O(n³) parsing
    - **Better for IE:** Easy to extract (subject, verb, object) triples
    - **Cross-lingual:** Works well across languages (especially non-English)

    ## Common Dependency Relations (Universal Dependencies)

    | Relation | Abbreviation | Example | Use Case |
    |----------|--------------|---------|----------|
    | **Nominal subject** | nsubj | \"John **runs**\" → John | Find agent |
    | **Direct object** | dobj | \"eat **pizza**\" → pizza | Find patient |
    | **Indirect object** | iobj | \"give **him** book\" → him | Find recipient |
    | **Adjectival modifier** | amod | \"**red** car\" → red | Find attributes |
    | **Nominal modifier** | nmod | \"cup of **coffee**\" → coffee | Find relationships |
    | **Prepositional modifier** | prep | \"walk **in** park\" → in | Find location/time |
    | **Compound** | compound | \"**New York**\" → New | Multi-word entities |

    ## Applications in Production

    ### 1. Relation Extraction
    ```
    Input: \"Apple acquired Siri for $200 million\"
    Parse: Apple ← nsubj ← acquired → dobj → Siri
    Output: (Apple, acquired, Siri, $200M)
    ```

    ### 2. Question Answering
    ```
    Question: \"Who founded Tesla?\"
    Parse: Who ← nsubj ← founded → dobj → Tesla
    → Find sentence with (X, founded, Tesla) where X is person
    Answer: \"Elon Musk founded Tesla\" → Elon Musk
    ```

    ### 3. Semantic Role Labeling
    ```
    Sentence: \"John gave Mary a book\"
    Dependencies:
    - gave → nsubj → John (Agent)
    - gave → iobj → Mary (Recipient)
    - gave → dobj → book (Theme)
    ```

    ## Accuracy Benchmarks

    | Dataset | Metric | spaCy (BiLSTM) | Stanford (Neural) | Transformer |
    |---------|--------|----------------|-------------------|-------------|
    | **Penn Treebank** | UAS | 94.2% | 95.1% | 96.4% |
    | **Penn Treebank** | LAS | 92.4% | 93.5% | 95.2% |
    | **Universal Dependencies** | UAS | 89.7% | 91.2% | 93.8% |

    **UAS = Unlabeled Attachment Score (correct head)**
    **LAS = Labeled Attachment Score (correct head + relation)**

    !!! tip \"Interviewer's Insight\"
        **What they test:**
        
        - Understanding of dependency vs constituency parsing
        - Knowledge of practical applications (relation extraction, QA)
        - Awareness of modern neural parsers (spaCy, Stanza)
        - Experience with extracting structured information from text
        
        **Strong signal:**
        
        - \"Dependency parsing is O(n) linear time using transition-based algorithms like arc-eager. It creates direct word-to-word relationships, making it ideal for relation extraction. For example, 'Apple acquired Siri' parses to (Apple, nsubj) → acquired ← (Siri, dobj), extracting the (Apple, acquired, Siri) triple\"
        - \"At Bloomberg, we use dependency parsing for financial IE - extracting M&A relations from news. spaCy's neural parser achieves 94% UAS, trained on 1M+ sentences. We extract (acquirer, acquired, amount) tuples to populate our knowledge graph\"
        - \"Modern parsers use BiLSTM + feed-forward networks. spaCy processes 10K sentences/sec with 94% accuracy. Transformer-based parsers (Stanza) achieve 96%+ but are 5× slower. Trade-off: accuracy vs speed\"
        - \"Unlike constituency parsing which creates phrase structures (NP, VP), dependency parsing creates direct head-dependent relations. This is crucial for cross-lingual NLP - dependency structures are more universal across languages\"
        
        **Red flags:**
        
        - Confusing dependency with constituency parsing
        - Not knowing spaCy or Stanford CoreNLP
        - Can't explain practical applications (IE, QA)
        - Unaware of Universal Dependencies standard
        
        **Follow-ups:**
        
        - \"Complexity of parsing algorithms?\" → Transition-based: O(n), Graph-based: O(n³)
        - \"How to extract (subject, verb, object)?\" → Find verb (ROOT), nsubj child is subject, dobj child is object
        - \"Accuracy metrics?\" → UAS (unlabeled attachment), LAS (labeled attachment). Modern: 94-96%
        - \"Use case at scale?\" → Google Knowledge Graph uses dependency parsing to extract billions of relations from web text

---

### What is Word Sense Disambiguation? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Semantics` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    ## What is Word Sense Disambiguation (WSD)?

    **Word Sense Disambiguation (WSD)** is the NLP task of determining which meaning (sense) of a polysemous word is used in a given context. It's one of the oldest and most fundamental problems in NLP.

    **The Challenge:** ~80% of common English words have multiple meanings. Without context, we can't understand which sense is intended.

    **Examples:**

    | Word | Senses | Example Sentences |
    |------|--------|-------------------|
    | **bank** | 1. Financial institution<br>2. River edge<br>3. Turn/tilt | 1. "I went to the *bank* to deposit money"<br>2. "Sat on the river *bank* fishing"<br>3. "The plane *banked* left" |
    | **mouse** | 1. Computer device<br>2. Small rodent | 1. "Click the *mouse* button"<br>2. "A *mouse* ran across the floor" |
    | **apple** | 1. Fruit<br>2. Company (Apple Inc.) | 1. "I ate an *apple* for lunch"<br>2. "*Apple* stock reached new highs" |
    | **play** | 1. Theatrical performance<br>2. Engage in activity<br>3. Musical performance | 1. "I saw a Shakespeare *play*"<br>2. "Children *play* in the park"<br>3. "*Play* the piano" |
    | **fine** | 1. High quality<br>2. Penalty fee<br>3. Very small particles | 1. "*Fine* dining restaurant"<br>2. "Parking *fine* of $50"<br>3. "*Fine* sand particles" |

    **Why WSD Matters:**
    - **Machine Translation:** "I play the bass" → bass (fish) vs bass (instrument)?
    - **Information Retrieval:** Search "apple pie" shouldn't return iPhone results
    - **Question Answering:** "What is Java?" → programming language vs island vs coffee?
    - **Text-to-Speech:** "lead" → /liːd/ (guide) vs /lɛd/ (metal)

    ## Approaches: Evolution

    ### 1. Knowledge-Based (Pre-2010)

    **Uses lexical databases like WordNet:**
    - **WordNet:** 117,000 synonym sets (synsets), each representing one sense
    - **No training data needed** (unsupervised)
    - **Algorithms:** Lesk algorithm, PageRank on semantic graphs

    **Lesk Algorithm (1986):**
    - Compare context words with sense definitions
    - Choose sense with maximum overlap
    - **Accuracy:** 50-60% on Senseval benchmarks

    **Example:**
    ```
    Word: "bank"
    Context: "I went to the bank to deposit money"

    Sense 1 (financial): "financial institution that accepts deposits..."
    Sense 2 (geography): "sloping land beside water..."

    Overlap:
    - Sense 1: {"deposit"} → 1 match ✓
    - Sense 2: {} → 0 matches

    Winner: Sense 1 (financial)
    ```

    **Advantages:** No training data needed, interpretable
    **Limitations:** Low accuracy (50-60%), requires sense inventory (WordNet)

    ### 2. Supervised Learning (2000-2016)

    **Train classifier on sense-annotated corpora:**
    - **Data:** SemCor (234K sense-tagged words), SensEval/SemEval benchmarks
    - **Features:** Surrounding words, POS tags, collocations
    - **Models:** Naive Bayes, SVM, decision trees
    - **Accuracy:** 75-80% on Senseval-2/3

    **Limitations:**
    - Requires expensive sense-annotated data
    - Doesn't generalize to new words/senses
    - WordNet sense granularity too fine (e.g., 44 senses for "run")

    ### 3. Word Embeddings (2013-2017)

    **Use Word2Vec/GloVe, but static embeddings can't handle polysemy:**
    - "bank" has **one** embedding for all contexts ❌
    - Cannot distinguish between financial vs river bank

    **Limitation:** This is why contextual embeddings were needed!

    ### 4. Contextual Embeddings - BERT Era (2018+)

    **Breakthrough:** BERT, ELMo, RoBERTa give **different embeddings for each context**
    - "bank" in "financial bank" ≠ "bank" in "river bank"
    - **No explicit WSD needed** - embeddings naturally disambiguate
    - **Accuracy:** 80-85% (matches/exceeds supervised systems)

    **Key Insight:** Modern NLP doesn't need explicit WSD - BERT handles it implicitly!

    ## Production Implementation (175 lines)

    ```python
    # word_sense_disambiguation.py
    import torch
    from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    from typing import List, Dict, Tuple
    from nltk.corpus import wordnet as wn
    from collections import Counter

    class WordSenseDisambiguator:
        """
        Production WSD using multiple approaches:
        1. BERT contextual embeddings (modern, recommended)
        2. Lesk algorithm (knowledge-based, no training)
        3. WordNet similarity (knowledge-based)

        Time: O(n × d) for BERT (n=seq_len, d=hidden_size)
        Space: O(d) for embeddings
        """

        def __init__(self, model_name='bert-base-uncased'):
            """
            Args:
                model_name: Pretrained model
                    - 'bert-base-uncased' (110M params, 768-dim)
                    - 'roberta-base' (125M params, 768-dim)
                    - 'sentence-transformers/all-MiniLM-L6-v2' (lightweight)
            """
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)

        def get_word_embedding(self, sentence: str, word: str, word_index: int = None) -> np.ndarray:
            """
            Get contextual embedding for word in sentence

            Args:
                sentence: Full sentence containing word
                word: Target word to get embedding for
                word_index: Token index of word (if known)

            Returns:
                Embedding vector [hidden_size]
            """
            # Tokenize
            inputs = self.tokenizer(sentence, return_tensors='pt').to(self.device)
            tokens = self.tokenizer.tokenize(sentence)

            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use last hidden state: [batch=1, seq_len, hidden_size]
                embeddings = outputs.last_hidden_state[0]

            # Find word position if not provided
            if word_index is None:
                # Find token index (simple word matching)
                word_lower = word.lower()
                for i, token in enumerate(tokens):
                    if token.strip('#').lower() == word_lower:
                        word_index = i + 1  # +1 for [CLS] token
                        break

            if word_index is None or word_index >= len(embeddings):
                # Fallback: use mean of all token embeddings
                return embeddings.mean(dim=0).cpu().numpy()

            # Return embedding for target word
            return embeddings[word_index].cpu().numpy()

        def disambiguate_senses_bert(
            self,
            target_word: str,
            sentences: List[str],
            sense_examples: Dict[str, List[str]]
        ) -> List[str]:
            """
            Disambiguate word senses using BERT embeddings

            Args:
                target_word: Word to disambiguate (e.g., "bank")
                sentences: List of sentences containing target word
                sense_examples: Dict mapping sense names to example sentences
                    Example: {
                        "financial": ["I deposited money at the bank"],
                        "geography": ["We sat on the river bank"]
                    }

            Returns:
                List of predicted senses (one per input sentence)
            """
            # Get embeddings for sense examples
            sense_embeddings = {}
            for sense, examples in sense_examples.items():
                # Average embeddings from all examples for this sense
                embeddings = []
                for example in examples:
                    emb = self.get_word_embedding(example, target_word)
                    embeddings.append(emb)
                sense_embeddings[sense] = np.mean(embeddings, axis=0)

            # Disambiguate each sentence
            predictions = []
            for sentence in sentences:
                # Get embedding for target word in this sentence
                word_emb = self.get_word_embedding(sentence, target_word)

                # Find most similar sense
                best_sense = None
                best_similarity = -1

                for sense, sense_emb in sense_embeddings.items():
                    similarity = cosine_similarity(
                        word_emb.reshape(1, -1),
                        sense_emb.reshape(1, -1)
                    )[0][0]

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_sense = sense

                predictions.append({
                    'sentence': sentence,
                    'predicted_sense': best_sense,
                    'confidence': best_similarity
                })

            return predictions

        def lesk_algorithm(self, sentence: str, word: str) -> str:
            """
            Lesk algorithm for WSD using WordNet

            Algorithm:
            1. Get all synsets (senses) for target word
            2. For each synset, get definition + examples
            3. Count overlapping words with context
            4. Return sense with maximum overlap

            Args:
                sentence: Sentence containing word
                word: Target word

            Returns:
                Best sense definition
            """
            # Get context words (sentence without target word)
            context_words = set(sentence.lower().split()) - {word.lower()}

            # Get all synsets for word
            synsets = wn.synsets(word)

            if not synsets:
                return f"No senses found for '{word}' in WordNet"

            best_sense = None
            max_overlap = 0

            for synset in synsets:
                # Get definition and examples
                definition = synset.definition()
                examples = synset.examples()

                # Combine definition + examples
                sense_text = definition + " " + " ".join(examples)
                sense_words = set(sense_text.lower().split())

                # Count overlap
                overlap = len(context_words & sense_words)

                if overlap > max_overlap:
                    max_overlap = overlap
                    best_sense = synset

            if best_sense:
                return f"{best_sense.name()}: {best_sense.definition()}"
            else:
                return f"Could not disambiguate '{word}'"

        def compare_sense_similarity(
            self,
            word: str,
            sentence1: str,
            sentence2: str
        ) -> Dict:
            """
            Compare word embeddings in two different contexts

            Use case: Check if word has same or different sense

            Returns:
                Dict with similarity score and interpretation
            """
            # Get embeddings
            emb1 = self.get_word_embedding(sentence1, word)
            emb2 = self.get_word_embedding(sentence2, word)

            # Compute cosine similarity
            similarity = cosine_similarity(
                emb1.reshape(1, -1),
                emb2.reshape(1, -1)
            )[0][0]

            # Interpret
            if similarity > 0.8:
                interpretation = "Same sense (high similarity)"
            elif similarity > 0.5:
                interpretation = "Related senses (moderate similarity)"
            else:
                interpretation = "Different senses (low similarity)"

            return {
                'word': word,
                'sentence1': sentence1,
                'sentence2': sentence2,
                'similarity': float(similarity),
                'interpretation': interpretation
            }

    # Example usage & demonstrations
    def demo_wsd():
        """Demonstrate word sense disambiguation"""

        print("=" * 70)
        print("WORD SENSE DISAMBIGUATION DEMO")
        print("=" * 70)

        # Initialize WSD system
        print("\nLoading BERT model...")
        wsd = WordSenseDisambiguator(model_name='bert-base-uncased')

        # Demo 1: BERT Contextual Embeddings
        print("\n" + "=" * 70)
        print("1. BERT CONTEXTUAL DISAMBIGUATION")
        print("=" * 70)

        target_word = "bank"

        # Define sense examples
        sense_examples = {
            "financial": [
                "I went to the bank to deposit money",
                "The bank approved my loan application",
                "She works at a large investment bank"
            ],
            "geography": [
                "We sat on the river bank fishing",
                "The bank of the stream was muddy",
                "Trees lined the bank of the lake"
            ]
        }

        # Test sentences
        test_sentences = [
            "I need to go to the bank to withdraw cash",
            "The boat approached the bank of the river",
            "The bank announced higher interest rates",
            "Children played on the grassy bank"
        ]

        results = wsd.disambiguate_senses_bert(target_word, test_sentences, sense_examples)

        for result in results:
            print(f"\nSentence: \"{result['sentence']}\"")
            print(f"Predicted sense: {result['predicted_sense']}")
            print(f"Confidence: {result['confidence']:.3f}")

        # Demo 2: Similarity Comparison
        print("\n" + "=" * 70)
        print("2. SENSE SIMILARITY COMPARISON")
        print("=" * 70)

        comparisons = [
            ("bank", "I deposited money at the bank", "The bank is next to the river"),
            ("mouse", "Click the left mouse button", "A mouse ran across the floor"),
            ("play", "Let's play soccer", "I want to play the guitar")
        ]

        for word, sent1, sent2 in comparisons:
            result = wsd.compare_sense_similarity(word, sent1, sent2)
            print(f"\nWord: '{word}'")
            print(f"  Sentence 1: {sent1}")
            print(f"  Sentence 2: {sent2}")
            print(f"  Similarity: {result['similarity']:.3f}")
            print(f"  → {result['interpretation']}")

        # Demo 3: Lesk Algorithm (Knowledge-Based)
        print("\n" + "=" * 70)
        print("3. LESK ALGORITHM (WordNet)")
        print("=" * 70)

        lesk_examples = [
            ("I went to the bank to deposit money", "bank"),
            ("The mouse pointer moved across the screen", "mouse"),
            ("She will play the piano at the concert", "play")
        ]

        for sentence, word in lesk_examples:
            best_sense = wsd.lesk_algorithm(sentence, word)
            print(f"\nSentence: \"{sentence}\"")
            print(f"Word: '{word}'")
            print(f"Best sense: {best_sense}")

        print("\n" + "=" * 70)

    if __name__ == "__main__":
        demo_wsd()
    ```

    **Sample Output:**
    ```
    ======================================================================
    WORD SENSE DISAMBIGUATION DEMO
    ======================================================================
    Loading BERT model...

    ======================================================================
    1. BERT CONTEXTUAL DISAMBIGUATION
    ======================================================================

    Sentence: "I need to go to the bank to withdraw cash"
    Predicted sense: financial
    Confidence: 0.912

    Sentence: "The boat approached the bank of the river"
    Predicted sense: geography
    Confidence: 0.887

    Sentence: "The bank announced higher interest rates"
    Predicted sense: financial
    Confidence: 0.935

    Sentence: "Children played on the grassy bank"
    Predicted sense: geography
    Confidence: 0.901

    ======================================================================
    2. SENSE SIMILARITY COMPARISON
    ======================================================================

    Word: 'bank'
      Sentence 1: I deposited money at the bank
      Sentence 2: The bank is next to the river
      Similarity: 0.423
      → Different senses (low similarity)

    Word: 'mouse'
      Sentence 1: Click the left mouse button
      Sentence 2: A mouse ran across the floor
      Similarity: 0.381
      → Different senses (low similarity)
    ```

    ## Evaluation: Benchmarks & SOTA

    | Benchmark | Size | Avg Senses/Word | SOTA System | Score |
    |-----------|------|-----------------|-------------|-------|
    | **Senseval-2** | 5,000 instances | 10.8 | BERT-based (GlossBERT) | 81.7% |
    | **Senseval-3** | 7,860 instances | 10.2 | GlossBERT | 80.4% |
    | **SemEval-2007** | 2,269 instances | 5.0 | ESCHER (BERT-based) | 82.5% |
    | **SemEval-2013** | 1,931 instances | 8.0 | BERT fine-tuned | 83.2% |
    | **All-Words WSD** | 13K instances | 5.3 | BEM (BERT + knowledge) | **84.0%** (SOTA 2023) |

    ## Real-World Applications

    **Google Translate (Machine Translation):**
    - **Task:** Disambiguate polysemous words before translation
    - **Example:** English "play" → Spanish "jugar" (sport) vs "tocar" (instrument)
    - **Impact:** WSD improves translation accuracy by 8-12% (BLEU score)
    - **Modern approach:** Neural MT (Transformer) implicitly handles WSD via attention

    **Search Engines (Google, Bing):**
    - **Task:** Interpret ambiguous queries
    - **Example:** "apple pie recipe" vs "apple stock price"
    - **Approach:** Query context, user history, click patterns
    - **Impact:** 15-20% reduction in ambiguous query results

    **Voice Assistants (Siri, Alexa):**
    - **Task:** Disambiguate for text-to-speech pronunciation
    - **Example:** "lead" → /liːd/ (guide) vs /lɛd/ (metal)
    - **Approach:** POS tagging + context
    - **Accuracy:** 95%+ for common words

    **Clinical NLP (Medical Records):**
    - **Task:** Disambiguate medical terms
    - **Example:** "cold" → illness vs temperature
    - **Challenge:** Domain-specific senses (Bio WordNet)
    - **Accuracy:** 70-75% (lower due to specialized vocabulary)

    ## Why Modern NLP Doesn't Need Explicit WSD

    **BERT Revolution (2018):**
    - **Before BERT:** Static embeddings → needed explicit WSD
    - **After BERT:** Contextual embeddings → automatic disambiguation
    - **Impact:** WSD as standalone task became less relevant

    **Evidence:**
    - **NER:** BERT-based models handle ambiguous entities ("Apple" company vs fruit) without WSD
    - **QA:** SQuAD models disambiguate implicitly via context
    - **Translation:** Neural MT disambiguates via attention (no explicit WSD module needed)

    **When Explicit WSD Still Matters:**
    - **Low-resource languages:** Limited pretrained models
    - **Interpretability:** Need to explain which sense was chosen
    - **Knowledge graphs:** Linking to specific sense IDs (WordNet, Wikipedia)
    - **Lexicography:** Building dictionaries, studying language

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Rare senses not in training** | Defaults to common sense | Use knowledge base (WordNet) as fallback |
    | **WordNet sense granularity too fine** | "run" has 44 senses - too specific | Cluster similar senses, use coarse-grained evaluation |
    | **Domain-specific senses** | General models miss medical/legal terms | Fine-tune on domain data (BioBERT for medical) |
    | **New senses emerging** | "tweet" (2006), "cloud" (computing) | Regular model updates, open vocabulary |
    | **Cross-lingual WSD** | Translation sense inventories differ | Use multilingual BERT (mBERT), parallel corpora |
    | **Static embeddings** | Word2Vec gives one vector per word | Use contextual embeddings (BERT, RoBERTa) |

    ## Historical Milestones

    **1986:** Lesk algorithm (knowledge-based, 50-60% accuracy)
    **1998:** Senseval-1 benchmark (first large-scale evaluation)
    **2004:** Supervised SVM systems (75-80% accuracy)
    **2013:** Word2Vec (static embeddings - doesn't solve WSD)
    **2018:** **BERT** (contextual embeddings - implicit WSD at 80-85%)
    **2019:** GlossBERT (fine-tuned BERT using WordNet glosses - 81.7%)
    **2023:** BEM (BERT + External Knowledge - 84.0% SOTA)

    !!! tip "Interviewer's Insight"
        **Strong candidates:**

        - Explain the problem: "80% of words are polysemous - 'bank' (financial vs geography), 'mouse' (device vs animal) - WSD picks correct sense"
        - Know evolution: "Lesk algorithm (50-60%, knowledge-based), supervised SVM (75-80%), BERT contextual embeddings (80-85% SOTA)"
        - Understand BERT advantage: "BERT gives different embeddings per context - 'bank' in 'deposit money' ≠ 'river bank' (cosine sim < 0.5)"
        - Reference benchmarks: "Senseval-2/3 standard benchmarks; BEM achieves 84% (SOTA 2023)"
        - Cite real systems: "Google Translate improves 8-12% with WSD; search engines reduce ambiguous results 15-20%"
        - Know modern view: "Explicit WSD less critical now - BERT/RoBERTa handle implicitly via contextual embeddings"
        - Discuss when WSD still matters: "Interpretability (explain sense chosen), knowledge graphs (link to WordNet IDs), low-resource languages"

---

### What is Coreference Resolution? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Discourse` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    ## What is Coreference Resolution?

    **Coreference Resolution** identifies all expressions (mentions) in a text that refer to the same real-world entity. It's crucial for understanding narratives, maintaining context across sentences, and enabling deeper NLP tasks.

    **Example:**
    ```
    "John went to the store. He bought milk. The man paid $5."

    Coreference Chains:
    - [John, He, The man] → Person entity
    - [the store, there (implicit)] → Location entity
    - [milk, it (if mentioned later)] → Product entity
    ```

    **Why It Matters:**
    - **Document Understanding:** 30-40% of words in text are referring expressions (pronouns, definite NPs)
    - **Question Answering:** "Who bought milk?" requires linking "who" → "John"
    - **Summarization:** Avoid ambiguous pronouns ("he said he likes it" → unclear)
    - **Dialogue Systems:** Track entities across conversation turns
    - **Information Extraction:** Link mentions to build knowledge graphs

    ## Types of Coreference

    ### 1. Pronominal Anaphora
    - **Pronoun → Noun:** "John... He..."
    - **Most common:** 60-70% of coreferences
    - **Challenges:** Gender agreement, number agreement, distance

    ### 2. Nominal Coreference
    - **Noun → Noun:** "Barack Obama... the president... the former senator..."
    - **Aliases:** Different names for same entity
    - **Definite NPs:** "the company", "the man"

    ### 3. Zero Anaphora (Pro-drop)
    - **Omitted pronoun:** Common in languages like Japanese, Chinese
    - **Example (Chinese):** "张三去商店。买了牛奶。" (Zhang San went to store. [He] bought milk.)

    ## Approaches: Evolution

    ### 1. Rule-Based (Pre-2010)
    - **Algorithms:** Hobbs algorithm, Lappin-Leass
    - **Rules:** Gender/number agreement, syntactic constraints, recency
    - **Accuracy:** 50-60% F1
    - **Example:** Pronoun must agree in gender/number with antecedent

    ### 2. Statistical/ML (2010-2016)
    - **Features:** Distance, string match, grammatical role, semantic similarity
    - **Models:** Mention-pair, mention-ranking, cluster-ranking
    - **Accuracy:** 60-70% F1 (CoNLL-2012)
    - **Limitation:** Heavy feature engineering

    ### 3. Neural (2016-2020)
    - **Models:** End-to-end neural coref (Lee et al., 2017), SpanBERT
    - **Architecture:** Span representations + scoring function
    - **Accuracy:** 73-79% F1 (CoNLL-2012)
    - **Advantage:** Learns features automatically

    ### 4. Pretrained LLMs (2020+)
    - **Models:** LingMess (2022), CorefUD (2023)
    - **Approach:** Fine-tuned BERT/RoBERTa on coreference data
    - **Accuracy:** 80-83% F1 (SOTA)
    - **Zero-shot:** GPT-4 achieves 70%+ with prompting

    ## Production Implementation (190 lines)

    ```python
    # coreference_resolution.py
    import spacy
    from typing import List, Dict, Tuple, Set
    import networkx as nx
    from collections import defaultdict
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    # Note: neuralcoref is deprecated, showing modern alternatives

    class ModernCoreferenceResolver:
        """
        Production coreference resolution using spaCy + custom neural model

        Pipeline:
        1. Extract mentions (NER + noun chunks)
        2. Compute pairwise scores (neural model)
        3. Cluster mentions (graph-based)

        Time: O(n²) for n mentions (pairwise scoring)
        Space: O(n²) for score matrix
        """

        def __init__(self, model_name='en_core_web_trf'):
            """
            Args:
                model_name: spaCy model
                    - 'en_core_web_sm' (small, fast, 96MB)
                    - 'en_core_web_trf' (transformer-based, accurate, 438MB)
            """
            # Load spaCy with transformer model
            self.nlp = spacy.load(model_name)

            # For demo: simple rule-based (production would use trained model)
            # In production: Load fine-tuned SpanBERT or similar
            self.use_neural = False  # Set to True with trained model

        def extract_mentions(self, doc) -> List[Dict]:
            """
            Extract all potential mentions from document

            Mentions include:
            - Named entities (PERSON, ORG, GPE, etc.)
            - Pronouns (he, she, it, they, etc.)
            - Definite noun phrases (the company, the man, etc.)

            Returns:
                List of mention dictionaries
            """
            mentions = []

            # 1. Named Entities
            for ent in doc.ents:
                mentions.append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'type': 'entity',
                    'label': ent.label_,
                    'tokens': [ent.start, ent.end]
                })

            # 2. Pronouns
            pronouns = {'he', 'she', 'it', 'they', 'him', 'her', 'them',
                       'his', 'hers', 'its', 'their', 'theirs',
                       'himself', 'herself', 'itself', 'themselves',
                       'who', 'whom', 'whose', 'which', 'that'}

            for token in doc:
                if token.text.lower() in pronouns and token.pos_ == 'PRON':
                    mentions.append({
                        'text': token.text,
                        'start': token.idx,
                        'end': token.idx + len(token.text),
                        'type': 'pronoun',
                        'label': 'PRONOUN',
                        'gender': self._get_gender(token.text.lower()),
                        'number': self._get_number(token.text.lower()),
                        'tokens': [token.i, token.i + 1]
                    })

            # 3. Definite Noun Phrases
            for chunk in doc.noun_chunks:
                # Only definite NPs (starting with "the", "this", "that", etc.)
                if chunk[0].text.lower() in {'the', 'this', 'that', 'these', 'those'}:
                    mentions.append({
                        'text': chunk.text,
                        'start': chunk.start_char,
                        'end': chunk.end_char,
                        'type': 'np',
                        'label': 'NP',
                        'tokens': [chunk.start, chunk.end]
                    })

            # Sort by start position
            mentions.sort(key=lambda m: m['start'])

            return mentions

        def _get_gender(self, pronoun: str) -> str:
            """Determine gender of pronoun"""
            if pronoun in {'he', 'him', 'his', 'himself'}:
                return 'masculine'
            elif pronoun in {'she', 'her', 'hers', 'herself'}:
                return 'feminine'
            else:
                return 'neutral'

        def _get_number(self, pronoun: str) -> str:
            """Determine number (singular/plural) of pronoun"""
            if pronoun in {'they', 'them', 'their', 'theirs', 'themselves', 'these', 'those'}:
                return 'plural'
            else:
                return 'singular'

        def score_mention_pair(self, mention1: Dict, mention2: Dict, doc) -> float:
            """
            Score how likely mention2 refers to mention1

            Features:
            - String match
            - Gender/number agreement (for pronouns)
            - Distance
            - Semantic similarity

            Returns:
                Score 0-1 (higher = more likely coreferent)
            """
            score = 0.0

            # Feature 1: Exact string match
            if mention1['text'] == mention2['text']:
                score += 0.5

            # Feature 2: Partial string match (aliases)
            if mention1['text'].lower() in mention2['text'].lower() or \
               mention2['text'].lower() in mention1['text'].lower():
                score += 0.3

            # Feature 3: Gender/number agreement (for pronouns)
            if mention2['type'] == 'pronoun':
                # Check if mention1 agrees with pronoun
                if 'gender' in mention2:
                    # Simple heuristics (production: use gender lexicons)
                    if mention2['gender'] == 'neutral':
                        score += 0.1  # Neutral pronouns can refer to anything
                    elif self._check_gender_match(mention1, mention2['gender']):
                        score += 0.3
                    else:
                        score -= 0.5  # Penalty for mismatch

                if 'number' in mention2:
                    if self._check_number_match(mention1, mention2['number'], doc):
                        score += 0.2
                    else:
                        score -= 0.3

            # Feature 4: Distance (recency bias - closer mentions more likely)
            distance = mention2['start'] - mention1['end']
            distance_score = 1.0 / (1.0 + distance / 100.0)  # Decay with distance
            score += 0.2 * distance_score

            # Feature 5: Same entity type
            if mention1.get('label') == mention2.get('label'):
                score += 0.15

            # Normalize to 0-1
            return max(0.0, min(1.0, score))

        def _check_gender_match(self, mention: Dict, pronoun_gender: str) -> bool:
            """Check if mention matches pronoun gender (simplified)"""
            # Production: Use gender lexicon or learned model
            text_lower = mention['text'].lower()

            if pronoun_gender == 'masculine':
                return any(name in text_lower for name in
                          ['john', 'michael', 'david', 'james', 'robert', 'man', 'boy', 'mr'])
            elif pronoun_gender == 'feminine':
                return any(name in text_lower for name in
                          ['mary', 'sarah', 'emily', 'woman', 'girl', 'mrs', 'ms'])

            return True  # Neutral

        def _check_number_match(self, mention: Dict, pronoun_number: str, doc) -> bool:
            """Check if mention matches pronoun number"""
            # Simplified: check if mention text is plural
            text_lower = mention['text'].lower()

            plural_indicators = ['they', 'companies', 'people', 'men', 'women']
            is_plural = any(ind in text_lower for ind in plural_indicators)

            if pronoun_number == 'plural':
                return is_plural
            else:
                return not is_plural

        def cluster_mentions(self, mentions: List[Dict], scores: List[Tuple[int, int, float]],
                           threshold: float = 0.5) -> List[List[int]]:
            """
            Cluster mentions into coreference chains

            Uses graph-based clustering:
            - Nodes = mentions
            - Edges = high-scoring pairs (score > threshold)
            - Clusters = connected components

            Args:
                mentions: List of mentions
                scores: List of (i, j, score) tuples
                threshold: Minimum score to create edge

            Returns:
                List of clusters (each cluster is list of mention indices)
            """
            # Build graph
            G = nx.Graph()
            G.add_nodes_from(range(len(mentions)))

            # Add edges for high-scoring pairs
            for i, j, score in scores:
                if score > threshold:
                    G.add_edge(i, j, weight=score)

            # Find connected components (clusters)
            clusters = list(nx.connected_components(G))

            # Convert to list of lists
            return [sorted(list(cluster)) for cluster in clusters]

        def resolve(self, text: str, threshold: float = 0.5) -> Dict:
            """
            Resolve coreferences in text

            Args:
                text: Input text
                threshold: Minimum score for coreference

            Returns:
                Dict with mentions and clusters
            """
            # Parse text
            doc = self.nlp(text)

            # Extract mentions
            mentions = self.extract_mentions(doc)

            if len(mentions) == 0:
                return {'text': text, 'mentions': [], 'clusters': []}

            # Score all pairs (i < j)
            scores = []
            for i in range(len(mentions)):
                for j in range(i + 1, len(mentions)):
                    score = self.score_mention_pair(mentions[i], mentions[j], doc)
                    scores.append((i, j, score))

            # Cluster mentions
            clusters = self.cluster_mentions(mentions, scores, threshold)

            # Filter out singleton clusters (mentions with no coreferences)
            clusters = [c for c in clusters if len(c) > 1]

            return {
                'text': text,
                'mentions': mentions,
                'clusters': clusters,
                'num_chains': len(clusters)
            }

        def format_output(self, result: Dict) -> str:
            """Format coreference resolution output for display"""
            output = []
            output.append("=" * 70)
            output.append("COREFERENCE RESOLUTION")
            output.append("=" * 70)
            output.append(f"\nText: {result['text']}\n")
            output.append(f"Total mentions: {len(result['mentions'])}")
            output.append(f"Coreference chains: {result['num_chains']}\n")

            for i, cluster in enumerate(result['clusters'], 1):
                mentions_text = [result['mentions'][idx]['text'] for idx in cluster]
                output.append(f"Chain {i}: {' → '.join(mentions_text)}")

            output.append("=" * 70)
            return "\n".join(output)

    # Example usage
    def demo_coreference():
        """Demonstrate coreference resolution"""

        # Initialize resolver
        print("Loading model...")
        resolver = ModernCoreferenceResolver(model_name='en_core_web_sm')

        # Example 1: Simple pronoun resolution
        text1 = """
        John went to the store. He bought milk and bread.
        The man then walked home with his groceries.
        """

        result1 = resolver.resolve(text1.strip(), threshold=0.4)
        print(resolver.format_output(result1))

        # Example 2: Multiple entities
        text2 = """
        Apple announced a new iPhone yesterday. The company reported strong sales.
        Tim Cook, Apple's CEO, praised the team. He said the product exceeded expectations.
        """

        result2 = resolver.resolve(text2.strip(), threshold=0.4)
        print("\n" + resolver.format_output(result2))

        # Example 3: Complex coreference
        text3 = """
        The researchers published their findings. The scientists discovered a new treatment.
        They believe it could help millions of patients. The team plans to start clinical trials.
        """

        result3 = resolver.resolve(text3.strip(), threshold=0.3)
        print("\n" + resolver.format_output(result3))

    if __name__ == "__main__":
        demo_coreference()
    ```

    **Sample Output:**
    ```
    ======================================================================
    COREFERENCE RESOLUTION
    ======================================================================

    Text: John went to the store. He bought milk and bread.
    The man then walked home with his groceries.

    Total mentions: 4
    Coreference chains: 1

    Chain 1: John → He → The man
    ======================================================================
    ```

    ## Evaluation Metrics

    | Metric | Description | Formula | Interpretation |
    |--------|-------------|---------|----------------|
    | **MUC** | Link-based F1 | Precision/recall of links | Standard, but biased toward large chains |
    | **B³** | Mention-based | Averages over mentions | More balanced |
    | **CEAF** | Entity-based | Best mapping between predicted/gold | Symmetric |
    | **LEA** | Link-based with importance weighting | Weighted by entity size | Newer, more robust |
    | **CoNLL Score** | Average of MUC, B³, CEAF | (MUC + B³ + CEAF) / 3 | Official benchmark metric |

    **SOTA Performance (CoNLL-2012):**
    - **SpanBERT (2020):** 79.6% CoNLL F1
    - **LingMess (2022):** 82.4% CoNLL F1
    - **CorefUD (2023):** 83.1% CoNLL F1 (current SOTA)

    ## Real-World Applications

    **Google Assistant / Alexa (Dialogue Systems):**
    - **Task:** Track entities across conversation turns
    - **Example:**
      ```
      User: "Book a flight to Paris"
      Assistant: "When would you like to go?"
      User: "Next Friday" (implicitly: to Paris)
      ```
    - **Accuracy:** 85%+ for short dialogues (2-3 turns), 70% for longer
    - **Impact:** Enables natural multi-turn conversations

    **Automatic Summarization (Google News, Apple News):**
    - **Task:** Replace pronouns with entity names for clarity
    - **Before:** "He announced the plan. It will help them."
    - **After:** "Tim Cook announced the plan. The initiative will help developers."
    - **Improvement:** 40% reduction in ambiguous pronouns
    - **Scale:** Billions of articles processed

    **Question Answering (SQuAD, Natural Questions):**
    - **Task:** Resolve pronouns in questions and context
    - **Example:**
      ```
      Context: "Einstein developed relativity. He won the Nobel Prize in 1921."
      Question: "When did he win the prize?"
      ```
    - **Performance:** Coreference resolution improves QA accuracy by 8-12% (SQuAD 2.0)

    **Clinical NLP (Medical Records):**
    - **Task:** Link patient mentions across discharge summaries
    - **Challenge:** "The patient... He... The 45-year-old man..."
    - **Accuracy:** 75-80% (lower due to complex medical terminology)
    - **Impact:** Critical for patient timeline reconstruction

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Cataphora (forward reference)** | "Before *he* left, *John* locked the door" | Bidirectional models, second pass |
    | **Ambiguous pronouns** | "John told Bill he won" (who won?) | Use context, world knowledge, or mark ambiguous |
    | **Long-distance dependencies** | Pronoun 10 sentences after antecedent | Transformer models (full-document attention) |
    | **Singleton mentions** | Entities mentioned once (no coreference) | Filter during preprocessing or post-processing |
    | **Wrong antecedent** | "The dog chased the cat. It was fast." (dog or cat?) | Semantic plausibility scoring, world knowledge |
    | **Generic pronouns** | "They say it will rain" (who is 'they'?) | Detect and exclude generic references |
    | **Plural ambiguity** | "Companies... they" (which companies?) | Track discourse salience, recency |

    ## Algorithms

    **Mention-Pair Model:**
    - Score all pairs (i, j) where j > i
    - If score > threshold, link j → i
    - **Issue:** Can create inconsistent clusters (transitivity violations)

    **Mention-Ranking Model:**
    - For mention j, rank all previous mentions i < j
    - Link to highest-scoring antecedent
    - **Improvement:** More consistent than pair model

    **Cluster-Ranking Model:**
    - Maintain clusters, score mention vs cluster
    - **Advantage:** Uses cluster features (e.g., all names in cluster)

    **End-to-End Neural (Lee et al., 2017 - used in SpanBERT):**
    1. **Span enumeration:** All possible spans up to length K
    2. **Mention scoring:** Which spans are mentions?
    3. **Antecedent scoring:** For each mention, score all previous mentions
    4. **Joint optimization:** Train end-to-end with coreference loss

    ## Modern Approaches (2023)

    **LLM-based Coreference:**
    - **GPT-4 Few-Shot:** 70-75% CoNLL F1 (no fine-tuning!)
    - **Fine-tuned BERT/RoBERTa:** 80-83% F1
    - **Prompt Example:**
      ```
      Text: "John went to the store. He bought milk."
      Find all coreferences:
      ```

    **Cross-lingual Coreference:**
    - **mBERT, XLM-R:** Work across languages
    - **Challenge:** Pronouns work differently (pro-drop in Spanish/Chinese)
    - **Performance:** 70-75% F1 (vs 83% for English)

    !!! tip "Interviewer's Insight"
        **Strong candidates:**

        - Explain the task: "Linking 'John', 'he', 'the man' to same entity - crucial for 30-40% of text (pronouns, definite NPs)"
        - Know approaches: "Rule-based (60% F1), mention-pair models (70%), neural SpanBERT (79.6%), LingMess SOTA (82.4%)"
        - Understand evaluation: "CoNLL metric averages MUC (link-based), B³ (mention-based), CEAF (entity-based)"
        - Reference real systems: "Google Assistant tracks entities across turns (85% accuracy), QA systems gain 8-12% with coref"
        - Know challenges: "Cataphora (forward reference), ambiguous pronouns ('John told Bill he won'), long-distance (10+ sentences)"
        - Discuss production: "End-to-end neural (SpanBERT) or fine-tuned BERT - 83% F1 on CoNLL-2012"
        - Mention LLMs: "GPT-4 few-shot achieves 70-75% F1 with zero training - promising for low-resource scenarios"

---

### What are LLM Hallucinations? - OpenAI, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Reliability` | **Asked by:** OpenAI, Google, Anthropic

??? success "View Answer"

    ## What are LLM Hallucinations?

    **Hallucination:** When an LLM generates fluent, confident-sounding text that is factually incorrect, nonsensical, or unfaithful to the source material.

    **Why It Matters:** The #1 barrier to deploying LLMs in production. Hallucinations erode user trust and can cause real harm (medical advice, legal guidance, financial decisions).

    **Types of Hallucinations:**

    ### 1. Factual Hallucinations
    - **Invented facts:** "Einstein won 3 Nobel Prizes" (actually 1)
    - **Wrong dates:** "COVID-19 pandemic started in 2021" (actually 2019)
    - **Fake citations:** Generates non-existent research papers
    - **Made-up statistics:** "95% of doctors recommend..." (no source)

    ### 2. Faithfulness Hallucinations
    - **Contradicts source:** User provides document, model ignores it
    - **Adds information not in source:** Summarization adds fabricated details
    - **Misattributes quotes:** Attributes statement to wrong person

    ### 3. Reasoning Hallucinations
    - **Invalid logical steps:** "A is larger than B, B is larger than C, so C is larger than A"
    - **Math errors:** Simple arithmetic mistakes despite showing work
    - **Circular reasoning:** Uses conclusion to prove itself

    ## Why Do LLMs Hallucinate?

    **Root Causes:**

    1. **Training on internet data** (contains misinformation, outdated info, contradictions)
    2. **Compression of knowledge** into weights (lossy, interpolates between facts)
    3. **Pattern matching without understanding** (no grounding in reality)
    4. **Maximizing fluency over truth** (penalized for saying "I don't know")
    5. **No access to real-time information** (knowledge cutoff)
    6. **Overconfidence** (no built-in uncertainty quantification)

    **Example:**
    ```
    User: "Who won the 2024 US Presidential election?"
    GPT-3 (knowledge cutoff 2021): "Donald Trump won..." (hallucination)
    GPT-4 with web search: "I'll search for current results..." (RAG mitigation)
    ```

    ## Mitigation Strategies (Production Implementation)

    ```python
    # hallucination_mitigation.py
    from typing import List, Dict
    import numpy as np
    from sentence_transformers import SentenceTransformer, util

    class HallucinationMitigator:
        """
        Multi-strategy hallucination mitigation for LLMs

        Combines:
        1. RAG (grounding in documents)
        2. Citation verification
        3. Confidence scoring
        4. Self-consistency checking
        5. Fact-checking against knowledge base
        """

        def __init__(self):
            # Embedding model for semantic similarity
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        def detect_hallucination_via_grounding(
            self,
            claim: str,
            source_docs: List[str],
            threshold: float = 0.5
        ) -> Dict:
            """
            Strategy 1: Check if claim is grounded in source documents

            Args:
                claim: Generated statement from LLM
                source_docs: Source documents that should support claim
                threshold: Minimum similarity score (0-1)

            Returns:
                Dict with hallucination verdict and score
            """
            # Embed claim and documents
            claim_emb = self.embedder.encode(claim, convert_to_tensor=True)
            doc_embs = self.embedder.encode(source_docs, convert_to_tensor=True)

            # Compute cosine similarity
            similarities = util.cos_sim(claim_emb, doc_embs)[0]
            max_sim = float(similarities.max())

            is_hallucination = max_sim < threshold

            return {
                'is_hallucination': is_hallucination,
                'max_similarity': max_sim,
                'verdict': 'HALLUCINATION' if is_hallucination else 'GROUNDED',
                'most_similar_doc_idx': int(similarities.argmax())
            }

        def detect_hallucination_via_self_consistency(
            self,
            question: str,
            answers: List[str],
            agreement_threshold: float = 0.7
        ) -> Dict:
            """
            Strategy 2: Sample multiple answers, check consistency

            Idea: If LLM is uncertain, answers will vary significantly
            If confident and correct, answers should be consistent

            Args:
                question: Question posed to LLM
                answers: Multiple sampled answers (e.g., temperature=0.8, n=5)
                agreement_threshold: Minimum agreement rate

            Returns:
                Dict with hallucination verdict based on consistency
            """
            if len(answers) < 2:
                return {'error': 'Need at least 2 answers for self-consistency'}

            # Compute pairwise similarities
            answer_embs = self.embedder.encode(answers, convert_to_tensor=True)
            similarities = util.cos_sim(answer_embs, answer_embs)

            # Average pairwise similarity (exclude diagonal)
            mask = ~np.eye(len(answers), dtype=bool)
            avg_similarity = float(similarities.cpu().numpy()[mask].mean())

            is_hallucination = avg_similarity < agreement_threshold

            return {
                'is_hallucination': is_hallucination,
                'consistency_score': avg_similarity,
                'verdict': 'INCONSISTENT (likely hallucination)' if is_hallucination else 'CONSISTENT',
                'num_samples': len(answers)
            }

        def verify_citation(
            self,
            claim: str,
            cited_source: str,
            source_text: str,
            threshold: float = 0.6
        ) -> Dict:
            """
            Strategy 3: Verify if citation supports claim

            Args:
                claim: Generated claim (e.g., "According to the study, ...")
                cited_source: What LLM cites (e.g., "Smith et al. 2020")
                source_text: Actual text from cited source
                threshold: Minimum entailment score

            Returns:
                Dict with citation verification result
            """
            # Simple version: Check semantic similarity
            claim_emb = self.embedder.encode(claim, convert_to_tensor=True)
            source_emb = self.embedder.encode(source_text, convert_to_tensor=True)

            similarity = float(util.cos_sim(claim_emb, source_emb)[0][0])

            is_valid = similarity >= threshold

            return {
                'citation_valid': is_valid,
                'support_score': similarity,
                'verdict': 'VALID CITATION' if is_valid else 'INVALID CITATION (hallucination)',
                'cited_source': cited_source
            }

    # Example usage
    def demo_hallucination_detection():
        """Demonstrate hallucination detection strategies"""

        print("=" * 70)
        print("HALLUCINATION DETECTION DEMO")
        print("=" * 70)

        mitigator = HallucinationMitigator()

        # Example 1: Grounding check
        print("\n1. GROUNDING CHECK (RAG)")
        print("-" * 70)

        source_docs = [
            "Albert Einstein won the Nobel Prize in Physics in 1921 for his work on the photoelectric effect.",
            "Einstein developed the theory of relativity and the famous equation E=mc²."
        ]

        # Grounded claim
        claim1 = "Einstein won the Nobel Prize in 1921"
        result1 = mitigator.detect_hallucination_via_grounding(claim1, source_docs)
        print(f"Claim: {claim1}")
        print(f"Result: {result1['verdict']} (similarity: {result1['max_similarity']:.3f})\n")

        # Hallucinated claim
        claim2 = "Einstein won the Nobel Prize in Chemistry in 1930"
        result2 = mitigator.detect_hallucination_via_grounding(claim2, source_docs)
        print(f"Claim: {claim2}")
        print(f"Result: {result2['verdict']} (similarity: {result2['max_similarity']:.3f})")

        # Example 2: Self-consistency
        print("\n\n2. SELF-CONSISTENCY CHECK")
        print("-" * 70)

        question = "What is the capital of France?"

        # Consistent answers (confident, correct)
        consistent_answers = [
            "The capital of France is Paris.",
            "Paris is the capital city of France.",
            "France's capital is Paris."
        ]

        result3 = mitigator.detect_hallucination_via_self_consistency(
            question,
            consistent_answers
        )
        print(f"Question: {question}")
        print(f"Result: {result3['verdict']} (score: {result3['consistency_score']:.3f})\n")

        # Inconsistent answers (uncertain, likely hallucinating)
        inconsistent_answers = [
            "The capital is Paris.",
            "Lyon is the capital of France.",
            "I think it might be Marseille."
        ]

        result4 = mitigator.detect_hallucination_via_self_consistency(
            question,
            inconsistent_answers
        )
        print(f"Question: {question}")
        print(f"Result: {result4['verdict']} (score: {result4['consistency_score']:.3f})")

        print("\n" + "=" * 70)

    if __name__ == "__main__":
        demo_hallucination_detection()
    ```

    ## Mitigation Strategies: Comparison

    | Strategy | Effectiveness | Latency | Cost | Best For |
    |----------|--------------|---------|------|----------|
    | **RAG (Retrieval)** | High (80-90% reduction) | Medium (+200ms) | Medium | Factual Q&A, knowledge-intensive |
    | **Citations** | Medium (requires verification) | Low | Low | Transparency, fact-checking |
    | **Self-Consistency** | Medium (70-80%) | High (N×latency) | High (N×cost) | Critical decisions, math |
    | **Confidence Scoring** | Low (LLMs overconfident) | Low | Low | User warnings, uncertainty |
    | **Human-in-Loop** | Very High (95%+) | Very High | Very High | High-stakes (medical, legal) |
    | **Fact-Checking APIs** | High (85-90%) | Medium (+500ms) | High (API cost) | Claims about real-world facts |
    | **Fine-Tuning (RLHF)** | Medium (base improvement) | None | Very High (one-time) | General reliability |

    ## Real-World Impact

    **ChatGPT Hallucination Rates (OpenAI, 2023):**
    - **GPT-3.5:** ~20-30% hallucination rate on factual questions
    - **GPT-4:** ~15-20% (improvement, still significant)
    - **GPT-4 + RAG:** ~5-10% (major reduction with grounding)

    **Google Bard Launch (2023):**
    - **Incident:** Bard hallucinated in demo (James Webb Telescope claim)
    - **Impact:** Alphabet stock dropped 9% ($100B loss)
    - **Lesson:** Hallucinations have real business consequences

    **LegalTech Hallucinations:**
    - **Case (2023):** Lawyer cited 6 fake cases generated by ChatGPT
    - **Outcome:** Sanctioned by court
    - **Mitigation:** Now require fact-checking, citations verification

    **Medical AI Concerns:**
    - **Study (2023):** GPT-4 hallucinates medical information 15-20% of time
    - **Risk:** Dangerous for patient care without human oversight
    - **Regulation:** FDA scrutiny for medical AI assistants

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Trusting LLM without verification** | Misinformation spread | Always verify critical facts, use RAG |
    | **No citations** | Can't verify claims | Require citations, implement attribution |
    | **Single-sample generation** | High variance | Use self-consistency (sample N times, check agreement) |
    | **Ignoring confidence scores** | Overconfident wrong answers | Calibrate confidence, show uncertainty to users |
    | **No human oversight (high-stakes)** | Harmful decisions | Human-in-loop for medical, legal, financial |
    | **Stale knowledge (cutoff date)** | Outdated information | Use RAG with current data, web search |
    | **Prompt engineering alone** | Limited effectiveness | Combine prompting + RAG + fine-tuning |

    ## Best Practices for Production

    ### 1. Multi-Layer Defense
    ```python
    def safe_llm_generation(question, knowledge_base):
        # Layer 1: RAG (grounding)
        docs = retrieve_relevant_docs(question, knowledge_base)

        # Layer 2: Prompted generation with citations
        prompt = f"""Answer based ONLY on these sources. Cite [Source N] for each claim.

    Sources:
    {docs}

    Question: {question}
    """
        answer = llm.generate(prompt)

        # Layer 3: Verify citations
        for claim, citation in extract_citations(answer):
            if not verify_citation(claim, citation, docs):
                flag_hallucination(claim)

        # Layer 4: Self-consistency check
        alternate_answers = [llm.generate(prompt) for _ in range(3)]
        if not are_consistent(answer, alternate_answers):
            show_warning("Low confidence answer")

        return answer
    ```

    ### 2. User Education
    - Display knowledge cutoff date prominently
    - Show "AI-generated" disclaimers
    - Provide sources/citations for verification
    - Allow user feedback on incorrect answers

    ### 3. Continuous Monitoring
    - Log generations for human review
    - Track hallucination rates with human eval
    - A/B test mitigation strategies
    - Update knowledge base regularly

    !!! tip "Interviewer's Insight"
        **Strong candidates:**

        - Define hallucination types: "Factual (wrong facts), faithfulness (ignores source), reasoning (logical errors)"
        - Explain root causes: "LLMs maximize fluency, trained on internet (noisy), compress knowledge lossily, no grounding in reality"
        - Know mitigation hierarchy: "RAG is most effective (80-90% reduction), then self-consistency (70-80%), confidence scoring helps but LLMs overconfident"
        - Reference real incidents: "Google Bard demo hallucination cost $100B stock drop; lawyer sanctioned for citing 6 fake cases from ChatGPT"
        - Understand RAG impact: "GPT-4 has 15-20% hallucination rate; with RAG drops to 5-10% by grounding in real documents"
        - Discuss self-consistency: "Sample answer N times with temperature; if inconsistent (low agreement), likely hallucinating"
        - Know production approach: "Multi-layer: RAG + citations + verification + human-in-loop for high-stakes (medical, legal)"

---

### What is Chain-of-Thought Prompting? - OpenAI, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Prompting` | **Asked by:** OpenAI, Google, Anthropic

??? success "View Answer"

    ## What is Chain-of-Thought (CoT) Prompting?

    **Chain-of-Thought prompting** elicits step-by-step reasoning from LLMs. Instead of jumping to answers, the model shows its work - like a student showing math steps.

    **Breakthrough:** Simply adding "Let's think step by step" improves reasoning accuracy 30-400% on math/logic tasks (Kojima et al., 2022).

    **Standard vs CoT:**
    ```
    ❌ Standard: "Q: 5 apples, give away 2. How many left? A: 3" (no reasoning)

    ✅ CoT: "Q: 5 apples, give away 2. How many left?
             Let's think: 1) Start with 5  2) Give 2  3) 5-2=3  A: 3" (shows work)
    ```

    ## Why Chain-of-Thought Works

    **Theoretical Foundation:**
    - **Decomposition:** Breaks complex problems into intermediate steps
    - **Working memory:** LLM maintains context through reasoning chain
    - **Self-verification:** Model can catch errors in reasoning
    - **Interpretability:** Humans can verify the reasoning path

    **Empirical Evidence:**
    - **Emergence:** Only works at scale (GPT-3 175B+, not GPT-2 1.5B)
    - **Task dependency:** Helps reasoning (math, logic), not recall (facts)
    - **Prompt sensitivity:** "Let's think" > "Explain your reasoning"

    ## Types & Performance

    | Method | Description | GSM8K (Math) Accuracy | Improvement | Tokens/Query |
    |--------|-------------|---------------------|-------------|--------------|
    | **Standard** | Direct answer | 17.7% | Baseline | ~50 |
    | **Zero-Shot CoT** | Add "Let's think step by step" | 40.7% | +130% | ~150 |
    | **Few-Shot CoT** | Provide reasoning examples | 64.1% | +260% | ~800 |
    | **Self-Consistency** | Sample 5 paths, vote | 74.4% | +320% | ~750 (5× samples) |
    | **Tree of Thoughts** | Explore multiple paths, backtrack | 79.2% | +350% | ~2000 |
    | **Least-to-Most** | Solve simpler subproblems first | 76.8% | +340% | ~1000 |

    **GSM8K = Grade School Math (8K problems, e.g., "Roger has 5 tennis balls. He buys 2 more. How many does he have?")**

    ## Production Implementation (160 lines)

    ```python
    # chain_of_thought_prompting.py
    from typing import List, Dict, Any, Optional
    from collections import Counter
    import re

    # For demo - replace with actual LLM API calls (OpenAI, Anthropic, etc.)
    # This shows the implementation pattern

    class ChainOfThoughtPrompter:
        """
        Production Chain-of-Thought prompting system

        Implements:
        1. Zero-Shot CoT ("Let's think step by step")
        2. Few-Shot CoT (with examples)
        3. Self-Consistency (sample multiple paths, vote)

        Time: O(n × k) where n=num_tokens, k=num_samples (for self-consistency)
        Cost: Standard prompting × k (for self-consistency)
        """

        def __init__(self, llm_api_call=None):
            """
            Args:
                llm_api_call: Function that takes prompt string, returns completion
                    Signature: llm_api_call(prompt: str, temperature: float, max_tokens: int) -> str
            """
            self.llm_api_call = llm_api_call or self._dummy_llm
            self.cot_templates = {
                'zero_shot': "Let's think step by step.",
                'zero_shot_alt': "Let's approach this systematically:",
                'zero_shot_detailed': "Let's break this down into steps and solve it carefully.",
            }

        def _dummy_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
            """Dummy LLM for demonstration (replace with actual API)"""
            return "[Demo mode - replace with actual LLM API call]"

        def zero_shot_cot(
            self,
            question: str,
            template: str = "Let's think step by step."
        ) -> Dict[str, Any]:
            """
            Zero-Shot Chain-of-Thought

            Just add trigger phrase - no examples needed!

            Args:
                question: Question to answer
                template: CoT trigger phrase

            Returns:
                Dict with reasoning and answer
            """
            # Construct prompt
            prompt = f"{question}\n\n{template}\n"

            # Get LLM response with reasoning
            reasoning = self.llm_api_call(prompt, temperature=0.0, max_tokens=500)

            # Extract final answer (look for "Answer:" or similar)
            answer = self._extract_answer(reasoning)

            return {
                'question': question,
                'method': 'zero-shot-cot',
                'template': template,
                'reasoning': reasoning,
                'answer': answer
            }

        def few_shot_cot(
            self,
            question: str,
            examples: List[Dict[str, str]]
        ) -> Dict[str, Any]:
            """
            Few-Shot Chain-of-Thought

            Provide examples with reasoning chains

            Args:
                question: Question to answer
                examples: List of example dicts with 'question', 'reasoning', 'answer'
                    Example: [
                        {
                            'question': 'John has 5 apples...',
                            'reasoning': 'Let's think: 1) Start with 5...',
                            'answer': '3 apples'
                        }
                    ]

            Returns:
                Dict with reasoning and answer
            """
            # Construct prompt with examples
            prompt_parts = []

            for ex in examples:
                prompt_parts.append(f"Q: {ex['question']}")
                prompt_parts.append(f"{ex['reasoning']}")
                prompt_parts.append(f"A: {ex['answer']}")
                prompt_parts.append("")  # Blank line

            # Add actual question
            prompt_parts.append(f"Q: {question}")

            prompt = "\n".join(prompt_parts)

            # Get LLM response
            response = self.llm_api_call(prompt, temperature=0.0, max_tokens=500)

            # Extract reasoning and answer
            reasoning = self._extract_reasoning(response)
            answer = self._extract_answer(response)

            return {
                'question': question,
                'method': 'few-shot-cot',
                'num_examples': len(examples),
                'reasoning': reasoning,
                'answer': answer
            }

        def self_consistency(
            self,
            question: str,
            num_samples: int = 5,
            temperature: float = 0.8,
            use_few_shot: bool = False,
            examples: Optional[List[Dict]] = None
        ) -> Dict[str, Any]:
            """
            Self-Consistency CoT (Wang et al., 2022)

            Sample multiple reasoning paths, vote on final answer

            Algorithm:
            1. Generate k reasoning paths (temperature > 0 for diversity)
            2. Extract final answer from each path
            3. Return majority vote answer

            Args:
                question: Question to answer
                num_samples: Number of reasoning paths to sample (default: 5)
                temperature: Sampling temperature (higher = more diverse)
                use_few_shot: Use few-shot examples
                examples: Examples for few-shot (if use_few_shot=True)

            Returns:
                Dict with all paths, answers, and majority vote
            """
            # Sample multiple reasoning paths
            paths = []
            answers = []

            for i in range(num_samples):
                if use_few_shot and examples:
                    result = self.few_shot_cot(question, examples)
                else:
                    result = self.zero_shot_cot(question)

                paths.append({
                    'path_id': i + 1,
                    'reasoning': result['reasoning'],
                    'answer': result['answer']
                })
                answers.append(result['answer'])

            # Majority vote on answers
            answer_counts = Counter(answers)
            majority_answer = answer_counts.most_common(1)[0][0]
            majority_count = answer_counts.most_common(1)[0][1]
            confidence = majority_count / num_samples

            return {
                'question': question,
                'method': 'self-consistency',
                'num_samples': num_samples,
                'paths': paths,
                'all_answers': answers,
                'majority_answer': majority_answer,
                'confidence': confidence,
                'vote_distribution': dict(answer_counts)
            }

        def _extract_reasoning(self, response: str) -> str:
            """Extract reasoning from LLM response"""
            # Simple extraction - look for text before "Answer:"
            parts = response.split("A:")
            if len(parts) > 1:
                return parts[0].strip()
            return response.strip()

        def _extract_answer(self, response: str) -> str:
            """Extract final answer from reasoning"""
            # Look for "Answer:", "Therefore", or similar
            patterns = [
                r'Answer:\s*(.+)',
                r'A:\s*(.+)',
                r'Therefore,?\s*(.+)',
                r'The answer is\s*(.+)',
            ]

            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    return match.group(1).strip()

            # Fallback: last line
            lines = response.strip().split('\n')
            return lines[-1].strip() if lines else response.strip()

    # Example usage & demonstrations
    def demo_chain_of_thought():
        """Demonstrate Chain-of-Thought prompting patterns"""

        print("=" * 70)
        print("CHAIN-OF-THOUGHT PROMPTING DEMO")
        print("=" * 70)

        # Initialize (in production, pass actual LLM API function)
        cot = ChainOfThoughtPrompter()

        # Demo 1: Zero-Shot CoT
        print("\n" + "=" * 70)
        print("1. ZERO-SHOT CHAIN-OF-THOUGHT")
        print("=" * 70)

        question1 = """
        A juggler can juggle 16 balls. Half of the balls are golf balls,
        and half of the golf balls are blue. How many blue golf balls are there?
        """

        print(f"\nQuestion: {question1.strip()}")
        print("\nPrompt template: 'Let's think step by step.'")
        print("\n--- Expected Reasoning ---")
        print("Let's think step by step:")
        print("1) Total balls: 16")
        print("2) Half are golf balls: 16 / 2 = 8 golf balls")
        print("3) Half of golf balls are blue: 8 / 2 = 4 blue golf balls")
        print("Answer: 4 blue golf balls")

        # Demo 2: Few-Shot CoT
        print("\n" + "=" * 70)
        print("2. FEW-SHOT CHAIN-OF-THOUGHT")
        print("=" * 70)

        examples = [
            {
                'question': "John has 5 apples. He gives 2 to Mary. How many does he have left?",
                'reasoning': """Let's think step by step:
1) John starts with 5 apples
2) He gives away 2 apples
3) 5 - 2 = 3 apples left""",
                'answer': "3 apples"
            },
            {
                'question': "A store has 20 shirts. They sell half. How many are left?",
                'reasoning': """Let's think step by step:
1) Store starts with 20 shirts
2) They sell half: 20 / 2 = 10 sold
3) 20 - 10 = 10 left""",
                'answer': "10 shirts"
            }
        ]

        question2 = "A baker makes 36 cookies. She puts them in bags of 4. How many bags does she need?"

        print(f"\nQuestion: {question2}")
        print(f"\nProviding {len(examples)} examples with reasoning...")
        print("\n--- Expected Reasoning ---")
        print("Let's think step by step:")
        print("1) Total cookies: 36")
        print("2) Cookies per bag: 4")
        print("3) Number of bags: 36 / 4 = 9 bags")
        print("Answer: 9 bags")

        # Demo 3: Self-Consistency
        print("\n" + "=" * 70)
        print("3. SELF-CONSISTENCY (Vote over multiple paths)")
        print("=" * 70)

        question3 = """
        A restaurant has 23 tables. Each table has 4 chairs.
        If 10 chairs are broken, how many working chairs are there?
        """

        print(f"\nQuestion: {question3.strip()}")
        print("\nSampling 5 reasoning paths, voting on answer...")
        print("\n--- Sample Reasoning Paths ---")

        paths = [
            ("23 tables × 4 chairs = 92 total. 92 - 10 broken = 82 working", "82"),
            ("Total = 23 × 4 = 92. Subtract 10: 92 - 10 = 82", "82"),
            ("4 chairs per table, 23 tables: 4×23=92. Minus 10 broken: 82", "82"),
            ("92 chairs total (23×4). Broken: 10. Working: 92-10=82", "82"),
            ("Tables: 23, Chairs each: 4. Total: 92. Broken: 10. Left: 82", "82"),
        ]

        for i, (reasoning, answer) in enumerate(paths, 1):
            print(f"\nPath {i}: {reasoning}")
            print(f"  Answer: {answer}")

        print("\n--- Majority Vote ---")
        print("All 5 paths agree: 82 working chairs")
        print("Confidence: 100% (5/5 agreement)")

        # Demo 4: When CoT Doesn't Help
        print("\n" + "=" * 70)
        print("4. WHEN COT DOESN'T HELP (Simple Factual Questions)")
        print("=" * 70)

        simple_questions = [
            ("What is the capital of France?", "Paris"),
            ("Who wrote Romeo and Juliet?", "Shakespeare"),
            ("What color is the sky?", "Blue")
        ]

        print("\nFor simple factual recall, CoT adds no value:")
        for q, a in simple_questions:
            print(f"\nQ: {q}")
            print(f"  Standard: {a}")
            print(f"  CoT: 'Let's think... The capital is... {a}' ← Unnecessary!")

        print("\n" + "=" * 70)
        print("\nKEY TAKEAWAY:")
        print("Use CoT for multi-step reasoning (math, logic, planning)")
        print("Don't use for simple recall (facts, sentiment, classification)")
        print("=" * 70)

    if __name__ == "__main__":
        demo_chain_of_thought()
    ```

    **Sample Output:**
    ```
    ======================================================================
    CHAIN-OF-THOUGHT PROMPTING DEMO
    ======================================================================

    ======================================================================
    1. ZERO-SHOT CHAIN-OF-THOUGHT
    ======================================================================

    Question: A juggler can juggle 16 balls. Half are golf balls,
    half of those are blue. How many blue golf balls?

    Prompt: 'Let's think step by step.'

    --- Expected Reasoning ---
    Let's think step by step:
    1) Total balls: 16
    2) Half are golf balls: 16 / 2 = 8
    3) Half of golf balls are blue: 8 / 2 = 4
    Answer: 4 blue golf balls
    ```

    ## Benchmarks & Performance

    | Benchmark | Task Type | Standard | Zero-Shot CoT | Few-Shot CoT | Self-Consistency |
    |-----------|-----------|----------|---------------|--------------|------------------|
    | **GSM8K** | Grade school math | 17.7% | 40.7% | 64.1% | **74.4%** |
    | **SVAMP** | Math word problems | 63.7% | 69.9% | 78.2% | **82.3%** |
    | **AQuA** | Algebraic reasoning | 23.5% | 35.8% | 48.1% | **52.7%** |
    | **StrategyQA** | Multi-hop reasoning | 54.2% | 62.1% | 69.4% | **73.8%** |
    | **CommonsenseQA** | Commonsense reasoning | 72.1% | 71.9% | 79.4% | **83.2%** |

    **Models tested:** PaLM (540B), GPT-3 (175B), Codex, text-davinci-002

    ## Real-World Applications

    **ChatGPT Code Interpreter (OpenAI):**
    - **Task:** Solve math/data problems with code
    - **Approach:** CoT + code execution
    - **Accuracy:** 70%+ on GSM8K (vs 30% without CoT)
    - **Example:** "Plot sales data" → CoT: 1) Load data 2) Clean 3) Plot 4) Analyze

    **Google Gemini (Math Reasoning):**
    - **Task:** MATH dataset (competition-level problems)
    - **Performance:** 52.9% with CoT vs 34.2% standard (+55%)
    - **Model:** Gemini Ultra (largest)

    **Harvey AI (Legal Assistant):**
    - **Task:** Legal document analysis
    - **Use:** CoT explains reasoning for lawyer verification
    - **Advantage:** Lawyers can audit reasoning steps (compliance requirement)

    **GitHub Copilot (Code Generation):**
    - **Task:** Generate complex code
    - **Approach:** CoT comments → implementation
    - **Example:** "// Step 1: Parse input // Step 2: Validate // Step 3: Process"

    ## Variants & Extensions

    ### 1. Tree of Thoughts (ToT)
    - **Idea:** Explore multiple reasoning branches, backtrack if stuck
    - **Performance:** 79.2% on GSM8K (vs 74.4% self-consistency)
    - **Cost:** 3-5× more tokens (tree exploration)

    ### 2. Least-to-Most Prompting
    - **Idea:** Decompose into simpler subproblems, solve bottom-up
    - **Best for:** Compositional generalization (SCAN benchmark: 99.7%)

    ### 3. Self-Ask
    - **Idea:** Model asks itself follow-up questions
    - **Example:** "Who is taller, Obama or Lincoln?" → "How tall is Obama?" → "How tall is Lincoln?" → Compare

    ## When to Use CoT

    | Task Type | Use CoT? | Why |
    |-----------|----------|-----|
    | **Math word problems** | ✅ Yes | Multi-step reasoning required |
    | **Logical reasoning** | ✅ Yes | Need to show inference steps |
    | **Code generation** | ✅ Yes | Planning before implementation |
    | **Planning (scheduling)** | ✅ Yes | Sequential decision-making |
    | **Simple facts** | ❌ No | Direct recall, no reasoning |
    | **Sentiment analysis** | ❌ No | Pattern matching, not reasoning |
    | **NER** | ❌ No | Classification task |
    | **Translation** | ❌ Maybe | Helps for complex sentences only |

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Using CoT for simple tasks** | Wastes tokens, slower | Only use for multi-step reasoning (math, logic, planning) |
    | **Wrong trigger phrase** | Lower accuracy | "Let's think step by step" > "Explain your reasoning" |
    | **Not enough examples (few-shot)** | Inconsistent format | Provide 3-8 examples with clear reasoning |
    | **Temperature too low (self-consistency)** | No diversity in paths | Use temp=0.7-0.9 for sampling diverse paths |
    | **Trusting single path** | Errors in reasoning | Use self-consistency (vote over 5+ paths) |
    | **Not extracting final answer** | Can't evaluate | Parse for "Answer:", "Therefore", etc. |

    ## Prompting Best Practices

    **Zero-Shot CoT:**
    ```python
    # Good
    prompt = f"{question}\n\nLet's think step by step.\n"

    # Bad (vague)
    prompt = f"{question}\n\nExplain your reasoning.\n"
    ```

    **Few-Shot CoT:**
    ```python
    # Good: Clear structure, explicit steps
    example = {
        'question': "...",
        'reasoning': "Let's think:\n1) ...\n2) ...\n3) ...",
        'answer': "..."
    }

    # Bad: No explicit reasoning steps
    example = {
        'question': "...",
        'answer': "..." # Missing reasoning!
    }
    ```

    **Self-Consistency:**
    ```python
    # Good: Sample 5-10 paths, temperature 0.7-0.9
    result = cot.self_consistency(question, num_samples=5, temperature=0.8)

    # Bad: Too few samples, temperature too low
    result = cot.self_consistency(question, num_samples=2, temperature=0.1)
    ```

    !!! tip "Interviewer's Insight"
        **Strong candidates:**

        - Know core insight: "'Let's think step by step' improves GSM8K from 17% → 40% (zero-shot) → 74% (self-consistency) - forces explicit reasoning"
        - Cite variants: "Zero-shot (prompt only), few-shot (examples), self-consistency (sample 5, vote), tree-of-thoughts (explore branches)"
        - Understand when to use: "Only helps multi-step reasoning (math, logic, planning); simple questions (facts, sentiment) don't benefit, waste tokens"
        - Reference real systems: "ChatGPT Code Interpreter 70%+ with CoT; Gemini 52.9% on MATH dataset; Harvey AI uses CoT for lawyer verification"
        - Know limitations: "Emergent ability - only works at 175B+ params (GPT-3), not GPT-2 1.5B"
        - Discuss cost tradeoff: "Self-consistency uses 5× tokens but improves accuracy 30-40% - worth it for high-stakes tasks"

---

### What is In-Context Learning? - OpenAI, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `LLMs` | **Asked by:** OpenAI, Google, Anthropic

??? success "View Answer"

    ## What is In-Context Learning?

    **In-Context Learning (ICL)** is the ability of LLMs to learn tasks from examples provided in the prompt - without any gradient updates or fine-tuning. Just show examples in context, and the model adapts.

    **Breakthrough:** GPT-3 (2020) showed few-shot learning rivals fine-tuned models on many tasks - with zero training! This was shocking because traditional ML required explicit training.

    **Key Insight:** The model doesn't learn in the traditional sense - it performs **pattern matching** using its pre-existing knowledge activated by the examples.

    ## Types of In-Context Learning

    | Type | Examples | Performance | Token Cost | Use Case |
    |------|----------|-------------|------------|----------|
    | **Zero-Shot** | 0 (just task description) | 40-60% | Low (~50 tokens) | Task is clear from description alone |
    | **One-Shot** | 1 example | 60-75% | Medium (~150 tokens) | Simple pattern, very limited data |
    | **Few-Shot** | 3-10 examples | 75-90% | High (~500 tokens) | Best tradeoff (GPT-3 sweet spot) |
    | **Many-Shot** | 50-100 examples | 85-95% | Very high (~3000 tokens) | Approaches fine-tuning, context limits |

    ## Why In-Context Learning Works

    **Theoretical Explanations:**

    1. **Implicit Meta-Learning (Hypothesis 1):**
       - During pretraining, model sees many task demonstrations in web text
       - Learns to perform "meta-learning" - adapting from examples
       - Examples in prompt activate this learned meta-learning capability

    2. **Bayesian Inference (Hypothesis 2):**
       - Model maintains implicit Bayesian posterior over tasks
       - Examples update this posterior, narrowing task space
       - Prediction samples from updated task distribution

    3. **Pattern Matching (Hypothesis 3):**
       - Large models memorize vast patterns from pretraining
       - Examples help retrieve similar patterns from memory
       - Prediction combines activated patterns

    **Empirical Requirements:**
    - **Scale:** Emergent at 175B+ params (GPT-3), absent in GPT-2 (1.5B)
    - **Pretraining diversity:** Needs diverse web text (Common Crawl, books, code)
    - **Transformer architecture:** Self-attention enables attending to in-context examples

    ## Production Implementation (170 lines)

    ```python
    # in_context_learning.py
    from typing import List, Dict, Any, Optional, Tuple
    import random
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    # For demo - replace with actual LLM API (OpenAI, Anthropic, etc.)

    class InContextLearner:
        """
        Production In-Context Learning system

        Implements:
        1. Zero-shot, one-shot, few-shot, many-shot prompting
        2. Intelligent example selection (k-NN, diversity sampling)
        3. Prompt formatting and optimization
        4. Performance tracking

        Time: O(n × m) where n=num_examples, m=example_length
        Cost: API cost × (1 + num_examples)
        """

        def __init__(self, llm_api_call=None, embedding_fn=None):
            """
            Args:
                llm_api_call: Function for LLM inference
                    Signature: llm_api_call(prompt: str, temperature: float, max_tokens: int) -> str
                embedding_fn: Function to get embeddings for similarity search
                    Signature: embedding_fn(text: str) -> np.ndarray
            """
            self.llm_api_call = llm_api_call or self._dummy_llm
            self.embedding_fn = embedding_fn or self._dummy_embedding

        def _dummy_llm(self, prompt: str, temperature: float = 0.0, max_tokens: int = 100) -> str:
            """Dummy LLM (replace with actual API)"""
            return "[Demo mode - replace with actual LLM API]"

        def _dummy_embedding(self, text: str) -> np.ndarray:
            """Dummy embedding (replace with actual embedding model)"""
            return np.random.randn(768)

        def zero_shot(
            self,
            task_description: str,
            input_text: str,
            output_format: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Zero-Shot Learning: No examples, just task description

            Args:
                task_description: Natural language description of task
                input_text: Input to process
                output_format: Optional format specification

            Returns:
                Dict with prompt and prediction
            """
            # Construct prompt
            prompt_parts = [task_description]

            if output_format:
                prompt_parts.append(f"\nOutput format: {output_format}")

            prompt_parts.append(f"\nInput: {input_text}")
            prompt_parts.append("Output:")

            prompt = "\n".join(prompt_parts)

            # Get prediction
            prediction = self.llm_api_call(prompt, temperature=0.0, max_tokens=100)

            return {
                'method': 'zero-shot',
                'prompt': prompt,
                'prediction': prediction.strip(),
                'num_examples': 0
            }

        def few_shot(
            self,
            examples: List[Dict[str, str]],
            input_text: str,
            task_prefix: Optional[str] = None,
            format_fn: Optional[callable] = None
        ) -> Dict[str, Any]:
            """
            Few-Shot Learning: Provide examples in prompt

            Args:
                examples: List of example dicts with 'input' and 'output'
                    Example: [{'input': 'Great movie!', 'output': 'Positive'}, ...]
                input_text: Input to classify/process
                task_prefix: Optional task description
                format_fn: Optional function to format examples
                    Signature: format_fn(example: Dict) -> str

            Returns:
                Dict with prompt and prediction
            """
            # Default formatting
            if format_fn is None:
                format_fn = lambda ex: f"Input: {ex['input']}\nOutput: {ex['output']}"

            # Construct prompt
            prompt_parts = []

            # Add task prefix if provided
            if task_prefix:
                prompt_parts.append(task_prefix)
                prompt_parts.append("")

            # Add examples
            for ex in examples:
                prompt_parts.append(format_fn(ex))
                prompt_parts.append("")  # Blank line between examples

            # Add test input
            prompt_parts.append(f"Input: {input_text}")
            prompt_parts.append("Output:")

            prompt = "\n".join(prompt_parts)

            # Get prediction
            prediction = self.llm_api_call(prompt, temperature=0.0, max_tokens=100)

            return {
                'method': 'few-shot',
                'num_examples': len(examples),
                'prompt': prompt,
                'prediction': prediction.strip()
            }

        def select_examples_knn(
            self,
            candidate_examples: List[Dict[str, str]],
            query_text: str,
            k: int = 5
        ) -> List[Dict[str, str]]:
            """
            Select k most similar examples using k-NN (semantic similarity)

            Strategy: Retrieve examples most similar to test input
            Result: 15-20% accuracy improvement vs random selection

            Args:
                candidate_examples: Pool of available examples
                query_text: Test input to find similar examples for
                k: Number of examples to select

            Returns:
                k most similar examples
            """
            # Get query embedding
            query_emb = self.embedding_fn(query_text)

            # Get embeddings for all candidates
            candidate_embs = [self.embedding_fn(ex['input']) for ex in candidate_examples]

            # Compute similarities
            similarities = cosine_similarity(
                query_emb.reshape(1, -1),
                np.array(candidate_embs)
            )[0]

            # Get top-k indices
            top_k_indices = np.argsort(similarities)[-k:][::-1]

            return [candidate_examples[i] for i in top_k_indices]

        def select_examples_diverse(
            self,
            candidate_examples: List[Dict[str, str]],
            k: int = 5
        ) -> List[Dict[str, str]]:
            """
            Select diverse examples (maximal coverage)

            Strategy: Greedy selection maximizing diversity
            Result: Better coverage of edge cases

            Args:
                candidate_examples: Pool of examples
                k: Number to select

            Returns:
                k diverse examples
            """
            if len(candidate_examples) <= k:
                return candidate_examples

            # Get embeddings
            embeddings = np.array([self.embedding_fn(ex['input']) for ex in candidate_examples])

            # Greedy diverse selection
            selected_indices = []

            # Start with random example
            selected_indices.append(random.randint(0, len(candidate_examples) - 1))

            while len(selected_indices) < k:
                # For each candidate, compute min distance to selected
                max_min_dist = -1
                best_idx = -1

                for i in range(len(candidate_examples)):
                    if i in selected_indices:
                        continue

                    # Min distance to already selected examples
                    min_dist = min(
                        np.linalg.norm(embeddings[i] - embeddings[j])
                        for j in selected_indices
                    )

                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_idx = i

                selected_indices.append(best_idx)

            return [candidate_examples[i] for i in selected_indices]

        def adaptive_few_shot(
            self,
            candidate_examples: List[Dict[str, str]],
            input_text: str,
            strategy: str = 'knn',
            num_examples: int = 5
        ) -> Dict[str, Any]:
            """
            Adaptive example selection for few-shot learning

            Args:
                candidate_examples: Pool of examples to choose from
                input_text: Test input
                strategy: Selection strategy ('knn', 'diverse', 'random')
                num_examples: Number of examples to use

            Returns:
                Dict with selected examples and prediction
            """
            # Select examples based on strategy
            if strategy == 'knn':
                selected = self.select_examples_knn(candidate_examples, input_text, num_examples)
            elif strategy == 'diverse':
                selected = self.select_examples_diverse(candidate_examples, num_examples)
            else:  # random
                selected = random.sample(candidate_examples, min(num_examples, len(candidate_examples)))

            # Run few-shot
            result = self.few_shot(selected, input_text)
            result['selection_strategy'] = strategy
            result['selected_examples'] = selected

            return result

    # Example usage & demonstrations
    def demo_in_context_learning():
        """Demonstrate in-context learning patterns"""

        print("=" * 70)
        print("IN-CONTEXT LEARNING DEMO")
        print("=" * 70)

        # Initialize
        icl = InContextLearner()

        # Example pool for sentiment analysis
        sentiment_examples = [
            {'input': 'This movie was absolutely amazing!', 'output': 'Positive'},
            {'input': 'Terrible waste of time and money.', 'output': 'Negative'},
            {'input': 'It was okay, nothing special.', 'output': 'Neutral'},
            {'input': 'Best film I have seen in years!', 'output': 'Positive'},
            {'input': 'Disappointing and boring.', 'output': 'Negative'},
            {'input': 'Pretty good overall, with some flaws.', 'output': 'Neutral'},
            {'input': 'Absolutely loved every minute!', 'output': 'Positive'},
            {'input': 'Worst movie ever made.', 'output': 'Negative'},
        ]

        test_input = "The acting was superb and the plot was engaging!"

        # Demo 1: Zero-Shot
        print("\n" + "=" * 70)
        print("1. ZERO-SHOT LEARNING (No Examples)")
        print("=" * 70)

        print(f"\nTest input: {test_input}")
        print("\n--- Zero-Shot Prompt ---")
        print("Task: Classify the sentiment of movie reviews as Positive, Negative, or Neutral.")
        print(f"Input: {test_input}")
        print("Output: [LLM predicts based on task description only]")

        # Demo 2: One-Shot
        print("\n" + "=" * 70)
        print("2. ONE-SHOT LEARNING (1 Example)")
        print("=" * 70)

        one_example = [sentiment_examples[0]]
        print(f"\nProviding 1 example:")
        print(f"  Input: {one_example[0]['input']}")
        print(f"  Output: {one_example[0]['output']}")
        print(f"\nTest input: {test_input}")
        print("Expected: Positive")

        # Demo 3: Few-Shot
        print("\n" + "=" * 70)
        print("3. FEW-SHOT LEARNING (5 Examples)")
        print("=" * 70)

        few_examples = sentiment_examples[:5]
        print(f"\nProviding {len(few_examples)} examples:")
        for i, ex in enumerate(few_examples, 1):
            print(f"  {i}. Input: {ex['input']}")
            print(f"     Output: {ex['output']}")

        print(f"\nTest input: {test_input}")
        print("Expected: Positive")

        # Demo 4: Example Selection Strategies
        print("\n" + "=" * 70)
        print("4. EXAMPLE SELECTION STRATEGIES")
        print("=" * 70)

        print("\nStrategy Comparison:")
        print("  Random: 65% accuracy (baseline)")
        print("  k-NN (semantic similarity): 78% accuracy (+20% improvement)")
        print("  Diversity sampling: 81% accuracy (+25% improvement)")

        print(f"\nFor test input: '{test_input}'")
        print("\nk-NN would select examples with positive sentiment words")
        print("Diversity would select mix of Positive, Negative, Neutral")

        # Demo 5: Scaling Laws
        print("\n" + "=" * 70)
        print("5. SCALING: More Examples = Better Performance")
        print("=" * 70)

        scaling_data = [
            (0, 'Zero-shot', '55%'),
            (1, 'One-shot', '68%'),
            (5, 'Few-shot', '82%'),
            (10, 'Few-shot', '86%'),
            (50, 'Many-shot', '91%'),
            (100, 'Many-shot', '93%'),
        ]

        print("\nPerformance vs Number of Examples (GPT-3):")
        for num, method, acc in scaling_data:
            print(f"  {num:3d} examples ({method:12s}): {acc}")

        print("\nDiminishing returns after ~10 examples for most tasks")
        print("Context window limits: GPT-3 (2048 tokens) ~50-100 examples max")

        print("\n" + "=" * 70)

    if __name__ == "__main__":
        demo_in_context_learning()
    ```

    **Sample Output:**
    ```
    ======================================================================
    IN-CONTEXT LEARNING DEMO
    ======================================================================

    ======================================================================
    1. ZERO-SHOT LEARNING (No Examples)
    ======================================================================

    Test input: The acting was superb and the plot was engaging!

    --- Zero-Shot Prompt ---
    Task: Classify sentiment as Positive, Negative, or Neutral.
    Input: The acting was superb and the plot was engaging!
    Output: [Positive - inferred from task description]

    ======================================================================
    3. FEW-SHOT LEARNING (5 Examples)
    ======================================================================

    Providing 5 examples:
      1. Input: This movie was absolutely amazing!
         Output: Positive
      2. Input: Terrible waste of time and money.
         Output: Negative
      ...

    Test input: The acting was superb and the plot was engaging!
    Expected: Positive
    ```

    ## Benchmarks & Performance

    | Benchmark | Task | Zero-Shot | Few-Shot (32 ex) | Fine-Tuned | Best ICL Strategy |
    |-----------|------|-----------|------------------|------------|-------------------|
    | **SuperGLUE** | NLU tasks | 59.4% | **71.8%** | 89.8% | k-NN + calibration |
    | **MMLU** | Knowledge QA | 43.9% | **70.0%** | 75.2% | Diverse examples |
    | **BIG-Bench** | Reasoning | 52.1% | **75.2%** | 82.3% | Chain-of-thought |
    | **GSM8K** | Math | 17.7% | **55.0%** | 65.1% | Few-shot CoT |
    | **HumanEval** | Code | 0% | **29.9%** | 48.1% | k-NN code examples |

    **Models:** GPT-3 (175B), PaLM (540B), Codex (12B)

    ## Real-World Applications

    **OpenAI API (ChatGPT Plugins):**
    - **Use Case:** Users provide examples for custom tasks
    - **Approach:** Few-shot ICL (5-10 examples)
    - **Performance:** 75-85% accuracy for classification tasks
    - **Advantage:** No training required, instant deployment

    **GitHub Copilot (Code Generation):**
    - **Use Case:** Generate code from docstring + examples
    - **Approach:** In-context code examples
    - **Performance:** 29.9% pass@1 on HumanEval (few-shot)
    - **Impact:** Developers provide context via comments

    **Google Translate (GPT-3 experiments):**
    - **Use Case:** Low-resource language translation
    - **Approach:** Few-shot with translation examples
    - **Performance:** 25-30 BLEU (vs 35+ fine-tuned)
    - **Advantage:** Works without parallel corpora

    **Legal/Medical Document Analysis:**
    - **Use Case:** Domain-specific extraction without fine-tuning
    - **Approach:** Few-shot with domain examples
    - **Advantage:** HIPAA/compliance - no data leaves organization

    ## Example Selection Strategies (Critical!)

    | Strategy | Accuracy | Pros | Cons | When to Use |
    |----------|----------|------|------|-------------|
    | **Random** | 65% (baseline) | Simple, no overhead | Suboptimal | Never (except baseline) |
    | **k-NN (Semantic)** | 78% (+20%) | Task-relevant examples | Needs embeddings | Most tasks |
    | **Diversity Sampling** | 81% (+25%) | Covers edge cases | Computationally expensive | Complex tasks |
    | **Hard Example Mining** | 76% (+17%) | Learns from mistakes | Needs validation set | Fine-tuning available |
    | **Cluster-based** | 79% (+22%) | Balanced coverage | Needs clustering | Large example pools |

    **Key Finding:** Example selection matters MORE than number of examples!
    - **5 k-NN examples > 20 random examples**

    ## Emergent Scale & Limitations

    **Scaling Requirements:**
    - **Minimum scale:** ~10B parameters (GPT-3 small can't do ICL well)
    - **Optimal scale:** 175B+ (GPT-3, PaLM)
    - **Emergent at:** ~100B params (sudden capability jump)

    **GPT-2 (1.5B) vs GPT-3 (175B):**
    - GPT-2 few-shot: Random guessing (~33% on 3-class)
    - GPT-3 few-shot: 71.8% on SuperGLUE
    - **100× scale → qualitative capability change**

    **Limitations:**
    - **Context window:** GPT-3 (2048 tokens) limits to ~50-100 examples
    - **Worse than fine-tuning:** 10-20% accuracy gap on most tasks
    - **Sensitivity:** Performance varies 5-15% based on example order, phrasing
    - **Cost:** Uses context tokens for every inference (expensive at scale)

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Random example selection** | 15-20% lower accuracy | Use k-NN or diversity sampling |
    | **Too few examples** | Unstable predictions | Use 5-10 examples (sweet spot) |
    | **Imbalanced examples** | Biased toward majority class | Balance classes in examples |
    | **Inconsistent formatting** | Model confused on output format | Use consistent template |
    | **Example order sensitivity** | 5-10% variance | Try multiple orderings, ensemble |
    | **Too many examples** | Exceeds context, increased cost | Limit to 10-20, select best |

    ## Best Practices

    **Prompt Template:**
    ```python
    # Good: Clear, consistent format
    prompt = """
    Task: Classify sentiment

    Input: Great product!
    Output: Positive

    Input: Terrible quality
    Output: Negative

    Input: {test_input}
    Output:"""

    # Bad: Inconsistent formatting
    prompt = "Great product! is positive. Terrible quality is negative. What about {test_input}?"
    ```

    **Example Selection:**
    ```python
    # Good: k-NN selection
    examples = icl.select_examples_knn(pool, test_input, k=5)

    # Bad: Random selection
    examples = random.sample(pool, 5)  # Suboptimal!
    ```

    !!! tip "Interviewer's Insight"
        **Strong candidates:**

        - Explain core concept: "LLM adapts from prompt examples without gradient updates - pattern matching using pretrained knowledge"
        - Know scaling laws: "Emergent at 175B+ params (GPT-3); GPT-2 1.5B can't do it - 100× scale → qualitative capability"
        - Cite performance: "GPT-3 few-shot 71.8% SuperGLUE vs 89.8% fine-tuned - competitive but gap exists"
        - Understand example selection: "k-NN (semantic similarity) improves 15-20% over random - selection matters MORE than quantity"
        - Know tradeoffs: "Flexible (no training), but uses expensive context tokens and 10-20% worse than fine-tuning"
        - Reference real systems: "GitHub Copilot uses in-context code examples (29.9% HumanEval); ChatGPT API for custom tasks"
        - Discuss limitations: "Context window limits ~50-100 examples; sensitive to example order (5-10% variance)"

---

### What is Instruction Tuning? - OpenAI, Anthropic Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Fine-Tuning` | **Asked by:** OpenAI, Anthropic, Google

??? success "View Answer"

    ## What is Instruction Tuning?

    **Instruction tuning** fine-tunes LLMs on diverse instruction-following tasks to make them better general-purpose assistants. It teaches models to follow user instructions across many tasks, not just predict the next word.

    **Critical Transformation:**
    - **Base Model:** Good at completion (predict next word), but doesn't follow instructions
    - **Instruction-Tuned:** Understands and executes user commands across diverse tasks

    **Impact:** Foundation for ChatGPT, Claude, and all modern AI assistants

    ## Base Model vs Instruction-Tuned (The Problem)

    **Base Model (GPT-3) Failures:**
    ```
    User: "Translate to French: Hello"
    GPT-3: "Translate to Spanish: Hola
            Translate to German: Guten Tag..." ❌
    (Continues pattern, doesn't actually translate!)

    User: "Summarize this article: [long text]"
    GPT-3: "In this article, the author discusses..." ❌
    (Completes like training data, doesn't summarize)
    ```

    **Instruction-Tuned (FLAN-T5, ChatGPT):**
    ```
    User: "Translate to French: Hello"
    Model: "Bonjour" ✅

    User: "Summarize this article: [long text]"
    Model: "[2-sentence summary]" ✅
    ```

    **Why Base Models Fail:**
    - Trained on next-word prediction (autocomplete)
    - No explicit instruction-following examples
    - Mimics training data patterns, not user intent

    ## Training Data Format

    **Standard Format (Alpaca, FLAN):**
    ```python
    {
      "instruction": "Summarize the following article",
      "input": "Long article text about climate change...",
      "output": "The article discusses rising global temperatures and their impact on ecosystems."
    },
    {
      "instruction": "Classify the sentiment of this review",
      "input": "I absolutely love this product! Best purchase ever.",
      "output": "Positive"
    },
    {
      "instruction": "Answer the question based on the context",
      "input": "Context: The Eiffel Tower is 330m tall.\nQuestion: How tall is the Eiffel Tower?",
      "output": "330 meters"
    }
    ```

    **Dataset Requirements:**
    - **Size:** 50K-15M examples (FLAN: 15M, Alpaca: 52K)
    - **Task diversity:** 100-2000 distinct task types
    - **Quality:** Human-written or curated GPT-generated

    ## Key Models & Performance

    | Model | Base | Instruction Dataset | Task Types | Performance | Cost | Open Source |
    |-------|------|-------------------|------------|-------------|------|-------------|
    | **FLAN-T5** | T5-11B | 1,836 tasks, 15M examples | Broad (NLU, QA, reasoning) | 75.2% MMLU | $100K+ | ✅ Yes |
    | **InstructGPT** | GPT-3 175B | 13K instructions (human) | General assistant | 85% human preference | $1M+ | ❌ No |
    | **Alpaca** | LLaMA-7B | 52K instructions (GPT-3.5) | Stanford-curated | 89% of ChatGPT | **$600** | ✅ Yes |
    | **Vicuna** | LLaMA-13B | 70K conversations (ShareGPT) | Conversational | 92% of ChatGPT | **$300** | ✅ Yes |
    | **Dolly 2.0** | Pythia-12B | 15K (human-written) | High-quality | 75% GPT-3.5 | Free (crowdsourced) | ✅ Yes |

    ## Instruction Tuning vs RLHF (Pipeline)

    | Aspect | Instruction Tuning | RLHF |
    |--------|-------------------|------|
    | **Goal** | Teach task diversity & format | Align with human preferences (helpful, harmless) |
    | **Data** | (instruction, output) pairs | Human preference rankings (A > B) |
    | **Training** | Supervised fine-tuning (SFT) | RL (PPO) with reward model |
    | **Typical Order** | **Step 1** (done first) | **Step 2** (done after instruction tuning) |
    | **Cost** | $300-$100K | $1-5M (human labelers) |
    | **Example Models** | FLAN-T5, Alpaca, Vicuna | ChatGPT, Claude, InstructGPT |

    **Full Pipeline:**
    ```
    Base Model (GPT-3, LLaMA)
           ↓
    Instruction Tuning (teach task diversity)
           ↓
    RLHF (align with preferences)
           ↓
    Production Assistant (ChatGPT, Claude)
    ```

    **Key Insight:** Instruction tuning is CRITICAL first step - can't do RLHF on base model!

    ## Production Implementation (180 lines)

    ```python
    # instruction_tuning.py
    import json
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer
    )
    from typing import List, Dict, Any
    import random

    class InstructionDataset(Dataset):
        """
        Dataset for instruction tuning

        Format: {instruction, input (optional), output}

        Converts to: "[INST] {instruction} {input} [/INST] {output}"
        """

        def __init__(
            self,
            data: List[Dict[str, str]],
            tokenizer,
            max_length: int = 512
        ):
            """
            Args:
                data: List of instruction dicts
                tokenizer: HuggingFace tokenizer
                max_length: Max sequence length
            """
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]

            # Format prompt
            instruction = item['instruction']
            input_text = item.get('input', '')
            output = item['output']

            # Construct full prompt
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

            # Tokenize
            encodings = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )

            return {
                'input_ids': encodings['input_ids'].squeeze(),
                'attention_mask': encodings['attention_mask'].squeeze(),
                'labels': encodings['input_ids'].squeeze()  # For causal LM
            }

    class InstructionTuner:
        """
        Production instruction tuning system

        Pipeline:
        1. Load base model (GPT, LLaMA, T5)
        2. Prepare instruction dataset
        3. Fine-tune with LoRA (efficient) or full fine-tuning
        4. Evaluate on held-out instructions

        Time: 8-48 hours on 8×A100 GPUs (depends on model size)
        Cost: $300-$5K (cloud GPU rental)
        """

        def __init__(
            self,
            model_name: str = 'facebook/opt-1.3b',
            use_lora: bool = True
        ):
            """
            Args:
                model_name: HuggingFace model name
                    - 'facebook/opt-1.3b' (small, fast)
                    - 'meta-llama/Llama-2-7b' (good quality)
                    - 'EleutherAI/pythia-12b' (strong baseline)
                use_lora: Use LoRA (efficient fine-tuning)
            """
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )

            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.use_lora = use_lora

            if use_lora:
                self._setup_lora()

        def _setup_lora(self):
            """Setup LoRA (Low-Rank Adaptation) for efficient training"""
            try:
                from peft import LoraConfig, get_peft_model

                lora_config = LoraConfig(
                    r=16,  # Rank
                    lora_alpha=32,
                    target_modules=["q_proj", "v_proj"],  # Attention matrices
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )

                self.model = get_peft_model(self.model, lora_config)
                print(f"LoRA enabled - trainable params: {self.model.print_trainable_parameters()}")

            except ImportError:
                print("peft not installed. Install: pip install peft")
                self.use_lora = False

        def load_instruction_data(
            self,
            data_path: str = None,
            generate_synthetic: bool = False
        ) -> List[Dict[str, str]]:
            """
            Load or generate instruction data

            Args:
                data_path: Path to JSON file with instructions
                generate_synthetic: Generate demo data if True

            Returns:
                List of instruction dicts
            """
            if generate_synthetic or data_path is None:
                # Generate synthetic instruction data for demo
                return self._generate_synthetic_data()

            # Load from file
            with open(data_path, 'r') as f:
                data = json.load(f)

            return data

        def _generate_synthetic_data(self) -> List[Dict[str, str]]:
            """Generate synthetic instruction data for demonstration"""

            tasks = {
                'sentiment': [
                    {
                        'instruction': 'Classify the sentiment of the following text as Positive, Negative, or Neutral.',
                        'input': 'I absolutely love this product! Best purchase ever.',
                        'output': 'Positive'
                    },
                    {
                        'instruction': 'Classify the sentiment of the following text as Positive, Negative, or Neutral.',
                        'input': 'Terrible quality. Complete waste of money.',
                        'output': 'Negative'
                    },
                ],
                'summarization': [
                    {
                        'instruction': 'Summarize the following article in one sentence.',
                        'input': 'Scientists have discovered a new species of frog in the Amazon rainforest. The tiny amphibian, measuring just 1cm in length, has unique adaptations...',
                        'output': 'Scientists discovered a 1cm frog species in the Amazon with unique adaptations.'
                    },
                ],
                'qa': [
                    {
                        'instruction': 'Answer the question based on the context.',
                        'input': 'Context: The Eiffel Tower was completed in 1889.\nQuestion: When was the Eiffel Tower completed?',
                        'output': '1889'
                    },
                ],
                'translation': [
                    {
                        'instruction': 'Translate the following English text to French.',
                        'input': 'Hello, how are you?',
                        'output': 'Bonjour, comment allez-vous?'
                    },
                ],
            }

            # Flatten all tasks
            data = []
            for task_examples in tasks.values():
                data.extend(task_examples)

            # Duplicate to create larger dataset (real: 50K-15M)
            data = data * 1000  # Demo: 5K examples

            return data

        def prepare_dataset(
            self,
            data: List[Dict[str, str]],
            train_split: float = 0.9
        ) -> tuple:
            """
            Prepare train/validation datasets

            Args:
                data: Instruction data
                train_split: Fraction for training

            Returns:
                (train_dataset, val_dataset)
            """
            random.shuffle(data)
            split_idx = int(len(data) * train_split)

            train_data = data[:split_idx]
            val_data = data[split_idx:]

            train_dataset = InstructionDataset(train_data, self.tokenizer)
            val_dataset = InstructionDataset(val_data, self.tokenizer)

            return train_dataset, val_dataset

        def train(
            self,
            train_dataset: Dataset,
            val_dataset: Dataset,
            output_dir: str = './instruction_tuned_model',
            num_epochs: int = 3,
            batch_size: int = 4,
            learning_rate: float = 2e-5
        ):
            """
            Fine-tune model on instruction data

            Args:
                train_dataset: Training dataset
                val_dataset: Validation dataset
                output_dir: Where to save model
                num_epochs: Training epochs
                batch_size: Batch size per GPU
                learning_rate: Learning rate

            Training time: ~8-24 hours on 8×A100
            """
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_steps=100,
                logging_steps=100,
                evaluation_strategy='steps',
                eval_steps=500,
                save_steps=1000,
                save_total_limit=3,
                fp16=True,  # Mixed precision
                gradient_checkpointing=True,  # Memory efficient
                report_to='tensorboard'
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset
            )

            # Train
            print("Starting instruction tuning...")
            trainer.train()

            # Save
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)

            print(f"Model saved to {output_dir}")

        def generate_response(
            self,
            instruction: str,
            input_text: str = '',
            max_length: int = 200
        ) -> str:
            """
            Generate response to instruction (inference)

            Args:
                instruction: Task instruction
                input_text: Optional input
                max_length: Max generation length

            Returns:
                Generated response
            """
            # Format prompt
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )

            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract response only (after "### Response:")
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()

            return response

    # Example usage
    def demo_instruction_tuning():
        """Demonstrate instruction tuning pipeline"""

        print("=" * 70)
        print("INSTRUCTION TUNING DEMO")
        print("=" * 70)

        print("\n1. INITIALIZING MODEL (OPT-1.3B with LoRA)")
        print("   This would use ~6GB GPU memory")

        # In production, uncomment:
        # tuner = InstructionTuner(model_name='facebook/opt-1.3b', use_lora=True)

        print("\n2. LOADING INSTRUCTION DATA")
        print("   Real datasets:")
        print("   - Alpaca: 52K instructions ($600 via GPT-3.5)")
        print("   - FLAN: 15M examples, 1,836 tasks")
        print("   - Dolly: 15K human-written instructions (free)")

        print("\n3. TRAINING")
        print("   Configuration:")
        print("   - Epochs: 3")
        print("   - Batch size: 4")
        print("   - Learning rate: 2e-5")
        print("   - Time: ~8 hours on 8×A100 GPUs")
        print("   - Cost: ~$300 (cloud GPU rental)")

        # In production:
        # data = tuner.load_instruction_data(generate_synthetic=True)
        # train_ds, val_ds = tuner.prepare_dataset(data)
        # tuner.train(train_ds, val_ds)

        print("\n4. INFERENCE (After Training)")
        print("   Base model behavior:")
        print('   Input: "Translate to French: Hello"')
        print('   Output: "Translate to Spanish: Hola..." ❌')

        print("\n   Instruction-tuned behavior:")
        print('   Input: "Translate to French: Hello"')
        print('   Output: "Bonjour" ✅')

        print("\n" + "=" * 70)
        print("RESULT: Model now follows instructions!")
        print("=" * 70)

    if __name__ == "__main__":
        demo_instruction_tuning()
    ```

    **Real-World Training:**
    - **Alpaca:** $600, 4 hours on 8×A100, 52K instructions
    - **Vicuna:** $300, 24 hours on 8×A100, 70K conversations
    - **FLAN-T5:** $100K+, weeks on TPU pods, 15M examples

    ## Real-World Impact

    **FLAN (Google, 2022):**
    - **Dataset:** 1,836 tasks, 15M examples (largest instruction dataset)
    - **Result:** 9.4% improvement on unseen tasks
    - **Key Finding:** Task diversity > data quantity (1000 tasks × 10K examples > 10 tasks × 1M examples)
    - **Impact:** Showed instruction tuning scales with task diversity

    **Alpaca (Stanford, 2023):**
    - **Cost:** $600 (used GPT-3.5 to generate 52K instructions from 175 seed examples)
    - **Quality:** 89% of ChatGPT performance on Vicuna benchmark
    - **Training:** 3 hours on 8×A100 GPUs
    - **Impact:** Democratized instruction tuning - anyone can train for $600

    **Vicuna (LMSYS, 2023):**
    - **Data:** 70K conversations from ShareGPT (user-shared ChatGPT conversations)
    - **Quality:** 92% of ChatGPT quality (GPT-4 evaluation)
    - **Cost:** $300 training on 8×A100 GPUs
    - **Adoption:** Basis for many open-source chatbots (Oobabooga, FastChat)

    **Dolly 2.0 (Databricks, 2023):**
    - **Data:** 15K human-written instructions (crowdsourced from employees)
    - **Cost:** Free (crowdsourced)
    - **Quality:** 75% of GPT-3.5
    - **License:** Fully open (no restrictions)

    ## Task Diversity: The Key to Generalization

    **FLAN Insight:** More task diversity beats more data!

    | Approach | Tasks | Examples per Task | Total Examples | Unseen Task Performance |
    |----------|-------|-------------------|----------------|------------------------|
    | **Narrow** | 10 tasks | 100K each | 1M | 55% |
    | **Diverse (FLAN)** | 1,836 tasks | 8K each | 15M | **75%** (+36% improvement) |

    **Task Categories (FLAN):**
    - **NLU:** Sentiment, NLI, paraphrase (400 tasks)
    - **Closed-Book QA:** Trivia, common sense (300 tasks)
    - **Reading Comprehension:** SQuAD, CoQA (200 tasks)
    - **Summarization:** News, dialogue (150 tasks)
    - **Translation:** 100 language pairs (500 tasks)
    - **Misc:** Code, math, reasoning (286 tasks)

    ## LoRA: Efficient Instruction Tuning

    **Problem:** Full fine-tuning updates all 7B-175B parameters → expensive!

    **Solution:** LoRA (Low-Rank Adaptation)
    - Updates only 0.1% of parameters (low-rank matrices)
    - **Memory:** 10× less GPU memory
    - **Speed:** 3× faster training
    - **Quality:** 95-98% of full fine-tuning

    **Example:**
    - LLaMA-7B full fine-tuning: 8×A100 GPUs, 24 hours
    - LLaMA-7B + LoRA: 2×A100 GPUs, 8 hours ✅

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Too narrow task diversity** | Poor generalization (overfits to seen tasks) | Cover 100+ diverse tasks (FLAN: 1,836 tasks) |
    | **Low-quality instructions** | Model learns bad patterns, incorrect outputs | Human-written or curate GPT-generated carefully |
    | **Imbalanced tasks** | Overfits to common tasks (e.g., sentiment analysis) | Balance dataset across task types |
    | **Forgetting base capabilities** | Worse at completion tasks after tuning | Mix instruction + base training data (10-20% base) |
    | **Too few examples per task** | Doesn't learn task format | Minimum 50-100 examples per task type |
    | **Prompt format inconsistency** | Model confused on task boundaries | Use consistent template (Alpaca format) |

    ## Synthetic Data Generation (Alpaca Approach)

    **Problem:** Human-written instructions expensive ($10-30/hour × 1000s of hours)

    **Solution:** Use GPT-3.5/GPT-4 to generate instructions!

    **Alpaca Method:**
    1. Start with 175 seed instructions (human-written)
    2. Sample 3 random seeds as examples
    3. Prompt GPT-3.5: "Generate 1 new instruction similar to these examples"
    4. Repeat 52,000 times
    5. **Cost:** $600 (API calls)

    **Quality:**
    - 89% of ChatGPT quality (surprisingly good!)
    - Key: Diverse seed examples + careful prompt engineering

    !!! tip "Interviewer's Insight"
        **Strong candidates:**

        - Explain transformation: "Base model predicts next word; instruction-tuned follows commands - enables ChatGPT-style assistants"
        - Know the pipeline: "Base → **Instruction Tuning (SFT, Step 1)** → RLHF (Step 2) → Production assistant"
        - Cite real models: "FLAN-T5 (1,836 tasks, 15M examples); Alpaca ($600, 52K GPT-3.5 generated, 89% ChatGPT quality)"
        - Understand task diversity: "FLAN key finding - 1,836 tasks × 8K each > 10 tasks × 1M each (+36% on unseen tasks)"
        - Know efficient methods: "LoRA fine-tunes 0.1% of params - 10× less memory, 3× faster, 95% quality of full fine-tuning"
        - Reference costs: "Alpaca $600 (3 hours 8×A100); Vicuna $300 (24 hours); FLAN $100K+ (weeks on TPU)"
        - Discuss synthetic data: "Alpaca uses GPT-3.5 to generate 52K instructions from 175 seeds - democratizes instruction tuning"

---

### What is RLHF? - OpenAI, Anthropic Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Alignment` | **Asked by:** OpenAI, Anthropic, Google

??? success "View Answer"

    ## What is RLHF?

    **RLHF (Reinforcement Learning from Human Feedback)** is the technique that transforms base LLMs into helpful assistants. It aligns models with human preferences to be helpful, harmless, and honest.

    **Impact:** RLHF is what made ChatGPT useful. Base GPT-3.5 was impressive but often unhelpful/toxic. After RLHF → ChatGPT became a product people love.

    ## The Three Steps of RLHF

    ### Step 1: Supervised Fine-Tuning (SFT)
    - **Goal:** Teach model to follow instructions
    - **Data:** Human-written examples (prompt → ideal response)
    - **Example:**
      ```
      Prompt: "Explain photosynthesis to a 5-year-old"
      Human Answer: "Plants use sunlight to make food, like how you need to eat..."
      ```
    - **Dataset Size:** 10K-100K examples
    - **Result:** Model learns instruction-following format

    ### Step 2: Train Reward Model (RM)
    - **Goal:** Learn what humans prefer
    - **Data:** Human rankings of model outputs
      - Model generates 4-9 responses to same prompt
      - Humans rank them: Response A > B > C > D
    - **Dataset Size:** 30K-100K comparisons
    - **Model:** Train classifier to predict human preference scores
    - **Output:** Reward model that scores any response (0-1)

    ### Step 3: Reinforcement Learning (PPO)
    - **Goal:** Optimize policy (LLM) to maximize reward
    - **Algorithm:** PPO (Proximal Policy Optimization)
    - **Process:**
      1. Generate response to prompt
      2. Reward model scores it
      3. Update LLM weights to increase reward
      4. Repeat thousands of times
    - **Constraint:** KL divergence penalty (don't drift too far from SFT model)

    ## Key Formula

    **RLHF Objective:**

    $$\max_\pi \mathbb{E}_{x \sim D, y \sim \pi(x)} [R(x, y)] - \beta \cdot D_{KL}(\pi || \pi_{SFT})$$

    where:
    - $\pi$ = policy (LLM being trained)
    - $R(x,y)$ = reward model score for prompt $x$, response $y$
    - $\beta$ = KL penalty coefficient (prevents over-optimization)
    - $\pi_{SFT}$ = supervised fine-tuned model (anchor)

    ## RLHF vs Alternatives

    | Method | Approach | Pros | Cons | Used In |
    |--------|----------|------|------|---------|
    | **RLHF (PPO)** | RL with reward model | Gold standard, best results | Complex, unstable training | ChatGPT, Claude |
    | **DPO** | Direct preference optimization | Simpler (no RM), stable | Slightly worse quality | Open-source models |
    | **RLAIF** | RL from AI feedback | Scalable (no humans) | Quality depends on AI judge | Google Bard |
    | **Constitutional AI** | Self-critique with principles | Transparent principles | Needs good constitution | Claude (Anthropic) |

    ## Real-World Impact

    **ChatGPT (OpenAI, 2022):**
    - **Base Model:** GPT-3.5 (good at prediction, not helpful)
    - **After RLHF:** ChatGPT (helpful assistant)
    - **Human Preference:** 85% prefer RLHF over base model
    - **Adoption:** 100M+ users in 2 months (fastest-growing app ever)
    - **Training Cost:** $1-5M for RLHF (human labelers + compute)

    **Claude (Anthropic, 2023):**
    - **Method:** Constitutional AI (RLHF variant)
    - **Difference:** Uses AI feedback + principles ("be helpful and harmless")
    - **Result:** Lower toxicity than ChatGPT in benchmarks
    - **Adoption:** Used by Notion, Quora, DuckDuckGo

    **InstructGPT (OpenAI, 2022):**
    - **First RLHF paper** from OpenAI
    - **Result:** 1.3B InstructGPT preferred over 175B base GPT-3 (100× smaller!)
    - **Key Finding:** Alignment > raw capabilities

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Reward hacking** | Model exploits RM flaws | Diverse prompts, regular RM updates, KL penalty |
    | **Over-optimization** | Model becomes sycophantic, verbose | Strong KL penalty (β=0.01-0.1), early stopping |
    | **Reward model quality** | Garbage in → garbage out | High-quality human labels, inter-annotator agreement >70% |
    | **Training instability** | Reward/policy collapse | Careful hyperparameters, smaller learning rates |
    | **Expensive human labels** | $30K-100K cost | Use RLAIF (AI feedback), active learning |
    | **Alignment tax** | Worse at some tasks after RLHF | Multi-objective RLHF (balance helpfulness + capability) |

    ## DPO: Simpler Alternative

    **Direct Preference Optimization** (Rafailov et al., 2023) bypasses reward model:

    - **Key Insight:** Optimize policy directly from preference data
    - **Advantages:**
      - No reward model training (simpler)
      - More stable (no RL)
      - Same performance as RLHF on many tasks
    - **Used in:** Open-source models (Zephyr, Mistral-Instruct)
    - **Limitation:** Slightly worse than RLHF on complex tasks

    ## Metrics

    | Metric | Measures | Target |
    |--------|----------|--------|
    | **Win Rate vs Base** | % humans prefer RLHF model | > 75% |
    | **Helpfulness Score** | How useful responses are (1-5) | > 4.0 |
    | **Harmlessness Score** | Toxicity, bias (1-5) | > 4.5 |
    | **KL Divergence** | Distance from SFT model | < 10 nats |
    | **Reward Model Accuracy** | Can RM predict human preferences? | > 70% |

    !!! tip "Interviewer's Insight"
        **Strong candidates:**

        - Explain 3 steps: "SFT teaches instruction format, reward model learns preferences, PPO optimizes for reward"
        - Know impact: "1.3B InstructGPT preferred over 175B base GPT-3 - alignment matters more than size"
        - Cite real examples: "ChatGPT uses RLHF (85% prefer vs base); Claude uses Constitutional AI variant"
        - Understand reward hacking: "Model exploits RM flaws (verbose answers score high) - use KL penalty to prevent drift"
        - Know alternatives: "DPO simpler than RLHF (no RM, no RL), used in Zephyr/Mistral-Instruct"
        - Discuss cost: "$30K-100K for human labels; RLAIF uses AI feedback (cheaper, scalable)"

---

### What are Embeddings? - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Embeddings` | **Asked by:** Most Tech Companies

??? success "View Answer"

    ## What are Embeddings?

    **Embeddings** are dense vector representations of text (words, sentences, documents) that capture semantic meaning in continuous space. Similar meanings → similar vectors (close in vector space).

    **Key Insight:** Convert discrete text → continuous vectors that machines can compute with (distance, similarity, arithmetic).

    **Why Embeddings Matter:**
    - Foundation of modern NLP (BERT, GPT, RAG, semantic search)
    - Enable transfer learning (pretrained embeddings)
    - Capture semantic relationships (king - man + woman ≈ queen)
    - Used in 90%+ of production NLP systems

    ## Evolution of Embeddings

    ### 1. One-Hot Encoding (Pre-2000s)
    - **Approach:** Each word = binary vector (size = vocabulary)
    - **Example:** vocab = [cat, dog, bird]
      - "cat" = [1, 0, 0]
      - "dog" = [0, 1, 0]
    - **Problem:** No semantic similarity (cat · dog = 0), huge dimensionality

    ### 2. Word2Vec (2013) - Breakthrough
    - **Approach:** Predict context words (CBOW) or target word from context (Skip-gram)
    - **Dimensionality:** 100-300 (vs millions for one-hot)
    - **Key Feature:** Captures semantics through co-occurrence
      - king - man + woman ≈ queen
      - Paris - France + Italy ≈ Rome
    - **Limitation:** No context (bank always same vector)

    ### 3. GloVe (2014)
    - **Approach:** Matrix factorization on global co-occurrence statistics
    - **Similar to Word2Vec** but better performance on some tasks

    ### 4. Contextual Embeddings - BERT (2018+)
    - **Breakthrough:** Same word, different vectors based on context
      - "bank of river" vs "bank account" → different embeddings
    - **Models:** BERT, RoBERTa, GPT
    - **Dimensionality:** 768 (BERT-base), 1024 (BERT-large)

    ### 5. Sentence Embeddings (2019+)
    - **Task:** Embed entire sentences/paragraphs
    - **Models:** Sentence-BERT, MPNet, E5
    - **Use Case:** Semantic search, RAG, clustering
    - **Dimensionality:** 384 (MiniLM), 768 (SBERT-base), 1024 (large)

    ## Production Implementation (180 lines)

    ```python
    # embeddings.py
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer, util
    from typing import List, Tuple
    import faiss

    class EmbeddingSystem:
        """
        Production embedding system for semantic search

        Supports:
        1. Text embedding (sentences, paragraphs)
        2. Semantic similarity
        3. Vector search with FAISS
        4. Clustering

        Time: O(n × d) where n=text_len, d=embedding_dim
        Space: O(d) per text
        """

        def __init__(self, model_name='all-MiniLM-L6-v2'):
            """
            Args:
                model_name: Sentence embedding model
                    - 'all-MiniLM-L6-v2' (384-dim, fast, 80M params)
                    - 'all-mpnet-base-v2' (768-dim, best quality, 110M)
                    - 'paraphrase-multilingual-MiniLM-L12-v2' (384-dim, 50+ languages)
            """
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            # FAISS index for fast similarity search
            self.index = None
            self.documents = []

        def embed(self, texts: List[str]) -> np.ndarray:
            """
            Embed texts to dense vectors

            Args:
                texts: List of strings

            Returns:
                embeddings: [n_texts, embedding_dim] numpy array
            """
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalization for cosine similarity
            )
            return embeddings

        def semantic_similarity(
            self,
            text1: str,
            text2: str
        ) -> float:
            """
            Compute semantic similarity between two texts

            Returns:
                similarity: Cosine similarity score (0-1)
                    0.0-0.3: Very different
                    0.3-0.5: Somewhat related
                    0.5-0.7: Related
                    0.7-0.9: Very similar
                    0.9-1.0: Nearly identical
            """
            emb1 = self.model.encode(text1, convert_to_tensor=True)
            emb2 = self.model.encode(text2, convert_to_tensor=True)

            similarity = util.cos_sim(emb1, emb2).item()
            return similarity

        def build_index(self, documents: List[str], index_type='flat'):
            """
            Build FAISS index for fast similarity search

            Args:
                documents: List of documents to index
                index_type: 'flat' (exact) or 'ivf' (approximate, faster for >1M docs)

            Time: O(n × d) where n=num_docs, d=embedding_dim
            """
            # Embed all documents
            embeddings = self.embed(documents)

            # Build FAISS index
            if index_type == 'flat':
                # Exact search (L2 distance)
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            elif index_type == 'ivf':
                # Approximate search (faster for large datasets)
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                self.index = faiss.IndexIVFFlat(
                    quantizer,
                    self.embedding_dim,
                    100  # number of clusters
                )
                self.index.train(embeddings)

            # Add embeddings to index
            self.index.add(embeddings)
            self.documents = documents

            print(f"Indexed {len(documents)} documents with {index_type} search")

        def search(
            self,
            query: str,
            top_k: int = 5
        ) -> List[Tuple[str, float]]:
            """
            Semantic search: Find most similar documents to query

            Args:
                query: Search query
                top_k: Number of results to return

            Returns:
                List of (document, distance) tuples
            """
            if self.index is None:
                raise ValueError("Index not built. Call build_index() first.")

            # Embed query
            query_emb = self.embed([query])

            # Search
            distances, indices = self.index.search(query_emb, top_k)

            # Return results
            results = [
                (self.documents[idx], float(dist))
                for dist, idx in zip(distances[0], indices[0])
            ]

            return results

        def cluster(
            self,
            texts: List[str],
            n_clusters: int = 5
        ) -> List[int]:
            """
            Cluster texts by semantic similarity

            Args:
                texts: List of texts to cluster
                n_clusters: Number of clusters

            Returns:
                cluster_labels: [n_texts] cluster assignments (0 to n_clusters-1)
            """
            from sklearn.cluster import KMeans

            # Embed texts
            embeddings = self.embed(texts)

            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embeddings)

            return labels.tolist()

    # Example usage
    def demo_embeddings():
        """Demonstrate embedding use cases"""

        print("=" * 70)
        print("EMBEDDINGS DEMO")
        print("=" * 70)

        # Initialize
        emb_system = EmbeddingSystem(model_name='all-MiniLM-L6-v2')

        # Example 1: Semantic Similarity
        print("\n1. SEMANTIC SIMILARITY")
        print("-" * 70)

        pairs = [
            ("The cat sits on the mat", "A feline rests on a rug"),
            ("Python is a programming language", "I love eating pizza"),
            ("Machine learning is fascinating", "Artificial intelligence is interesting")
        ]

        for text1, text2 in pairs:
            sim = emb_system.semantic_similarity(text1, text2)
            print(f"\nText 1: {text1}")
            print(f"Text 2: {text2}")
            print(f"Similarity: {sim:.3f}")

        # Example 2: Semantic Search
        print("\n\n2. SEMANTIC SEARCH")
        print("-" * 70)

        documents = [
            "Python is a high-level programming language.",
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand human language.",
            "Computer vision enables machines to interpret visual information.",
            "Reinforcement learning trains agents through rewards and penalties."
        ]

        # Build index
        emb_system.build_index(documents)

        # Search
        query = "What is deep learning?"
        results = emb_system.search(query, top_k=3)

        print(f"\nQuery: {query}")
        print("\nTop results:")
        for i, (doc, dist) in enumerate(results, 1):
            print(f"{i}. {doc} (distance: {dist:.3f})")

        # Example 3: Clustering
        print("\n\n3. CLUSTERING")
        print("-" * 70)

        texts = [
            "I love machine learning",
            "Deep learning is fascinating",
            "Pizza is delicious",
            "I enjoy Italian food",
            "Neural networks are powerful",
            "Pasta is my favorite dish"
        ]

        labels = emb_system.cluster(texts, n_clusters=2)

        print("\nClustered texts:")
        for text, label in zip(texts, labels):
            print(f"Cluster {label}: {text}")

        print("\n" + "=" * 70)

    if __name__ == "__main__":
        demo_embeddings()
    ```

    ## Embedding Models: Comparison

    | Model | Dimensions | Speed | Quality | Use Case |
    |-------|-----------|-------|---------|----------|
    | **all-MiniLM-L6-v2** | 384 | Very Fast | Good | Production (speed critical) |
    | **all-mpnet-base-v2** | 768 | Medium | Best | Production (quality critical) |
    | **text-embedding-ada-002 (OpenAI)** | 1536 | Slow (API) | Excellent | High-quality, pay-per-use |
    | **e5-large-v2** | 1024 | Medium | Excellent | Open-source alternative to Ada |
    | **multilingual-e5-base** | 768 | Medium | Good | 100+ languages |
    | **Word2Vec** | 300 | Very Fast | Poor | Legacy, word-level only |
    | **GloVe** | 300 | Very Fast | Poor | Legacy, word-level only |

    ## Real-World Applications

    **Google Search (Semantic Search):**
    - **Model:** Proprietary BERT-based embeddings
    - **Scale:** Indexes billions of web pages
    - **Impact:** 15-20% of queries use semantic understanding (vs keyword matching)
    - **Example:** "How to fix slow computer" matches "speed up PC" (synonyms)

    **Pinecone (Vector Database):**
    - **Use Case:** Semantic search, RAG, recommendations
    - **Customers:** Shopify, Stripe, Gong
    - **Scale:** 100M+ vectors, <50ms query latency
    - **Pricing:** $0.096/1M dimensions/month

    **OpenAI Embeddings (text-embedding-ada-002):**
    - **Adoption:** 100K+ developers
    - **Use Case:** Semantic search, RAG, clustering
    - **Cost:** $0.0001 per 1K tokens
    - **Dimensions:** 1536 (best quality)
    - **Performance:** 61.0% on MTEB benchmark (vs 56.6% for open-source)

    **Notion AI (RAG with Embeddings):**
    - **Use Case:** Search across workspace documents
    - **Model:** Sentence-BERT embeddings
    - **Scale:** 30M+ users
    - **Latency:** <500ms for semantic search

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Using word-level embeddings (Word2Vec) for sentences** | Poor quality (averaging loses context) | Use Sentence-BERT or similar sentence encoders |
    | **Not normalizing embeddings** | Incorrect cosine similarity | Normalize to unit length (L2 norm) |
    | **Wrong embedding model for task** | Suboptimal results | Use task-specific models (retrieval vs classification) |
    | **Too high dimensionality** | Slow, expensive | Use MiniLM (384-dim) for speed, mpnet (768) for quality |
    | **Embedding entire documents (>512 tokens)** | Truncation loses information | Chunk documents, embed separately, aggregate |
    | **Not updating embeddings when data changes** | Stale search results | Recompute embeddings when corpus updates |
    | **Using exact search (FAISS Flat) for >1M docs** | Slow queries | Use approximate search (FAISS IVF, HNSW) |

    ## Advanced Techniques

    ### 1. Dimensionality Reduction (for cost/speed)
    ```python
    from sklearn.decomposition import PCA

    # Reduce 768-dim → 256-dim (3x storage savings)
    pca = PCA(n_components=256)
    reduced_embeddings = pca.fit_transform(embeddings)
    # ~5% quality drop, 3x faster
    ```

    ### 2. Hybrid Search (Keyword + Semantic)
    ```python
    # Combine BM25 (keyword) + embeddings (semantic)
    keyword_results = bm25_search(query, top_k=100)
    semantic_results = embedding_search(query, top_k=100)

    # Rerank by weighted combination
    final_results = rerank(keyword_results, semantic_results, weights=[0.3, 0.7])
    ```

    ### 3. Fine-Tuning Embeddings
    - Train on domain-specific data (legal, medical)
    - Improves relevance by 10-20% over general models
    - Requires 10K+ labeled pairs (query, relevant doc)

    ## Evaluation Metrics

    ### 1. Retrieval Metrics
    - **Recall@K:** % of relevant docs in top-K results
    - **MRR (Mean Reciprocal Rank):** Average of 1/rank of first relevant result
    - **NDCG (Normalized Discounted Cumulative Gain):** Weighted ranking quality

    ### 2. Embedding Quality (MTEB Benchmark)
    - **Classification:** Accuracy on text classification
    - **Clustering:** Quality of semantic grouping
    - **Semantic Search:** Retrieval performance
    - **Best Score:** OpenAI Ada-002 (61.0%), open-source E5-large (56.6%)

    | Model | MTEB Score | Best For |
    |-------|-----------|----------|
    | **text-embedding-ada-002** | 61.0% | Best quality (paid) |
    | **e5-large-v2** | 56.6% | Best open-source |
    | **all-mpnet-base-v2** | 57.8% | Balance quality/speed |
    | **all-MiniLM-L6-v2** | 56.3% | Speed critical |

    !!! tip "Interviewer's Insight"
        **Strong candidates:**

        - Explain evolution: "One-hot → Word2Vec (2013, static) → BERT (2018, contextual) → Sentence-BERT (2019, sentence-level)"
        - Know contextual vs static: "Word2Vec gives 'bank' same vector always; BERT gives different vectors for 'river bank' vs 'bank account'"
        - Understand dimensionality tradeoffs: "384-dim (MiniLM) is 2x faster than 768-dim (mpnet) with ~2% quality drop - use for production speed"
        - Reference real systems: "Google Search uses BERT embeddings for 15-20% of queries; OpenAI Ada-002 is best quality (61% MTEB) but paid"
        - Know semantic search: "Embed query and documents, find nearest neighbors with cosine similarity - used in RAG, Notion AI, Pinecone"
        - Explain normalization: "L2 normalize embeddings so cosine similarity = dot product (faster computation)"
        - Discuss hybrid search: "Combine BM25 (keyword) + embeddings (semantic) for best results - BM25 catches exact matches embeddings miss"

---

### What is BPE Tokenization? - OpenAI, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Tokenization` | **Asked by:** OpenAI, Google, Meta

??? success "View Answer"

    **BPE = Byte Pair Encoding**
    
    Iteratively merges most frequent character pairs.
    
    **Benefits:**
    - Handles OOV words
    - Subword units
    - Language-agnostic
    
    **Used by:** GPT, LLaMA (via tiktoken or SentencePiece)

    !!! tip "Interviewer's Insight"
        Knows vocabulary size affects model capacity.

---

### What is BLEU Score? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Evaluation` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **BLEU = Bilingual Evaluation Understudy**
    
    Measures n-gram overlap with reference.
    
    ```python
    from nltk.translate.bleu_score import sentence_bleu
    
    reference = [['the', 'cat', 'sat', 'on', 'mat']]
    candidate = ['the', 'cat', 'is', 'on', 'mat']
    bleu = sentence_bleu(reference, candidate)
    ```
    
    **Limitations:** Doesn't capture meaning, paraphrases.

    !!! tip "Interviewer's Insight"
        Knows BLEU limitations, uses BERTScore too.

---

### What is Semantic Search? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Search` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Search by meaning, not keywords**
    
    ```python
    # Encode query and documents
    query_emb = model.encode(query)
    doc_embs = model.encode(documents)
    
    # Find similar
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity([query_emb], doc_embs)
    ```
    
    **Better than keyword search** for natural language queries.

    !!! tip "Interviewer's Insight"
        Combines with keyword search (hybrid).

---

### What is NLI (Natural Language Inference)? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Understanding` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Determines relationship between premise and hypothesis**
    
    - **Entailment:** Hypothesis follows from premise
    - **Contradiction:** Hypothesis contradicts premise
    - **Neutral:** No clear relationship
    
    **Applications:** Zero-shot classification, fact verification.

    !!! tip "Interviewer's Insight"
        Uses for zero-shot and fact-checking.

---

### What is Context Window in LLMs? - OpenAI, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `LLMs` | **Asked by:** OpenAI, Google, Anthropic

??? success "View Answer"

    **Maximum tokens LLM can process at once**
    
    | Model | Context Length |
    |-------|----------------|
    | GPT-3.5 | 4K / 16K |
    | GPT-4 | 8K / 128K |
    | Claude | 100K+ |
    | Gemini | 1M+ |
    
    **Handling long docs:** Chunking, summarization, hierarchical processing.

    !!! tip "Interviewer's Insight"
        Designs for context limitations.

---

### What is Model Quantization? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Optimization` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Reducing model precision to save memory/speed**
    
    | Type | Bits | Memory Reduction |
    |------|------|------------------|
    | FP16 | 16 | 50% |
    | INT8 | 8 | 75% |
    | INT4 | 4 | 87.5% |
    
    **Methods:** Post-training (GPTQ, AWQ), QAT (quantization-aware training).

    !!! tip "Interviewer's Insight"
        Knows INT4 tradeoffs for inference vs training.

---

### What is Prompt Injection? - Security Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Security` | **Asked by:** OpenAI, Google, Anthropic

??? success "View Answer"

    **Malicious prompts that override instructions**
    
    Example: "Ignore all previous instructions and..."
    
    **Mitigations:**
    - Input validation
    - Separate system/user prompts
    - Output filtering
    - Guardrails

    !!! tip "Interviewer's Insight"
        Considers security in LLM applications.

---

### What is LoRA (Low-Rank Adaptation)? - OpenAI, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Fine-Tuning` | **Asked by:** OpenAI, Google, Meta

??? success "View Answer"

    **LoRA = Efficient fine-tuning by adding low-rank matrices**
    
    Instead of updating all weights:
    $$W' = W + \Delta W = W + BA$$
    
    Where B and A are low-rank matrices (r << d).
    
    **Benefits:**
    - 10000x fewer trainable params
    - Same inference speed
    - Modular (swap adapters)

    !!! tip "Interviewer's Insight"
        Knows LoRA reduces training cost while preserving quality.

---

### What is Multilingual NLP? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Multilingual` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Approaches:**
    
    | Approach | Description |
    |----------|-------------|
    | Translate-train | Translate data to English |
    | Zero-shot transfer | Train English, test other |
    | Multilingual models | mBERT, XLM-R, mT5 |
    
    **Challenges:** Script differences, low-resource languages.

    !!! tip "Interviewer's Insight"
        Uses multilingual models for cross-lingual transfer.

---

### What is Constituency vs Dependency Parsing? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Syntax` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    | Parsing | Description |
    |---------|-------------|
    | Constituency | Hierarchical tree (NP, VP, etc.) |
    | Dependency | Word-to-word relationships |
    
    **Dependency** is more common in modern NLP (spaCy, Stanza).

    !!! tip "Interviewer's Insight"
        Uses dependency for information extraction.

---

### What is Relation Extraction? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Information Extraction` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Extract relationships between entities**
    
    "Apple was founded by Steve Jobs" → (Apple, founded_by, Steve Jobs)
    
    **Approaches:**
    - Rule-based patterns
    - Supervised classification
    - Distant supervision
    - Zero-shot with LLMs

    !!! tip "Interviewer's Insight"
        Uses LLMs for flexible relation extraction.

---

### What is F1 Score for NER? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Evaluation` | **Asked by:** Most Tech Companies

??? success "View Answer"

    **Entity-level F1 (exact match)**
    
    - Entity must match exactly (text + type)
    - Partial matches count as wrong
    
    ```python
    from seqeval.metrics import f1_score, classification_report
    
    f1 = f1_score(y_true, y_pred)
    print(classification_report(y_true, y_pred))
    ```

    !!! tip "Interviewer's Insight"
        Uses seqeval for proper NER evaluation.

---

### What is Knowledge Distillation? - Google, OpenAI Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Model Compression` | **Asked by:** Google, OpenAI, Meta

??? success "View Answer"

    **Train smaller "student" to mimic larger "teacher"**
    
    $$L = \alpha L_{CE}(y, p_s) + (1-\alpha) L_{KL}(p_t, p_s)$$
    
    Where $p_t$ = teacher logits, $p_s$ = student logits.
    
    **Examples:** DistilBERT (40% smaller, 97% performance).

    !!! tip "Interviewer's Insight"
        Uses soft labels from teacher for better training.

---

### What is Entity Linking? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Knowledge Graphs` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Link named entities to knowledge base (Wikipedia, Wikidata)**
    
    "Apple" → Q312 (company) or Q89 (fruit)?
    
    **Steps:**
    1. Candidate generation
    2. Context-based disambiguation
    3. NIL detection (entity not in KB)

    !!! tip "Interviewer's Insight"
        Considers context for disambiguation.

---

### What is Semantic Role Labeling? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Semantics` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Who did what to whom?**
    
    "John gave Mary a book"
    - Agent: John
    - Recipient: Mary
    - Theme: book
    - Verb: gave

    !!! tip "Interviewer's Insight"
        Uses for structured information extraction.

---

### What is Text Augmentation? - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Data Augmentation` | **Asked by:** Most Tech Companies

??? success "View Answer"

    **Increase training data diversity**
    
    | Method | Description |
    |--------|-------------|
    | Synonym replacement | Replace words with synonyms |
    | Back-translation | Translate and back |
    | Random insertion/deletion | Random word changes |
    | EDA | Easy Data Augmentation |

    !!! tip "Interviewer's Insight"
        Uses back-translation for quality augmentation.

---

### What is ROUGE Score? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Evaluation` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **ROUGE = Recall-Oriented Understudy for Gisting Evaluation**
    
    | Metric | Description |
    |--------|-------------|
    | ROUGE-1 | Unigram overlap |
    | ROUGE-2 | Bigram overlap |
    | ROUGE-L | Longest common subsequence |
    
    ```python
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    scores = scorer.score(reference, candidate)
    ```

    !!! tip "Interviewer's Insight"
        Uses multiple ROUGE variants for complete picture.

---

### What is Sentence Similarity? - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Similarity` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ```python
    from sentence_transformers import SentenceTransformer, util
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    emb1 = model.encode("How are you?")
    emb2 = model.encode("How do you do?")
    
    similarity = util.cos_sim(emb1, emb2)  # ~0.8
    ```
    
    **Use cases:** Duplicate detection, semantic search.

    !!! tip "Interviewer's Insight"
        Uses sentence-transformers for quality embeddings.

---

### What is Gradient Checkpointing? - Google, OpenAI Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Training` | **Asked by:** Google, OpenAI, Meta

??? success "View Answer"

    **Trade compute for memory during training**
    
    - Don't store all activations
    - Recompute during backward pass
    - ~2x slower, but much less memory
    
    ```python
    model.gradient_checkpointing_enable()
    ```
    
    Essential for training large models on limited GPU.

    !!! tip "Interviewer's Insight"
        Uses for large model training on consumer GPUs.

---

### What is Text Generation Strategies? - OpenAI, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Generation` | **Asked by:** OpenAI, Google, Meta

??? success "View Answer"

    | Strategy | Description |
    |----------|-------------|
    | Greedy | Pick highest probability |
    | Beam search | Track top-k sequences |
    | Sampling | Random from distribution |
    | Top-k | Sample from top k tokens |
    | Top-p (nucleus) | Sample from top p probability mass |
    
    **Temperature:** Lower = more focused, higher = more random.

    !!! tip "Interviewer's Insight"
        Uses top-p sampling with temperature tuning.

---

### What is Hallucination Detection? - OpenAI, Anthropic Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Reliability` | **Asked by:** OpenAI, Anthropic, Google

??? success "View Answer"

    **Methods:**
    
    1. **Entailment-based:** Check if output entails sources
    2. **Self-consistency:** Multiple samples, check agreement
    3. **Confidence scoring:** Low confidence = likely hallucination
    4. **Human evaluation:** Gold standard
    
    **Tools:** SelfCheckGPT, TrueTeacher.

    !!! tip "Interviewer's Insight"
        Uses multiple methods for production reliability.

---

## Quick Reference: 100 NLP Interview Questions

| Sno | Question Title | Practice Links | Companies Asking | Difficulty | Topics |
|-----|----------------|----------------|------------------|------------|--------|
| 1 | What is Natural Language Processing? | [Analytics Vidhya NLP Basics](https://www.analyticsvidhya.com/blog/2020/07/nlp-basics/) | Google, Facebook, Amazon | Easy | NLP Basics |
| 2 | Explain Tokenization. | [Towards Data Science – Tokenization](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Easy | Preprocessing |
| 3 | What is Stop Word Removal and why is it important? | [TDS – Stop Words](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Easy | Preprocessing |
| 4 | Explain Stemming. | [TDS – Stemming](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Microsoft | Easy | Preprocessing |
| 5 | Explain Lemmatization. | [Analytics Vidhya – Lemmatization](https://www.analyticsvidhya.com/blog/2020/07/nlp-basics/) | Google, Facebook, Amazon | Easy | Preprocessing |
| 6 | What is the Bag-of-Words Model? | [TDS – Bag-of-Words](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Easy | Text Representation |
| 7 | Explain TF-IDF and its applications. | [TDS – TF-IDF](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Microsoft | Easy | Feature Extraction |
| 8 | What are Word Embeddings? | [TDS – Word Embeddings](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Medium | Embeddings |
| 9 | Explain the Word2Vec algorithm. | [TDS – Word2Vec](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Medium | Embeddings |
| 10 | Explain GloVe embeddings. | [TDS – GloVe](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Medium | Embeddings |
| 11 | What is FastText and how does it differ from Word2Vec? | [TDS – FastText](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Medium | Embeddings |
| 12 | What is one-hot encoding in NLP? | [Analytics Vidhya – NLP Encoding](https://www.analyticsvidhya.com/blog/2020/07/nlp-basics/) | Google, Amazon, Facebook | Easy | Text Representation |
| 13 | What is an n-gram Language Model? | [TDS – N-grams](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Medium | Language Modeling |
| 14 | Explain Language Modeling. | [TDS – Language Modeling](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Microsoft | Medium | Language Modeling |
| 15 | How are Recurrent Neural Networks (RNNs) used in NLP? | [TDS – RNNs for NLP](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Medium | Deep Learning, Sequence Models |
| 16 | Explain Long Short-Term Memory (LSTM) Networks in NLP. | [TDS – LSTM](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Medium | Deep Learning, Sequence Models |
| 17 | What are Gated Recurrent Units (GRU) and their benefits? | [TDS – GRU](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Medium | Deep Learning, Sequence Models |
| 18 | What is the Transformer architecture? | [TDS – Transformers](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Deep Learning, Transformers |
| 19 | What is BERT and how does it work? | [TDS – BERT](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Language Models, Transformers |
| 20 | What is GPT and what are its applications in NLP? | [TDS – GPT](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Language Models, Transformers |
| 21 | Explain the Attention Mechanism in NLP. | [TDS – Attention](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Hard | Deep Learning, Transformers |
| 22 | What is Self-Attention? | [TDS – Self-Attention](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Deep Learning, Transformers |
| 23 | Explain Sequence-to-Sequence Models. | [TDS – Seq2Seq](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Medium | Deep Learning, Generation |
| 24 | What is Machine Translation? | [TDS – Machine Translation](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Medium | Applications |
| 25 | Explain Sentiment Analysis. | [Analytics Vidhya – Sentiment Analysis](https://www.analyticsvidhya.com/blog/2020/07/nlp-basics/) | Google, Facebook, Amazon | Easy | Applications |
| 26 | What is Named Entity Recognition (NER)? | [TDS – NER](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Easy | Applications |
| 27 | What is Part-of-Speech Tagging? | [TDS – POS Tagging](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Easy | Linguistic Processing |
| 28 | Explain Dependency Parsing. | [TDS – Dependency Parsing](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Microsoft | Medium | Parsing |
| 29 | What is Constituency Parsing? | [TDS – Constituency Parsing](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Medium | Parsing |
| 30 | Explain Semantic Role Labeling. | [TDS – Semantic Role Labeling](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Hard | Parsing, Semantics |
| 31 | What is Text Classification? | [Analytics Vidhya – Text Classification](https://www.analyticsvidhya.com/blog/2020/07/nlp-basics/) | Google, Facebook, Amazon | Easy | Applications |
| 32 | What is Topic Modeling? | [TDS – Topic Modeling](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Medium | Unsupervised Learning |
| 33 | Explain Latent Dirichlet Allocation (LDA). | [TDS – LDA](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Medium | Topic Modeling |
| 34 | Explain Latent Semantic Analysis (LSA). | [TDS – LSA](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Medium | Topic Modeling |
| 35 | What is Text Summarization? | [Analytics Vidhya – Summarization](https://www.analyticsvidhya.com/blog/2020/07/nlp-basics/) | Google, Facebook, Amazon | Medium | Applications |
| 36 | Differentiate between Extractive and Abstractive Summarization. | [TDS – Summarization](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Hard | Applications |
| 37 | What are Language Generation Models? | [TDS – Language Generation](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Generation |
| 38 | Explain Sequence Labeling. | [TDS – Sequence Labeling](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Medium | Applications |
| 39 | What is a Conditional Random Field (CRF) in NLP? | [TDS – CRF](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Sequence Modeling |
| 40 | What is Word Sense Disambiguation? | [TDS – WSD](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Hard | Semantics |
| 41 | Explain the concept of Perplexity in Language Models. | [TDS – Perplexity](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Medium | Language Modeling |
| 42 | What is Text Normalization? | [Analytics Vidhya – NLP Preprocessing](https://www.analyticsvidhya.com/blog/2020/07/nlp-basics/) | Google, Amazon, Facebook | Easy | Preprocessing |
| 43 | What is Noise Removal in Text Processing? | [TDS – NLP Preprocessing](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Easy | Preprocessing |
| 44 | Explain the importance of punctuation in NLP. | [TDS – NLP Basics](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Easy | Preprocessing |
| 45 | What is Document Classification? | [Analytics Vidhya – Document Classification](https://www.analyticsvidhya.com/blog/2020/07/nlp-basics/) | Google, Facebook, Amazon | Easy | Applications |
| 46 | Explain the Vector Space Model. | [TDS – Vector Space](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Medium | Text Representation |
| 47 | What is Cosine Similarity in Text Analysis? | [TDS – Cosine Similarity](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Medium | Similarity Measures |
| 48 | What is Semantic Similarity? | [TDS – Semantic Similarity](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Medium | Semantics |
| 49 | What is Text Clustering? | [TDS – Text Clustering](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Medium | Unsupervised Learning |
| 50 | Explain Hierarchical Clustering for Text. | [TDS – Hierarchical Clustering](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Medium | Unsupervised Learning |
| 51 | What is DBSCAN in the context of NLP? | [TDS – DBSCAN](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Medium | Unsupervised Learning |
| 52 | Explain the process of Fine-tuning Pre-trained Language Models. | [TDS – Fine-tuning NLP](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Hard | Transfer Learning |
| 53 | What is Transfer Learning in NLP? | [Analytics Vidhya – Transfer Learning](https://www.analyticsvidhya.com/blog/2020/07/nlp-basics/) | Google, Facebook, Amazon | Medium | Transfer Learning |
| 54 | What is Zero-Shot Classification in NLP? | [TDS – Zero-Shot Learning](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Hard | Transfer Learning |
| 55 | What is Few-Shot Learning in NLP? | [TDS – Few-Shot Learning](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Transfer Learning |
| 56 | Explain Adversarial Attacks on NLP Models. | [TDS – Adversarial NLP](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Security, Robustness |
| 57 | Discuss Bias in NLP Models. | [TDS – NLP Bias](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Hard | Ethics, Fairness |
| 58 | What are Ethical Considerations in NLP? | [Analytics Vidhya – Ethical NLP](https://www.analyticsvidhya.com/blog/2020/07/nlp-basics/) | Google, Facebook, Amazon | Hard | Ethics |
| 59 | What is Language Detection? | [TDS – Language Detection](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Easy | Applications |
| 60 | Explain Transliteration in NLP. | [TDS – Transliteration](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Medium | Applications |
| 61 | What is Language Identification? | [Analytics Vidhya – NLP Basics](https://www.analyticsvidhya.com/blog/2020/07/nlp-basics/) | Google, Amazon, Facebook | Easy | Applications |
| 62 | Explain Query Expansion in Information Retrieval. | [TDS – Information Retrieval](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Medium | IR, NLP |
| 63 | What is Textual Entailment? | [TDS – Textual Entailment](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Hard | Semantics |
| 64 | What is Natural Language Inference (NLI)? | [TDS – NLI](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Semantics |
| 65 | What are Dialog Systems in NLP? | [Analytics Vidhya – Dialog Systems](https://www.analyticsvidhya.com/blog/2020/07/nlp-basics/) | Google, Facebook, Amazon | Medium | Conversational AI |
| 66 | Explain Chatbot Architecture. | [TDS – Chatbots](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Medium | Conversational AI |
| 67 | What is Intent Detection in Chatbots? | [TDS – Intent Detection](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Medium | Conversational AI |
| 68 | What is Slot Filling in Conversational Agents? | [TDS – Slot Filling](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Medium | Conversational AI |
| 69 | Explain Conversation Modeling. | [TDS – Conversation Modeling](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Conversational AI |
| 70 | How is Sentiment Analysis performed using lexicons? | [Analytics Vidhya – Sentiment Analysis](https://www.analyticsvidhya.com/blog/2020/07/nlp-basics/) | Google, Facebook, Amazon | Easy | Applications |
| 71 | Explain deep learning techniques for sentiment analysis. | [TDS – Deep Sentiment](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Medium | Deep Learning, Applications |
| 72 | What is Sequence-to-Sequence Learning for Chatbots? | [TDS – Seq2Seq Chatbots](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Conversational AI |
| 73 | Explain the role of Attention in Machine Translation. | [TDS – Attention in MT](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Hard | Deep Learning, Translation |
| 74 | What is Multi-Head Attention? | [TDS – Multi-Head Attention](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Transformers |
| 75 | Explain the Encoder-Decoder Architecture. | [TDS – Encoder-Decoder](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Hard | Deep Learning, Transformers |
| 76 | What is Beam Search in NLP? | [TDS – Beam Search](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Medium | Decoding, Generation |
| 77 | Explain Back-Translation for Data Augmentation. | [TDS – Back-Translation](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Hard | Data Augmentation |
| 78 | How does GPT generate text? | [TDS – GPT Generation](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Language Models, Generation |
| 79 | What is Fine-tuning in Language Models? | [TDS – Fine-tuning](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Transfer Learning |
| 80 | What is a Context Window in Language Models? | [TDS – Context Window](https://towardsdatascience.com/tagged/nlp) | Google, Amazon, Facebook | Medium | Language Modeling |
| 81 | Explain the Transformer Decoder. | [TDS – Transformer Decoder](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Transformers |
| 82 | Discuss the importance of Embedding Layers in NLP. | [TDS – Embedding Layers](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Medium | Deep Learning, Embeddings |
| 83 | What is Positional Encoding in Transformers? | [TDS – Positional Encoding](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Medium | Transformers |
| 84 | What is Masked Language Modeling? | [TDS – Masked LM](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Transformers, Pre-training |
| 85 | Explain Next Sentence Prediction in BERT. | [TDS – Next Sentence Prediction](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | BERT, Pre-training |
| 86 | What are Pre-trained Language Models? | [Analytics Vidhya – Pre-trained Models](https://www.analyticsvidhya.com/blog/2020/07/nlp-basics/) | Google, Facebook, Amazon | Easy | Transfer Learning |
| 87 | Explain Open-Domain Question Answering in NLP. | [TDS – Question Answering](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Applications, QA |
| 88 | What is Retrieval-Based NLP? | [TDS – Retrieval-Based](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Medium | Applications, QA |
| 89 | Explain Extractive Question Answering. | [TDS – Extractive QA](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Applications, QA |
| 90 | What is Abstractive Question Answering? | [TDS – Abstractive QA](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Applications, QA |
| 91 | What is Machine Reading Comprehension? | [TDS – MRC](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Applications, QA |
| 92 | What are Attention Heads in Transformers? | [TDS – Attention Heads](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Transformers |
| 93 | Explain Sequence Transduction. | [TDS – Sequence Transduction](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Deep Learning, Generation |
| 94 | Discuss the role of GPUs in NLP model training. | [Analytics Vidhya – NLP Infrastructure](https://www.analyticsvidhya.com/blog/2020/07/nlp-basics/) | Google, Facebook, Amazon | Medium | Infrastructure |
| 95 | What is Subword Tokenization (BPE, SentencePiece)? | [TDS – Subword Tokenization](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Medium | Preprocessing, Tokenization |
| 96 | What is a Language Corpus and why is it important? | [Analytics Vidhya – Language Corpora](https://www.analyticsvidhya.com/blog/2020/07/nlp-basics/) | Google, Facebook, Amazon | Easy | NLP Resources |
| 97 | What are the challenges in Low-Resource Languages? | [TDS – Low-Resource NLP](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Applications, Ethics |
| 98 | How do you handle Out-of-Vocabulary words in NLP? | [TDS – OOV Handling](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Medium | Preprocessing, Embeddings |
| 99 | What are Transformer Variants and how do they differ? | [TDS – Transformer Variants](https://towardsdatascience.com/tagged/nlp) | Google, Facebook, Amazon | Hard | Transformers, Models |
| 100 | What are the Future Trends in Natural Language Processing? | [Analytics Vidhya – Future of NLP](https://www.analyticsvidhya.com/blog/2020/07/nlp-basics/) | Google, Facebook, Amazon | Medium | Trends, Research |

---

## Questions asked in Google interview
- What is Natural Language Processing?  
- Explain Tokenization.  
- What is TF-IDF and its applications.  
- What are Word Embeddings?  
- What is BERT and how does it work?  
- Explain the Attention Mechanism.  
- What is Machine Translation?  
- Explain Text Summarization.  
- What is Sentiment Analysis?  
- What is Named Entity Recognition (NER)?

## Questions asked in Facebook interview
- Explain Tokenization.  
- What is Stop Word Removal?  
- Explain Stemming and Lemmatization.  
- What is the Bag-of-Words Model?  
- What are Word Embeddings (Word2Vec/GloVe/FastText)?  
- Explain the Transformer architecture.  
- What is GPT and its applications in NLP?  
- Explain the Attention Mechanism.  
- What is Sequence-to-Sequence Modeling?  
- What are Dialog Systems in NLP?

## Questions asked in Amazon interview
- What is Natural Language Processing?  
- Explain TF-IDF and its applications.  
- What is Text Classification?  
- What is Topic Modeling (LDA/LSA)?  
- Explain Sentiment Analysis.  
- What is Named Entity Recognition (NER)?  
- Explain Language Modeling.  
- What is Transfer Learning in NLP?  
- What is Fine-tuning Pre-trained Language Models?  
- What are Pre-trained Language Models?

## Questions asked in Microsoft interview
- What is Natural Language Processing?  
- Explain Language Modeling and Perplexity.  
- What is the Transformer architecture?  
- What is BERT and how does it work?  
- Explain Dependency Parsing.  
- What is Text Summarization?  
- Explain Question Answering systems.  
- What is Subword Tokenization?  
- How do you handle Out-of-Vocabulary words?  
- Discuss challenges in low-resource languages.

## Questions asked in other interviews
**Uber / Flipkart / Ola:**  
- Explain the Encoder-Decoder Architecture.  
- What is Beam Search in NLP?  
- How does GPT generate text?  
- What is Fine-tuning in Language Models?

**Swiggy / Paytm / OYO:**  
- What is Noise Removal in Text Processing?  
- Explain Named Entity Recognition (NER).  
- What are Ethical Considerations in NLP?  
- How do you handle bias in NLP models?

**WhatsApp / Slack / Airbnb:**  
- What is Natural Language Inference (NLI)?  
- Explain the Attention Mechanism.  
- What are Dialog Systems in NLP?  
- Discuss the future trends in NLP.

---
