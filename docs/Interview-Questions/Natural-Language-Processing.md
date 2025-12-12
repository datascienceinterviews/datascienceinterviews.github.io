---
title: Natural Language Processing (NLP) Interview Questions
description: A curated list of 100 Natural Language Processing interview questions for cracking data science interviews
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

    **BERT = Bidirectional Encoder Representations from Transformers**
    
    **Pre-training Objectives:**
    1. **MLM (Masked Language Modeling):** Predict masked tokens (15%)
    2. **NSP (Next Sentence Prediction):** Binary classification
    
    ```python
    from transformers import BertTokenizer, BertModel
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    inputs = tokenizer("Hello world", return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # [batch, seq_len, 768]
    ```
    
    **Use [CLS] token** for classification, **token embeddings** for sequence labeling.

    !!! tip "Interviewer's Insight"
        Knows MLM masking strategy and [CLS]/[SEP] token purposes.

---

### Explain Word Embeddings (Word2Vec, GloVe) - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Embeddings` | **Asked by:** Most Tech Companies

??? success "View Answer"

    **Word2Vec:**
    - **CBOW:** Predict word from context
    - **Skip-gram:** Predict context from word
    
    **GloVe:** Global vectors from co-occurrence matrix.
    
    ```python
    from gensim.models import Word2Vec, KeyedVectors
    
    # Train Word2Vec
    model = Word2Vec(sentences, vector_size=100, window=5)
    
    # Load pre-trained GloVe
    glove = KeyedVectors.load_word2vec_format('glove.txt')
    
    # Analogies: king - man + woman ≈ queen
    model.wv.most_similar(positive=['king', 'woman'], negative=['man'])
    ```

    !!! tip "Interviewer's Insight"
        Understands training objectives and analogy property.

---

### What is TF-IDF? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Feature Extraction` | **Asked by:** Most Tech Companies

??? success "View Answer"

    $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)$$
    
    - **TF:** Term frequency in document
    - **IDF:** Inverse document frequency (rarity across corpus)
    
    ```python
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(documents)
    ```
    
    **Limitation:** Doesn't capture semantics (unlike embeddings).

    !!! tip "Interviewer's Insight"
        Knows when to use TF-IDF vs embeddings.

---

### What is the Attention Mechanism? - Google, OpenAI Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Deep Learning` | **Asked by:** Google, OpenAI, Meta

??? success "View Answer"

    **Purpose:** Allow model to focus on relevant parts of input.
    
    **Types:**
    - **Self-attention:** Query, Key, Value from same sequence
    - **Cross-attention:** Query from decoder, K/V from encoder
    
    **Scaled Dot-Product:**
    ```python
    import torch.nn.functional as F
    
    def attention(Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V)
    ```

    !!! tip "Interviewer's Insight"
        Can implement attention from scratch and explain masking.

---

### Explain Named Entity Recognition (NER) - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Applications` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **NER = Identify and classify named entities (person, org, location, etc.)**
    
    **Approaches:**
    1. **Rule-based:** Regex, gazetteers
    2. **ML:** CRF, HMM
    3. **Deep Learning:** BiLSTM-CRF, BERT-based
    
    ```python
    from transformers import pipeline
    
    ner = pipeline("ner", model="dslim/bert-base-NER")
    results = ner("Apple was founded by Steve Jobs in California")
    # [{'entity': 'B-ORG', 'word': 'Apple'}, ...]
    ```
    
    **BIO Tagging:** B-PERSON, I-PERSON, O

    !!! tip "Interviewer's Insight"
        Knows BIO tagging scheme and CRF layer purpose.

---

### What is Tokenization? Compare Methods - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Preprocessing` | **Asked by:** Most Tech Companies

??? success "View Answer"

    | Method | Description | Example |
    |--------|-------------|---------|
    | Whitespace | Split by spaces | Simple but limited |
    | WordPiece | Subword (BERT) | "playing" → "play" + "##ing" |
    | BPE | Byte-Pair Encoding (GPT) | Merges frequent pairs |
    | SentencePiece | Language-agnostic (T5) | Works without pre-tokenization |
    
    ```python
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.tokenize("unbelievable")  # ['un', '##bel', '##iev', '##able']
    ```

    !!! tip "Interviewer's Insight"
        Knows subword tokenization handles OOV words.

---

### Explain Sentiment Analysis Approaches - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Applications` | **Asked by:** Most Tech Companies

??? success "View Answer"

    **Levels:**
    - Document-level: Overall sentiment
    - Sentence-level: Per-sentence
    - Aspect-based: Sentiment per aspect ("food good, service bad")
    
    **Approaches:**
    1. **Lexicon-based:** VADER, TextBlob
    2. **Traditional ML:** SVM + TF-IDF
    3. **Deep Learning:** BERT fine-tuned
    
    ```python
    from transformers import pipeline
    
    classifier = pipeline("sentiment-analysis")
    result = classifier("I love this product!")
    # [{'label': 'POSITIVE', 'score': 0.999}]
    ```

    !!! tip "Interviewer's Insight"
        Considers aspect-based sentiment for nuanced analysis.

---

### What is GPT? How Does It Differ from BERT? - OpenAI, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Language Models` | **Asked by:** OpenAI, Google, Meta

??? success "View Answer"

    | Aspect | BERT | GPT |
    |--------|------|-----|
    | Architecture | Encoder-only | Decoder-only |
    | Attention | Bidirectional | Causal (left-to-right) |
    | Pre-training | MLM + NSP | Next token prediction |
    | Best for | Classification, NER | Generation, few-shot |
    
    **GPT Training:**
    $$P(w_1, ..., w_n) = \prod_{i=1}^n P(w_i | w_1, ..., w_{i-1})$$
    
    **GPT advantages:** Generative, few-shot learning, emergent abilities.

    !!! tip "Interviewer's Insight"
        Knows architectural differences and when to use each.

---

### Explain Text Summarization - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Applications` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Types:**
    - **Extractive:** Select important sentences
    - **Abstractive:** Generate new sentences
    
    **Extractive approach:**
    ```python
    from sumy.summarizers.lex_rank import LexRankSummarizer
    ```
    
    **Abstractive with T5:**
    ```python
    from transformers import pipeline
    
    summarizer = pipeline("summarization", model="t5-base")
    summary = summarizer(long_text, max_length=100)
    ```
    
    **Metrics:** ROUGE-1, ROUGE-2, ROUGE-L, BERTScore

    !!! tip "Interviewer's Insight"
        Knows ROUGE metrics and extractive vs abstractive tradeoffs.

---

### What is Perplexity? - Google, OpenAI Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Evaluation` | **Asked by:** Google, OpenAI, Meta

??? success "View Answer"

    **Perplexity = Exponentiated average negative log-likelihood**
    
    $$PPL = \exp\left(-\frac{1}{N}\sum_{i=1}^N \log P(w_i | w_{<i})\right)$$
    
    **Interpretation:** Lower is better; average branching factor.
    
    **GPT-2:** ~35 on WebText
    **GPT-3:** ~20-25
    
    **Caveat:** A model can have low perplexity but generate repetitive text.

    !!! tip "Interviewer's Insight"
        Knows perplexity limitations and doesn't rely solely on it.

---

### Explain Sequence-to-Sequence Models - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Deep Learning` | **Asked by:** Most Tech Companies

??? success "View Answer"

    **Encoder-Decoder architecture for sequence transformation:**
    
    - Machine translation
    - Summarization
    - Question answering
    
    ```
    Input → [Encoder] → Context → [Decoder] → Output
    ```
    
    **Attention improvement:** Decoder attends to all encoder states.
    
    **Modern:** Transformer-based (T5, BART, mT5)

    !!! tip "Interviewer's Insight"
        Knows attention solved the bottleneck problem.

---

### What is Fine-Tuning vs Prompt Engineering? - Google, OpenAI Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `LLMs` | **Asked by:** Google, OpenAI, Meta

??? success "View Answer"

    | Approach | When to Use | Pros | Cons |
    |----------|-------------|------|------|
    | Prompt Engineering | Few examples, no training | Fast, cheap | Limited customization |
    | Fine-Tuning | Specific task, many examples | Best performance | Expensive, needs data |
    | RAG | Need current/private data | Grounded | Retrieval latency |
    
    **Prompt Engineering Techniques:**
    - Few-shot examples
    - Chain-of-Thought
    - System prompts

    !!! tip "Interviewer's Insight"
        Chooses approach based on data availability and requirements.

---

### What is RAG (Retrieval-Augmented Generation)? - Google, OpenAI Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `RAG` | **Asked by:** Google, OpenAI, Meta, Amazon

??? success "View Answer"

    **RAG = Retrieve relevant context, then generate**
    
    ```
    Query → [Retriever] → Context → [LLM + Context] → Answer
    ```
    
    **Benefits:**
    - Reduces hallucination
    - Uses up-to-date information
    - Enables citations
    
    **Components:** Document chunking, embeddings, vector store, retriever.

    !!! tip "Interviewer's Insight"
        Knows chunking strategies and evaluation metrics.

---

### Explain Positional Encoding - Google, OpenAI Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Transformers` | **Asked by:** Google, OpenAI, Meta

??? success "View Answer"

    **Purpose:** Transformers have no recurrence, need position info.
    
    **Sinusoidal Encoding:**
    $$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
    $$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$
    
    **Modern:** Learned positional embeddings, RoPE, ALiBi.

    !!! tip "Interviewer's Insight"
        Knows RoPE for extended context lengths.

---

### What is Topic Modeling (LDA)? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Unsupervised` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **LDA = Latent Dirichlet Allocation**
    
    Discovers topics as distributions over words.
    
    ```python
    from sklearn.decomposition import LatentDirichletAllocation
    
    lda = LatentDirichletAllocation(n_components=10)
    topics = lda.fit_transform(tfidf_matrix)
    ```
    
    **Modern alternatives:** BERTopic, Top2Vec.

    !!! tip "Interviewer's Insight"
        Uses BERTopic for better semantic topics.

---

### What is Question Answering? - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Applications` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Types:**
    - **Extractive:** Span from context
    - **Abstractive:** Generated answer
    - **Open-domain:** No given context
    
    ```python
    from transformers import pipeline
    
    qa = pipeline("question-answering")
    result = qa(question="...", context="...")
    ```

    !!! tip "Interviewer's Insight"
        Knows extractive vs abstractive tradeoffs.

---

### What is Text Classification? - Most Tech Companies Interview Question

**Difficulty:** 🟢 Easy | **Tags:** `Classification` | **Asked by:** Most Tech Companies

??? success "View Answer"

    **Common Tasks:**
    - Spam detection
    - Sentiment analysis
    - Intent classification
    - Topic categorization
    
    **Approaches:** TF-IDF + SVM, BERT fine-tuning, SetFit (few-shot).

    !!! tip "Interviewer's Insight"
        Uses appropriate complexity for data size.

---

### What is Zero-Shot Classification? - OpenAI, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Transfer Learning` | **Asked by:** OpenAI, Google, Meta

??? success "View Answer"

    **No task-specific training data needed**
    
    ```python
    from transformers import pipeline
    
    classifier = pipeline("zero-shot-classification")
    result = classifier(
        "I love playing tennis",
        candidate_labels=["sports", "cooking", "travel"]
    )
    ```
    
    **Models:** BART-MNLI, DeBERTa-MNLI.

    !!! tip "Interviewer's Insight"
        Knows NLI-based zero-shot classification.

---

### What is Machine Translation? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Translation` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Evolution:**
    - Rule-based → Statistical → Neural (Seq2Seq + Attention)
    - Modern: Transformers (mT5, NLLB)
    
    **Metrics:** BLEU, chrF, COMET (neural).
    
    **Challenges:** Low-resource languages, domain adaptation.

    !!! tip "Interviewer's Insight"
        Knows BLEU limitations and neural metrics.

---

### What is Dependency Parsing? - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Linguistic` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Analyzes grammatical structure**
    
    ```python
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("The quick brown fox jumps")
    
    for token in doc:
        print(token.text, token.dep_, token.head.text)
    ```
    
    **Applications:** Information extraction, relation extraction.

    !!! tip "Interviewer's Insight"
        Uses for structured information extraction.

---

### What is Word Sense Disambiguation? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Semantics` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Determining which sense of a word is used**
    
    Example: "bank" → financial institution or river bank?
    
    **Approaches:**
    - Knowledge-based (WordNet)
    - Supervised learning
    - Contextual embeddings (BERT naturally handles this)

    !!! tip "Interviewer's Insight"
        Notes BERT embeddings are context-dependent.

---

### What is Coreference Resolution? - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Discourse` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Linking mentions to same entity**
    
    "John went to the store. He bought milk." → He = John
    
    ```python
    import spacy
    import neuralcoref
    
    nlp = spacy.load("en_core_web_sm")
    neuralcoref.add_to_pipe(nlp)
    doc = nlp("John bought milk. He likes it.")
    ```

    !!! tip "Interviewer's Insight"
        Important for document understanding.

---

### What are LLM Hallucinations? - OpenAI, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Reliability` | **Asked by:** OpenAI, Google, Anthropic

??? success "View Answer"

    **LLM generates plausible but factually incorrect text**
    
    **Mitigation:**
    - RAG (grounding)
    - Citations/sources
    - Confidence scoring
    - Self-verification
    - Human-in-the-loop

    !!! tip "Interviewer's Insight"
        Uses multiple strategies for production reliability.

---

### What is Chain-of-Thought Prompting? - OpenAI, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Prompting` | **Asked by:** OpenAI, Google, Anthropic

??? success "View Answer"

    **Asking LLM to show reasoning steps**
    
    ```
    Q: If I have 5 apples and give away 2, how many left?
    
    Let's think step by step:
    1. Start with 5 apples
    2. Give away 2
    3. 5 - 2 = 3
    
    Answer: 3 apples
    ```
    
    Improves reasoning accuracy significantly.

    !!! tip "Interviewer's Insight"
        Uses for complex reasoning tasks.

---

### What is In-Context Learning? - OpenAI, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `LLMs` | **Asked by:** OpenAI, Google, Anthropic

??? success "View Answer"

    **Learning from examples in prompt (no weight updates)**
    
    ```
    Translate English to French:
    "Hello" -> "Bonjour"
    "Goodbye" -> "Au revoir"
    "Thank you" -> ?
    ```
    
    **Types:** Zero-shot, one-shot, few-shot.

    !!! tip "Interviewer's Insight"
        Knows few-shot example selection matters.

---

### What is Instruction Tuning? - OpenAI, Anthropic Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Fine-Tuning` | **Asked by:** OpenAI, Anthropic, Google

??? success "View Answer"

    **Fine-tuning on instruction-following examples**
    
    Training data format:
    ```
    {"instruction": "Summarize the text", 
     "input": "Long text...", 
     "output": "Summary..."}
    ```
    
    **Models:** FLAN, InstructGPT, Alpaca.

    !!! tip "Interviewer's Insight"
        Knows difference from base model training.

---

### What is RLHF? - OpenAI, Anthropic Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Alignment` | **Asked by:** OpenAI, Anthropic, Google

??? success "View Answer"

    **RLHF = Reinforcement Learning from Human Feedback**
    
    **Steps:**
    1. Collect human preferences
    2. Train reward model
    3. Fine-tune LLM with PPO
    
    **Purpose:** Align LLM to be helpful, harmless, honest.

    !!! tip "Interviewer's Insight"
        Knows RLHF alternatives: DPO, Constitutional AI.

---

### What are Embeddings? - Most Tech Companies Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Embeddings` | **Asked by:** Most Tech Companies

??? success "View Answer"

    **Dense vector representations capturing semantics**
    
    ```python
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode("Hello world")  # 384-dim vector
    ```
    
    **Use cases:** Semantic search, clustering, RAG.

    !!! tip "Interviewer's Insight"
        Chooses embedding model for specific task.

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
