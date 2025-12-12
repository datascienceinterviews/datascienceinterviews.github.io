---
title: ML System Design Interview Questions
description: 50+ ML system design interview questions - recommendation systems, real-time ML pipelines, feature stores, model serving, A/B testing infrastructure for senior ML engineer roles.
---

# System Design Interview Questions (DS & ML)

<!-- ![Total Questions](https://img.shields.io/badge/Total%20Questions-0-blue?style=flat&labelColor=black&color=blue)
![Unanswered Questions](https://img.shields.io/badge/Unanswered%20Questions-0-blue?style=flat&labelColor=black&color=yellow)
![Answered Questions](https://img.shields.io/badge/Answered%20Questions-0-blue?style=flat&labelColor=black&color=success) -->


This document provides a curated list of system design questions tailored for Data Science and Machine Learning interviews. The questions focus on designing scalable, robust, and maintainable systems—from end-to-end ML pipelines and data ingestion frameworks to model serving, monitoring, and MLOps architectures. Use the practice links provided to dive deeper into each topic.

---

## Premium Interview Questions

### Design a Recommendation System - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `ML Systems`, `Recommendations` | **Asked by:** Google, Amazon, Netflix, Meta

??? success "View Answer"

    **Scale Requirements:**
    - **Users:** 100M+ daily active users
    - **Items:** 10M+ products/content
    - **Latency:** <50ms p99
    - **Throughput:** 1M+ QPS
    - **Personalization:** Real-time signals

    **Detailed Architecture:**

    ```
    ┌─────────────┐
    │ User Activity│ (clicks, views, purchases, time spent)
    └──────┬──────┘
           ↓
    ┌─────────────────────────────────────────┐
    │        Feature Engineering              │
    │  - Real-time: last 1hr behavior         │
    │  - Batch: 7d/30d aggregates             │
    │  - User profile: demographics, history  │
    │  - Context: time, device, location      │
    └──────┬──────────────────────────────────┘
           ↓
    ┌─────────────────────────────────────────┐
    │     Candidate Generation (Retrieval)    │
    │  ┌────────────────────────────────────┐ │
    │  │ 1. Collaborative Filtering (ALS)   │ │ → 1000 candidates
    │  │ 2. Content-based (embeddings)      │ │
    │  │ 3. Trending/Popular items          │ │
    │  │ 4. Graph-based (item2item)         │ │
    │  └────────────────────────────────────┘ │
    └──────┬──────────────────────────────────┘
           ↓
    ┌─────────────────────────────────────────┐
    │          Ranking (Scoring)              │
    │  Two-Tower Neural Network               │
    │  - User tower: user embeddings          │
    │  - Item tower: item embeddings          │
    │  - Features: 100+ features              │
    │  - Model: DLRM, DCN, DeepFM             │
    └──────┬──────────────────────────────────┘ → Top 100
           ↓
    ┌─────────────────────────────────────────┐
    │         Re-ranking (Filtering)          │
    │  - Diversity: avoid similar items       │
    │  - Business rules: inventory, policies  │
    │  - Explore/exploit: Thompson sampling   │
    │  - Deduplication                        │
    └──────┬──────────────────────────────────┘ → Top 20
           ↓
    ┌─────────────┐
    │   Results   │
    └─────────────┘
    ```

    **Implementation Details:**

    ```python
    class RecommendationSystem:
        def __init__(self):
            self.feature_store = FeatureStore()
            self.candidate_gen = CandidateGenerator()
            self.ranker = TwoTowerRanker()
            self.reranker = Reranker()

        def get_recommendations(self, user_id: str, context: dict) -> List[str]:
            # 1. Feature retrieval (<10ms)
            user_features = self.feature_store.get_user_features(user_id)
            context_features = self._extract_context(context)

            # 2. Candidate generation (<20ms)
            # Retrieve ~1000 candidates from multiple sources
            cf_candidates = self.candidate_gen.collaborative_filter(user_id, k=500)
            content_candidates = self.candidate_gen.content_based(user_features, k=300)
            trending = self.candidate_gen.get_trending(k=200)

            all_candidates = set(cf_candidates + content_candidates + trending)

            # 3. Ranking (<15ms)
            # Score all candidates with neural network
            candidate_features = self.feature_store.get_item_features(all_candidates)
            scores = self.ranker.predict(user_features, candidate_features, context_features)

            top_100 = sorted(zip(all_candidates, scores), key=lambda x: x[1], reverse=True)[:100]

            # 4. Re-ranking (<5ms)
            # Apply business rules and diversification
            final_recs = self.reranker.rerank(
                candidates=top_100,
                diversity_weight=0.3,
                explore_rate=0.1
            )

            return [item_id for item_id, _ in final_recs[:20]]

    # Candidate Generation with ANN
    class CandidateGenerator:
        def collaborative_filter(self, user_id: str, k: int) -> List[str]:
            """Use Approximate Nearest Neighbors for fast retrieval"""
            user_embedding = self.get_user_embedding(user_id)  # 128-dim vector

            # HNSW index for fast ANN search
            # Search through 10M items in <5ms
            similar_items = self.ann_index.search(user_embedding, k=k)
            return similar_items

    # Two-Tower Ranking Model
    class TwoTowerRanker:
        def __init__(self):
            self.user_tower = UserTower(input_dim=200, output_dim=128)
            self.item_tower = ItemTower(input_dim=150, output_dim=128)

        def predict(self, user_feats, item_feats, context_feats):
            user_emb = self.user_tower(user_feats)
            item_emb = self.item_tower(item_feats)

            # Dot product for scoring
            scores = torch.matmul(user_emb, item_emb.T)
            return scores
    ```

    **Key Components Deep Dive:**

    | Component | Technology | Scale | Purpose |
    |-----------|-----------|-------|---------|
    | **Feature Store** | Redis, DynamoDB | <5ms p99 | Real-time feature serving |
    | **ANN Index** | FAISS, ScaNN | 10M vectors | Fast similarity search |
    | **Ranking Model** | TensorFlow Serving | 5ms inference | Score candidates |
    | **A/B Testing** | Custom platform | 1000+ concurrent tests | Online evaluation |
    | **Monitoring** | Prometheus, Grafana | Real-time | Track metrics |

    **Cold Start Solutions:**

    ```python
    def handle_cold_start(user_id: str, user_data: dict):
        """Strategies for new users/items"""

        # New User:
        if is_new_user(user_id):
            # 1. Use demographic-based recommendations
            recs = get_popular_for_demographic(user_data['age'], user_data['location'])

            # 2. Quick onboarding survey
            preferences = get_user_preferences(user_id)
            recs += content_based_on_preferences(preferences)

            # 3. Thompson sampling for exploration
            recs += explore_diverse_content(explore_rate=0.5)

        # New Item:
        if is_new_item(item_id):
            # 1. Content-based: use item metadata
            similar_items = find_similar_by_content(item_id)

            # 2. Cold start boost in ranking
            boost_score = 0.1  # Temporary boost

            # 3. Show to exploratory users first
            target_users = get_early_adopter_users()
    ```

    **Metrics & Evaluation:**

    | Metric Category | Examples | Target |
    |----------------|----------|--------|
    | **Online Metrics** | CTR, Conversion, Watch time | CTR: 5-15% |
    | **Engagement** | Session length, Return rate | +10% retention |
    | **Business** | Revenue, GMV | +5% revenue |
    | **Diversity** | ILS (Intra-list similarity) | ILS < 0.7 |
    | **Freshness** | Avg item age | <3 days |
    | **Serendipity** | Unexpected but relevant | 20% of recs |

    **Common Pitfalls:**

    ❌ **Filter bubble:** Showing only similar items → Add diversity
    ❌ **Popularity bias:** Always recommending popular items → Balance with personalization
    ❌ **Position bias:** Higher positions get more clicks → Debias training data
    ❌ **Feedback loop:** Model reinforces itself → Use exploration
    ❌ **Recency bias:** Only recent items → Balance with evergreen content

    **Trade-offs:**

    | Aspect | Option A | Option B | Netflix's Choice |
    |--------|----------|----------|------------------|
    | Candidate Gen | Collaborative Filter | Deep Learning | Both (ensemble) |
    | Ranking | LightGBM | Neural Network | Neural (DLRM) |
    | Serving | CPU | GPU | CPU for latency |
    | Update Freq | Real-time | Batch (daily) | Near real-time (hourly) |

    **Real-World Examples:**

    - **Netflix:** 80% of watch time from recommendations, saves $1B/year in retention
    - **Amazon:** 35% of revenue from recommendations
    - **YouTube:** 70% of watch time from recommendations
    - **Spotify:** Discover Weekly has 40M+ active users

    !!! tip "Interviewer's Insight"
        **What they're testing:** Multi-stage architecture understanding, cold-start problem, scale considerations.

        **Strong answer signals:**
        - Explains funnel approach (1000 → 100 → 20)
        - Discusses latency budget breakdown
        - Knows specific algorithms (ALS, FAISS, Two-Tower)
        - Addresses cold-start for both users and items
        - Mentions diversity/exploration tradeoffs
        - Talks about position bias and debiasing
        - Discusses A/B testing challenges (novelty effect, network effects)

---

### Design a Real-Time Fraud Detection System - Amazon, PayPal Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Real-Time`, `Anomaly Detection` | **Asked by:** Amazon, PayPal, Stripe

??? success "View Answer"

    **Scale Requirements:**
    - **Transactions:** 10M+ per day (115 TPS, 1000+ peak)
    - **Latency:** <100ms p99 (to not block checkout)
    - **False Positive Rate:** <1% (user experience)
    - **Fraud Catch Rate:** >80% (business requirement)
    - **Data Volume:** 1TB+ transaction data/day

    **Detailed Architecture:**

    ```
    ┌──────────────┐
    │ Transaction  │ (amount, merchant, location, device, etc.)
    └──────┬───────┘
           ↓
    ┌─────────────────────────────────────────┐
    │      Kafka Stream (partitioned)         │
    │   - Partition by user_id for ordering   │
    │   - Retention: 7 days for replay        │
    └──────┬──────────────────────────────────┘
           ↓
    ┌─────────────────────────────────────────┐
    │    Real-Time Feature Engineering        │
    │  (Flink / Spark Streaming)              │
    │                                         │
    │  1. Velocity Features:                  │
    │     - Transactions last 5/30/60 min     │
    │     - Amount spent last 1 hour          │
    │     - Unique merchants last 24h         │
    │                                         │
    │  2. Anomaly Features:                   │
    │     - Unusual location (>500km from     │
    │       last transaction)                 │
    │     - New device fingerprint            │
    │     - Unusual time (3am for daytime user)│
    │                                         │
    │  3. Network Features:                   │
    │     - Merchant risk score               │
    │     - IP reputation                     │
    │     - Email/phone shared with fraudsters│
    └──────┬──────────────────────────────────┘
           ↓
    ┌─────────────────────────────────────────┐
    │         Feature Store Lookup            │
    │   Online:  Redis (1-5ms)                │
    │   Batch:   Cassandra/BigQuery           │
    │                                         │
    │   - User historical patterns            │
    │   - Device fingerprints                 │
    │   - Merchant metadata                   │
    └──────┬──────────────────────────────────┘
           ↓
    ┌─────────────────────────────────────────┐
    │         Multi-Layer Detection           │
    │                                         │
    │  Layer 1: Rule Engine (<10ms)           │
    │   ├─ Blacklist check                    │
    │   ├─ Amount thresholds                  │
    │   └─ Basic velocity rules               │
    │   → Block: 5% of fraud                  │
    │                                         │
    │  Layer 2: ML Model (<50ms)              │
    │   ├─ Gradient Boosting (XGBoost)        │
    │   ├─ Features: 200+                     │
    │   └─ Score: 0-1 fraud probability       │
    │   → Catch: 70% of fraud                 │
    │                                         │
    │  Layer 3: Deep Learning (<80ms)         │
    │   ├─ LSTM for sequence modeling         │
    │   ├─ Graph Neural Network               │
    │   └─ Catches complex patterns           │
    │   → Catch additional: 10% of fraud      │
    └──────┬──────────────────────────────────┘
           ↓
    ┌─────────────────────────────────────────┐
    │         Decision Logic                  │
    │                                         │
    │  if score > 0.9:                        │
    │      → BLOCK (hard decline)             │
    │  elif score > 0.7:                      │
    │      → CHALLENGE (2FA, 3DS)             │
    │  elif score > 0.5:                      │
    │      → REVIEW (async manual review)     │
    │  else:                                  │
    │      → APPROVE                           │
    └──────┬──────────────────────────────────┘
           ↓
    ┌─────────────────────────────────────────┐
    │       Feedback Loop & Labeling          │
    │   - User disputes (chargebacks)         │
    │   - Manual review decisions             │
    │   - Confirmed fraud cases               │
    │   → Retrain models weekly               │
    └─────────────────────────────────────────┘
    ```

    **Implementation:**

    ```python
    class FraudDetectionSystem:
        def __init__(self):
            self.rule_engine = RuleEngine()
            self.ml_model = load_model('xgboost_v23.pkl')
            self.deep_model = load_model('lstm_v5.pt')
            self.feature_store = FeatureStore()
            self.decision_thresholds = {
                'block': 0.9,
                'challenge': 0.7,
                'review': 0.5
            }

        async def detect_fraud(self, transaction: dict) -> dict:
            start_time = time.time()

            # Step 1: Quick rule check (<5ms)
            rule_result = self.rule_engine.check(transaction)
            if rule_result['action'] == 'BLOCK':
                return {
                    'decision': 'BLOCK',
                    'reason': rule_result['reason'],
                    'latency_ms': (time.time() - start_time) * 1000
                }

            # Step 2: Feature engineering (parallel)
            features = await asyncio.gather(
                self._compute_realtime_features(transaction),
                self._fetch_historical_features(transaction['user_id']),
                self._fetch_merchant_features(transaction['merchant_id'])
            )
            feature_vector = self._combine_features(*features)  # 200+ features

            # Step 3: ML scoring (<30ms)
            ml_score = self.ml_model.predict_proba(feature_vector)[0][1]

            # Step 4: Deep learning (only for borderline cases)
            if 0.4 < ml_score < 0.8:
                # Get transaction sequence for user
                sequence = await self._get_transaction_sequence(transaction['user_id'])
                dl_score = self.deep_model.predict(sequence)
                final_score = 0.6 * ml_score + 0.4 * dl_score
            else:
                final_score = ml_score

            # Step 5: Make decision
            decision = self._make_decision(final_score)

            # Step 6: Log for monitoring
            self._log_decision(transaction, final_score, decision)

            return {
                'decision': decision,
                'score': final_score,
                'latency_ms': (time.time() - start_time) * 1000
            }

        def _make_decision(self, score: float) -> str:
            if score > self.decision_thresholds['block']:
                return 'BLOCK'
            elif score > self.decision_thresholds['challenge']:
                return 'CHALLENGE'  # Ask for 2FA
            elif score > self.decision_thresholds['review']:
                return 'REVIEW'  # Manual review queue
            else:
                return 'APPROVE'

    # Real-time Feature Engineering
    class RealtimeFeatureEngine:
        def compute_velocity_features(self, user_id: str) -> dict:
            """Compute velocity over different time windows"""
            now = time.time()

            # Count transactions in time windows
            txns_5min = redis_client.zcount(f'txn:{user_id}', now - 300, now)
            txns_30min = redis_client.zcount(f'txn:{user_id}', now - 1800, now)
            txns_1hour = redis_client.zcount(f'txn:{user_id}', now - 3600, now)

            # Amount velocity
            amounts_1hour = redis_client.zrangebyscore(
                f'amt:{user_id}', now - 3600, now
            )
            total_amount_1hour = sum(float(a) for a in amounts_1hour)

            return {
                'txn_count_5min': txns_5min,
                'txn_count_30min': txns_30min,
                'txn_count_1hour': txns_1hour,
                'total_amount_1hour': total_amount_1hour,
                'avg_amount_1hour': total_amount_1hour / max(txns_1hour, 1)
            }

        def compute_anomaly_features(self, transaction: dict, user_profile: dict) -> dict:
            """Detect anomalies based on user history"""
            features = {}

            # Location anomaly
            last_location = user_profile.get('last_location')
            curr_location = (transaction['lat'], transaction['lon'])
            if last_location:
                distance_km = haversine_distance(last_location, curr_location)
                time_diff_hours = (transaction['timestamp'] - user_profile['last_txn_time']) / 3600
                features['distance_from_last'] = distance_km
                features['impossible_travel'] = 1 if distance_km > 1000 and time_diff_hours < 2 else 0

            # Amount anomaly (Z-score)
            avg_amount = user_profile.get('avg_transaction_amount', 100)
            std_amount = user_profile.get('std_transaction_amount', 50)
            features['amount_zscore'] = (transaction['amount'] - avg_amount) / std_amount

            # Time anomaly
            typical_hours = user_profile.get('typical_transaction_hours', [9, 10, 11, 14, 15, 16])
            current_hour = datetime.fromtimestamp(transaction['timestamp']).hour
            features['unusual_time'] = 1 if current_hour not in typical_hours else 0

            return features
    ```

    **Feature Engineering Details:**

    | Feature Type | Examples | Window | Storage |
    |--------------|----------|--------|---------|
    | **Velocity** | Transaction count, amount sum | 5min, 30min, 1h, 24h | Redis sorted sets |
    | **Anomaly** | Distance from last txn, unusual time | Real-time | Computed on-the-fly |
    | **Historical** | Avg transaction amount, preferred merchants | 30d, 90d | Cassandra |
    | **Network** | IP reputation, email risk score | Updated daily | PostgreSQL |
    | **Behavioral** | Spending pattern, transaction sequence | 90d | Feature store |

    **Model Architecture:**

    ```python
    # XGBoost Model (Primary)
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=10,  # Handle class imbalance (1:10 fraud:legit)
        eval_metric='auc'
    )

    # Features: 200+
    feature_groups = {
        'transaction': 20,      # amount, merchant, category
        'velocity': 30,         # counts and amounts over time windows
        'anomaly': 15,          # deviations from user profile
        'network': 40,          # IP, device, email risk
        'behavioral': 50,       # spending patterns
        'merchant': 25,         # merchant risk, category
        'temporal': 20          # time-based features
    }

    # LSTM for Sequential Modeling
    class FraudLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=50, hidden_size=128, num_layers=2, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        def forward(self, sequence):
            # sequence: [batch, seq_len, 50 features]
            lstm_out, _ = self.lstm(sequence)
            last_hidden = lstm_out[:, -1, :]  # Take last timestep
            return self.fc(last_hidden)
    ```

    **Decision Threshold Tuning:**

    | Threshold | FPR | Fraud Catch Rate | Business Impact |
    |-----------|-----|------------------|-----------------|
    | 0.95 | 0.1% | 50% | Block $10M fraud, lose $1M revenue |
    | 0.90 | 0.5% | 70% | Block $14M fraud, lose $5M revenue |
    | 0.85 | 1.0% | 80% | Block $16M fraud, lose $10M revenue |
    | 0.80 | 2.0% | 85% | Block $17M fraud, lose $20M revenue |

    **Common Pitfalls:**

    ❌ **Class imbalance:** Fraud is 0.1-1% of transactions → Use SMOTE, class weights
    ❌ **Data leakage:** Using future information → Strict point-in-time features
    ❌ **Concept drift:** Fraud patterns change weekly → Retrain frequently
    ❌ **False positives:** Blocking good customers → Tune thresholds carefully
    ❌ **Label delay:** Chargebacks take 30-60 days → Use confirmed fraud + disputes

    **Real-World Numbers (Stripe, PayPal):**

    - **Fraud rate:** 0.5-1.5% of transactions
    - **Chargeback cost:** $20-50 per transaction (fees + lost goods)
    - **False positive cost:** Lost revenue + customer churn
    - **Detection latency:** 50-100ms typical
    - **Model update frequency:** Weekly to daily
    - **Feature count:** 100-500 features

    **Monitoring & Alerting:**

    ```python
    # Key metrics to monitor
    metrics = {
        'fraud_catch_rate': 0.80,  # Alert if drops below 75%
        'false_positive_rate': 0.01,  # Alert if exceeds 1.5%
        'p99_latency_ms': 100,  # Alert if exceeds 150ms
        'model_score_distribution': None,  # Alert on significant shift
        'feature_null_rate': 0.02,  # Alert if exceeds 5%
        'data_drift_psi': 0.15  # Alert if PSI > 0.25
    }
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Real-time ML systems, feature engineering under latency constraints, handling class imbalance.

        **Strong answer signals:**
        - Multi-layer defense (rules + ML + DL)
        - Discusses velocity features and time windows
        - Addresses cold start (new users, new merchants)
        - Talks about false positive cost vs fraud cost tradeoff
        - Mentions feedback loop and model retraining
        - Explains how to handle label delay (chargebacks)
        - Discusses A/B testing challenges (can't show fraud to users!)

---

### Design an ML Feature Store - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `MLOps`, `Infrastructure` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Scale Requirements:**
    - **Features:** 10,000+ features across 100+ ML models
    - **Online Serving:** <5ms p99 latency
    - **Throughput:** 1M+ feature requests/second
    - **Training Data:** Petabyte-scale offline feature retrieval
    - **Freshness:** Real-time features (<1 min latency)

    **Detailed Architecture:**

    ```
    ┌──────────────────────────────────────────────────────┐
    │              Feature Definition Layer                 │
    │  - Python SDK for defining features                   │
    │  - Schema validation and type checking                │
    │  - Version control integration                        │
    └──────────────┬───────────────────────────────────────┘
                   ↓
    ┌──────────────────────────────────────────────────────┐
    │           Feature Computation Layer                   │
    │                                                       │
    │  Batch (Spark/Dask):          Streaming (Flink):     │
    │  - Daily aggregates            - Real-time counts     │
    │  - Historical features         - Windowed aggregates  │
    │  - Complex transformations     - Event-driven updates │
    └──────────────┬───────────────────────────────────────┘
                   ↓
    ┌──────────────────────────────────────────────────────┐
    │              Feature Storage Layer                    │
    │                                                       │
    │  ┌────────────────────┐   ┌─────────────────────┐   │
    │  │   Online Store     │   │   Offline Store     │   │
    │  │  (Low Latency)     │   │   (Training Data)   │   │
    │  │                    │   │                     │   │
    │  │ Redis/DynamoDB     │   │ S3/BigQuery/Delta  │   │
    │  │ - Key-value lookup │   │ - Point-in-time    │   │
    │  │ - <5ms p99         │   │   joins            │   │
    │  │ - Hot features     │   │ - Historical data  │   │
    │  └────────────────────┘   └─────────────────────┘   │
    └──────────────┬───────────────────────────────────────┘
                   ↓
    ┌──────────────────────────────────────────────────────┐
    │            Feature Registry (Metadata)                │
    │  - Schema & types                                     │
    │  - Lineage (data sources → features → models)        │
    │  - Statistics (min, max, missing %)                  │
    │  - Access control & governance                        │
    └───────────────────────────────────────────────────────┘
    ```

    **Implementation:**

    ```python
    from datetime import datetime, timedelta
    from typing import List, Dict
    import redis
    import pandas as pd

    # Feature Definition
    class FeatureStore:
        def __init__(self):
            self.online_store = redis.Redis(host='localhost', port=6379)
            self.offline_store = BigQueryClient()
            self.registry = FeatureRegistry()

        # Define a feature
        @feature(
            name="user_purchase_count_7d",
            entity="user",
            value_type=ValueType.INT64,
            ttl=timedelta(days=7),
            online=True,
            offline=True
        )
        def user_purchase_count_7d(self, user_id: str, timestamp: datetime) -> int:
            """Count user purchases in last 7 days"""
            start_date = timestamp - timedelta(days=7)

            # For batch/training (point-in-time correct)
            if self.context == "offline":
                query = f"""
                SELECT user_id, COUNT(*) as purchase_count
                FROM purchases
                WHERE user_id = '{user_id}'
                  AND purchase_timestamp >= '{start_date}'
                  AND purchase_timestamp < '{timestamp}'
                GROUP BY user_id
                """
                return self.offline_store.query(query)

            # For online serving (real-time)
            else:
                # Pre-computed and cached in Redis
                key = f"user:{user_id}:purchase_count_7d"
                return int(self.online_store.get(key) or 0)

        # Get features for online serving
        def get_online_features(
            self,
            entity_rows: List[Dict],  # e.g., [{"user_id": "123"}, ...]
            feature_refs: List[str]   # e.g., ["user_purchase_count_7d", ...]
        ) -> pd.DataFrame:
            """
            Fast batch retrieval for inference
            Target latency: <5ms for 10 features
            """
            results = []

            # Parallel Redis MGET for performance
            pipeline = self.online_store.pipeline()

            for row in entity_rows:
                entity_key = f"user:{row['user_id']}"
                for feature in feature_refs:
                    key = f"{entity_key}:{feature}"
                    pipeline.get(key)

            # Execute all at once
            values = pipeline.execute()

            # Parse results
            idx = 0
            for row in entity_rows:
                feature_dict = {"user_id": row["user_id"]}
                for feature in feature_refs:
                    feature_dict[feature] = values[idx]
                    idx += 1
                results.append(feature_dict)

            return pd.DataFrame(results)

        # Get features for training (point-in-time correct)
        def get_historical_features(
            self,
            entity_df: pd.DataFrame,  # user_id, timestamp
            feature_refs: List[str]
        ) -> pd.DataFrame:
            """
            Point-in-time correct joins for training data
            Prevents data leakage
            """
            # Generate SQL with point-in-time joins
            query = self._build_pit_query(entity_df, feature_refs)

            # Execute on data warehouse
            result = self.offline_store.query(query)

            return result

        def _build_pit_query(self, entity_df, features):
            """
            Build SQL for point-in-time correct feature retrieval

            Example: If training data point is at 2024-01-15,
            only use features computed from data BEFORE 2024-01-15
            """
            base_query = """
            WITH entity_timestamps AS (
                SELECT user_id, event_timestamp
                FROM training_events
            )
            """

            # For each feature, join with timestamp constraint
            for feature in features:
                base_query += f"""
                LEFT JOIN LATERAL (
                    SELECT {feature}
                    FROM feature_values_{feature}
                    WHERE entity_id = entity_timestamps.user_id
                      AND feature_timestamp <= entity_timestamps.event_timestamp
                    ORDER BY feature_timestamp DESC
                    LIMIT 1
                ) AS {feature}_values ON TRUE
                """

            return base_query

    # Batch Feature Computation (Spark)
    class BatchFeatureCompute:
        def compute_daily_features(self, date: datetime):
            """Run daily to compute batch features"""

            # Example: Compute user purchase count for all users
            query = """
            SELECT
                user_id,
                COUNT(*) as purchase_count_7d,
                SUM(amount) as total_spent_7d,
                AVG(amount) as avg_order_value_7d
            FROM purchases
            WHERE purchase_date BETWEEN {date - 7d} AND {date}
            GROUP BY user_id
            """

            df = spark.sql(query)

            # Write to both stores
            self._write_to_online_store(df)
            self._write_to_offline_store(df, date)

        def _write_to_online_store(self, df: DataFrame):
            """Write to Redis for low-latency serving"""
            # Batch write to Redis
            pipeline = redis_client.pipeline()

            for row in df.collect():
                key = f"user:{row.user_id}:purchase_count_7d"
                pipeline.set(key, row.purchase_count_7d, ex=7*24*3600)  # 7 day TTL

            pipeline.execute()

        def _write_to_offline_store(self, df: DataFrame, date: datetime):
            """Write to data warehouse for training"""
            # Append to partitioned table
            df.write.partitionBy("date").mode("append").saveAsTable(
                "feature_store.user_features"
            )

    # Streaming Feature Computation (Flink)
    class StreamingFeatureCompute:
        def process_realtime_event(self, event: dict):
            """Process events in real-time (Kafka → Flink → Redis)"""
            user_id = event['user_id']

            # Update velocity features
            current_count = redis_client.get(f"user:{user_id}:txn_count_1hr") or 0
            redis_client.incr(f"user:{user_id}:txn_count_1hr")
            redis_client.expire(f"user:{user_id}:txn_count_1hr", 3600)

            # Update windowed aggregates
            redis_client.zadd(
                f"user:{user_id}:recent_purchases",
                {event['purchase_id']: event['timestamp']}
            )

            # Remove old events outside window
            cutoff = time.time() - 3600
            redis_client.zremrangebyscore(
                f"user:{user_id}:recent_purchases",
                0,
                cutoff
            )
    ```

    **Key Components Deep Dive:**

    | Component | Technology | Purpose | Scale |
    |-----------|-----------|---------|-------|
    | **Online Store** | Redis Cluster | Real-time serving | <5ms p99, 1M QPS |
    | **Offline Store** | BigQuery/Delta Lake | Training data | PB-scale, point-in-time joins |
    | **Registry** | PostgreSQL | Metadata & lineage | 10K+ features |
    | **Batch Compute** | Spark | Daily aggregates | Process TB data |
    | **Stream Compute** | Flink/Spark Streaming | Real-time updates | 100K events/sec |
    | **Feature SDK** | Python | Define features | Type-safe, versioned |

    **Point-in-Time Correctness:**

    ```python
    # WRONG: Data leakage - using future information
    def get_features_WRONG(user_id, prediction_timestamp):
        # This query looks at ALL data, including future data!
        return db.query(f"""
            SELECT AVG(purchase_amount)
            FROM purchases
            WHERE user_id = '{user_id}'
        """)

    # CORRECT: Point-in-time join
    def get_features_CORRECT(user_id, prediction_timestamp):
        # Only use data from BEFORE prediction time
        return db.query(f"""
            SELECT AVG(purchase_amount)
            FROM purchases
            WHERE user_id = '{user_id}'
              AND purchase_timestamp < '{prediction_timestamp}'
        """)
    ```

    **Feature Freshness Trade-offs:**

    | Feature Type | Computation | Latency | Use Case |
    |--------------|-------------|---------|----------|
    | **Batch** | Daily Spark job | 24 hours | Historical patterns |
    | **Mini-batch** | Hourly job | 1 hour | Near real-time |
    | **Streaming** | Flink/Kafka | <1 minute | Velocity features |
    | **On-demand** | Computed at request | <5ms | Session features |

    **Common Pitfalls:**

    ❌ **Data leakage:** Not using point-in-time joins → Wrong model performance
    ❌ **Train-serve skew:** Different feature computation in training vs serving
    ❌ **Missing features:** No handling for entities without features → Model errors
    ❌ **Stale features:** Not monitoring feature freshness → Degraded predictions
    ❌ **Schema changes:** Breaking changes to feature definitions → Production errors

    **Monitoring:**

    ```python
    # Key metrics
    feature_metrics = {
        'online_latency_p99_ms': 5,
        'online_error_rate': 0.001,
        'feature_null_rate': {
            'user_purchase_count_7d': 0.02,  # 2% nulls acceptable
            'user_age': 0.10  # 10% nulls (optional feature)
        },
        'feature_staleness_minutes': {
            'batch_features': 24 * 60,  # Daily
            'streaming_features': 5      # 5 min max
        },
        'train_serve_skew': 0.05  # Feature distributions should match
    }
    ```

    **Real-World Examples:**

    - **Uber:** Michelangelo feature store, 10K+ features, serves 100M+ predictions/day
    - **Airbnb:** Zipline feature store, reduces feature engineering from weeks to days
    - **DoorDash:** Feature store reduced model development time by 50%
    - **Netflix:** Feature store serves 1B+ feature requests/day

    **Tools Comparison:**

    | Tool | Pros | Cons | Best For |
    |------|------|------|----------|
    | **Feast** | Open-source, flexible | Limited UI | Custom deployments |
    | **Tecton** | Enterprise, managed | Expensive | Large orgs |
    | **Vertex AI** | GCP integrated | Vendor lock-in | GCP users |
    | **SageMaker** | AWS integrated | Limited features | AWS users |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of train-serve consistency, point-in-time correctness, scaling challenges.

        **Strong answer signals:**
        - Explains point-in-time joins for preventing data leakage
        - Discusses online vs offline stores with specific latency numbers
        - Mentions feature freshness and staleness monitoring
        - Knows about train-serve skew and how to detect it
        - Talks about feature versioning and backward compatibility
        - Discusses feature sharing across teams and governance

---

### Design a Model Serving System - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Deployment`, `Serving` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Scale Requirements:**
    - **Throughput:** 100K+ requests/second
    - **Latency:** <50ms p99 (< 10ms for simple models)
    - **Models:** 100+ models simultaneously
    - **GPU Utilization:** >70% (expensive hardware)
    - **Availability:** 99.99% uptime

    **Detailed Architecture:**

    ```
    ┌─────────────────────────────────────────┐
    │          API Gateway / Load Balancer    │
    │  - Rate limiting (1000 QPS/user)        │
    │  - Authentication & authorization        │
    │  - Traffic routing by model version     │
    └──────────────┬──────────────────────────┘
                   ↓
    ┌─────────────────────────────────────────┐
    │         Model Server Fleet              │
    │  (Kubernetes pods with auto-scaling)    │
    │                                         │
    │  ┌────────────┐  ┌────────────┐       │
    │  │  Server 1  │  │  Server 2  │  ...  │
    │  │  CPU/GPU   │  │  CPU/GPU   │       │
    │  │            │  │            │       │
    │  │ Model A v1 │  │ Model A v2 │       │
    │  │ Model B    │  │ Model C    │       │
    │  └────────────┘  └────────────┘       │
    └──────────────┬──────────────────────────┘
                   ↓
    ┌─────────────────────────────────────────┐
    │          Optimization Layer             │
    │  - Request batching (collect 10-100ms)  │
    │  - Result caching (Redis)               │
    │  - Feature caching                      │
    └──────────────┬──────────────────────────┘
                   ↓
    ┌─────────────────────────────────────────┐
    │         Model Registry & Storage        │
    │  - S3/GCS: Model artifacts              │
    │  - Versioning & metadata                │
    │  - Lazy loading / preloading            │
    └─────────────────────────────────────────┘

    Monitoring:
    ┌─────────────────────────────────────────┐
    │  Prometheus + Grafana + Alerts          │
    │  - Latency (p50, p95, p99)              │
    │  - Throughput (QPS)                     │
    │  - GPU utilization                      │
    │  - Model drift                          │
    │  - Error rates                          │
    └─────────────────────────────────────────┘
    ```

    **Implementation:**

    ```python
    from fastapi import FastAPI, HTTPException
    from typing import List, Dict
    import torch
    import numpy as np
    import asyncio
    from collections import defaultdict
    import time

    app = FastAPI()

    class ModelServer:
        def __init__(self):
            self.models = {}  # model_name -> model
            self.batchers = {}  # model_name -> RequestBatcher
            self.cache = RedisCache()
            self.metrics = PrometheusMetrics()

        async def load_model(self, model_name: str, version: str):
            """Load model from registry"""
            # Download from S3/GCS
            model_path = f"s3://models/{model_name}/{version}/model.pt"

            if torch.cuda.is_available():
                device = torch.device("cuda")
                model = torch.load(model_path, map_location=device)
                model = torch.jit.script(model)  # TorchScript for optimization
            else:
                device = torch.device("cpu")
                model = torch.load(model_path, map_location=device)
                # Quantize for CPU inference
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )

            model.eval()
            self.models[model_name] = model
            self.batchers[model_name] = RequestBatcher(max_batch_size=32, max_wait_ms=50)

            print(f"Loaded {model_name} v{version} on {device}")

        @app.post("/predict/{model_name}")
        async def predict(self, model_name: str, features: Dict):
            """
            Prediction endpoint with batching and caching
            """
            start_time = time.time()

            # Step 1: Check cache
            cache_key = self._compute_cache_key(model_name, features)
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                self.metrics.increment("cache_hit", model=model_name)
                return {"prediction": cached_result, "cached": True}

            # Step 2: Add to batch
            future = asyncio.Future()
            await self.batchers[model_name].add_request(features, future)

            # Wait for batch processing
            prediction = await future

            # Step 3: Cache result
            await self.cache.set(cache_key, prediction, ttl=3600)

            latency_ms = (time.time() - start_time) * 1000
            self.metrics.observe("prediction_latency", latency_ms, model=model_name)

            return {"prediction": prediction, "cached": False}

    class RequestBatcher:
        """
        Batch requests for GPU efficiency
        Trade-off: Slight latency increase for much higher throughput
        """
        def __init__(self, max_batch_size=32, max_wait_ms=50):
            self.max_batch_size = max_batch_size
            self.max_wait_ms = max_wait_ms
            self.queue = []
            self.processing = False

        async def add_request(self, features, future):
            """Add request to batch queue"""
            self.queue.append((features, future))

            # Start batch processing if not already running
            if not self.processing:
                asyncio.create_task(self._process_batch())

            # Or if queue is full
            if len(self.queue) >= self.max_batch_size:
                asyncio.create_task(self._process_batch())

        async def _process_batch(self):
            """Process accumulated requests as batch"""
            if self.processing or len(self.queue) == 0:
                return

            self.processing = True

            # Wait for more requests (up to max_wait_ms)
            await asyncio.sleep(self.max_wait_ms / 1000)

            # Get batch
            batch_size = min(len(self.queue), self.max_batch_size)
            batch = self.queue[:batch_size]
            self.queue = self.queue[batch_size:]

            # Prepare batch tensor
            features_list = [item[0] for item in batch]
            futures = [item[1] for item in batch]

            # Convert to tensor
            batch_tensor = torch.tensor(
                np.array([self._features_to_array(f) for f in features_list])
            )

            # Run inference
            with torch.no_grad():
                predictions = model(batch_tensor)

            # Return results to individual futures
            for i, future in enumerate(futures):
                future.set_result(predictions[i].item())

            self.processing = False

            # Process remaining queue if any
            if len(self.queue) > 0:
                asyncio.create_task(self._process_batch())

    # GPU Optimization
    class GPUOptimizedServer:
        """Optimize for GPU serving"""

        def __init__(self):
            self.model = None
            self.use_amp = True  # Automatic Mixed Precision

        def load_optimized_model(self, model_path: str):
            """Load model with optimizations"""

            # TensorRT optimization (NVIDIA)
            import torch_tensorrt

            model = torch.load(model_path)

            # Compile with TensorRT
            trt_model = torch_tensorrt.compile(
                model,
                inputs=[torch_tensorrt.Input(shape=[1, 784])],
                enabled_precisions={torch.float16},  # FP16 for speed
                workspace_size=1 << 30  # 1GB
            )

            self.model = trt_model

        @torch.cuda.amp.autocast()  # Mixed precision
        def predict(self, batch_tensor):
            """Inference with AMP"""
            with torch.no_grad():
                return self.model(batch_tensor)

    # A/B Testing Support
    class ABTestingServer:
        """Route traffic to different model versions"""

        def __init__(self):
            self.model_versions = {
                'model_a': {'v1': 0.9, 'v2': 0.1},  # 90% v1, 10% v2
                'model_b': {'v1': 0.5, 'v2': 0.5}   # 50/50 split
            }

        def get_model_version(self, model_name: str, user_id: str) -> str:
            """Deterministic assignment based on user_id"""
            import hashlib

            # Hash user_id to get consistent assignment
            hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            bucket = (hash_value % 100) / 100.0

            # Assign to version based on bucket
            cumulative = 0
            for version, weight in self.model_versions[model_name].items():
                cumulative += weight
                if bucket < cumulative:
                    return version

            return 'v1'  # Default

    # Auto-scaling based on metrics
    class AutoScaler:
        """Scale model servers based on load"""

        def should_scale_up(self, metrics):
            """Decide if we need more servers"""
            conditions = [
                metrics['cpu_usage'] > 80,
                metrics['gpu_usage'] > 85,
                metrics['p99_latency_ms'] > 100,
                metrics['queue_size'] > 1000
            ]

            return any(conditions)

        def should_scale_down(self, metrics):
            """Decide if we can reduce servers"""
            conditions = [
                metrics['cpu_usage'] < 30,
                metrics['gpu_usage'] < 30,
                metrics['p99_latency_ms'] < 20,
                metrics['queue_size'] < 100
            ]

            return all(conditions)
    ```

    **Latency Optimization Techniques:**

    | Technique | Latency Gain | Throughput Gain | Trade-off |
    |-----------|--------------|-----------------|-----------|
    | **Request Batching** | +10-50ms | 5-10x | Latency vs throughput |
    | **Model Quantization** | 2-4x faster | 2-4x | Slight accuracy drop |
    | **TensorRT/ONNX** | 2-5x faster | 2-5x | Hardware specific |
    | **Result Caching** | 10-100x faster | 10-100x | Staleness |
    | **Feature Caching** | 5-20ms saved | N/A | Memory usage |
    | **Mixed Precision (FP16)** | 2-3x faster | 2-3x | GPU only |

    **Model Format Comparison:**

    | Format | Speed | Portability | Use Case |
    |--------|-------|-------------|----------|
    | **PyTorch (.pt)** | Baseline | Python only | Development |
    | **TorchScript** | 1.5-2x | Python/C++ | Production (PyTorch) |
    | **ONNX** | 2-3x | Any framework | Cross-platform |
    | **TensorRT** | 3-5x | NVIDIA GPU only | High-performance GPU |
    | **Quantized INT8** | 3-4x (CPU) | CPU optimized | Edge/mobile |

    **Common Pitfalls:**

    ❌ **Cold start:** Model loading takes 10-30s → Warm pools, lazy loading
    ❌ **GPU underutilization:** <50% utilization → Use batching, shared GPUs
    ❌ **Memory leaks:** OOM after hours → Proper cleanup, monitoring
    ❌ **Version conflicts:** Model dependencies clash → Containerization
    ❌ **No graceful degradation:** Model unavailable → Fallback to simpler model

    **Monitoring Dashboard:**

    ```python
    # Key metrics to track
    serving_metrics = {
        'latency_p50_ms': 10,
        'latency_p95_ms': 30,
        'latency_p99_ms': 50,
        'qps': 10000,
        'error_rate': 0.001,
        'gpu_utilization_%': 75,
        'gpu_memory_used_gb': 10,
        'batch_size_avg': 24,
        'cache_hit_rate': 0.30,
        'model_load_time_s': 15
    }

    # Alerts
    alerts = {
        'p99_latency > 100ms': 'High latency',
        'error_rate > 0.01': 'High error rate',
        'gpu_util < 40%': 'Underutilized GPU',
        'qps drops > 50%': 'Traffic drop'
    }
    ```

    **Real-World Examples:**

    - **Google:** TensorFlow Serving handles billions of predictions/day
    - **Amazon:** SageMaker serves models with auto-scaling, multi-model endpoints
    - **Uber:** Michelangelo serves 100M+ predictions/day with <10ms p99
    - **Netflix:** Serves 1000+ models for recommendations, <50ms latency

    **Deployment Patterns:**

    | Pattern | Pros | Cons | Use Case |
    |---------|------|------|----------|
    | **Single model per server** | Simple, isolated | Expensive | High-value models |
    | **Multi-model per server** | Cost-effective | Resource contention | Many small models |
    | **Serverless (Lambda)** | No management | Cold start, limited | Infrequent inference |
    | **Edge deployment** | Low latency, offline | Limited compute | Mobile apps |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of GPU optimization, batching strategies, latency vs throughput trade-offs.

        **Strong answer signals:**
        - Discusses dynamic batching with specific wait times
        - Knows model optimization formats (TensorRT, ONNX, quantization)
        - Mentions A/B testing for model versions
        - Talks about GPU utilization and multi-model serving
        - Discusses graceful degradation and fallback strategies
        - Knows about cold start problem and solutions

---

### Design a Model Monitoring System - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `MLOps`, `Monitoring` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Scale Requirements:**
    - **Models Monitored:** 100+ models in production
    - **Predictions:** 1B+ predictions/day
    - **Monitoring Frequency:** Real-time (streaming) + batch (daily)
    - **Alert Latency:** <5 minutes for critical issues
    - **Data Retention:** 90 days detailed, 1 year aggregated

    **Detailed Architecture:**

    ```
    ┌────────────────────────────────────────────────┐
    │         Production Predictions                 │
    │  (Model serving logs every prediction)         │
    └──────────────┬─────────────────────────────────┘
                   ↓
    ┌────────────────────────────────────────────────┐
    │          Streaming Pipeline (Kafka)            │
    │  - Prediction logs                             │
    │  - Features used                               │
    │  - Model version                               │
    │  - Latency, errors                             │
    └──────────────┬─────────────────────────────────┘
                   ↓
    ┌────────────────────────────────────────────────┐
    │         Real-Time Monitoring Layer             │
    │  (Flink/Spark Streaming)                       │
    │                                                │
    │  1. Data Quality Checks:                       │
    │     - Schema validation                        │
    │     - Missing value detection                  │
    │     - Range/distribution checks                │
    │                                                │
    │  2. Data Drift Detection:                      │
    │     - PSI (Population Stability Index)         │
    │     - KL Divergence                            │
    │     - Kolmogorov-Smirnov test                  │
    │                                                │
    │  3. Performance Monitoring:                    │
    │     - Latency (p50, p95, p99)                  │
    │     - Throughput (QPS)                         │
    │     - Error rates                              │
    └──────────────┬─────────────────────────────────┘
                   ↓
    ┌────────────────────────────────────────────────┐
    │         Ground Truth Collection                │
    │  (Delayed labels via user feedback)            │
    │  - User clicks/conversions                     │
    │  - Manual labels                               │
    │  - Downstream outcomes                         │
    └──────────────┬─────────────────────────────────┘
                   ↓
    ┌────────────────────────────────────────────────┐
    │         Model Performance Analysis             │
    │  (Daily batch jobs)                            │
    │                                                │
    │  - Accuracy, Precision, Recall                 │
    │  - AUC, F1 score                               │
    │  - Per-segment performance                     │
    │  - Calibration metrics                         │
    └──────────────┬─────────────────────────────────┘
                   ↓
    ┌────────────────────────────────────────────────┐
    │        Alerting & Visualization                │
    │                                                │
    │  - Prometheus + Grafana dashboards             │
    │  - PagerDuty alerts                            │
    │  - Weekly performance reports                  │
    └────────────────────────────────────────────────┘
    ```

    **Implementation:**

    ```python
    import numpy as np
    from scipy import stats
    from typing import Dict, List
    import pandas as pd

    class ModelMonitor:
        def __init__(self, model_name: str):
            self.model_name = model_name
            self.baseline_stats = self._load_baseline_stats()
            self.alert_thresholds = {
                'psi': 0.2,
                'kl_divergence': 0.1,
                'accuracy_drop': 0.05,
                'p99_latency_ms': 100,
                'error_rate': 0.01
            }

        # 1. DATA QUALITY MONITORING
        def check_data_quality(self, batch: pd.DataFrame) -> Dict:
            """Real-time data quality checks"""
            issues = []

            # Schema validation
            expected_cols = set(self.baseline_stats['feature_names'])
            actual_cols = set(batch.columns)
            if expected_cols != actual_cols:
                issues.append({
                    'type': 'SCHEMA_DRIFT',
                    'severity': 'CRITICAL',
                    'message': f'Missing columns: {expected_cols - actual_cols}'
                })

            # Missing values
            missing_pct = batch.isnull().sum() / len(batch)
            high_missing = missing_pct[missing_pct > 0.1]
            if len(high_missing) > 0:
                issues.append({
                    'type': 'HIGH_MISSING_VALUES',
                    'severity': 'WARNING',
                    'features': high_missing.to_dict()
                })

            # Range validation
            for col in batch.select_dtypes(include=[np.number]).columns:
                baseline_min = self.baseline_stats['ranges'][col]['min']
                baseline_max = self.baseline_stats['ranges'][col]['max']

                current_min = batch[col].min()
                current_max = batch[col].max()

                if current_min < baseline_min * 0.5 or current_max > baseline_max * 2:
                    issues.append({
                        'type': 'OUT_OF_RANGE',
                        'severity': 'WARNING',
                        'feature': col,
                        'baseline': f'[{baseline_min}, {baseline_max}]',
                        'current': f'[{current_min}, {current_max}]'
                    })

            return {'issues': issues, 'passed': len(issues) == 0}

        # 2. DATA DRIFT DETECTION
        def detect_data_drift(self, current_data: pd.DataFrame) -> Dict:
            """Detect feature distribution drift"""
            drift_results = {}

            for feature in current_data.columns:
                if feature in self.baseline_stats['distributions']:
                    # PSI (Population Stability Index)
                    psi = self._calculate_psi(
                        self.baseline_stats['distributions'][feature],
                        current_data[feature]
                    )

                    # KL Divergence
                    kl_div = self._calculate_kl_divergence(
                        self.baseline_stats['distributions'][feature],
                        current_data[feature]
                    )

                    # Kolmogorov-Smirnov test
                    ks_stat, ks_pvalue = stats.ks_2samp(
                        self.baseline_stats['distributions'][feature],
                        current_data[feature]
                    )

                    drift_results[feature] = {
                        'psi': psi,
                        'kl_divergence': kl_div,
                        'ks_statistic': ks_stat,
                        'ks_pvalue': ks_pvalue,
                        'drifted': psi > self.alert_thresholds['psi']
                    }

            return drift_results

        def _calculate_psi(self, baseline: np.ndarray, current: np.ndarray, bins=10) -> float:
            """
            Population Stability Index
            PSI < 0.1: No significant drift
            0.1 < PSI < 0.2: Moderate drift
            PSI > 0.2: Significant drift
            """
            # Create bins from baseline
            breakpoints = np.percentile(baseline, np.linspace(0, 100, bins + 1))
            breakpoints[-1] += 0.001  # Include max value

            # Calculate distributions
            baseline_counts = np.histogram(baseline, bins=breakpoints)[0]
            current_counts = np.histogram(current, bins=breakpoints)[0]

            # Convert to percentages
            baseline_pct = baseline_counts / len(baseline)
            current_pct = current_counts / len(current)

            # Avoid division by zero
            baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
            current_pct = np.where(current_pct == 0, 0.0001, current_pct)

            # PSI formula
            psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
            return psi

        def _calculate_kl_divergence(self, baseline: np.ndarray, current: np.ndarray, bins=50) -> float:
            """KL Divergence: D_KL(P||Q)"""
            # Create histograms
            hist_range = (min(baseline.min(), current.min()),
                         max(baseline.max(), current.max()))

            p, _ = np.histogram(baseline, bins=bins, range=hist_range, density=True)
            q, _ = np.histogram(current, bins=bins, range=hist_range, density=True)

            # Normalize and avoid zeros
            p = p / p.sum()
            q = q / q.sum()
            p = np.where(p == 0, 1e-10, p)
            q = np.where(q == 0, 1e-10, q)

            # KL divergence
            kl = np.sum(p * np.log(p / q))
            return kl

        # 3. MODEL PERFORMANCE MONITORING
        def monitor_model_performance(
            self,
            predictions: np.ndarray,
            actuals: np.ndarray,
            prediction_times: List[float]
        ) -> Dict:
            """Monitor model accuracy and performance"""
            from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

            metrics = {}

            # Classification metrics (if labels available)
            if actuals is not None:
                metrics['accuracy'] = accuracy_score(actuals, predictions > 0.5)
                metrics['auc'] = roc_auc_score(actuals, predictions)

                precision, recall, f1, _ = precision_recall_fscore_support(
                    actuals, predictions > 0.5, average='binary'
                )
                metrics['precision'] = precision
                metrics['recall'] = recall
                metrics['f1'] = f1

                # Check for degradation
                baseline_accuracy = self.baseline_stats['accuracy']
                if metrics['accuracy'] < baseline_accuracy - self.alert_thresholds['accuracy_drop']:
                    self._trigger_alert({
                        'type': 'ACCURACY_DROP',
                        'severity': 'CRITICAL',
                        'baseline': baseline_accuracy,
                        'current': metrics['accuracy'],
                        'drop': baseline_accuracy - metrics['accuracy']
                    })

            # Latency monitoring
            latency_p50 = np.percentile(prediction_times, 50)
            latency_p95 = np.percentile(prediction_times, 95)
            latency_p99 = np.percentile(prediction_times, 99)

            metrics['latency_ms'] = {
                'p50': latency_p50,
                'p95': latency_p95,
                'p99': latency_p99
            }

            if latency_p99 > self.alert_thresholds['p99_latency_ms']:
                self._trigger_alert({
                    'type': 'HIGH_LATENCY',
                    'severity': 'WARNING',
                    'p99_latency': latency_p99,
                    'threshold': self.alert_thresholds['p99_latency_ms']
                })

            return metrics

        # 4. PREDICTION DRIFT (MODEL OUTPUT DISTRIBUTION)
        def monitor_prediction_drift(self, predictions: np.ndarray) -> Dict:
            """Check if prediction distribution has changed"""
            # For classification: check score distribution
            score_buckets = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            current_dist = np.histogram(predictions, bins=score_buckets)[0]
            current_dist = current_dist / current_dist.sum()

            baseline_dist = self.baseline_stats['prediction_distribution']

            # Chi-square test
            chi_stat, p_value = stats.chisquare(current_dist, baseline_dist)

            return {
                'chi_square_statistic': chi_stat,
                'p_value': p_value,
                'drifted': p_value < 0.05,  # Significant at 5% level
                'current_distribution': current_dist.tolist(),
                'baseline_distribution': baseline_dist.tolist()
            }

        # 5. BUSINESS METRICS MONITORING
        def monitor_business_metrics(self, predictions: pd.DataFrame, outcomes: pd.DataFrame) -> Dict:
            """Monitor business impact"""
            # Example: For a recommendation system
            metrics = {
                'ctr': outcomes['clicked'].mean(),
                'conversion_rate': outcomes['converted'].mean(),
                'revenue_per_impression': outcomes['revenue'].mean(),
                'engagement_time': outcomes['time_spent'].mean()
            }

            # Compare with baseline
            for metric, value in metrics.items():
                baseline = self.baseline_stats['business_metrics'][metric]
                change_pct = (value - baseline) / baseline * 100

                if abs(change_pct) > 10:  # 10% change threshold
                    self._trigger_alert({
                        'type': 'BUSINESS_METRIC_CHANGE',
                        'severity': 'WARNING',
                        'metric': metric,
                        'baseline': baseline,
                        'current': value,
                        'change_pct': change_pct
                    })

            return metrics

        def _trigger_alert(self, alert: Dict):
            """Send alert to monitoring system"""
            print(f"🚨 ALERT: {alert['type']} - {alert['severity']}")
            # Send to PagerDuty, Slack, etc.
            self._send_to_pagerduty(alert)
            self._send_to_slack(alert)
    ```

    **Monitoring Dashboard Metrics:**

    | Category | Metrics | Frequency | Alert Threshold |
    |----------|---------|-----------|-----------------|
    | **Data Quality** | Missing %, Schema drift | Real-time | Missing > 10% |
    | **Data Drift** | PSI, KL divergence | Hourly | PSI > 0.2 |
    | **Model Performance** | Accuracy, AUC, F1 | Daily | Accuracy drop > 5% |
    | **Latency** | p50, p95, p99 | Real-time | p99 > 100ms |
    | **Throughput** | QPS, Requests/day | Real-time | Drop > 20% |
    | **Business Metrics** | CTR, Conversion, Revenue | Daily | Change > 10% |
    | **Prediction Drift** | Score distribution | Daily | Chi-square p < 0.05 |
    | **Error Rate** | 4xx, 5xx errors | Real-time | Error rate > 1% |

    **Drift Detection Thresholds:**

    ```python
    drift_severity = {
        'psi': {
            'low': (0, 0.1),      # No action needed
            'medium': (0.1, 0.2),  # Investigate
            'high': (0.2, float('inf'))  # Retrain model
        },
        'kl_divergence': {
            'low': (0, 0.05),
            'medium': (0.05, 0.1),
            'high': (0.1, float('inf'))
        }
    }
    ```

    **Common Pitfalls:**

    ❌ **No ground truth collection:** Can't measure accuracy → Implement feedback loops
    ❌ **Alert fatigue:** Too many false alerts → Tune thresholds carefully
    ❌ **Only monitoring overall metrics:** Masked subgroup degradation → Monitor per-segment
    ❌ **Ignoring business metrics:** Technical metrics don't capture value → Track CTR, revenue
    ❌ **No automated response:** Manual investigation is slow → Auto-trigger retraining

    **Real-World Examples:**

    - **Uber:** Monitors 1000+ models, detects drift within 1 hour, auto-triggers retraining
    - **Netflix:** Per-title model monitoring, catches regional content drift
    - **Airbnb:** Monitors search ranking models, detects seasonal drift automatically
    - **Stripe:** Real-time fraud model monitoring, <5 min alert latency

    **Automated Remediation:**

    ```python
    class AutoRemediation:
        def handle_drift(self, drift_severity: str):
            """Automated response to drift"""
            if drift_severity == 'high':
                # Trigger model retraining
                self.trigger_retraining_pipeline()

                # Meanwhile, rollback to previous version
                self.rollback_model_version()

            elif drift_severity == 'medium':
                # Increase monitoring frequency
                self.increase_monitoring_frequency()

                # Alert data science team
                self.alert_team()
    ```

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of drift detection, monitoring at scale, automated alerting.

        **Strong answer signals:**
        - Discusses multiple drift detection methods (PSI, KL, KS test)
        - Mentions both data drift and concept drift
        - Talks about delayed ground truth labels
        - Knows about per-segment monitoring (not just overall)
        - Discusses business metrics in addition to technical metrics
        - Mentions automated retraining triggers
        - Talks about alert fatigue and threshold tuning

---

### Design a Distributed Training System - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Deep Learning`, `Scale` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Scale Requirements:**
    - **Model Size:** 1B - 175B parameters (GPT-3 scale)
    - **Dataset:** 1TB - 1PB training data
    - **GPUs:** 100-10,000 GPUs
    - **Training Time:** Days to weeks
    - **Throughput:** 1000+ samples/second
    - **Communication:** 100 GB/s+ bandwidth

    **Detailed Architecture:**

    ```
    ┌──────────────────────────────────────────────────────┐
    │             Orchestration Layer                       │
    │  Kubernetes + Kubeflow / Ray / Slurm                  │
    │  - Resource allocation                                │
    │  - Fault tolerance & checkpointing                    │
    │  - Job scheduling                                     │
    └──────────────┬───────────────────────────────────────┘
                   ↓
    ┌──────────────────────────────────────────────────────┐
    │          Data Parallelism (Most Common)               │
    │                                                       │
    │  GPU 1: Model copy 1 → Batch 1 → Gradients           │
    │  GPU 2: Model copy 2 → Batch 2 → Gradients           │
    │  GPU 3: Model copy 3 → Batch 3 → Gradients           │
    │  GPU 4: Model copy 4 → Batch 4 → Gradients           │
    │                         ↓                             │
    │              All-Reduce (Average gradients)           │
    │                         ↓                             │
    │              Update all model copies                  │
    └───────────────────────────────────────────────────────┘

    For VERY large models:
    ┌──────────────────────────────────────────────────────┐
    │          Model Parallelism (Layers split)             │
    │                                                       │
    │  GPU 1: Layers 1-25    → Forward → Activation        │
    │  GPU 2: Layers 26-50   → Forward → Activation        │
    │  GPU 3: Layers 51-75   → Forward → Activation        │
    │  GPU 4: Layers 76-100  → Forward → Output            │
    │                                                       │
    │  Backward pass flows in reverse                       │
    └───────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────┐
    │       Pipeline Parallelism (Micro-batching)           │
    │                                                       │
    │  Time  │  GPU 1  │  GPU 2  │  GPU 3  │  GPU 4       │
    │  ──────┼─────────┼─────────┼─────────┼──────────    │
    │   t1   │ Batch 1 │    -    │    -    │    -         │
    │   t2   │ Batch 2 │ Batch 1 │    -    │    -         │
    │   t3   │ Batch 3 │ Batch 2 │ Batch 1 │    -         │
    │   t4   │ Batch 4 │ Batch 3 │ Batch 2 │ Batch 1      │
    │                                                       │
    │  Minimize bubble (idle time) with micro-batches       │
    └───────────────────────────────────────────────────────┘
    ```

    **Implementation:**

    ```python
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler

    # 1. DATA PARALLEL (Most Common) - PyTorch
    def setup_distributed():
        """Initialize distributed training"""
        # Initialize process group
        dist.init_process_group(
            backend='nccl',  # NVIDIA Collective Communications Library
            init_method='env://',  # Use environment variables
            world_size=int(os.environ['WORLD_SIZE']),  # Total GPUs
            rank=int(os.environ['RANK'])  # This GPU's rank
        )

    def train_data_parallel(model, train_dataset, epochs=10):
        """Data parallel training"""
        # Setup
        setup_distributed()
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f'cuda:{local_rank}')

        # Wrap model with DDP
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank])

        # Distributed sampler (each GPU gets different data)
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True
        )

        dataloader = DataLoader(
            train_dataset,
            batch_size=32,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(epochs):
            # Set epoch for shuffling
            sampler.set_epoch(epoch)

            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)

                # Forward pass
                output = model(data)
                loss = F.cross_entropy(output, target)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()  # Gradients are automatically all-reduced by DDP

                # Update weights
                optimizer.step()

                # Logging (only rank 0)
                if dist.get_rank() == 0 and batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

            # Checkpoint (only rank 0)
            if dist.get_rank() == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f'checkpoint_epoch_{epoch}.pt')

    # 2. MODEL PARALLEL - For Large Models
    class ModelParallelTransformer(nn.Module):
        """Split large model across GPUs"""
        def __init__(self, num_layers=96, hidden_size=12288):
            super().__init__()

            # Split layers across 4 GPUs
            layers_per_gpu = num_layers // 4

            # GPU 0: First 25% of layers
            self.layers_0 = nn.Sequential(*[
                TransformerBlock(hidden_size) for _ in range(layers_per_gpu)
            ]).to('cuda:0')

            # GPU 1: Next 25%
            self.layers_1 = nn.Sequential(*[
                TransformerBlock(hidden_size) for _ in range(layers_per_gpu)
            ]).to('cuda:1')

            # GPU 2: Next 25%
            self.layers_2 = nn.Sequential(*[
                TransformerBlock(hidden_size) for _ in range(layers_per_gpu)
            ]).to('cuda:2')

            # GPU 3: Last 25%
            self.layers_3 = nn.Sequential(*[
                TransformerBlock(hidden_size) for _ in range(layers_per_gpu)
            ]).to('cuda:3')

            self.output = nn.Linear(hidden_size, vocab_size).to('cuda:3')

        def forward(self, x):
            # Move through GPUs sequentially
            x = x.to('cuda:0')
            x = self.layers_0(x)

            x = x.to('cuda:1')
            x = self.layers_1(x)

            x = x.to('cuda:2')
            x = self.layers_2(x)

            x = x.to('cuda:3')
            x = self.layers_3(x)
            x = self.output(x)

            return x

    # 3. PIPELINE PARALLEL - Deepspeed, Megatron-LM
    from deepspeed.pipe import PipelineModule, LayerSpec

    def pipeline_parallel():
        """Pipeline parallelism with DeepSpeed"""
        # Define model as sequence of layers
        layers = [
            LayerSpec(TransformerBlock, args=(hidden_size,))
            for _ in range(96)
        ]

        # DeepSpeed will automatically partition across GPUs
        model = PipelineModule(
            layers=layers,
            num_stages=4,  # 4 GPUs
            partition_method='uniform'  # or 'balanced'
        )

        # Training with micro-batches
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            config={
                'train_micro_batch_size_per_gpu': 4,
                'gradient_accumulation_steps': 4,
                'pipeline': {
                    'pipe_partitioned': True,
                    'grad_partitioned': True
                }
            }
        )

        for batch in dataloader:
            loss = engine(batch)
            engine.backward(loss)
            engine.step()

    # 4. ZERO OPTIMIZER (Memory Optimization)
    from deepspeed import DeepSpeedConfig

    def train_with_zero():
        """ZeRO: Memory-optimized distributed training"""
        # ZeRO Stage 1: Partition optimizer states
        # ZeRO Stage 2: + Partition gradients
        # ZeRO Stage 3: + Partition model parameters

        config = {
            "train_batch_size": 128,
            "gradient_accumulation_steps": 4,
            "zero_optimization": {
                "stage": 3,  # Full ZeRO
                "offload_optimizer": {
                    "device": "cpu",  # Offload to CPU RAM
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu"
                }
            },
            "fp16": {
                "enabled": True  # Mixed precision
            }
        }

        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=config
        )

    # 5. GRADIENT ACCUMULATION (Simulate larger batch)
    def train_with_gradient_accumulation(model, dataloader, accumulation_steps=4):
        """Accumulate gradients before update"""
        optimizer.zero_grad()

        for i, (data, target) in enumerate(dataloader):
            output = model(data)
            loss = criterion(output, target)

            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
            loss.backward()

            # Update every N steps
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
    ```

    **Parallelism Strategy Decision Tree:**

    | Model Size | Data Size | Strategy | Example |
    |------------|-----------|----------|---------|
    | <1B params | Large | **Data Parallel** | ResNet, BERT-base |
    | 1-10B params | Large | **Data Parallel + ZeRO** | GPT-2, BERT-large |
    | 10-100B params | Large | **Model + Data Parallel** | GPT-3, BLOOM |
    | >100B params | Large | **Pipeline + Model + Data** | GPT-4, PaLM |

    **Communication Patterns:**

    | Method | Communication | Use Case | Efficiency |
    |--------|---------------|----------|------------|
    | **All-Reduce** | All-to-all gradient sync | Data parallel | High |
    | **Point-to-Point** | Sequential activation passing | Model parallel | Medium |
    | **Broadcast** | Scatter parameters | Parameter server | Medium |
    | **Reduce-Scatter** | Gradient partitioning | ZeRO optimizer | High |

    **Optimization Techniques:**

    ```python
    # 1. Mixed Precision Training (FP16)
    from torch.cuda.amp import autocast, GradScaler

    scaler = GradScaler()

    for data, target in dataloader:
        optimizer.zero_grad()

        # Forward in FP16
        with autocast():
            output = model(data)
            loss = criterion(output, target)

        # Backward with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # 2. Gradient Checkpointing (Memory Savings)
    from torch.utils.checkpoint import checkpoint

    class CheckpointedBlock(nn.Module):
        def forward(self, x):
            # Don't store activations, recompute in backward
            return checkpoint(self._forward, x)

        def _forward(self, x):
            return self.layer(x)

    # 3. Gradient Compression
    class GradientCompressor:
        def compress(self, tensor, compression_ratio=0.01):
            """Top-k gradient sparsification"""
            numel = tensor.numel()
            k = max(1, int(numel * compression_ratio))

            # Keep only top-k gradients
            values, indices = torch.topk(tensor.abs().flatten(), k)
            compressed = torch.zeros_like(tensor.flatten())
            compressed[indices] = tensor.flatten()[indices]

            return compressed.reshape(tensor.shape)
    ```

    **Fault Tolerance:**

    ```python
    class FaultTolerantTrainer:
        def __init__(self, checkpoint_freq=100):
            self.checkpoint_freq = checkpoint_freq

        def save_checkpoint(self, epoch, model, optimizer, path):
            """Save training state"""
            if dist.get_rank() == 0:  # Only rank 0 saves
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state_all()
                }, path)

        def load_checkpoint(self, path, model, optimizer):
            """Resume from checkpoint"""
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            torch.set_rng_state(checkpoint['rng_state'])
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
            return checkpoint['epoch']
    ```

    **Performance Metrics:**

    | Metric | Target | Calculation |
    |--------|--------|-------------|
    | **Throughput** | 1000+ samples/sec | Samples / Time |
    | **GPU Utilization** | >80% | Compute time / Total time |
    | **Communication Overhead** | <20% | Comm time / Total time |
    | **Scaling Efficiency** | >90% | Speedup(N GPUs) / N |
    | **Memory Efficiency** | >70% GPU RAM used | Used memory / Total memory |

    **Common Pitfalls:**

    ❌ **Small batch size per GPU:** Underutilizes GPU → Use at least 32-64
    ❌ **Slow data loading:** GPU waits for CPU → Use multiple workers, pin_memory
    ❌ **Not using mixed precision:** 2x slower → Use FP16/BF16
    ❌ **Synchronization bottlenecks:** Frequent all-reduce → Gradient accumulation
    ❌ **Imbalanced pipeline stages:** GPU idle time → Balance layer distribution

    **Real-World Examples:**

    - **Google PaLM (540B):** 6144 TPUs, model + data + pipeline parallelism
    - **Meta LLAMA-2 (70B):** 2000 A100 GPUs, ZeRO-3 + pipeline parallelism
    - **OpenAI GPT-3 (175B):** 10,000 V100 GPUs, model parallelism
    - **Stability AI (2B):** 256 A100 GPUs, data parallel with DeepSpeed

    **Cost Optimization:**

    | GPU Type | Price/hr | Speed | Best For |
    |----------|----------|-------|----------|
    | **V100** | $2-3 | Baseline | Legacy workloads |
    | **A100** | $4-6 | 2x V100 | Most efficient |
    | **H100** | $8-10 | 3x V100 | Cutting edge |
    | **TPU v4** | $3-5 | Comparable to A100 | Google ecosystem |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Knowledge of distributed training strategies, communication patterns, optimization techniques.

        **Strong answer signals:**
        - Knows when to use data vs model vs pipeline parallelism
        - Discusses communication overhead and all-reduce
        - Mentions ZeRO optimizer for memory efficiency
        - Talks about gradient checkpointing and mixed precision
        - Knows about fault tolerance and checkpointing
        - Discusses scaling efficiency metrics
        - Mentions pipeline bubbles and how to minimize them

---

### Design an A/B Testing Platform - Netflix, Airbnb Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Experimentation` | **Asked by:** Netflix, Airbnb, Uber

??? success "View Answer"

    **Scale Requirements:**
    - **Concurrent Experiments:** 100-1000+ active tests
    - **Users:** 100M+ users in experiments
    - **Events:** 10B+ events/day
    - **Experiment Duration:** 1-4 weeks typical
    - **Statistical Power:** 80%+ with 5% significance
    - **Analysis Latency:** Real-time dashboards + daily reports

    **Detailed Architecture:**

    ```
    ┌────────────────────────────────────────────────────┐
    │           Experiment Configuration                 │
    │  - Define variants (A, B, C)                       │
    │  - Traffic allocation (50/50, 90/10, etc.)         │
    │  - Target audience (location, platform, etc.)      │
    │  - Metrics (primary, secondary, guardrails)        │
    │  - Duration & sample size calculation              │
    └──────────────┬─────────────────────────────────────┘
                   ↓
    ┌────────────────────────────────────────────────────┐
    │        Assignment Service (User Bucketing)         │
    │                                                    │
    │  Input: user_id, experiment_id                     │
    │  Output: variant (A or B)                          │
    │                                                    │
    │  hash(user_id + experiment_id) % 100               │
    │    → Deterministic, consistent assignment          │
    │    → Same user always gets same variant            │
    │    → No database lookup needed                     │
    └──────────────┬─────────────────────────────────────┘
                   ↓
    ┌────────────────────────────────────────────────────┐
    │         User Experience (Application)              │
    │                                                    │
    │  if variant == 'A':                                │
    │      show_old_checkout_flow()                      │
    │  elif variant == 'B':                              │
    │      show_new_checkout_flow()                      │
    └──────────────┬─────────────────────────────────────┘
                   ↓
    ┌────────────────────────────────────────────────────┐
    │          Event Tracking (Kafka Stream)             │
    │                                                    │
    │  - Exposure events (user saw variant)              │
    │  - Action events (clicks, purchases, etc.)         │
    │  - Metadata (timestamp, user_id, variant, etc.)   │
    └──────────────┬─────────────────────────────────────┘
                   ↓
    ┌────────────────────────────────────────────────────┐
    │       Data Pipeline (Batch Processing)             │
    │                                                    │
    │  Daily Spark jobs:                                 │
    │  - Join exposure + outcome events                  │
    │  - Calculate metrics per variant                   │
    │  - Run statistical tests                           │
    │  - Detect Sample Ratio Mismatch (SRM)             │
    └──────────────┬─────────────────────────────────────┘
                   ↓
    ┌────────────────────────────────────────────────────┐
    │         Statistical Analysis Engine                │
    │                                                    │
    │  - T-test for continuous metrics                   │
    │  - Z-test for proportions                          │
    │  - Sequential testing (early stopping)             │
    │  - Multiple testing correction (Bonferroni)        │
    │  - Variance reduction (CUPED, stratification)      │
    └──────────────┬─────────────────────────────────────┘
                   ↓
    ┌────────────────────────────────────────────────────┐
    │      Dashboard & Reporting (Real-time)             │
    │                                                    │
    │  - Experiment status & health                      │
    │  - Metric movements (% change, confidence)         │
    │  - Statistical significance & p-values             │
    │  - Sample Ratio Mismatch alerts                    │
    │  - Interaction effects detection                   │
    └────────────────────────────────────────────────────┘
    ```

    **Implementation:**

    ```python
    import hashlib
    import numpy as np
    from scipy import stats
    from typing import Dict, List, Tuple

    # 1. ASSIGNMENT SERVICE
    class ExperimentAssignmentService:
        """Deterministic user assignment to experiment variants"""

        def __init__(self):
            self.experiments = {}  # experiment_id -> config

        def assign_variant(self, user_id: str, experiment_id: str) -> str:
            """
            Deterministic assignment using hash function
            Same user always gets same variant
            """
            experiment = self.experiments[experiment_id]

            # Hash user_id + experiment_id for randomization
            hash_input = f"{user_id}_{experiment_id}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

            # Convert to bucket (0-99)
            bucket = hash_value % 100

            # Assign to variant based on traffic allocation
            cumulative = 0
            for variant, allocation in experiment['traffic_allocation'].items():
                cumulative += allocation
                if bucket < cumulative:
                    return variant

            return 'control'  # Default

        def should_include_user(
            self,
            user: Dict,
            experiment_config: Dict
        ) -> bool:
            """Check if user qualifies for experiment"""
            targeting = experiment_config['targeting']

            # Check filters
            if 'countries' in targeting:
                if user['country'] not in targeting['countries']:
                    return False

            if 'platforms' in targeting:
                if user['platform'] not in targeting['platforms']:
                    return False

            if 'user_segments' in targeting:
                if user['segment'] not in targeting['user_segments']:
                    return False

            return True

    # 2. EVENT TRACKING
    class ExperimentEventTracker:
        """Track exposure and outcome events"""

        def track_exposure(
            self,
            user_id: str,
            experiment_id: str,
            variant: str,
            timestamp: int
        ):
            """Log when user is exposed to experiment"""
            event = {
                'event_type': 'exposure',
                'user_id': user_id,
                'experiment_id': experiment_id,
                'variant': variant,
                'timestamp': timestamp
            }
            self._send_to_kafka('experiment_events', event)

        def track_outcome(
            self,
            user_id: str,
            experiment_id: str,
            metric_name: str,
            metric_value: float,
            timestamp: int
        ):
            """Log outcome metric (conversion, revenue, etc.)"""
            event = {
                'event_type': 'outcome',
                'user_id': user_id,
                'experiment_id': experiment_id,
                'metric_name': metric_name,
                'metric_value': metric_value,
                'timestamp': timestamp
            }
            self._send_to_kafka('experiment_events', event)

    # 3. STATISTICAL ANALYSIS
    class ExperimentAnalyzer:
        """Analyze experiment results"""

        def __init__(self):
            self.alpha = 0.05  # Significance level (5%)
            self.power = 0.80  # Statistical power (80%)

        def calculate_sample_size(
            self,
            baseline_rate: float,
            minimum_detectable_effect: float,
            alpha: float = 0.05,
            power: float = 0.80
        ) -> int:
            """
            Calculate required sample size per variant
            For detecting a minimum effect with desired power
            """
            from statsmodels.stats.power import zt_ind_solve_power

            # Effect size (Cohen's h for proportions)
            p1 = baseline_rate
            p2 = baseline_rate * (1 + minimum_detectable_effect)

            effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))

            # Calculate sample size
            n = zt_ind_solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                alternative='two-sided'
            )

            return int(np.ceil(n))

        def analyze_experiment(
            self,
            control_metrics: np.ndarray,
            treatment_metrics: np.ndarray
        ) -> Dict:
            """
            Run statistical test on experiment results
            """
            n_control = len(control_metrics)
            n_treatment = len(treatment_metrics)

            mean_control = np.mean(control_metrics)
            mean_treatment = np.mean(treatment_metrics)

            # Relative lift
            relative_lift = (mean_treatment - mean_control) / mean_control

            # T-test for continuous metrics
            t_stat, p_value = stats.ttest_ind(
                treatment_metrics,
                control_metrics,
                equal_var=False  # Welch's t-test
            )

            # Confidence interval (95%)
            se_diff = np.sqrt(
                np.var(control_metrics) / n_control +
                np.var(treatment_metrics) / n_treatment
            )
            ci_lower = (mean_treatment - mean_control) - 1.96 * se_diff
            ci_upper = (mean_treatment - mean_control) + 1.96 * se_diff

            is_significant = p_value < self.alpha

            return {
                'control_mean': mean_control,
                'treatment_mean': mean_treatment,
                'absolute_lift': mean_treatment - mean_control,
                'relative_lift': relative_lift,
                'p_value': p_value,
                'is_significant': is_significant,
                'confidence_interval': (ci_lower, ci_upper),
                'sample_size_control': n_control,
                'sample_size_treatment': n_treatment
            }

        def check_sample_ratio_mismatch(
            self,
            n_control: int,
            n_treatment: int,
            expected_ratio: float = 0.5
        ) -> Dict:
            """
            Sample Ratio Mismatch (SRM) detection
            Checks if traffic split matches expected ratio
            """
            total = n_control + n_treatment
            expected_control = total * expected_ratio
            expected_treatment = total * (1 - expected_ratio)

            # Chi-square test
            observed = [n_control, n_treatment]
            expected = [expected_control, expected_treatment]

            chi_stat, p_value = stats.chisquare(observed, expected)

            has_srm = p_value < 0.001  # Very strict threshold

            return {
                'n_control': n_control,
                'n_treatment': n_treatment,
                'expected_ratio': expected_ratio,
                'actual_ratio': n_control / total,
                'p_value': p_value,
                'has_srm': has_srm
            }

        def apply_cuped(
            self,
            post_metrics: np.ndarray,
            pre_metrics: np.ndarray
        ) -> np.ndarray:
            """
            CUPED (Controlled-experiment Using Pre-Experiment Data)
            Variance reduction technique using covariates
            """
            # Calculate theta (optimal coefficient)
            cov = np.cov(post_metrics, pre_metrics)[0, 1]
            var_pre = np.var(pre_metrics)
            theta = cov / var_pre

            # Adjust post-experiment metric
            adjusted_metrics = post_metrics - theta * (pre_metrics - np.mean(pre_metrics))

            # Variance reduction
            var_original = np.var(post_metrics)
            var_adjusted = np.var(adjusted_metrics)
            variance_reduction = 1 - (var_adjusted / var_original)

            print(f"Variance reduced by {variance_reduction:.1%}")

            return adjusted_metrics

        def sequential_testing(
            self,
            control_data: List[float],
            treatment_data: List[float],
            looks: int = 5
        ) -> Dict:
            """
            Sequential testing for early stopping
            Allows peeking at results without inflating false positive rate
            """
            # Always-valid p-values (mixture sequential probability ratio test)
            results = []

            for i in range(1, looks + 1):
                # Get data up to this point
                idx = int(len(control_data) * i / looks)
                control_subset = control_data[:idx]
                treatment_subset = treatment_data[:idx]

                # Run test
                result = self.analyze_experiment(
                    np.array(control_subset),
                    np.array(treatment_subset)
                )

                # Adjusted alpha for multiple looks (Bonferroni correction)
                adjusted_alpha = self.alpha / looks
                result['adjusted_alpha'] = adjusted_alpha
                result['can_stop'] = result['p_value'] < adjusted_alpha

                results.append(result)

                if result['can_stop']:
                    print(f"Can stop early at look {i}/{looks}")
                    break

            return results

    # 4. INTERACTION EFFECTS
    class InteractionEffectsDetector:
        """Detect when multiple experiments interfere"""

        def detect_interaction(
            self,
            exp1_assignment: np.ndarray,  # 0 or 1
            exp2_assignment: np.ndarray,  # 0 or 1
            outcome: np.ndarray
        ) -> Dict:
            """
            2-way ANOVA to detect interaction effects
            """
            from scipy.stats import f_oneway

            # Four groups: (exp1=0, exp2=0), (exp1=1, exp2=0), etc.
            group_00 = outcome[(exp1_assignment == 0) & (exp2_assignment == 0)]
            group_01 = outcome[(exp1_assignment == 0) & (exp2_assignment == 1)]
            group_10 = outcome[(exp1_assignment == 1) & (exp2_assignment == 0)]
            group_11 = outcome[(exp1_assignment == 1) & (exp2_assignment == 1)]

            # Main effect of exp1
            exp1_control = np.concatenate([group_00, group_01])
            exp1_treatment = np.concatenate([group_10, group_11])
            _, p_exp1 = stats.ttest_ind(exp1_treatment, exp1_control)

            # Main effect of exp2
            exp2_control = np.concatenate([group_00, group_10])
            exp2_treatment = np.concatenate([group_01, group_11])
            _, p_exp2 = stats.ttest_ind(exp2_treatment, exp2_control)

            # Interaction effect
            # If interaction exists: effect of exp1 differs based on exp2
            effect_exp1_when_exp2_control = np.mean(group_10) - np.mean(group_00)
            effect_exp1_when_exp2_treatment = np.mean(group_11) - np.mean(group_01)
            interaction_magnitude = abs(
                effect_exp1_when_exp2_treatment - effect_exp1_when_exp2_control
            )

            return {
                'exp1_significant': p_exp1 < 0.05,
                'exp2_significant': p_exp2 < 0.05,
                'interaction_magnitude': interaction_magnitude,
                'has_interaction': interaction_magnitude > 0.01  # Threshold
            }
    ```

    **Key Formulas:**

    | Concept | Formula | Purpose |
    |---------|---------|---------|
    | **Sample Size** | $n = \frac{2(Z_{\alpha/2} + Z_\beta)^2 \sigma^2}{\delta^2}$ | Required users per variant |
    | **T-statistic** | $t = \frac{\bar{X}_B - \bar{X}_A}{\sqrt{s^2(\frac{1}{n_A} + \frac{1}{n_B})}}$ | Statistical significance |
    | **Confidence Interval** | $CI = \bar{X} \pm Z_{\alpha/2} \times SE$ | Range of true effect |
    | **Relative Lift** | $\frac{\bar{X}_B - \bar{X}_A}{\bar{X}_A} \times 100\%$ | % improvement |
    | **Statistical Power** | $1 - \beta$ | Probability of detecting true effect |

    **Common Pitfalls:**

    ❌ **Peeking:** Looking at results too early → Inflated false positives (use sequential testing)
    ❌ **Sample Ratio Mismatch:** Unequal traffic split → Check randomization
    ❌ **Multiple testing:** Testing many metrics → Apply Bonferroni correction
    ❌ **Not accounting for novelty effect:** New feature gets attention → Run for 2+ weeks
    ❌ **Ignoring interaction effects:** Conflicting experiments → Use orthogonal assignment

    **Real-World Examples:**

    - **Netflix:** 1000+ concurrent tests, watches for interactions, uses CUPED for variance reduction
    - **Airbnb:** ERF (Experiment Reporting Framework), automated SRM detection, layered experiments
    - **Uber:** XP platform, sequential testing, handles >100M users
    - **Booking.com:** 1000+ active experiments, isolated experiment layers

    **Advanced Techniques:**

    ```python
    # Stratified Sampling (Variance Reduction)
    def stratified_analysis(df, strata_col='country'):
        """Analyze within strata, then combine"""
        results = []
        for stratum in df[strata_col].unique():
            subset = df[df[strata_col] == stratum]
            result = analyze_experiment(
                subset[subset['variant'] == 'A']['metric'],
                subset[subset['variant'] == 'B']['metric']
            )
            results.append((stratum, result))
        return results

    # Bayesian A/B Testing (Alternative to frequentist)
    def bayesian_ab_test(control_conversions, control_trials,
                         treatment_conversions, treatment_trials):
        """Bayesian approach with Beta priors"""
        from scipy.stats import beta

        # Posterior distributions
        control_posterior = beta(control_conversions + 1, control_trials - control_conversions + 1)
        treatment_posterior = beta(treatment_conversions + 1, treatment_trials - treatment_conversions + 1)

        # Probability treatment > control
        samples_control = control_posterior.rvs(100000)
        samples_treatment = treatment_posterior.rvs(100000)
        prob_treatment_better = (samples_treatment > samples_control).mean()

        return prob_treatment_better
    ```

    **Metrics Taxonomy:**

    | Metric Type | Examples | Guardrails? |
    |-------------|----------|-------------|
    | **Primary** | Conversion rate, Revenue | Decision metric |
    | **Secondary** | CTR, Engagement time | Supplementary insights |
    | **Guardrail** | Page load time, Error rate | Must not degrade |
    | **Debugging** | Feature usage, Funnel steps | Understand "why" |

    !!! tip "Interviewer's Insight"
        **What they're testing:** Understanding of randomization, statistical power, variance reduction, multiple testing.

        **Strong answer signals:**
        - Discusses deterministic assignment with hash functions
        - Mentions Sample Ratio Mismatch (SRM) detection
        - Knows about CUPED for variance reduction
        - Talks about multiple testing correction (Bonferroni)
        - Discusses interaction effects between experiments
        - Mentions sequential testing for early stopping
        - Knows about novelty effects and proper experiment duration
        - Discusses layered experiments for isolation

---

### Design a Data Pipeline for ML - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Data Engineering` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ## Scale Requirements

    - **Data Volume:** 10TB-1PB daily ingestion
    - **Throughput:** 100K-1M events/second
    - **Latency:** Batch (hourly/daily), Streaming (<1 min end-to-end)
    - **Features:** 1K-10K features, 100M-10B rows
    - **Pipeline SLA:** 99.9% uptime, <5% data loss tolerance
    - **Data Quality:** 99%+ accuracy, <0.1% duplicate rate

    ## Architecture

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                     Data Sources                                 │
    │  [Databases] [APIs] [Event Streams] [Files] [3rd Party]        │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                 Ingestion Layer (Airflow/Prefect)                │
    │                                                                  │
    │  Batch:          CDC:              Streaming:                   │
    │  Sqoop/Fivetran  Debezium         Kafka Connect                 │
    │  (hourly/daily)  (real-time)      (real-time)                   │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                  Raw Data Lake (S3/GCS/ADLS)                     │
    │                                                                  │
    │  /raw/yyyy/mm/dd/hh/source_name/data.parquet                   │
    │  - Immutable, append-only                                       │
    │  - Partitioned by date + source                                 │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │            Data Quality & Validation Layer                       │
    │                                                                  │
    │  Schema validation → Null checks → Range checks                 │
    │  → Duplicate detection → Anomaly detection                      │
    │  Great Expectations / Deequ / Custom                            │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │          Processing Layer (Spark/Dask/DBT)                       │
    │                                                                  │
    │  ETL/ELT:                      Feature Engineering:             │
    │  - Cleaning & deduplication    - Aggregations                   │
    │  - Schema normalization        - Joins (point-in-time)          │
    │  - Filtering & sampling        - Transformations                │
    │  - Enrichment                  - Embeddings                     │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │       Curated Data & Feature Store (Delta Lake/Hudi)            │
    │                                                                  │
    │  Offline Store:            Online Store:                        │
    │  S3/BigQuery/Snowflake     Redis/DynamoDB/Cassandra             │
    │  (training data)           (low-latency serving)                │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                   ML Training & Serving                          │
    │                                                                  │
    │  [Training Jobs] ← Historical features                          │
    │  [Inference] ← Real-time features                               │
    └─────────────────────────────────────────────────────────────────┘

            ┌────────────────────────────────────┐
            │     Cross-Cutting Concerns         │
            │                                    │
            │  - Metadata & Lineage (DataHub)   │
            │  - Monitoring (Datadog/Grafana)   │
            │  - Versioning (DVC/Delta)         │
            │  - Access Control (IAM/RBAC)      │
            └────────────────────────────────────┘
    ```

    ## Production Implementation (320 lines)

    ```python
    # airflow_ml_pipeline.py
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
    from datetime import datetime, timedelta
    import great_expectations as ge
    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F
    import logging
    from typing import Dict, List, Tuple
    from dataclasses import dataclass

    # ============= Configuration =============
    @dataclass
    class PipelineConfig:
        """Pipeline configuration with all parameters"""
        raw_data_path: str = "s3://ml-data/raw"
        processed_data_path: str = "s3://ml-data/processed"
        feature_store_path: str = "s3://ml-data/features"
        data_quality_threshold: float = 0.95
        max_null_percentage: float = 0.05
        deduplication_keys: List[str] = None

        def __post_init__(self):
            if self.deduplication_keys is None:
                self.deduplication_keys = ['user_id', 'timestamp']

    config = PipelineConfig()

    # ============= Data Quality Checks =============
    class DataQualityChecker:
        """Comprehensive data quality validation"""

        def __init__(self, spark: SparkSession):
            self.spark = spark
            self.logger = logging.getLogger(__name__)

        def validate_schema(self, df, expected_schema: Dict) -> Tuple[bool, List[str]]:
            """Validate DataFrame schema against expected"""
            issues = []
            df_schema = {field.name: str(field.dataType) for field in df.schema}

            for col, dtype in expected_schema.items():
                if col not in df_schema:
                    issues.append(f"Missing column: {col}")
                elif df_schema[col] != dtype:
                    issues.append(f"Type mismatch for {col}: expected {dtype}, got {df_schema[col]}")

            return len(issues) == 0, issues

        def check_nulls(self, df, max_null_pct: float = 0.05) -> Tuple[bool, Dict]:
            """Check null percentage for each column"""
            total_rows = df.count()
            null_stats = {}
            failed_cols = []

            for col in df.columns:
                null_count = df.filter(F.col(col).isNull()).count()
                null_pct = null_count / total_rows
                null_stats[col] = null_pct

                if null_pct > max_null_pct:
                    failed_cols.append(col)
                    self.logger.warning(f"Column {col} has {null_pct:.2%} nulls (threshold: {max_null_pct:.2%})")

            return len(failed_cols) == 0, null_stats

        def detect_duplicates(self, df, keys: List[str]) -> Tuple[int, float]:
            """Detect and count duplicates based on keys"""
            total_rows = df.count()
            duplicate_count = df.groupBy(keys).count().filter(F.col('count') > 1).count()
            duplicate_rate = duplicate_count / total_rows if total_rows > 0 else 0

            return duplicate_count, duplicate_rate

        def check_value_ranges(self, df, range_constraints: Dict) -> Tuple[bool, List[str]]:
            """Validate value ranges for numeric columns"""
            issues = []

            for col, (min_val, max_val) in range_constraints.items():
                out_of_range = df.filter(
                    (F.col(col) < min_val) | (F.col(col) > max_val)
                ).count()

                if out_of_range > 0:
                    issues.append(f"{col}: {out_of_range} values out of range [{min_val}, {max_val}]")

            return len(issues) == 0, issues

        def detect_anomalies(self, df, numeric_cols: List[str], std_threshold: float = 3.0):
            """Detect statistical anomalies using z-score"""
            for col in numeric_cols:
                stats = df.select(
                    F.mean(col).alias('mean'),
                    F.stddev(col).alias('std')
                ).first()

                if stats.std and stats.std > 0:
                    anomalies = df.filter(
                        F.abs((F.col(col) - stats.mean) / stats.std) > std_threshold
                    ).count()

                    if anomalies > 0:
                        self.logger.warning(f"{col}: {anomalies} anomalies detected (>{std_threshold}σ)")

        def run_great_expectations(self, df, checkpoint_name: str) -> bool:
            """Run Great Expectations validation suite"""
            try:
                context = ge.data_context.DataContext()
                batch = context.get_batch({'dataset': df, 'datasource': 'spark'})
                results = context.run_checkpoint(checkpoint_name=checkpoint_name)
                return results['success']
            except Exception as e:
                self.logger.error(f"Great Expectations failed: {e}")
                return False

    # ============= Feature Engineering Pipeline =============
    class FeatureEngineeringPipeline:
        """Production feature engineering with point-in-time correctness"""

        def __init__(self, spark: SparkSession):
            self.spark = spark
            self.logger = logging.getLogger(__name__)

        def create_time_features(self, df, timestamp_col: str = 'timestamp'):
            """Extract temporal features"""
            return df.withColumn('hour', F.hour(timestamp_col)) \
                     .withColumn('day_of_week', F.dayofweek(timestamp_col)) \
                     .withColumn('day_of_month', F.dayofmonth(timestamp_col)) \
                     .withColumn('month', F.month(timestamp_col)) \
                     .withColumn('is_weekend', F.dayofweek(timestamp_col).isin([1, 7]).cast('int'))

        def create_aggregation_features(self, df, group_keys: List[str],
                                       agg_col: str, windows: List[str]):
            """Create time-windowed aggregations with point-in-time correctness"""

            # Define window specifications
            window_specs = {
                '1h': 3600,
                '24h': 86400,
                '7d': 604800,
                '30d': 2592000
            }

            result_df = df

            for window in windows:
                if window in window_specs:
                    seconds = window_specs[window]

                    # Sliding window aggregation
                    window_spec = Window.partitionBy(group_keys) \
                                       .orderBy(F.col('timestamp').cast('long')) \
                                       .rangeBetween(-seconds, 0)

                    result_df = result_df.withColumn(
                        f'{agg_col}_sum_{window}',
                        F.sum(agg_col).over(window_spec)
                    ).withColumn(
                        f'{agg_col}_avg_{window}',
                        F.avg(agg_col).over(window_spec)
                    ).withColumn(
                        f'{agg_col}_count_{window}',
                        F.count(agg_col).over(window_spec)
                    ).withColumn(
                        f'{agg_col}_max_{window}',
                        F.max(agg_col).over(window_spec)
                    )

            return result_df

        def point_in_time_join(self, events_df, features_df,
                               join_keys: List[str], event_time_col: str = 'timestamp'):
            """Point-in-time correct join to prevent data leakage"""

            # For each event, get the latest feature values BEFORE the event timestamp
            window_spec = Window.partitionBy(join_keys) \
                                .orderBy(F.col('feature_timestamp').cast('long')) \
                                .rowsBetween(Window.unboundedPreceding, Window.currentRow)

            # Add sequence number to handle ties
            features_with_seq = features_df.withColumn(
                'seq', F.row_number().over(window_spec)
            )

            # Join using inequality condition
            joined = events_df.alias('e').join(
                features_with_seq.alias('f'),
                (events_df[join_keys[0]] == features_df[join_keys[0]]) &
                (F.col('f.feature_timestamp') <= F.col(f'e.{event_time_col}')),
                'left'
            )

            # Keep only the latest feature value before each event
            window_latest = Window.partitionBy(
                [f'e.{k}' for k in join_keys] + [f'e.{event_time_col}']
            ).orderBy(F.col('f.feature_timestamp').desc())

            result = joined.withColumn('rank', F.row_number().over(window_latest)) \
                          .filter(F.col('rank') == 1) \
                          .drop('rank', 'seq', 'feature_timestamp')

            return result

        def handle_missing_values(self, df, strategy: Dict[str, str]):
            """Handle missing values with column-specific strategies"""
            result_df = df

            for col, method in strategy.items():
                if method == 'mean':
                    mean_val = df.select(F.mean(col)).first()[0]
                    result_df = result_df.fillna({col: mean_val})
                elif method == 'median':
                    median_val = df.approxQuantile(col, [0.5], 0.01)[0]
                    result_df = result_df.fillna({col: median_val})
                elif method == 'mode':
                    mode_val = df.groupBy(col).count().orderBy('count', ascending=False).first()[0]
                    result_df = result_df.fillna({col: mode_val})
                elif method == 'zero':
                    result_df = result_df.fillna({col: 0})
                elif method == 'forward_fill':
                    window = Window.partitionBy().orderBy('timestamp').rowsBetween(Window.unboundedPreceding, 0)
                    result_df = result_df.withColumn(col, F.last(col, ignorenulls=True).over(window))

            return result_df

    # ============= Data Lineage Tracker =============
    class DataLineageTracker:
        """Track data lineage for reproducibility and debugging"""

        def __init__(self):
            self.lineage_graph = {}

        def record_transformation(self, output_path: str, input_paths: List[str],
                                 transformation_name: str, parameters: Dict):
            """Record a data transformation step"""
            self.lineage_graph[output_path] = {
                'inputs': input_paths,
                'transformation': transformation_name,
                'parameters': parameters,
                'timestamp': datetime.now().isoformat(),
                'spark_config': self._get_spark_config()
            }

            # Persist to DataHub or custom metadata store
            self._persist_lineage(output_path)

        def _get_spark_config(self) -> Dict:
            """Capture Spark configuration for reproducibility"""
            spark = SparkSession.getActiveSession()
            return {
                'spark.version': spark.version,
                'spark.executor.memory': spark.conf.get('spark.executor.memory'),
                'spark.executor.cores': spark.conf.get('spark.executor.cores')
            }

        def _persist_lineage(self, output_path: str):
            """Persist lineage metadata to external system (DataHub, Atlas, etc.)"""
            # Integration with DataHub/Apache Atlas
            pass

    # ============= Airflow DAG Definition =============
    default_args = {
        'owner': 'ml-team',
        'depends_on_past': False,
        'email': ['ml-alerts@company.com'],
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': 2,
        'retry_delay': timedelta(minutes=5),
    }

    dag = DAG(
        'ml_feature_pipeline',
        default_args=default_args,
        description='Production ML feature engineering pipeline',
        schedule_interval='0 */1 * * *',  # Hourly
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['ml', 'features', 'production'],
    )

    def ingest_data(**context):
        """Ingest data from various sources"""
        execution_date = context['execution_date']

        # Example: Ingest from database, APIs, S3
        # This is a placeholder - replace with actual ingestion logic

        output_path = f"{config.raw_data_path}/{execution_date.strftime('%Y/%m/%d/%H')}"
        logging.info(f"Ingesting data to {output_path}")

        return output_path

    def validate_data_quality(**context):
        """Run data quality checks"""
        spark = SparkSession.builder.appName("DataQualityCheck").getOrCreate()
        input_path = context['task_instance'].xcom_pull(task_ids='ingest_data')

        df = spark.read.parquet(input_path)
        checker = DataQualityChecker(spark)

        # Run all quality checks
        schema_valid, schema_issues = checker.validate_schema(df, expected_schema={
            'user_id': 'string',
            'timestamp': 'timestamp',
            'amount': 'double'
        })

        nulls_valid, null_stats = checker.check_nulls(df, max_null_pct=0.05)
        dup_count, dup_rate = checker.detect_duplicates(df, ['user_id', 'timestamp'])

        # Fail if quality below threshold
        if not schema_valid or not nulls_valid or dup_rate > 0.01:
            raise ValueError(f"Data quality check failed: {schema_issues}")

        logging.info(f"Data quality passed: {len(df.columns)} columns, {df.count()} rows")
        spark.stop()

    # Define Airflow tasks
    ingest_task = PythonOperator(
        task_id='ingest_data',
        python_callable=ingest_data,
        dag=dag,
    )

    quality_check_task = PythonOperator(
        task_id='validate_data_quality',
        python_callable=validate_data_quality,
        dag=dag,
    )

    feature_engineering_task = SparkSubmitOperator(
        task_id='feature_engineering',
        application='feature_engineering.py',
        conf={
            'spark.executor.memory': '8g',
            'spark.executor.cores': '4',
            'spark.dynamicAllocation.enabled': 'true'
        },
        dag=dag,
    )

    # Define task dependencies
    ingest_task >> quality_check_task >> feature_engineering_task
    ```

    ## Technology Stack Comparison

    | Layer | Tool Options | When to Use |
    |-------|-------------|-------------|
    | **Orchestration** | Airflow, Prefect, Dagster | Airflow: mature ecosystem; Prefect: dynamic DAGs; Dagster: asset-based |
    | **Batch Processing** | Spark, Dask, Ray | Spark: PB-scale; Dask: Python-native; Ray: ML workloads |
    | **Stream Processing** | Flink, Spark Streaming, Kafka Streams | Flink: exactly-once, low latency; Spark: batch+stream; Kafka: simple |
    | **Storage** | S3, GCS, ADLS, HDFS | Cloud: S3/GCS/ADLS; On-prem: HDFS |
    | **Format** | Parquet, ORC, Delta Lake, Hudi | Parquet: read-heavy; Delta/Hudi: ACID, time travel |
    | **Data Quality** | Great Expectations, Deequ, Soda | GE: Python; Deequ: Spark/Scala; Soda: SQL-based |
    | **Metadata** | DataHub, Apache Atlas, Amundsen | DataHub: modern; Atlas: Hadoop ecosystem; Amundsen: search-focused |

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Data Leakage** | Train/test contamination | Use point-in-time joins, strict temporal splits |
    | **Schema Drift** | Pipeline failures | Schema evolution with backward compatibility |
    | **Late-Arriving Data** | Incomplete features | Watermarks, reprocessing windows |
    | **Duplicate Records** | Inflated metrics | Deduplication with unique keys |
    | **Missing Values** | Biased models | Strategy per column (imputation/drop/flag) |
    | **Skewed Partitions** | Slow jobs | Salting, repartitioning, broadcast joins |
    | **No Data Versioning** | Irreproducible results | DVC, Delta Lake, manifest files |
    | **Insufficient Monitoring** | Silent failures | Data quality alerts, pipeline SLAs |

    ## Real-World Examples

    **Uber's Michelangelo:**
    - **Scale:** 10K+ features, 100M+ predictions/day
    - **Architecture:** Kafka → Flink → Cassandra (online), Hive (offline)
    - **Feature Store:** Point-in-time correct joins, feature monitoring
    - **Impact:** Reduced feature engineering time by 70%

    **Netflix's Data Pipeline:**
    - **Scale:** 500TB+ daily, 1.3PB total
    - **Tools:** S3 → Spark → Iceberg → Presto
    - **Features:** Schema evolution, time travel, data quality checks
    - **Impact:** Powers 800+ data scientists, 100K+ jobs/day

    **Airbnb's Zipline:**
    - **Scale:** 6K+ features, 10M+ bookings/day
    - **Architecture:** Airflow → Spark → Hive (offline), Redis (online)
    - **Innovation:** Feature freshness SLAs, automatic backfills
    - **Impact:** 80% reduction in feature development time

    ## Monitoring & Debugging

    ```python
    # Pipeline metrics to track
    metrics = {
        'data_volume': 'Input/output row counts',
        'latency': 'End-to-end pipeline duration',
        'data_quality': 'Null rate, duplicate rate, schema violations',
        'freshness': 'Time from data creation to availability',
        'resource_usage': 'CPU, memory, disk I/O per stage',
        'failure_rate': 'Task failures, retries, SLA misses'
    }

    # Alerting thresholds
    alerts = {
        'data_volume_drop': 'Alert if <80% of expected volume',
        'latency_spike': 'Alert if p99 > 2x baseline',
        'quality_drop': 'Alert if quality score < 95%',
        'freshness_lag': 'Alert if data >4 hours old'
    }
    ```

    !!! tip "Interviewer's Insight"
        Emphasizes point-in-time correctness, data quality, and lineage tracking. Discusses trade-offs between batch and streaming, shows knowledge of Great Expectations/Deequ, and understands schema evolution. Can explain how Uber/Netflix/Airbnb implement feature stores at scale.

---

### Design a Model Registry - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `MLOps` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    ## Scale Requirements

    - **Models:** 100-10K registered models
    - **Versions:** 10-1K versions per model
    - **Metadata:** 100KB-10MB per model (metrics, params, artifacts)
    - **Throughput:** 1K-100K model queries/day
    - **Storage:** 10GB-10TB (model binaries + artifacts)
    - **Latency:** <100ms for metadata queries, <1s for model downloads
    - **Users:** 10-1K data scientists/engineers

    ## Architecture

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Training Environment                          │
    │                                                                  │
    │  [Notebook/Script] → MLflow Client → Model Registry API        │
    │                                                                  │
    │  Logs: model, metrics, params, artifacts, tags                 │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                  Model Registry (MLflow Server)                  │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │            Metadata Store (PostgreSQL/MySQL)              │  │
    │  │                                                            │  │
    │  │  - Model names & versions                                │  │
    │  │  - Metrics (accuracy, F1, AUC)                           │  │
    │  │  - Parameters (hyperparameters)                          │  │
    │  │  - Tags & descriptions                                   │  │
    │  │  - Stage (None/Staging/Production/Archived)              │  │
    │  │  - Lineage (dataset version, code commit)                │  │
    │  │  - User & timestamp                                      │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │         Artifact Store (S3/GCS/Azure Blob/HDFS)           │  │
    │  │                                                            │  │
    │  │  - Model binaries (pickle, ONNX, SavedModel)             │  │
    │  │  - Feature preprocessors                                  │  │
    │  │  - Training/validation datasets (samples)                 │  │
    │  │  - Plots & visualizations                                 │  │
    │  │  - Model cards & documentation                            │  │
    │  └──────────────────────────────────────────────────────────┘  │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Model Lifecycle Management                    │
    │                                                                  │
    │  Stage Transitions:                                             │
    │  None → Staging → Production → Archived                        │
    │                                                                  │
    │  Approval Workflow:                                             │
    │  1. Register model (None)                                       │
    │  2. Validation tests → Staging                                  │
    │  3. A/B test → Production (with approval)                       │
    │  4. Superseded → Archived                                       │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                   Serving & Deployment                           │
    │                                                                  │
    │  [Model Serving] ← Load model by stage or version              │
    │  [CI/CD Pipeline] ← Trigger deploy on stage change             │
    │  [Monitoring] ← Track production model performance              │
    └─────────────────────────────────────────────────────────────────┘

             ┌──────────────────────────────────────┐
             │      Cross-Cutting Features          │
             │                                      │
             │  - Access Control (RBAC)            │
             │  - Model Comparison (side-by-side)  │
             │  - Search & Discovery               │
             │  - Webhooks (stage change alerts)   │
             │  - Model Card Generation            │
             │  - Reproducibility (env capture)    │
             └──────────────────────────────────────┘
    ```

    ## Production Implementation (280 lines)

    ```python
    # model_registry.py
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.models.signature import infer_signature
    from typing import Dict, List, Optional, Any
    import pandas as pd
    import numpy as np
    from datetime import datetime
    from dataclasses import dataclass
    import json
    import logging
    from enum import Enum

    # ============= Configuration =============
    class ModelStage(Enum):
        """Model lifecycle stages"""
        NONE = "None"
        STAGING = "Staging"
        PRODUCTION = "Production"
        ARCHIVED = "Archived"

    @dataclass
    class ModelRegistryConfig:
        """Model registry configuration"""
        tracking_uri: str = "http://mlflow-server:5000"
        artifact_location: str = "s3://ml-models"
        experiment_name: str = "default"
        min_accuracy_staging: float = 0.80
        min_accuracy_production: float = 0.90

    config = ModelRegistryConfig()

    # ============= Model Registry Client =============
    class ModelRegistry:
        """Production model registry with lifecycle management"""

        def __init__(self, config: ModelRegistryConfig):
            self.config = config
            mlflow.set_tracking_uri(config.tracking_uri)
            self.client = MlflowClient()
            self.logger = logging.getLogger(__name__)

        def register_model(
            self,
            model: Any,
            model_name: str,
            X_sample: np.ndarray,
            y_sample: np.ndarray,
            metrics: Dict[str, float],
            params: Dict[str, Any],
            tags: Optional[Dict[str, str]] = None,
            artifacts: Optional[Dict[str, str]] = None,
            description: str = ""
        ) -> str:
            """
            Register a new model with comprehensive metadata

            Returns: model_version (e.g., "1", "2", etc.)
            """
            # Start MLflow run
            with mlflow.start_run() as run:
                # Log parameters
                mlflow.log_params(params)

                # Log metrics
                mlflow.log_metrics(metrics)

                # Log tags
                if tags:
                    mlflow.set_tags(tags)

                # Infer model signature for input/output validation
                signature = infer_signature(X_sample, model.predict(X_sample))

                # Log model with signature
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    signature=signature,
                    registered_model_name=model_name
                )

                # Log additional artifacts (plots, datasets, etc.)
                if artifacts:
                    for name, path in artifacts.items():
                        mlflow.log_artifact(path, artifact_path=name)

                # Log dataset samples for reproducibility
                train_data = pd.DataFrame(X_sample)
                train_data['target'] = y_sample
                mlflow.log_input(
                    mlflow.data.from_pandas(train_data),
                    context="training"
                )

                run_id = run.info.run_id

            # Get the registered model version
            model_version = self._get_latest_version(model_name)

            # Add model description
            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=model_version,
                    description=description
                )

            # Log lineage information
            self._log_lineage(model_name, model_version, params)

            self.logger.info(f"Registered {model_name} v{model_version} (run_id: {run_id})")
            return model_version

        def transition_stage(
            self,
            model_name: str,
            version: str,
            stage: ModelStage,
            archive_existing: bool = True
        ) -> bool:
            """
            Transition model to a new stage with validation

            Returns: True if transition successful
            """
            try:
                # Validate model meets requirements for the stage
                if not self._validate_for_stage(model_name, version, stage):
                    self.logger.error(f"Model {model_name} v{version} failed validation for {stage.value}")
                    return False

                # Archive existing models in target stage if requested
                if archive_existing and stage in [ModelStage.STAGING, ModelStage.PRODUCTION]:
                    self._archive_existing_models(model_name, stage)

                # Transition to new stage
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage=stage.value,
                    archive_existing_versions=archive_existing
                )

                # Send notification (webhook, Slack, email, etc.)
                self._notify_stage_change(model_name, version, stage)

                self.logger.info(f"Transitioned {model_name} v{version} to {stage.value}")
                return True

            except Exception as e:
                self.logger.error(f"Stage transition failed: {e}")
                return False

        def get_model(
            self,
            model_name: str,
            version: Optional[str] = None,
            stage: Optional[ModelStage] = None
        ) -> Any:
            """
            Load a model by version or stage

            If both version and stage are None, returns latest production model
            """
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                model_uri = f"models:/{model_name}/{stage.value}"
            else:
                model_uri = f"models:/{model_name}/Production"

            try:
                model = mlflow.sklearn.load_model(model_uri)
                self.logger.info(f"Loaded model from {model_uri}")
                return model
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                raise

        def compare_models(
            self,
            model_name: str,
            versions: List[str],
            metrics: List[str]
        ) -> pd.DataFrame:
            """
            Compare multiple versions of a model side-by-side
            """
            comparison_data = []

            for version in versions:
                try:
                    # Get model version details
                    mv = self.client.get_model_version(model_name, version)

                    # Get run metrics
                    run = self.client.get_run(mv.run_id)
                    metrics_data = {m: run.data.metrics.get(m) for m in metrics}

                    comparison_data.append({
                        'version': version,
                        'stage': mv.current_stage,
                        'created': datetime.fromtimestamp(mv.creation_timestamp / 1000),
                        **metrics_data
                    })
                except Exception as e:
                    self.logger.warning(f"Skipping version {version}: {e}")

            return pd.DataFrame(comparison_data)

        def search_models(
            self,
            filter_string: str = "",
            max_results: int = 100
        ) -> List[Dict]:
            """
            Search for models using filter syntax

            Examples:
            - "name='fraud_detector'"
            - "tags.team='risk'"
            - "run.metrics.accuracy > 0.9"
            """
            results = self.client.search_model_versions(
                filter_string=filter_string,
                max_results=max_results
            )

            return [{
                'name': mv.name,
                'version': mv.version,
                'stage': mv.current_stage,
                'run_id': mv.run_id,
                'created': datetime.fromtimestamp(mv.creation_timestamp / 1000)
            } for mv in results]

        def get_model_lineage(
            self,
            model_name: str,
            version: str
        ) -> Dict[str, Any]:
            """
            Get full lineage: dataset, code, dependencies
            """
            mv = self.client.get_model_version(model_name, version)
            run = self.client.get_run(mv.run_id)

            lineage = {
                'model': {
                    'name': model_name,
                    'version': version,
                    'created': datetime.fromtimestamp(mv.creation_timestamp / 1000)
                },
                'training': {
                    'run_id': mv.run_id,
                    'user': run.info.user_id,
                    'start_time': datetime.fromtimestamp(run.info.start_time / 1000)
                },
                'data': {
                    'dataset_version': run.data.tags.get('dataset_version'),
                    'data_path': run.data.tags.get('data_path')
                },
                'code': {
                    'git_commit': run.data.tags.get('git_commit'),
                    'git_branch': run.data.tags.get('git_branch'),
                    'code_version': run.data.tags.get('code_version')
                },
                'params': run.data.params,
                'metrics': run.data.metrics,
                'tags': run.data.tags
            }

            return lineage

        def delete_model_version(
            self,
            model_name: str,
            version: str
        ):
            """
            Delete a specific model version (only if not in Production)
            """
            mv = self.client.get_model_version(model_name, version)

            if mv.current_stage == ModelStage.PRODUCTION.value:
                raise ValueError("Cannot delete model in Production stage")

            self.client.delete_model_version(model_name, version)
            self.logger.info(f"Deleted {model_name} v{version}")

        # ============= Private Helper Methods =============

        def _get_latest_version(self, model_name: str) -> str:
            """Get the latest version number for a model"""
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if not versions:
                return "1"
            return max([int(v.version) for v in versions])

        def _validate_for_stage(
            self,
            model_name: str,
            version: str,
            stage: ModelStage
        ) -> bool:
            """Validate model meets requirements for stage"""
            mv = self.client.get_model_version(model_name, version)
            run = self.client.get_run(mv.run_id)

            accuracy = run.data.metrics.get('accuracy', 0)

            if stage == ModelStage.STAGING:
                return accuracy >= self.config.min_accuracy_staging
            elif stage == ModelStage.PRODUCTION:
                return accuracy >= self.config.min_accuracy_production
            else:
                return True

        def _archive_existing_models(self, model_name: str, stage: ModelStage):
            """Archive all models currently in the target stage"""
            versions = self.client.search_model_versions(
                f"name='{model_name}' AND current_stage='{stage.value}'"
            )

            for mv in versions:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage=ModelStage.ARCHIVED.value
                )

        def _log_lineage(self, model_name: str, version: str, params: Dict):
            """Log lineage information to external system (DataHub, etc.)"""
            # Integration point for lineage tracking systems
            pass

        def _notify_stage_change(self, model_name: str, version: str, stage: ModelStage):
            """Send notification about stage change (Slack, PagerDuty, etc.)"""
            message = f"Model {model_name} v{version} transitioned to {stage.value}"
            self.logger.info(f"Notification: {message}")
            # Integration with notification systems

    # ============= Usage Example =============
    def example_workflow():
        """End-to-end example of model registry workflow"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification

        # Initialize registry
        registry = ModelRegistry(config)

        # 1. Train model
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        model = RandomForestClassifier(n_estimators=100, max_depth=10)
        model.fit(X, y)

        # Calculate metrics
        train_accuracy = model.score(X, y)

        # 2. Register model
        version = registry.register_model(
            model=model,
            model_name="fraud_detector",
            X_sample=X[:100],
            y_sample=y[:100],
            metrics={
                'accuracy': train_accuracy,
                'n_estimators': 100
            },
            params={
                'max_depth': 10,
                'min_samples_split': 2
            },
            tags={
                'team': 'risk',
                'git_commit': 'abc123',
                'dataset_version': 'v1.0'
            },
            description="Fraud detection model using Random Forest"
        )

        # 3. Transition to Staging
        registry.transition_stage(
            model_name="fraud_detector",
            version=version,
            stage=ModelStage.STAGING
        )

        # 4. Compare with other versions
        comparison = registry.compare_models(
            model_name="fraud_detector",
            versions=[version, str(int(version)-1)] if int(version) > 1 else [version],
            metrics=['accuracy', 'n_estimators']
        )
        print(comparison)

        # 5. Promote to Production (after validation)
        registry.transition_stage(
            model_name="fraud_detector",
            version=version,
            stage=ModelStage.PRODUCTION
        )

        # 6. Load production model for serving
        prod_model = registry.get_model(
            model_name="fraud_detector",
            stage=ModelStage.PRODUCTION
        )

        # 7. Get lineage
        lineage = registry.get_model_lineage("fraud_detector", version)
        print(json.dumps(lineage, indent=2, default=str))
    ```

    ## Technology Stack Comparison

    | Tool | Strengths | Weaknesses | Best For |
    |------|-----------|------------|----------|
    | **MLflow** | Open-source, vendor-neutral, rich ecosystem | Self-hosted complexity | Teams wanting full control |
    | **Weights & Biases** | Great UI, experiment tracking, collaboration | Closed-source, cost | Research teams, quick setup |
    | **AWS SageMaker** | AWS integration, managed service | Vendor lock-in | AWS-native environments |
    | **Azure ML** | Azure integration, AutoML | Vendor lock-in | Azure-native environments |
    | **Databricks MLflow** | Managed MLflow, Unity Catalog integration | Cost, Databricks dependency | Databricks users |
    | **Custom** | Full flexibility | High maintenance | Very specific requirements |

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **No Model Signature** | Input/output validation missing | Always log signature with `infer_signature()` |
    | **Lost Reproducibility** | Can't recreate model | Log dataset version, git commit, dependencies |
    | **Manual Stage Management** | Human error, slow releases | Automate with CI/CD + validation gates |
    | **No Access Control** | Security risk | Implement RBAC, audit logs |
    | **Stale Models in Prod** | Performance degradation | Auto-archive after 90 days, monitor drift |
    | **Large Model Binaries** | Slow downloads, storage cost | Use model compression, separate artifacts |
    | **Duplicate Models** | Clutter, confusion | Naming conventions, tags, search |
    | **No Model Cards** | Poor documentation | Auto-generate from metadata + manual notes |

    ## Real-World Examples

    **Uber's Michelangelo:**
    - **Scale:** 10K+ models, 1K+ daily registrations
    - **Features:** Multi-framework support, auto-versioning, stage management
    - **Architecture:** Custom registry + Hive metadata + S3 artifacts
    - **Impact:** Reduced model deployment time from weeks to hours

    **Netflix's Model Registry:**
    - **Scale:** 1K+ registered models, 100+ in production
    - **Features:** A/B testing integration, canary deployments
    - **Tools:** Custom registry built on S3 + DynamoDB
    - **Impact:** 10x faster model iteration cycles

    **Airbnb's ML Platform:**
    - **Scale:** 800+ models, 150+ teams
    - **Features:** MLflow + Zipline integration, auto-documentation
    - **Workflow:** Notebook → MLflow → CI/CD → Production
    - **Impact:** 5x increase in models deployed/quarter

    ## Model Card Generation

    ```python
    def generate_model_card(registry: ModelRegistry, model_name: str, version: str) -> str:
        """Auto-generate model card from registry metadata"""
        lineage = registry.get_model_lineage(model_name, version)
        mv = registry.client.get_model_version(model_name, version)

        card = f"""
        # Model Card: {model_name} v{version}

        ## Model Details
        - **Stage:** {mv.current_stage}
        - **Created:** {lineage['model']['created']}
        - **Owner:** {lineage['training']['user']}

        ## Intended Use
        - **Primary Use:** [Fill from tags/description]
        - **Out-of-Scope:** [Fill from tags/description]

        ## Training Data
        - **Dataset Version:** {lineage['data']['dataset_version']}
        - **Data Path:** {lineage['data']['data_path']}

        ## Performance
        {json.dumps(lineage['metrics'], indent=2)}

        ## Ethical Considerations
        - Bias: [Review required]
        - Fairness: [Review required]

        ## Caveats and Recommendations
        - [Based on model type and metrics]
        """
        return card
    ```

    !!! tip "Interviewer's Insight"
        Emphasizes model lifecycle management (None → Staging → Production), reproducibility through lineage tracking, and automation. Discusses model signatures for input validation, CI/CD integration for automated deployments, and shows knowledge of MLflow internals. Can explain trade-offs between hosted (W&B, SageMaker) vs self-hosted (MLflow) solutions.

---

### Design a Low-Latency Inference Service - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Performance` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ## Scale Requirements

    - **Throughput:** 10K-1M+ RPS (requests per second)
    - **Latency:** <50ms p99, <20ms p50, <100ms p99.9
    - **Models:** 10-100 models deployed concurrently
    - **Model Size:** 10MB-10GB per model
    - **Batch Size:** 1-128 requests (dynamic batching)
    - **GPU Utilization:** >70% target
    - **Availability:** 99.99% SLA

    ## Architecture

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                      Load Balancer (L7)                          │
    │                                                                  │
    │  - Round-robin with least-connections                           │
    │  - Health checks (every 10s)                                    │
    │  - Request routing by model_id                                  │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Inference Service (FastAPI)                   │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │              Request Handler (async)                      │  │
    │  │                                                            │  │
    │  │  1. Validate input                                        │  │
    │  │  2. Feature lookup (parallel)                             │  │
    │  │  3. Add to batch queue                                    │  │
    │  │  4. Wait for result (Future)                              │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │          Dynamic Batcher (background thread)              │  │
    │  │                                                            │  │
    │  │  Trigger batching when:                                   │  │
    │  │  - Queue size ≥ max_batch_size (e.g., 32)                │  │
    │  │  - OR timeout reached (e.g., 5ms)                         │  │
    │  │                                                            │  │
    │  │  Coalesces requests into single inference call            │  │
    │  └──────────────────────────────────────────────────────────┘  │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                   Model Inference Engine                         │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │         Model Cache (LRU, in-memory)                      │  │
    │  │                                                            │  │
    │  │  - Warm models (GPU VRAM)                                 │  │
    │  │  - Cold models (CPU RAM/Disk)                             │  │
    │  │  - Auto-eviction based on usage                           │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │          GPU Inference (TensorRT/ONNX)                    │  │
    │  │                                                            │  │
    │  │  - FP16/INT8 quantization                                 │  │
    │  │  - Kernel fusion                                          │  │
    │  │  - Dynamic shapes                                         │  │
    │  │  - Multi-stream execution                                 │  │
    │  └──────────────────────────────────────────────────────────┘  │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                  Feature Store (Redis/Aerospike)                 │
    │                                                                  │
    │  - Online features (<5ms p99)                                   │
    │  - Connection pooling                                           │
    │  - Batch get operations                                         │
    └─────────────────────────────────────────────────────────────────┘

             ┌──────────────────────────────────────┐
             │      Cross-Cutting Optimizations     │
             │                                      │
             │  - Response caching (Redis)         │
             │  - Feature caching (TTL: 1min)      │
             │  - Connection pooling               │
             │  - Async I/O (asyncio)              │
             │  - Zero-copy where possible         │
             └──────────────────────────────────────┘
    ```

    ## Latency Budget Breakdown

    ```
    Total: 50ms p99 target

    ┌─────────────────────────────────────────────────────────┐
    │  1. Network (Load Balancer → Service)      5ms          │
    │  2. Request Validation                     1ms          │
    │  3. Feature Lookup (Redis parallel)       10ms          │
    │  4. Batching Wait Time                     5ms (max)    │
    │  5. Model Inference (GPU)                 20ms          │
    │     - Input preprocessing                  2ms          │
    │     - GPU compute                         15ms          │
    │     - Output postprocessing                3ms          │
    │  6. Result Serialization                   2ms          │
    │  7. Network (Service → Client)             7ms          │
    └─────────────────────────────────────────────────────────┘

    Optimization priorities:
    1. GPU compute (15ms) → quantization, TensorRT
    2. Feature lookup (10ms) → caching, batch fetch
    3. Batching wait (5ms) → tuned timeout/batch size
    ```

    ## Production Implementation (300 lines)

    ```python
    # low_latency_inference.py
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import torch
    import tensorrt as trt
    import numpy as np
    from typing import List, Dict, Any, Optional
    import asyncio
    import redis.asyncio as aioredis
    from collections import deque
    from dataclasses import dataclass
    import time
    import logging
    from concurrent.futures import ThreadPoolExecutor
    import uvicorn

    # ============= Configuration =============
    @dataclass
    class InferenceConfig:
        """Low-latency inference configuration"""
        max_batch_size: int = 32
        batch_timeout_ms: int = 5  # ms
        feature_cache_ttl: int = 60  # seconds
        max_queue_size: int = 1000
        gpu_device: int = 0
        num_workers: int = 4
        warmup_requests: int = 100

    config = InferenceConfig()

    # ============= Request/Response Models =============
    class InferenceRequest(BaseModel):
        """Input request schema"""
        model_id: str
        features: Optional[Dict[str, Any]] = None
        feature_keys: Optional[List[str]] = None  # For feature store lookup
        use_cache: bool = True

    class InferenceResponse(BaseModel):
        """Output response schema"""
        predictions: List[float]
        model_version: str
        latency_ms: float
        cache_hit: bool = False

    # ============= Dynamic Batcher =============
    class DynamicBatcher:
        """
        Batches requests dynamically based on size and timeout
        Inspired by NVIDIA Triton and TensorFlow Serving
        """

        def __init__(self, config: InferenceConfig):
            self.config = config
            self.queue: deque = deque()
            self.pending_futures: Dict[int, asyncio.Future] = {}
            self.batch_id = 0
            self.lock = asyncio.Lock()
            self.logger = logging.getLogger(__name__)

        async def add_request(self, request: InferenceRequest) -> np.ndarray:
            """Add request to batch queue and wait for result"""
            request_id = id(request)
            future = asyncio.Future()

            async with self.lock:
                if len(self.queue) >= self.config.max_queue_size:
                    raise HTTPException(status_code=503, detail="Queue full")

                self.queue.append((request_id, request))
                self.pending_futures[request_id] = future

            # Wait for result with timeout
            try:
                result = await asyncio.wait_for(
                    future,
                    timeout=self.config.batch_timeout_ms * 10 / 1000  # 10x timeout for safety
                )
                return result
            except asyncio.TimeoutError:
                self.logger.error(f"Request {request_id} timed out")
                raise HTTPException(status_code=504, detail="Inference timeout")

        async def process_batches(self, model_engine):
            """Background task to process batches"""
            while True:
                batch_start = time.perf_counter()

                # Wait for batch to fill or timeout
                await asyncio.sleep(self.config.batch_timeout_ms / 1000)

                async with self.lock:
                    if not self.queue:
                        continue

                    # Extract batch (up to max_batch_size)
                    batch_size = min(len(self.queue), self.config.max_batch_size)
                    batch = [self.queue.popleft() for _ in range(batch_size)]

                if not batch:
                    continue

                # Run inference on batch
                try:
                    request_ids, requests = zip(*batch)
                    results = await model_engine.infer_batch(list(requests))

                    # Resolve futures with results
                    for request_id, result in zip(request_ids, results):
                        if request_id in self.pending_futures:
                            self.pending_futures[request_id].set_result(result)
                            del self.pending_futures[request_id]

                    batch_latency = (time.perf_counter() - batch_start) * 1000
                    self.logger.info(f"Processed batch of {batch_size} in {batch_latency:.2f}ms")

                except Exception as e:
                    self.logger.error(f"Batch inference failed: {e}")
                    # Reject all requests in batch
                    for request_id, _ in batch:
                        if request_id in self.pending_futures:
                            self.pending_futures[request_id].set_exception(e)
                            del self.pending_futures[request_id]

    # ============= Model Engine with TensorRT =============
    class TensorRTModelEngine:
        """
        Optimized model inference using TensorRT
        """

        def __init__(self, config: InferenceConfig):
            self.config = config
            self.models: Dict[str, Any] = {}  # model_id -> TRT engine
            self.device = torch.device(f"cuda:{config.gpu_device}")
            self.logger = logging.getLogger(__name__)
            self.warmup_done = False

        def load_model(self, model_id: str, model_path: str):
            """Load and optimize model with TensorRT"""
            self.logger.info(f"Loading model {model_id} from {model_path}")

            # Load PyTorch model
            model = torch.jit.load(model_path)
            model = model.to(self.device)
            model.eval()

            # Convert to TensorRT (simplified - actual conversion is more complex)
            # In production, use torch2trt or ONNX → TensorRT pipeline
            self.models[model_id] = {
                'model': model,
                'version': '1.0',
                'input_shape': (None, 128),  # Dynamic batch
                'warmup_done': False
            }

            # Warmup
            self._warmup_model(model_id)

        def _warmup_model(self, model_id: str):
            """Warmup model with dummy requests for kernel optimization"""
            model_info = self.models[model_id]
            model = model_info['model']

            self.logger.info(f"Warming up model {model_id}")
            with torch.no_grad():
                for batch_size in [1, 8, 16, 32]:
                    dummy_input = torch.randn(
                        batch_size, 128, device=self.device, dtype=torch.float16
                    )
                    for _ in range(10):
                        _ = model(dummy_input)

            torch.cuda.synchronize()
            model_info['warmup_done'] = True
            self.logger.info(f"Warmup complete for {model_id}")

        async def infer_batch(self, requests: List[InferenceRequest]) -> List[np.ndarray]:
            """Run inference on a batch of requests"""
            if not requests:
                return []

            # Assume all requests use same model (can be extended)
            model_id = requests[0].model_id

            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not loaded")

            # Prepare batch input
            inputs = []
            for req in requests:
                # In production, this would fetch from feature store
                input_tensor = np.random.randn(128).astype(np.float16)
                inputs.append(input_tensor)

            batch_input = torch.tensor(
                np.array(inputs), device=self.device, dtype=torch.float16
            )

            # Run inference with torch.cuda.nvtx for profiling
            with torch.no_grad():
                start = time.perf_counter()
                outputs = self.models[model_id]['model'](batch_input)
                torch.cuda.synchronize()  # Wait for GPU
                latency = (time.perf_counter() - start) * 1000

            self.logger.debug(f"Batch inference: {len(requests)} requests in {latency:.2f}ms")

            # Convert to numpy
            return [output.cpu().numpy() for output in outputs]

    # ============= Feature Store Client =============
    class FeatureStoreClient:
        """
        Async feature store client with caching
        """

        def __init__(self, redis_url: str = "redis://localhost"):
            self.redis = None
            self.redis_url = redis_url
            self.cache: Dict[str, Any] = {}  # Local cache
            self.cache_ttl = config.feature_cache_ttl
            self.logger = logging.getLogger(__name__)

        async def connect(self):
            """Initialize Redis connection"""
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,
                max_connections=50  # Connection pooling
            )

        async def get_features(
            self, feature_keys: List[str], use_cache: bool = True
        ) -> np.ndarray:
            """
            Fetch features with parallel Redis queries and local caching
            """
            if use_cache:
                # Check local cache first
                cached = self._get_from_cache(feature_keys)
                if cached is not None:
                    return cached

            # Batch fetch from Redis (pipeline for parallelism)
            start = time.perf_counter()
            pipeline = self.redis.pipeline()
            for key in feature_keys:
                pipeline.get(key)

            results = await pipeline.execute()
            latency = (time.perf_counter() - start) * 1000

            self.logger.debug(f"Feature fetch: {len(feature_keys)} keys in {latency:.2f}ms")

            # Parse results
            features = np.array([float(r) if r else 0.0 for r in results])

            # Update cache
            if use_cache:
                self._update_cache(feature_keys, features)

            return features

        def _get_from_cache(self, keys: List[str]) -> Optional[np.ndarray]:
            """Check local cache for features"""
            cache_key = tuple(keys)
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                if time.time() - entry['timestamp'] < self.cache_ttl:
                    return entry['value']
                else:
                    del self.cache[cache_key]
            return None

        def _update_cache(self, keys: List[str], value: np.ndarray):
            """Update local cache"""
            cache_key = tuple(keys)
            self.cache[cache_key] = {
                'value': value,
                'timestamp': time.time()
            }

    # ============= FastAPI Application =============
    app = FastAPI(title="Low-Latency Inference Service")

    # Global state
    batcher: Optional[DynamicBatcher] = None
    model_engine: Optional[TensorRTModelEngine] = None
    feature_store: Optional[FeatureStoreClient] = None

    @app.on_event("startup")
    async def startup():
        """Initialize services on startup"""
        global batcher, model_engine, feature_store

        # Initialize components
        model_engine = TensorRTModelEngine(config)
        batcher = DynamicBatcher(config)
        feature_store = FeatureStoreClient()

        # Load models
        model_engine.load_model("model_v1", "models/model_v1.pt")

        # Connect to feature store
        await feature_store.connect()

        # Start background batcher
        asyncio.create_task(batcher.process_batches(model_engine))

        logging.info("Inference service started")

    @app.post("/predict", response_model=InferenceResponse)
    async def predict(request: InferenceRequest) -> InferenceResponse:
        """
        Low-latency prediction endpoint
        """
        start_time = time.perf_counter()

        try:
            # Fetch features if needed
            if request.feature_keys:
                features = await feature_store.get_features(
                    request.feature_keys,
                    use_cache=request.use_cache
                )
                request.features = {'input': features.tolist()}

            # Add to batch queue
            result = await batcher.add_request(request)

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            return InferenceResponse(
                predictions=result.tolist(),
                model_version="1.0",
                latency_ms=latency_ms,
                cache_hit=False  # Would track actual cache hits
            )

        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "models_loaded": len(model_engine.models) if model_engine else 0,
            "queue_size": len(batcher.queue) if batcher else 0
        }

    # ============= Main Entry Point =============
    if __name__ == "__main__":
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            workers=config.num_workers,
            log_level="info"
        )
    ```

    ## Optimization Techniques Comparison

    | Technique | Speedup | Accuracy Impact | Complexity | When to Use |
    |-----------|---------|----------------|------------|-------------|
    | **FP16 (Half Precision)** | 2-3x | Minimal (<0.5%) | Low | Almost always on modern GPUs |
    | **INT8 Quantization** | 3-4x | Small (1-2%) | Medium | When latency critical, post-training |
    | **Dynamic Batching** | 3-10x throughput | None | Medium | High QPS scenarios |
    | **Model Distillation** | 2-5x | Medium (2-5%) | High | When training new model is ok |
    | **TensorRT Optimization** | 2-5x | Minimal | Medium | NVIDIA GPUs, production deployment |
    | **ONNX Runtime** | 1.5-3x | Minimal | Low | Cross-platform, CPU/GPU |
    | **Model Pruning** | 1.5-3x | Medium (2-5%) | High | When model is overparameterized |
    | **Feature Caching** | 2-5x | None | Low | When features stable (1min+) |
    | **Response Caching** | 10-100x | None | Low | When exact requests repeat |

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Cold Start** | 5-10s first request | Warmup models with dummy requests at startup |
    | **Small Batches** | Low GPU utilization | Dynamic batching with timeout |
    | **CPU Bottleneck** | GPU idle, high latency | Async I/O, multi-threading for preprocessing |
    | **Memory Fragmentation** | OOM errors | Preallocate tensors, use memory pools |
    | **Blocking I/O** | Queue buildup | Use async Redis, async feature fetching |
    | **Large Models** | High VRAM, slow load | Model quantization, layer freezing |
    | **No Request Timeout** | Unbounded latency | Set max wait time (e.g., 100ms) |
    | **Synchronous GPU Calls** | Underutilized GPU | Use CUDA streams for parallelism |

    ## Real-World Examples

    **Uber's Real-Time Prediction Service:**
    - **Scale:** 100K+ RPS, <10ms p99
    - **Optimizations:** TensorFlow Serving, TensorRT INT8, batching
    - **Architecture:** Go service → TF Serving → GPU cluster
    - **Impact:** Handles surge pricing, ETA prediction at scale

    **Meta's PyTorch Inference:**
    - **Scale:** 1M+ RPS, <50ms p99
    - **Optimizations:** TorchScript, ONNX, custom CUDA kernels
    - **Models:** 100+ models, dynamic batching per model
    - **Impact:** Powers ads ranking, content recommendation

    **Google's TF Serving:**
    - **Scale:** 10M+ QPS aggregate
    - **Features:** Dynamic batching, model versioning, multi-model
    - **Latency:** <1ms for small models (embeddings)
    - **Impact:** Industry standard for model serving

    ## Monitoring Metrics

    ```python
    metrics_to_track = {
        'latency': {
            'p50': 'Median latency',
            'p95': '95th percentile',
            'p99': '99th percentile',
            'p99.9': '99.9th percentile'
        },
        'throughput': {
            'rps': 'Requests per second',
            'batch_size_avg': 'Average batch size',
            'queue_depth': 'Pending requests'
        },
        'resource': {
            'gpu_utilization': 'GPU compute %',
            'gpu_memory': 'VRAM usage',
            'cpu_utilization': 'CPU %',
            'network_bandwidth': 'MB/s'
        },
        'errors': {
            'timeout_rate': '% requests timing out',
            'error_rate': '% requests failing',
            'queue_full_rate': '% requests rejected'
        }
    }
    ```

    !!! tip "Interviewer's Insight"
        Emphasizes latency budget breakdown, dynamic batching for GPU efficiency, and multi-level optimization (model, serving, infrastructure). Discusses trade-offs between FP16/INT8 quantization and accuracy. Shows knowledge of TensorRT, async I/O, and production serving patterns from Uber/Meta/Google.

---

### Design a Search System - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Search`, `Information Retrieval` | **Asked by:** Google, Amazon, LinkedIn

??? success "View Answer"

    ## Scale Requirements

    - **Index Size:** 1B-1T documents
    - **Query Volume:** 10K-1M QPS (queries per second)
    - **Latency:** <100ms p99, <50ms p50
    - **Index Update:** Real-time (<1s) or near-real-time (<1min)
    - **Relevance:** NDCG@10 > 0.75, MRR > 0.80
    - **Availability:** 99.99% SLA
    - **Storage:** 10TB-10PB (index + documents)

    ## Architecture

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                        User Query                                │
    │                   "machin learning books"                        │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │              Query Understanding Layer                           │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │  1. Spell Correction: "machine learning books"            │  │
    │  │  2. Query Expansion: +["ML", "deep learning", "AI"]       │  │
    │  │  3. Intent Classification: [product_search, confidence=0.9]│ │
    │  │  4. Entity Extraction: ["machine learning" -> TOPIC]      │  │
    │  └──────────────────────────────────────────────────────────┘  │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                  Retrieval Layer (Stage 1)                       │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │         Elasticsearch / Solr (Inverted Index)             │  │
    │  │                                                            │  │
    │  │  BM25 Scoring:                                            │  │
    │  │  - Term frequency (TF)                                    │  │
    │  │  - Inverse document frequency (IDF)                       │  │
    │  │  - Document length normalization                          │  │
    │  │                                                            │  │
    │  │  Filters:                                                 │  │
    │  │  - Category, price range, rating                          │  │
    │  │  - Availability, location                                 │  │
    │  │                                                            │  │
    │  │  Retrieve top 1000 candidates (~10-20ms)                  │  │
    │  └──────────────────────────────────────────────────────────┘  │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                  Ranking Layer (Stage 2)                         │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │          Learning-to-Rank (LambdaMART / Neural)           │  │
    │  │                                                            │  │
    │  │  Features (100-1000 features):                            │  │
    │  │  - Text relevance: BM25, TF-IDF, exact match              │  │
    │  │  - Quality signals: CTR, conversion rate, ratings         │  │
    │  │  - Freshness: recency, update time                        │  │
    │  │  - User context: location, device, history                │  │
    │  │  - Item popularity: views, sales, trending                │  │
    │  │                                                            │  │
    │  │  Model: GBDT (e.g., LightGBM) or DNN                      │  │
    │  │  Re-rank top 100 results (~30-50ms)                       │  │
    │  └──────────────────────────────────────────────────────────┘  │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                  Personalization Layer (Stage 3)                 │
    │                                                                  │
    │  - User preferences (past clicks, purchases)                    │
    │  - Collaborative filtering (users like you bought...)           │
    │  - Diversity & exploration (avoid filter bubble)                │
    │  - Business rules (promotions, ads, editorial picks)            │
    │                                                                  │
    │  Final top 20 results (~10ms)                                   │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                      Search Results                              │
    │                                                                  │
    │  1. "Hands-On Machine Learning" ⭐4.8 $39.99                   │
    │  2. "Deep Learning" by Goodfellow ⭐4.9 $49.99                 │
    │  3. ...                                                         │
    └─────────────────────────────────────────────────────────────────┘

             ┌──────────────────────────────────────┐
             │      Supporting Components           │
             │                                      │
             │  - Indexing Pipeline (Kafka → ES)   │
             │  - Query Logs (click tracking)      │
             │  - A/B Testing Framework            │
             │  - Ranking Model Training           │
             │  - Autocomplete / Suggestions       │
             │  - Synonym Management               │
             └──────────────────────────────────────┘
    ```

    ## Production Implementation (310 lines)

    ```python
    # search_system.py
    from elasticsearch import Elasticsearch, helpers
    from typing import List, Dict, Any, Optional
    import numpy as np
    from dataclasses import dataclass
    import re
    from collections import defaultdict
    import lightgbm as lgb
    from scipy.spatial.distance import cosine
    import logging
    from datetime import datetime
    import hashlib

    # ============= Configuration =============
    @dataclass
    class SearchConfig:
        """Search system configuration"""
        es_hosts: List[str] = None
        index_name: str = "products"
        max_candidates: int = 1000
        max_results: int = 20
        ltr_model_path: str = "models/ranker.txt"
        min_score_threshold: float = 0.1

        def __post_init__(self):
            if self.es_hosts is None:
                self.es_hosts = ["localhost:9200"]

    config = SearchConfig()

    # ============= Query Understanding =============
    class QueryUnderstanding:
        """Query preprocessing and understanding"""

        def __init__(self):
            self.logger = logging.getLogger(__name__)
            # Load spell correction dictionary (simplified)
            self.spelling_corrections = {
                'machin': 'machine',
                'lerning': 'learning',
                'python': 'python',
                'javascrpit': 'javascript'
            }
            # Synonym expansion
            self.synonyms = {
                'ml': ['machine learning', 'ML'],
                'ai': ['artificial intelligence', 'AI'],
                'dl': ['deep learning', 'DL']
            }

        def process_query(self, query: str) -> Dict[str, Any]:
            """
            Process raw query through multiple stages
            """
            # 1. Normalize
            normalized = self._normalize(query)

            # 2. Spell correction
            corrected = self._spell_correct(normalized)

            # 3. Tokenize
            tokens = self._tokenize(corrected)

            # 4. Expand with synonyms
            expanded_tokens = self._expand_synonyms(tokens)

            # 5. Extract entities (simplified NER)
            entities = self._extract_entities(corrected)

            # 6. Classify intent
            intent = self._classify_intent(corrected)

            return {
                'original': query,
                'normalized': normalized,
                'corrected': corrected,
                'tokens': tokens,
                'expanded_tokens': expanded_tokens,
                'entities': entities,
                'intent': intent
            }

        def _normalize(self, query: str) -> str:
            """Lowercase, trim, remove special chars"""
            return ' '.join(query.lower().strip().split())

        def _spell_correct(self, query: str) -> str:
            """Simple spell correction using dictionary"""
            words = query.split()
            corrected = []
            for word in words:
                if word in self.spelling_corrections:
                    corrected.append(self.spelling_corrections[word])
                    self.logger.info(f"Spell correction: {word} → {self.spelling_corrections[word]}")
                else:
                    corrected.append(word)
            return ' '.join(corrected)

        def _tokenize(self, query: str) -> List[str]:
            """Simple whitespace tokenization"""
            return query.split()

        def _expand_synonyms(self, tokens: List[str]) -> List[str]:
            """Expand tokens with synonyms"""
            expanded = list(tokens)
            for token in tokens:
                if token in self.synonyms:
                    expanded.extend(self.synonyms[token])
            return expanded

        def _extract_entities(self, query: str) -> Dict[str, List[str]]:
            """Extract named entities (simplified)"""
            entities = defaultdict(list)
            # Pattern matching for common entities
            if 'python' in query:
                entities['language'].append('Python')
            if 'machine learning' in query or 'ml' in query:
                entities['topic'].append('Machine Learning')
            return dict(entities)

        def _classify_intent(self, query: str) -> Dict[str, Any]:
            """Classify user intent (simplified)"""
            # In production, use a trained classifier
            if any(word in query for word in ['buy', 'purchase', 'price']):
                return {'type': 'transactional', 'confidence': 0.9}
            elif any(word in query for word in ['how to', 'what is', 'tutorial']):
                return {'type': 'informational', 'confidence': 0.85}
            else:
                return {'type': 'navigational', 'confidence': 0.7}

    # ============= Elasticsearch Retrieval =============
    class ElasticsearchRetriever:
        """BM25-based retrieval using Elasticsearch"""

        def __init__(self, config: SearchConfig):
            self.config = config
            self.es = Elasticsearch(config.es_hosts)
            self.logger = logging.getLogger(__name__)

        def create_index(self):
            """Create Elasticsearch index with custom mapping"""
            mapping = {
                "mappings": {
                    "properties": {
                        "title": {
                            "type": "text",
                            "analyzer": "standard",
                            "fields": {
                                "keyword": {"type": "keyword"},
                                "ngram": {
                                    "type": "text",
                                    "analyzer": "ngram_analyzer"
                                }
                            }
                        },
                        "description": {"type": "text", "analyzer": "standard"},
                        "category": {"type": "keyword"},
                        "price": {"type": "float"},
                        "rating": {"type": "float"},
                        "num_reviews": {"type": "integer"},
                        "created_at": {"type": "date"},
                        "tags": {"type": "keyword"}
                    }
                },
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "ngram_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "ngram_filter"]
                            }
                        },
                        "filter": {
                            "ngram_filter": {
                                "type": "ngram",
                                "min_gram": 3,
                                "max_gram": 4
                            }
                        }
                    }
                }
            }

            if not self.es.indices.exists(index=self.config.index_name):
                self.es.indices.create(index=self.config.index_name, body=mapping)
                self.logger.info(f"Created index: {self.config.index_name}")

        def index_documents(self, documents: List[Dict[str, Any]]):
            """Bulk index documents"""
            actions = [
                {
                    "_index": self.config.index_name,
                    "_id": doc.get('id', hashlib.md5(doc['title'].encode()).hexdigest()),
                    "_source": doc
                }
                for doc in documents
            ]
            helpers.bulk(self.es, actions)
            self.logger.info(f"Indexed {len(documents)} documents")

        def search(
            self,
            query_info: Dict[str, Any],
            filters: Optional[Dict] = None,
            size: int = 1000
        ) -> List[Dict[str, Any]]:
            """
            Execute BM25 search with filters
            """
            # Build Elasticsearch query
            must_clauses = [
                {
                    "multi_match": {
                        "query": query_info['corrected'],
                        "fields": ["title^3", "description", "tags^2"],
                        "type": "best_fields",
                        "tie_breaker": 0.3
                    }
                }
            ]

            # Add expanded query terms with lower weight
            if query_info.get('expanded_tokens'):
                expanded_query = ' '.join(query_info['expanded_tokens'])
                must_clauses.append({
                    "multi_match": {
                        "query": expanded_query,
                        "fields": ["title", "description"],
                        "type": "phrase",
                        "boost": 0.5
                    }
                })

            # Build filter clauses
            filter_clauses = []
            if filters:
                if 'category' in filters:
                    filter_clauses.append({"term": {"category": filters['category']}})
                if 'min_price' in filters or 'max_price' in filters:
                    range_filter = {"range": {"price": {}}}
                    if 'min_price' in filters:
                        range_filter['range']['price']['gte'] = filters['min_price']
                    if 'max_price' in filters:
                        range_filter['range']['price']['lte'] = filters['max_price']
                    filter_clauses.append(range_filter)

            query = {
                "query": {
                    "bool": {
                        "must": must_clauses,
                        "filter": filter_clauses
                    }
                },
                "size": size,
                "_source": True
            }

            response = self.es.search(index=self.config.index_name, body=query)
            results = [
                {
                    **hit['_source'],
                    'doc_id': hit['_id'],
                    'bm25_score': hit['_score']
                }
                for hit in response['hits']['hits']
            ]

            self.logger.info(f"Retrieved {len(results)} candidates")
            return results

    # ============= Learning-to-Rank =============
    class LearningToRank:
        """LTR re-ranking using LightGBM"""

        def __init__(self, config: SearchConfig):
            self.config = config
            self.model = None
            self.logger = logging.getLogger(__name__)
            self._load_model()

        def _load_model(self):
            """Load pre-trained LightGBM ranker"""
            try:
                self.model = lgb.Booster(model_file=self.config.ltr_model_path)
                self.logger.info("Loaded LTR model")
            except Exception as e:
                self.logger.warning(f"Could not load LTR model: {e}")
                self.model = None

        def extract_features(
            self,
            query_info: Dict[str, Any],
            document: Dict[str, Any],
            user_context: Optional[Dict] = None
        ) -> np.ndarray:
            """
            Extract ranking features for query-document pair
            """
            features = []

            # 1. Text relevance features
            features.append(document.get('bm25_score', 0))
            features.append(self._exact_match_score(query_info['corrected'], document['title']))
            features.append(self._query_coverage(query_info['tokens'], document['title']))

            # 2. Quality signals
            features.append(document.get('rating', 0))
            features.append(np.log1p(document.get('num_reviews', 0)))
            features.append(document.get('conversion_rate', 0))

            # 3. Freshness
            days_old = self._days_since_creation(document.get('created_at'))
            features.append(1.0 / (1.0 + days_old))  # Decay with age

            # 4. Popularity
            features.append(np.log1p(document.get('view_count', 0)))
            features.append(np.log1p(document.get('sales_count', 0)))

            # 5. User personalization (if available)
            if user_context:
                features.append(self._user_affinity(user_context, document))
            else:
                features.append(0)

            return np.array(features, dtype=np.float32)

        def _exact_match_score(self, query: str, text: str) -> float:
            """Score for exact query match in text"""
            text_lower = text.lower()
            query_lower = query.lower()
            if query_lower in text_lower:
                # Bonus for match at beginning
                if text_lower.startswith(query_lower):
                    return 2.0
                return 1.0
            return 0.0

        def _query_coverage(self, query_tokens: List[str], text: str) -> float:
            """Fraction of query tokens found in text"""
            text_lower = text.lower()
            matches = sum(1 for token in query_tokens if token in text_lower)
            return matches / len(query_tokens) if query_tokens else 0

        def _days_since_creation(self, created_at: Optional[str]) -> int:
            """Calculate days since document creation"""
            if not created_at:
                return 365  # Default to 1 year old
            try:
                created = datetime.fromisoformat(created_at)
                return (datetime.now() - created).days
            except:
                return 365

        def _user_affinity(self, user_context: Dict, document: Dict) -> float:
            """User-document affinity score"""
            # Simplified - in production, use collaborative filtering
            user_categories = user_context.get('preferred_categories', [])
            doc_category = document.get('category', '')
            return 1.0 if doc_category in user_categories else 0.0

        def rank(
            self,
            query_info: Dict[str, Any],
            candidates: List[Dict[str, Any]],
            user_context: Optional[Dict] = None,
            top_k: int = 20
        ) -> List[Dict[str, Any]]:
            """
            Re-rank candidates using LTR model
            """
            if not self.model or not candidates:
                return candidates[:top_k]

            # Extract features for all candidates
            feature_matrix = np.array([
                self.extract_features(query_info, doc, user_context)
                for doc in candidates
            ])

            # Predict scores
            scores = self.model.predict(feature_matrix)

            # Sort by score
            ranked_indices = np.argsort(scores)[::-1]
            ranked_results = [
                {**candidates[i], 'ltr_score': float(scores[i])}
                for i in ranked_indices[:top_k]
            ]

            return ranked_results

    # ============= Search Service =============
    class SearchService:
        """Main search service orchestrating all components"""

        def __init__(self, config: SearchConfig):
            self.config = config
            self.query_understanding = QueryUnderstanding()
            self.retriever = ElasticsearchRetriever(config)
            self.ranker = LearningToRank(config)
            self.logger = logging.getLogger(__name__)

        def search(
            self,
            query: str,
            filters: Optional[Dict] = None,
            user_context: Optional[Dict] = None
        ) -> Dict[str, Any]:
            """
            End-to-end search pipeline
            """
            import time
            start_time = time.time()

            # 1. Query understanding
            query_info = self.query_understanding.process_query(query)
            self.logger.info(f"Understood query: {query_info['corrected']}")

            # 2. Retrieval (Stage 1)
            candidates = self.retriever.search(
                query_info,
                filters=filters,
                size=self.config.max_candidates
            )

            # 3. Ranking (Stage 2)
            ranked_results = self.ranker.rank(
                query_info,
                candidates,
                user_context=user_context,
                top_k=self.config.max_results
            )

            latency_ms = (time.time() - start_time) * 1000

            return {
                'query': query_info['corrected'],
                'results': ranked_results,
                'total_candidates': len(candidates),
                'latency_ms': latency_ms,
                'spelling_corrected': query != query_info['corrected']
            }

    # ============= Usage Example =============
    def example_usage():
        """Example search workflow"""
        service = SearchService(config)

        # Create index
        service.retriever.create_index()

        # Index sample documents
        documents = [
            {
                'id': '1',
                'title': 'Hands-On Machine Learning',
                'description': 'Practical ML with Scikit-Learn and TensorFlow',
                'category': 'Books',
                'price': 39.99,
                'rating': 4.8,
                'num_reviews': 2500,
                'created_at': '2023-01-15',
                'tags': ['machine learning', 'python', 'AI']
            },
            {
                'id': '2',
                'title': 'Deep Learning',
                'description': 'Comprehensive guide to deep learning by Goodfellow',
                'category': 'Books',
                'price': 49.99,
                'rating': 4.9,
                'num_reviews': 1800,
                'created_at': '2023-03-20',
                'tags': ['deep learning', 'neural networks', 'AI']
            }
        ]
        service.retriever.index_documents(documents)

        # Execute search
        results = service.search(
            query="machin learning books",  # Typo intentional
            filters={'category': 'Books'},
            user_context={'preferred_categories': ['Books', 'Technology']}
        )

        print(f"Query: {results['query']}")
        print(f"Latency: {results['latency_ms']:.2f}ms")
        print(f"Results: {len(results['results'])}")
        for i, result in enumerate(results['results'][:3], 1):
            print(f"{i}. {result['title']} - ${result['price']} ⭐{result['rating']}")
    ```

    ## Ranking Stage Comparison

    | Stage | Algorithm | Candidates | Latency | Use Case |
    |-------|-----------|------------|---------|----------|
    | **Stage 1: Retrieval** | BM25, TF-IDF | 1M → 1K | <20ms | Fast pruning from large corpus |
    | **Stage 2: Re-ranking** | LightGBM, BERT | 1K → 100 | <50ms | Feature-rich scoring |
    | **Stage 3: Personalization** | Collaborative Filtering | 100 → 20 | <10ms | User-specific adjustments |

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **No Spell Correction** | Miss ~10% queries | Use Levenshtein distance, context-aware correction |
    | **Single-Stage Ranking** | Slow or poor relevance | Multi-stage: fast retrieval → expensive re-ranking |
    | **No Query Expansion** | Miss synonyms/variations | Synonym dictionaries, word embeddings |
    | **Static Ranking** | Stale results | Incorporate real-time signals (CTR, freshness) |
    | **No Personalization** | Generic results | User history, collaborative filtering |
    | **Index Hotspots** | Uneven load | Shard by hash, avoid temporal sharding |
    | **No Diversity** | Filter bubble | MMR (Maximal Marginal Relevance), genre mixing |
    | **Ignoring Long Tail** | Miss niche queries | Fuzzy matching, relaxed filters for 0 results |

    ## Real-World Examples

    **Google Search:**
    - **Scale:** Billions of documents, 100K+ QPS
    - **Architecture:** Multi-tiered serving (L1: memory, L2: SSD, L3: disk)
    - **Ranking:** 200+ signals, PageRank + BERT embeddings + user signals
    - **Latency:** <200ms p99 with global query routing
    - **Impact:** Gold standard for search relevance

    **Amazon Product Search:**
    - **Scale:** 600M+ products, 1M+ QPS
    - **Architecture:** Elasticsearch + custom ranking service
    - **Ranking:** 150+ features (text, behavior, business metrics)
    - **Personalization:** Purchase history, browsing, collaborative filtering
    - **Impact:** 35% of revenue from search-driven purchases

    **LinkedIn Talent Search:**
    - **Scale:** 800M+ profiles, 100K+ QPS
    - **Architecture:** Galene (custom search engine) + LTR
    - **Ranking:** 50+ features (skills, experience, network, activity)
    - **Innovation:** Standardization (normalize titles, skills)
    - **Impact:** 70% of hires go through search

    ## Evaluation Metrics

    ```python
    def evaluate_search_quality(predicted_rankings: List[List[int]],
                                 ground_truth: List[List[int]]) -> Dict[str, float]:
        """
        Evaluate search quality using standard IR metrics
        """
        from sklearn.metrics import ndcg_score

        metrics = {}

        # NDCG@K (Normalized Discounted Cumulative Gain)
        for k in [5, 10, 20]:
            ndcg = ndcg_score(ground_truth, predicted_rankings, k=k)
            metrics[f'ndcg@{k}'] = ndcg

        # MRR (Mean Reciprocal Rank)
        mrr = 0
        for pred, truth in zip(predicted_rankings, ground_truth):
            for rank, item in enumerate(pred, 1):
                if item in truth:
                    mrr += 1.0 / rank
                    break
        metrics['mrr'] = mrr / len(predicted_rankings)

        return metrics
    ```

    !!! tip "Interviewer's Insight"
        Emphasizes multi-stage ranking (BM25 → LTR → personalization) for latency-quality trade-off, query understanding for handling typos/synonyms, and learning-to-rank with 100+ features. Discusses inverted index structure, sharding strategies, and evaluation metrics (NDCG, MRR). Can explain how Google/Amazon/LinkedIn implement search at scale with specific architectural choices.

---

### Design a Data Warehouse - Amazon, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Data Engineering` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    ## Scale Requirements

    - **Data Volume:** 100TB-10PB total storage
    - **Daily Ingestion:** 1TB-100TB/day
    - **Tables:** 100-10K tables (10-100 fact, 50-500 dimension)
    - **Queries:** 1K-100K queries/day
    - **Latency:** <5s for dashboards, <30s for ad-hoc, <5min for reports
    - **Users:** 100-10K analysts/data scientists
    - **Retention:** 1-7 years historical data

    ## Architecture

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                      Source Systems                              │
    │                                                                  │
    │  [Databases] [APIs] [SaaS Apps] [Event Streams] [Files]        │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                   Ingestion Layer (ELT)                          │
    │                                                                  │
    │  Batch:                    CDC:                  Streaming:     │
    │  Fivetran, Airbyte        Debezium             Kafka Connect    │
    │  (daily/hourly)           (real-time)          (real-time)      │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │              Raw/Staging Layer (S3/GCS/ADLS)                     │
    │                                                                  │
    │  /raw/source_name/table_name/yyyy/mm/dd/data.parquet           │
    │  - Immutable source data                                        │
    │  - Partitioned by ingestion date                                │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │           Transformation Layer (DBT/Spark/Airflow)               │
    │                                                                  │
    │  DBT Models:                                                    │
    │  1. Staging: Clean, type-cast, standardize                     │
    │  2. Intermediate: Joins, aggregations, deduplication           │
    │  3. Marts: Business-ready star/snowflake schemas               │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │         Data Quality Checks (Great Expectations)          │  │
    │  │  - Schema validation                                      │  │
    │  │  - Referential integrity                                  │  │
    │  │  - Business rules (e.g., revenue > 0)                     │  │
    │  └──────────────────────────────────────────────────────────┘  │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │          Data Warehouse (BigQuery/Snowflake/Redshift)            │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │                 Star Schema Design                        │  │
    │  │                                                            │  │
    │  │       Fact Tables:                                        │  │
    │  │       ┌─────────────────────┐                             │  │
    │  │       │   fact_sales        │                             │  │
    │  │       ├─────────────────────┤                             │  │
    │  │       │ sale_id (PK)        │                             │  │
    │  │       │ date_key (FK) ──────┼───┐                         │  │
    │  │       │ product_key (FK) ───┼─┐ │                         │  │
    │  │       │ customer_key (FK) ──┼┐│ │                         │  │
    │  │       │ store_key (FK) ─────┼┼┼─┼──┐                      │  │
    │  │       │ quantity            ││││ │  │                      │  │
    │  │       │ revenue             ││││ │  │                      │  │
    │  │       │ cost                ││││ │  │                      │  │
    │  │       └─────────────────────┘│││ │  │                      │  │
    │  │                               │││ │  │                      │  │
    │  │       Dimension Tables:       │││ │  │                      │  │
    │  │       ┌──────────────┐        │││ │  │                      │  │
    │  │       │ dim_customer │◄───────┘││ │  │                      │  │
    │  │       ├──────────────┤         ││ │  │                      │  │
    │  │       │ customer_key │         ││ │  │                      │  │
    │  │       │ name         │         ││ │  │                      │  │
    │  │       │ email        │         ││ │  │                      │  │
    │  │       │ segment      │         ││ │  │                      │  │
    │  │       │ valid_from   │ (SCD)   ││ │  │                      │  │
    │  │       │ valid_to     │ (Type2) ││ │  │                      │  │
    │  │       └──────────────┘         ││ │  │                      │  │
    │  │                                ││ │  │                      │  │
    │  │       ┌──────────────┐         ││ │  │                      │  │
    │  │       │ dim_product  │◄────────┘│ │  │                      │  │
    │  │       ├──────────────┤          │ │  │                      │  │
    │  │       │ product_key  │          │ │  │                      │  │
    │  │       │ product_name │          │ │  │                      │  │
    │  │       │ category     │          │ │  │                      │  │
    │  │       │ brand        │          │ │  │                      │  │
    │  │       └──────────────┘          │ │  │                      │  │
    │  │                                 │ │  │                      │  │
    │  │       ┌──────────────┐          │ │  │                      │  │
    │  │       │ dim_date     │◄─────────┘ │  │                      │  │
    │  │       ├──────────────┤            │  │                      │  │
    │  │       │ date_key     │            │  │                      │  │
    │  │       │ date         │            │  │                      │  │
    │  │       │ day_of_week  │            │  │                      │  │
    │  │       │ month        │            │  │                      │  │
    │  │       │ quarter      │            │  │                      │  │
    │  │       │ is_holiday   │            │  │                      │  │
    │  │       └──────────────┘            │  │                      │  │
    │  │                                   │  │                      │  │
    │  │       ┌──────────────┐            │  │                      │  │
    │  │       │ dim_store    │◄───────────┘  │                      │  │
    │  │       ├──────────────┤               │                      │  │
    │  │       │ store_key    │               │                      │  │
    │  │       │ store_name   │               │                      │  │
    │  │       │ city         │               │                      │  │
    │  │       │ country      │               │                      │  │
    │  │       └──────────────┘               │                      │  │
    │  └──────────────────────────────────────┘                      │  │
    │                                                                  │
    │  Partitioning: fact_sales partitioned by date_key               │
    │  Clustering: clustered by (customer_key, product_key)           │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Semantic/Metrics Layer                        │
    │                                                                  │
    │  dbt Metrics / LookML / Cube.js:                                │
    │  - Total Revenue = SUM(revenue)                                 │
    │  - Average Order Value = AVG(revenue)                           │
    │  - Customer Lifetime Value = SUM(revenue) per customer          │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                   Consumption Layer                              │
    │                                                                  │
    │  [BI Tools: Tableau, Looker, Power BI]                          │
    │  [Data Science: Python, R, Notebooks]                           │
    │  [Reverse ETL: Back to operational systems]                     │
    └─────────────────────────────────────────────────────────────────┘
    ```

    ## Production Implementation (290 lines)

    ```python
    # data_warehouse.py
    from typing import List, Dict, Any, Optional
    from dataclasses import dataclass
    from datetime import datetime, date
    from enum import Enum
    import pandas as pd
    import logging

    # ============= Configuration =============
    class SCDType(Enum):
        """Slowly Changing Dimension types"""
        TYPE_0 = 0  # No changes
        TYPE_1 = 1  # Overwrite
        TYPE_2 = 2  # Add new row with versioning
        TYPE_3 = 3  # Add new column
        TYPE_4 = 4  # Separate history table

    @dataclass
    class WarehouseConfig:
        """Data warehouse configuration"""
        warehouse_type: str = "bigquery"  # bigquery, snowflake, redshift
        project_id: str = "my-project"
        dataset_id: str = "analytics"
        partition_field: str = "date_key"
        cluster_fields: List[str] = None

        def __post_init__(self):
            if self.cluster_fields is None:
                self.cluster_fields = ["customer_key", "product_key"]

    config = WarehouseConfig()

    # ============= Star Schema Design =============
    class StarSchemaDesigner:
        """Design and implement star schema"""

        def __init__(self):
            self.logger = logging.getLogger(__name__)

        def generate_fact_table_ddl(
            self,
            table_name: str,
            measures: List[str],
            dimensions: List[str],
            partition_by: Optional[str] = None,
            cluster_by: Optional[List[str]] = None
        ) -> str:
            """
            Generate DDL for fact table with partitioning and clustering
            """
            # BigQuery DDL
            ddl = f"""
    CREATE TABLE IF NOT EXISTS {config.project_id}.{config.dataset_id}.{table_name}
    (
        {table_name}_id INT64 NOT NULL,  -- Surrogate key

        -- Dimension foreign keys
        {chr(10).join(f'    {dim}_key INT64 NOT NULL,' for dim in dimensions)}

        -- Measures (metrics)
        {chr(10).join(f'    {measure} FLOAT64,' for measure in measures)}

        -- Audit columns
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP(),
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
    )
    """

            # Add partitioning
            if partition_by:
                ddl += f"\nPARTITION BY DATE({partition_by})"

            # Add clustering
            if cluster_by:
                ddl += f"\nCLUSTER BY {', '.join(cluster_by)}"

            ddl += ";"

            return ddl

        def generate_dimension_table_ddl(
            self,
            table_name: str,
            attributes: List[str],
            scd_type: SCDType = SCDType.TYPE_1
        ) -> str:
            """
            Generate DDL for dimension table with SCD support
            """
            ddl = f"""
    CREATE TABLE IF NOT EXISTS {config.project_id}.{config.dataset_id}.{table_name}
    (
        {table_name}_key INT64 NOT NULL,  -- Surrogate key
        {table_name}_id STRING NOT NULL,  -- Natural key (business key)

        -- Dimension attributes
        {chr(10).join(f'    {attr} STRING,' for attr in attributes)}
    """

            # SCD Type 2 specific columns
            if scd_type == SCDType.TYPE_2:
                ddl += """
        -- SCD Type 2 columns
        valid_from DATE NOT NULL,
        valid_to DATE,
        is_current BOOL NOT NULL DEFAULT TRUE,
        version INT64 NOT NULL DEFAULT 1,
    """

            # Audit columns
            ddl += """
        -- Audit columns
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP(),
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
    );
    """

            return ddl

    # ============= Slowly Changing Dimensions =============
    class SCDHandler:
        """Handle Slowly Changing Dimensions"""

        def __init__(self):
            self.logger = logging.getLogger(__name__)

        def apply_scd_type_1(
            self,
            existing_dim: pd.DataFrame,
            new_data: pd.DataFrame,
            natural_key: str,
            attributes: List[str]
        ) -> pd.DataFrame:
            """
            SCD Type 1: Overwrite (no history)

            Simply update the existing record with new values
            """
            # Merge on natural key
            merged = existing_dim.set_index(natural_key).combine_first(
                new_data.set_index(natural_key)
            ).reset_index()

            merged['updated_at'] = datetime.now()

            self.logger.info(f"Applied SCD Type 1 to {len(new_data)} records")
            return merged

        def apply_scd_type_2(
            self,
            existing_dim: pd.DataFrame,
            new_data: pd.DataFrame,
            natural_key: str,
            attributes: List[str],
            effective_date: date = None
        ) -> pd.DataFrame:
            """
            SCD Type 2: Add new row with versioning (preserves history)

            When a dimension changes:
            1. Mark old record as no longer current (is_current=False, valid_to=today)
            2. Insert new record with new values (is_current=True, valid_from=today)
            """
            if effective_date is None:
                effective_date = date.today()

            result_records = []

            for _, new_record in new_data.iterrows():
                # Find existing record with same natural key
                existing = existing_dim[
                    (existing_dim[natural_key] == new_record[natural_key]) &
                    (existing_dim['is_current'] == True)
                ]

                if existing.empty:
                    # New dimension member
                    new_record['valid_from'] = effective_date
                    new_record['valid_to'] = None
                    new_record['is_current'] = True
                    new_record['version'] = 1
                    new_record['created_at'] = datetime.now()
                    result_records.append(new_record)
                else:
                    # Check if any attributes changed
                    changed = False
                    for attr in attributes:
                        if existing.iloc[0][attr] != new_record[attr]:
                            changed = True
                            break

                    if changed:
                        # Expire old record
                        old_record = existing.iloc[0].copy()
                        old_record['valid_to'] = effective_date
                        old_record['is_current'] = False
                        old_record['updated_at'] = datetime.now()
                        result_records.append(old_record)

                        # Insert new record
                        new_record['valid_from'] = effective_date
                        new_record['valid_to'] = None
                        new_record['is_current'] = True
                        new_record['version'] = existing.iloc[0]['version'] + 1
                        new_record['created_at'] = datetime.now()
                        result_records.append(new_record)

                        self.logger.info(
                            f"Applied SCD Type 2: {natural_key}={new_record[natural_key]}, "
                            f"version {old_record['version']} → {new_record['version']}"
                        )
                    else:
                        # No change, keep existing
                        result_records.append(existing.iloc[0])

            return pd.DataFrame(result_records)

        def apply_scd_type_3(
            self,
            existing_dim: pd.DataFrame,
            new_data: pd.DataFrame,
            natural_key: str,
            tracked_attributes: List[str]
        ) -> pd.DataFrame:
            """
            SCD Type 3: Add column for previous value (limited history)

            E.g., current_city, previous_city
            """
            for _, new_record in new_data.iterrows():
                mask = existing_dim[natural_key] == new_record[natural_key]

                if mask.any():
                    for attr in tracked_attributes:
                        old_value = existing_dim.loc[mask, attr].iloc[0]
                        new_value = new_record[attr]

                        if old_value != new_value:
                            # Move current to previous
                            existing_dim.loc[mask, f'previous_{attr}'] = old_value
                            existing_dim.loc[mask, attr] = new_value
                            self.logger.info(f"SCD Type 3: {attr} changed from {old_value} to {new_value}")

            return existing_dim

    # ============= DBT Model Generator =============
    class DBTModelGenerator:
        """Generate DBT models for warehouse transformations"""

        def generate_staging_model(self, source_table: str, transformations: Dict[str, str]) -> str:
            """
            Generate DBT staging model (cleansing, type casting)
            """
            model = f"""
    -- models/staging/stg_{source_table}.sql
    {{{{
        config(
            materialized='view'
        )
    }}}}

    WITH source AS (
        SELECT * FROM {{{{ source('raw', '{source_table}') }}}}
    ),

    renamed AS (
        SELECT
            {chr(10).join(f'        {old} AS {new},' for old, new in transformations.items())}
            _loaded_at
        FROM source
    )

    SELECT * FROM renamed
    """
            return model

        def generate_fact_model(
            self,
            fact_name: str,
            source_tables: List[str],
            measures: List[str],
            dimensions: List[str]
        ) -> str:
            """
            Generate DBT fact table model
            """
            model = f"""
    -- models/marts/{fact_name}.sql
    {{{{
        config(
            materialized='incremental',
            partition_by={{
                'field': 'date_key',
                'data_type': 'date',
                'granularity': 'day'
            }},
            cluster_by=['customer_key', 'product_key']
        )
    }}}}

    WITH base AS (
        SELECT * FROM {{{{ ref('stg_{source_tables[0]}') }}}}
        {f"LEFT JOIN {{{{ ref('stg_{source_tables[1]}') }}}} USING (join_key)" if len(source_tables) > 1 else ''}
    ),

    final AS (
        SELECT
            {{ dbt_utils.generate_surrogate_key([
                {', '.join(f"'{dim}'" for dim in dimensions)}
            ]) }} AS {fact_name}_id,

            -- Dimension keys
            {chr(10).join(f'        {dim}_key,' for dim in dimensions)}

            -- Measures
            {chr(10).join(f'        {measure},' for measure in measures)}

            CURRENT_TIMESTAMP() AS created_at
        FROM base

        {{%- if is_incremental() %}}
        WHERE date_key > (SELECT MAX(date_key) FROM {{{{ this }}}})
        {{%- endif %}}
    )

    SELECT * FROM final
    """
            return model

        def generate_data_quality_tests(self, table_name: str, unique_cols: List[str], not_null_cols: List[str]) -> str:
            """
            Generate DBT data quality tests (YAML)
            """
            yaml = f"""
    # models/{table_name}.yml
    version: 2

    models:
      - name: {table_name}
        description: "Fact table for {table_name}"

        tests:
          - dbt_expectations.expect_table_row_count_to_be_between:
              min_value: 1000

        columns:
          {chr(10).join(f'''
          - name: {col}
            tests:
              - unique
              - not_null''' for col in unique_cols)}

          {chr(10).join(f'''
          - name: {col}
            tests:
              - not_null''' for col in not_null_cols)}

          - name: revenue
            tests:
              - dbt_expectations.expect_column_values_to_be_between:
                  min_value: 0
                  max_value: 1000000
    """
            return yaml

    # ============= Partitioning Strategy =============
    class PartitioningStrategy:
        """Determine optimal partitioning strategy"""

        def recommend_partition_strategy(
            self,
            table_size_gb: float,
            query_pattern: str,  # 'time_range', 'full_scan', 'point_lookup'
            date_range_years: int
        ) -> Dict[str, Any]:
            """
            Recommend partitioning strategy based on usage patterns
            """
            recommendations = {}

            # Size-based recommendations
            if table_size_gb < 10:
                recommendations['partition'] = None
                recommendations['reason'] = "Table too small, partitioning overhead not worth it"

            elif query_pattern == 'time_range':
                # Time-series queries benefit from date partitioning
                if date_range_years <= 1:
                    recommendations['partition'] = 'daily'
                elif date_range_years <= 3:
                    recommendations['partition'] = 'weekly'
                else:
                    recommendations['partition'] = 'monthly'

                recommendations['reason'] = "Time-range queries → date partitioning reduces scan"

            elif query_pattern == 'point_lookup':
                recommendations['partition'] = None
                recommendations['cluster_by'] = ['primary_key']
                recommendations['reason'] = "Point lookups → clustering more effective than partitioning"

            else:  # full_scan
                recommendations['partition'] = 'monthly'
                recommendations['reason'] = "Full scans → coarse partitioning for data lifecycle management"

            # Clustering recommendations
            if table_size_gb > 1:
                recommendations['cluster_by'] = ['customer_key', 'product_key']
                recommendations['cluster_reason'] = "Improves JOIN and filter performance"

            return recommendations

    # ============= Example Usage =============
    def example_warehouse_setup():
        """Example: Set up star schema for e-commerce analytics"""

        designer = StarSchemaDesigner()
        scd_handler = SCDHandler()
        dbt_gen = DBTModelGenerator()

        # 1. Generate fact table DDL
        fact_sales_ddl = designer.generate_fact_table_ddl(
            table_name="fact_sales",
            measures=["quantity", "revenue", "cost", "discount"],
            dimensions=["date", "customer", "product", "store"],
            partition_by="date_key",
            cluster_by=["customer_key", "product_key"]
        )
        print("Fact Table DDL:")
        print(fact_sales_ddl)

        # 2. Generate dimension table DDL with SCD Type 2
        dim_customer_ddl = designer.generate_dimension_table_ddl(
            table_name="dim_customer",
            attributes=["name", "email", "segment", "city", "country"],
            scd_type=SCDType.TYPE_2
        )
        print("\nDimension Table DDL (SCD Type 2):")
        print(dim_customer_ddl)

        # 3. Apply SCD Type 2 transformation
        existing_customers = pd.DataFrame({
            'customer_key': [1, 2],
            'customer_id': ['C001', 'C002'],
            'name': ['Alice', 'Bob'],
            'segment': ['Premium', 'Standard'],
            'is_current': [True, True],
            'version': [1, 1],
            'valid_from': [date(2024, 1, 1), date(2024, 1, 1)],
            'valid_to': [None, None]
        })

        new_customer_data = pd.DataFrame({
            'customer_id': ['C001', 'C003'],
            'name': ['Alice'],
            'segment': ['VIP'],  # Alice upgraded from Premium to VIP
        })

        updated_customers = scd_handler.apply_scd_type_2(
            existing_customers,
            new_customer_data,
            natural_key='customer_id',
            attributes=['name', 'segment']
        )
        print("\nSCD Type 2 Result:")
        print(updated_customers)

        # 4. Generate DBT fact model
        fact_model = dbt_gen.generate_fact_model(
            fact_name="fact_sales",
            source_tables=["sales_transactions", "products"],
            measures=["quantity", "revenue", "cost"],
            dimensions=["date", "customer", "product", "store"]
        )
        print("\nDBT Fact Model:")
        print(fact_model)
    ```

    ## Technology Comparison

    | Platform | Strengths | Weaknesses | Best For |
    |----------|-----------|------------|----------|
    | **BigQuery** | Serverless, fast, columnar, integrates with GCP | Can get expensive at scale, vendor lock-in | GCP users, fast analytics |
    | **Snowflake** | Multi-cloud, separation of compute/storage, zero-copy cloning | Cost can be high, cold start latency | Multi-cloud, scalability |
    | **Redshift** | AWS integration, mature, familiar (Postgres-based) | More manual tuning needed | AWS-native, budget-conscious |
    | **Databricks** | Unified analytics, ML integration, Delta Lake | Complexity, cost | ML-heavy workloads |
    | **Synapse** | Azure integration, Spark + SQL, serverless | Less mature than competitors | Azure-native environments |

    ## Schema Design Comparison

    | Schema | Structure | Pros | Cons | Use Case |
    |--------|-----------|------|------|----------|
    | **Star** | 1 fact + N dimensions (denormalized) | Simple queries, fast joins, BI-friendly | Data redundancy | Most BI/analytics workloads |
    | **Snowflake** | Normalized dimensions (dimension hierarchies) | Reduces redundancy, easier updates | More joins, complex queries | Highly normalized sources |
    | **Data Vault** | Hubs, Links, Satellites | Auditability, flexibility | Complex, slower queries | Regulatory/audit-heavy industries |
    | **One Big Table (OBT)** | Fully denormalized | Simplest queries, fastest | Massive redundancy, hard to update | Reporting, static datasets |

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **No Partitioning** | Full table scans, high cost | Partition by date (daily/monthly) |
    | **Wrong Grain** | Incorrect aggregations | Define fact table grain clearly (1 row = 1 sale) |
    | **No SCD Strategy** | Lost history or incorrect snapshots | Implement SCD Type 2 for critical dimensions |
    | **Surrogate vs Natural Keys** | Join failures, duplicates | Always use surrogate keys for dimensions |
    | **Missing Audit Columns** | Can't debug data issues | Add created_at, updated_at, loaded_by |
    | **No Data Quality Tests** | Bad data propagates | Implement dbt tests, Great Expectations |
    | **Over-Normalization** | Slow queries (too many joins) | Denormalize for query performance |
    | **Late-Arriving Facts** | Orphaned records | Handle late arrivals with default dimensions |

    ## Real-World Examples

    **Airbnb's Data Warehouse:**
    - **Scale:** 10PB+ in S3, 100K+ tables
    - **Architecture:** S3 (storage) + Presto/Hive (query) + Airflow (orchestration)
    - **Schema:** Star schema with 200+ fact tables
    - **Innovation:** Minerva (metadata service), automatic partitioning
    - **Impact:** Powers all business analytics, 1000+ data scientists

    **Netflix's Data Warehouse:**
    - **Scale:** 100PB+ in S3
    - **Architecture:** S3 (Iceberg format) + Spark + Trino
    - **Partitioning:** Dynamic partitioning by region + date
    - **Use Case:** A/B test analysis, content performance, personalization
    - **Impact:** Drives all content and product decisions

    **Uber's Data Warehouse:**
    - **Scale:** Multiple exabytes
    - **Architecture:** HDFS → Hive → Vertica/Presto
    - **Schema:** 100K+ tables, star/OBT hybrid
    - **Innovation:** Databook (data discovery), automated quality checks
    - **Impact:** Real-time surge pricing, driver matching analytics

    ## Monitoring & Optimization

    ```python
    def warehouse_health_metrics() -> Dict[str, str]:
        """Key metrics to monitor for warehouse health"""
        return {
            'storage': 'Total GB, growth rate, top 10 largest tables',
            'query_performance': 'P50/P95/P99 latency, slot utilization (BigQuery), warehouse credit usage (Snowflake)',
            'data_freshness': 'Max lag for each table (expected vs actual load time)',
            'data_quality': 'Test failure rate, null percentage, duplicate rate',
            'cost': 'Daily spend by team/project, cost per query, storage vs compute split',
            'usage': 'Queries/day, active users, most queried tables'
        }

    def optimize_query_performance(slow_query: str) -> List[str]:
        """Recommendations for slow query optimization"""
        return [
            "1. Check EXPLAIN plan for full table scans",
            "2. Add partition filter (WHERE date_key >= ...)",
            "3. Use clustering columns in WHERE/JOIN clauses",
            "4. Denormalize frequently joined dimensions",
            "5. Create aggregated summary tables (OLAP cubes)",
            "6. Use materialized views for common queries",
            "7. Limit SELECT * to only needed columns",
            "8. For BigQuery: use APPROX_COUNT_DISTINCT for cardinality"
        ]
    ```

    !!! tip "Interviewer's Insight"
        Emphasizes star schema design with proper grain definition, SCD Type 2 for dimension history, and partitioning/clustering strategies. Discusses DBT for transformations, data quality testing, and trade-offs between star/snowflake/data vault schemas. Can explain how Airbnb/Netflix/Uber implement warehouses at PB-scale with specific architectural patterns.

---

### Design a Stream Processing System - Uber, Netflix Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Streaming` | **Asked by:** Uber, Netflix, LinkedIn

??? success "View Answer"

    ## Scale Requirements

    - **Event Volume:** 1M-100M events/second
    - **Latency:** <1s end-to-end (event to output)
    - **Throughput:** 10GB-1TB/second
    - **State Size:** 10GB-10TB (distributed across cluster)
    - **Windows:** 1s-24h time windows
    - **Late Data:** Handle events up to 1h late
    - **Availability:** 99.99% SLA with checkpointing

    ## Architecture

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Event Sources                                 │
    │                                                                  │
    │  [User Actions] [IoT Sensors] [Logs] [Transactions] [Clicks]   │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │               Message Queue (Kafka/Pulsar)                       │
    │                                                                  │
    │  Topic: user_events                                             │
    │  - Partitions: 100 (parallelism)                                │
    │  - Replication: 3x                                              │
    │  - Retention: 7 days                                            │
    │  - Throughput: 1M msgs/sec                                      │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │         Stream Processing (Flink/Spark Streaming/Kafka Streams) │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │              Event Time Processing                        │  │
    │  │                                                            │  │
    │  │  Watermarks: Max(event_time) - 5min lag                   │  │
    │  │  - Handles late arrivals up to 5min                       │  │
    │  │  - Triggers window computation                            │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │                  Windowing Operations                      │  │
    │  │                                                            │  │
    │  │  Tumbling (5min):    [00:00-00:05] [00:05-00:10] ...     │  │
    │  │  Sliding (5min/1min): [00:00-00:05] [00:01-00:06] ...    │  │
    │  │  Session (gap 10min): User activity sessions              │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │              Stateful Operations                          │  │
    │  │                                                            │  │
    │  │  - Aggregations: SUM, AVG, COUNT per key                  │  │
    │  │  - Joins: Stream-Stream, Stream-Table                     │  │
    │  │  - Pattern detection: CEP (Complex Event Processing)      │  │
    │  │                                                            │  │
    │  │  State Backend: RocksDB (disk), Heap (memory)            │  │
    │  │  Checkpointing: Every 1min to S3/HDFS                     │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │            Exactly-Once Semantics                         │  │
    │  │                                                            │  │
    │  │  1. Checkpointing (Flink snapshots)                       │  │
    │  │  2. Two-phase commit (transactional sinks)                │  │
    │  │  3. Idempotent writes (deduplication keys)                │  │
    │  └──────────────────────────────────────────────────────────┘  │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                        Sinks (Outputs)                           │
    │                                                                  │
    │  [Feature Store] [Database] [Cache] [Alerts] [Dashboards]      │
    │  (Redis/Cassandra) (PostgreSQL) (Redis) (PagerDuty) (Grafana)  │
    └─────────────────────────────────────────────────────────────────┘

                ┌──────────────────────────────────────┐
                │      Monitoring & Observability      │
                │                                      │
                │  - Lag (consumer lag per partition) │
                │  - Throughput (records/sec)         │
                │  - Latency (event time - proc time) │
                │  - Checkpoint duration & failures   │
                │  - State size growth                │
                └──────────────────────────────────────┘
    ```

    ## Production Implementation (310 lines)

    ```python
    # stream_processing.py
    from pyflink.datastream import StreamExecutionEnvironment, TimeCharacteristic
    from pyflink.datastream.window import TumblingEventTimeWindows, SlidingEventTimeWindows, Time
    from pyflink.datastream.functions import MapFunction, AggregateFunction, ProcessWindowFunction
    from pyflink.common.watermark_strategy import WatermarkStrategy
    from pyflink.common.typeinfo import Types
    from pyflink.common.serialization import SimpleStringSchema
    from pyflink.datastream.connectors import FlinkKafkaConsumer, FlinkKafkaProducer
    from typing import Tuple, Iterable
    from dataclasses import dataclass
    from datetime import datetime, timedelta
    import json
    import logging

    # ============= Configuration =============
    @dataclass
    class StreamConfig:
        """Stream processing configuration"""
        kafka_bootstrap_servers: str = "localhost:9092"
        input_topic: str = "user_events"
        output_topic: str = "aggregated_metrics"
        checkpoint_interval_ms: int = 60000  # 1 minute
        max_out_of_orderness_ms: int = 300000  # 5 minutes
        parallelism: int = 10

    config = StreamConfig()

    # ============= Event Schema =============
    @dataclass
    class UserEvent:
        """User event schema"""
        user_id: str
        event_type: str
        value: float
        timestamp: int  # Unix timestamp (ms)
        metadata: dict

        @staticmethod
        def from_json(json_str: str) -> 'UserEvent':
            """Parse JSON string to UserEvent"""
            data = json.loads(json_str)
            return UserEvent(
                user_id=data['user_id'],
                event_type=data['event_type'],
                value=data.get('value', 0.0),
                timestamp=data['timestamp'],
                metadata=data.get('metadata', {})
            )

        def to_json(self) -> str:
            """Serialize UserEvent to JSON"""
            return json.dumps({
                'user_id': self.user_id,
                'event_type': self.event_type,
                'value': self.value,
                'timestamp': self.timestamp,
                'metadata': self.metadata
            })

    # ============= Watermark Strategy =============
    class UserEventWatermarkStrategy:
        """Custom watermark strategy for handling late events"""

        @staticmethod
        def create(max_out_of_orderness: timedelta):
            """
            Create watermark strategy with bounded out-of-orderness

            Watermark = max(event_time) - max_out_of_orderness
            Events with timestamp < watermark are considered late
            """
            return WatermarkStrategy \
                .for_bounded_out_of_orderness(max_out_of_orderness) \
                .with_timestamp_assigner(lambda event, ts: event.timestamp)

    # ============= Stream Processing Functions =============
    class ParseEventFunction(MapFunction):
        """Parse JSON events from Kafka"""

        def map(self, value: str) -> UserEvent:
            return UserEvent.from_json(value)

    class AggregateMetricsFunction(AggregateFunction):
        """
        Aggregate function for window computations
        Efficiently computes running aggregations
        """

        def create_accumulator(self) -> Tuple[int, float, float, float]:
            """Initialize accumulator: (count, sum, min, max)"""
            return (0, 0.0, float('inf'), float('-inf'))

        def add(self, value: UserEvent, accumulator: Tuple) -> Tuple:
            """Add new event to accumulator"""
            count, sum_val, min_val, max_val = accumulator
            return (
                count + 1,
                sum_val + value.value,
                min(min_val, value.value),
                max(max_val, value.value)
            )

        def get_result(self, accumulator: Tuple) -> dict:
            """Compute final result from accumulator"""
            count, sum_val, min_val, max_val = accumulator
            avg = sum_val / count if count > 0 else 0
            return {
                'count': count,
                'sum': sum_val,
                'avg': avg,
                'min': min_val,
                'max': max_val
            }

        def merge(self, acc1: Tuple, acc2: Tuple) -> Tuple:
            """Merge two accumulators (for parallel processing)"""
            return (
                acc1[0] + acc2[0],  # count
                acc1[1] + acc2[1],  # sum
                min(acc1[2], acc2[2]),  # min
                max(acc1[3], acc2[3])  # max
            )

    class EnrichWindowResults(ProcessWindowFunction):
        """
        Process window function to enrich aggregation results
        Has access to window metadata (start, end)
        """

        def process(self, key: str, context: ProcessWindowFunction.Context,
                   elements: Iterable[dict]) -> Iterable[str]:
            """
            Enrich aggregated results with window metadata
            """
            result = list(elements)[0]  # Single element from AggregateFunction

            window = context.window()
            output = {
                'user_id': key,
                'window_start': window.start,
                'window_end': window.end,
                'metrics': result,
                'processing_time': context.current_processing_time()
            }

            yield json.dumps(output)

    # ============= Complex Event Processing (CEP) =============
    class FraudDetectionPattern:
        """
        Detect fraud patterns using CEP
        Example: Multiple high-value transactions in short time
        """

        @staticmethod
        def detect_suspicious_pattern(events: Iterable[UserEvent]) -> bool:
            """
            Pattern: 3+ transactions > $1000 within 5 minutes
            """
            high_value_events = [e for e in events if e.value > 1000]

            if len(high_value_events) < 3:
                return False

            # Check if all within 5 minutes
            timestamps = sorted([e.timestamp for e in high_value_events])
            time_span_ms = timestamps[-1] - timestamps[0]
            return time_span_ms <= 300000  # 5 minutes

    # ============= State Management =============
    class StatefulCounter:
        """
        Maintain stateful counter across events
        Uses Flink's ValueState for fault-tolerant state
        """

        def __init__(self):
            self.state = None  # Initialized by Flink runtime

        def process(self, event: UserEvent, ctx) -> Iterable[Tuple[str, int]]:
            """
            Update counter state for each user
            """
            # Get current count (or 0 if first event)
            current_count = self.state.value() or 0

            # Increment counter
            new_count = current_count + 1
            self.state.update(new_count)

            # Emit result
            yield (event.user_id, new_count)

    # ============= Stream-Stream Join =============
    class StreamJoinExample:
        """
        Join two streams with time-bounded join window
        Example: Join clicks with purchases within 1 hour
        """

        @staticmethod
        def join_streams(click_stream, purchase_stream):
            """
            Join click stream with purchase stream
            Match clicks with purchases within 1 hour window
            """
            return click_stream.join(purchase_stream) \
                .where(lambda click: click.user_id) \
                .equal_to(lambda purchase: purchase.user_id) \
                .window(TumblingEventTimeWindows.of(Time.hours(1))) \
                .apply(lambda click, purchase: {
                    'user_id': click.user_id,
                    'click_time': click.timestamp,
                    'purchase_time': purchase.timestamp,
                    'time_to_convert_ms': purchase.timestamp - click.timestamp
                })

    # ============= Main Pipeline =============
    def create_streaming_pipeline():
        """
        Create production Flink streaming pipeline
        """
        # 1. Set up execution environment
        env = StreamExecutionEnvironment.get_execution_environment()
        env.set_parallelism(config.parallelism)

        # Enable checkpointing for exactly-once semantics
        env.enable_checkpointing(config.checkpoint_interval_ms)
        env.get_checkpoint_config().set_checkpoint_storage_dir("s3://checkpoints/")

        # Event time processing
        env.set_stream_time_characteristic(TimeCharacteristic.EventTime)

        # 2. Set up Kafka source
        kafka_props = {
            'bootstrap.servers': config.kafka_bootstrap_servers,
            'group.id': 'flink-consumer-group'
        }

        kafka_consumer = FlinkKafkaConsumer(
            topics=config.input_topic,
            deserialization_schema=SimpleStringSchema(),
            properties=kafka_props
        )

        # 3. Define watermark strategy
        watermark_strategy = UserEventWatermarkStrategy.create(
            timedelta(milliseconds=config.max_out_of_orderness_ms)
        )

        # 4. Create data stream
        event_stream = env.add_source(kafka_consumer) \
            .map(ParseEventFunction()) \
            .assign_timestamps_and_watermarks(watermark_strategy)

        # 5. Apply windowing and aggregations
        tumbling_aggregations = event_stream \
            .key_by(lambda event: event.user_id) \
            .window(TumblingEventTimeWindows.of(Time.minutes(5))) \
            .aggregate(
                AggregateMetricsFunction(),
                EnrichWindowResults()
            )

        # 6. Sliding window for overlapping computations
        sliding_aggregations = event_stream \
            .key_by(lambda event: event.user_id) \
            .window(SlidingEventTimeWindows.of(
                Time.minutes(10),  # window size
                Time.minutes(1)    # slide interval
            )) \
            .aggregate(AggregateMetricsFunction())

        # 7. Session windows (activity-based)
        # Groups events into sessions based on inactivity gap
        session_windows = event_stream \
            .key_by(lambda event: event.user_id) \
            .window(SessionWindows.with_gap(Time.minutes(30))) \
            .aggregate(AggregateMetricsFunction())

        # 8. Set up Kafka sink
        kafka_producer = FlinkKafkaProducer(
            topic=config.output_topic,
            serialization_schema=SimpleStringSchema(),
            producer_config=kafka_props
        )

        tumbling_aggregations.add_sink(kafka_producer)

        # 9. Execute pipeline
        env.execute("User Event Processing Pipeline")

    # ============= Alternative: Kafka Streams (Lighter Weight) =============
    def kafka_streams_example():
        """
        Alternative implementation using Kafka Streams
        Simpler for Kafka-native deployments
        """
        from confluent_kafka import Consumer, Producer
        import time

        # Consumer
        consumer_config = {
            'bootstrap.servers': config.kafka_bootstrap_servers,
            'group.id': 'kafka-streams-group',
            'auto.offset.reset': 'earliest'
        }

        consumer = Consumer(consumer_config)
        consumer.subscribe([config.input_topic])

        # Producer
        producer = Producer({'bootstrap.servers': config.kafka_bootstrap_servers})

        # Stateful aggregation (in-memory for simplicity)
        state = {}

        try:
            while True:
                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    continue

                # Parse event
                event = UserEvent.from_json(msg.value().decode('utf-8'))

                # Update state
                key = event.user_id
                if key not in state:
                    state[key] = {'count': 0, 'sum': 0}

                state[key]['count'] += 1
                state[key]['sum'] += event.value

                # Emit result
                result = {
                    'user_id': key,
                    'count': state[key]['count'],
                    'avg': state[key]['sum'] / state[key]['count']
                }

                producer.produce(
                    config.output_topic,
                    json.dumps(result).encode('utf-8')
                )

        except KeyboardInterrupt:
            pass
        finally:
            consumer.close()
            producer.flush()

    # ============= Example Usage =============
    if __name__ == "__main__":
        # Run Flink pipeline
        create_streaming_pipeline()

        # Or run Kafka Streams
        # kafka_streams_example()
    ```

    ## Windowing Comparison

    | Window Type | Behavior | Use Case | Example |
    |-------------|----------|----------|---------|
    | **Tumbling** | Fixed, non-overlapping | Periodic aggregations | Hourly metrics, daily summaries |
    | **Sliding** | Fixed, overlapping | Moving averages | Last 5min metrics every 1min |
    | **Session** | Gap-based, variable | User sessions | Activity grouped by 30min gaps |
    | **Global** | All data in one window | Rare; entire stream | Counting all events ever |

    ## State Backend Comparison

    | Backend | Storage | Performance | Use Case |
    |---------|---------|-------------|----------|
    | **Heap** | JVM memory | Fastest | Small state (<1GB), low latency |
    | **RocksDB** | Local disk | Slower, scalable | Large state (GB-TB), fault-tolerant |
    | **External** | S3, HDFS | Slowest | Very large state, recovery |

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **No Watermarks** | Windows never close | Configure watermarks with acceptable lag |
    | **Processing Time Windows** | Non-deterministic results | Use event time for reproducibility |
    | **Too Small Windows** | High overhead | Balance window size vs latency needs |
    | **No Checkpointing** | Data loss on failure | Enable checkpointing every 1-5min |
    | **Unbounded State Growth** | OOM errors | Use TTL for state, cleanup old keys |
    | **Skewed Keys** | Hotspot on single task | Pre-aggregate, use combiner, salting |
    | **Late Data Ignored** | Missed events | Configure allowed lateness, side outputs |
    | **No Backpressure Handling** | System overload | Rate limiting, buffering, auto-scaling |

    ## Real-World Examples

    **Uber's Stream Processing:**
    - **Scale:** 1M+ events/second, 100+ Flink jobs
    - **Use Cases:** Surge pricing, ETA calculation, fraud detection
    - **Architecture:** Kafka → Flink → Cassandra/Redis
    - **State:** 10TB+ distributed state in RocksDB
    - **Impact:** Real-time pricing updates, <1s latency

    **Netflix's Keystone:**
    - **Scale:** 8M+ events/second peak
    - **Use Cases:** Viewing history, recommendations, A/B tests
    - **Architecture:** Kafka → Flink → Elasticsearch/S3
    - **Features:** Exactly-once, session windows, 99.99% availability
    - **Impact:** Powers real-time personalization for 200M+ users

    **LinkedIn's Stream Processing:**
    - **Scale:** 1.5T+ messages/day
    - **Use Cases:** Feed updates, notifications, analytics
    - **Architecture:** Kafka Streams + Samza
    - **Innovation:** Venice (distributed state store)
    - **Impact:** Real-time feed ranking, <100ms updates

    ## Monitoring Metrics

    ```python
    def stream_processing_metrics() -> dict:
        """Key metrics for stream processing health"""
        return {
            'throughput': {
                'records_in_per_sec': 'Input rate from Kafka',
                'records_out_per_sec': 'Output rate to sinks',
                'bytes_per_sec': 'Network throughput'
            },
            'latency': {
                'event_time_lag': 'Watermark - current event time',
                'processing_lag': 'Processing time - event time',
                'end_to_end_latency': 'Event creation to sink output'
            },
            'resource_usage': {
                'cpu_utilization': 'Per task manager',
                'memory_heap': 'JVM heap usage',
                'state_size': 'RocksDB state size',
                'network_buffers': 'Backpressure indicator'
            },
            'checkpointing': {
                'checkpoint_duration': 'Time to complete checkpoint',
                'checkpoint_size': 'Checkpoint state size',
                'checkpoint_failures': 'Failed checkpoints count'
            },
            'kafka': {
                'consumer_lag': 'Per partition lag',
                'rebalance_count': 'Consumer group rebalances'
            }
        }
    ```

    !!! tip "Interviewer's Insight"
        Emphasizes event-time processing vs processing-time, watermarks for handling late data, and exactly-once semantics via checkpointing. Discusses windowing strategies (tumbling/sliding/session), state management (heap vs RocksDB), and backpressure handling. Can explain how Uber/Netflix/LinkedIn implement stream processing at massive scale with specific trade-offs (latency vs throughput, memory vs disk state).

---

### Design an ML Labeling Pipeline - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Data Quality` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ## Scale Requirements

    - **Data Volume:** 100K-10M samples to label
    - **Throughput:** 100-10K labels/day
    - **Annotators:** 10-1K human labelers
    - **Agreement:** >80% inter-annotator agreement (IAA)
    - **Quality:** >95% label accuracy
    - **Latency:** <2s for UI responsiveness
    - **Active Learning:** Reduce labeling by 50-70% via smart sampling

    ## Architecture

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                   Unlabeled Data Pool                            │
    │                                                                  │
    │  [Images] [Text] [Audio] [Video] [Structured Data]             │
    │  - 10M samples (raw)                                            │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │              Active Learning Sampler                             │
    │                                                                  │
    │  Sampling Strategies:                                           │
    │  1. Random (baseline)                                           │
    │  2. Uncertainty Sampling (low confidence predictions)           │
    │  3. Diversity Sampling (representative distribution)            │
    │  4. Query-by-Committee (model disagreement)                     │
    │  5. Expected Model Change (gradient-based)                      │
    │                                                                  │
    │  Priority Score = uncertainty * diversity * business_value      │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                  Annotation Interface (UI)                       │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │              Task-Specific UI                             │  │
    │  │                                                            │  │
    │  │  Classification: Multi-choice buttons                     │  │
    │  │  Object Detection: Bounding box tool                      │  │
    │  │  Segmentation: Polygon/brush tool                         │  │
    │  │  NER: Text highlighting                                   │  │
    │  │  Ranking: Drag-and-drop ordering                          │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  Features:                                                      │
    │  - Keyboard shortcuts (fast labeling)                           │
    │  - Pre-annotations (model predictions as starting point)        │
    │  - Guidelines & examples                                        │
    │  - Progress tracking                                            │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                Quality Assurance Layer                           │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │          Multi-Annotator Consensus                        │  │
    │  │                                                            │  │
    │  │  Strategy: 3 annotators per sample                        │  │
    │  │  - Majority vote (2/3 agree)                              │  │
    │  │  - Adjudication (expert resolves conflicts)               │  │
    │  │  - Dawid-Skene model (probabilistic consensus)            │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │          Inter-Annotator Agreement (IAA)                  │  │
    │  │                                                            │  │
    │  │  Metrics:                                                 │  │
    │  │  - Cohen's Kappa (2 annotators)                           │  │
    │  │  - Fleiss' Kappa (3+ annotators)                          │  │
    │  │  - Krippendorff's Alpha (ordinal/interval data)           │  │
    │  │                                                            │  │
    │  │  Alert if Kappa < 0.6 (poor agreement)                    │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │             Gold Standard Test Set                        │  │
    │  │                                                            │  │
    │  │  - 100-1000 expert-labeled samples                        │  │
    │  │  - Test each annotator periodically                       │  │
    │  │  - Track accuracy over time                               │  │
    │  │  - Retrain if accuracy < 90%                              │  │
    │  └──────────────────────────────────────────────────────────┘  │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Labeled Dataset                               │
    │                                                                  │
    │  Version Control:                                               │
    │  - v1.0: Initial 10K labels (baseline)                          │
    │  - v1.1: +5K labels, fixed 200 errors                           │
    │  - v2.0: New label schema, relabeled all                        │
    │                                                                  │
    │  Metadata: annotator_id, timestamp, confidence, version         │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                  Model Training & Feedback                       │
    │                                                                  │
    │  [Train Model] → [Evaluate] → [Identify Hard Examples]         │
    │                                        ↓                         │
    │                          [Feed back to Active Learning]         │
    └─────────────────────────────────────────────────────────────────┘
    ```

    ## Production Implementation (280 lines)

    ```python
    # labeling_pipeline.py
    from typing import List, Dict, Any, Optional, Tuple
    from dataclasses import dataclass
    from datetime import datetime
    import numpy as np
    from sklearn.metrics import cohen_kappa_score
    from collections import Counter
    import logging

    # ============= Configuration =============
    @dataclass
    class LabelingConfig:
        """Labeling pipeline configuration"""
        num_annotators_per_sample: int = 3
        min_agreement_threshold: float = 0.6  # Kappa score
        gold_standard_size: int = 1000
        active_learning_batch_size: int = 100
        min_annotator_accuracy: float = 0.90

    config = LabelingConfig()

    # ============= Active Learning Sampler =============
    class ActiveLearningSampler:
        """Sample most informative examples for labeling"""

        def __init__(self, model):
            self.model = model
            self.logger = logging.getLogger(__name__)

        def uncertainty_sampling(
            self,
            unlabeled_data: np.ndarray,
            n_samples: int
        ) -> List[int]:
            """
            Sample examples with highest prediction uncertainty

            Methods:
            - Least Confident: 1 - max(P(y|x))
            - Margin: P(y1|x) - P(y2|x)  (smallest margin)
            - Entropy: -∑ P(y|x) log P(y|x)
            """
            # Get prediction probabilities
            probs = self.model.predict_proba(unlabeled_data)

            # Entropy-based uncertainty
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)

            # Select top-k most uncertain
            uncertain_indices = np.argsort(entropy)[-n_samples:]

            self.logger.info(f"Selected {n_samples} uncertain samples (avg entropy: {entropy[uncertain_indices].mean():.3f})")
            return uncertain_indices.tolist()

        def diversity_sampling(
            self,
            unlabeled_data: np.ndarray,
            n_samples: int,
            embeddings: Optional[np.ndarray] = None
        ) -> List[int]:
            """
            Sample diverse examples using k-means clustering
            """
            from sklearn.cluster import KMeans

            if embeddings is None:
                embeddings = unlabeled_data

            # Cluster into n_samples clusters
            kmeans = KMeans(n_clusters=n_samples, random_state=42)
            kmeans.fit(embeddings)

            # Select one sample closest to each cluster center
            diverse_indices = []
            for i in range(n_samples):
                cluster_mask = kmeans.labels_ == i
                cluster_samples = np.where(cluster_mask)[0]

                if len(cluster_samples) > 0:
                    # Find closest to center
                    distances = np.linalg.norm(
                        embeddings[cluster_samples] - kmeans.cluster_centers_[i],
                        axis=1
                    )
                    closest_idx = cluster_samples[np.argmin(distances)]
                    diverse_indices.append(closest_idx)

            return diverse_indices

        def query_by_committee(
            self,
            unlabeled_data: np.ndarray,
            models: List[Any],
            n_samples: int
        ) -> List[int]:
            """
            Sample examples where models disagree most (ensemble variance)
            """
            # Get predictions from each model
            all_predictions = np.array([
                model.predict(unlabeled_data) for model in models
            ])

            # Calculate disagreement (variance)
            disagreement = np.var(all_predictions, axis=0)

            # Select top-k most disagreed
            disagreed_indices = np.argsort(disagreement)[-n_samples:]

            return disagreed_indices.tolist()

    # ============= Quality Assurance =============
    class QualityAssurance:
        """Ensure high-quality labels through consensus and validation"""

        def __init__(self, config: LabelingConfig):
            self.config = config
            self.logger = logging.getLogger(__name__)

        def compute_inter_annotator_agreement(
            self,
            annotations: List[List[int]]
        ) -> float:
            """
            Compute Fleiss' Kappa for multi-annotator agreement

            annotations: List of annotations per sample
            [[annotator1_labels], [annotator2_labels], ...]
            """
            from statsmodels.stats.inter_rater import fleiss_kappa

            # Convert to matrix format: (n_samples, n_categories)
            n_samples = len(annotations[0])
            n_annotators = len(annotations)

            # Count votes per category
            categories = set()
            for ann in annotations:
                categories.update(ann)
            n_categories = len(categories)

            # Build contingency table
            table = np.zeros((n_samples, n_categories))
            for sample_idx in range(n_samples):
                votes = [annotations[ann_idx][sample_idx] for ann_idx in range(n_annotators)]
                vote_counts = Counter(votes)
                for cat_idx, cat in enumerate(sorted(categories)):
                    table[sample_idx, cat_idx] = vote_counts.get(cat, 0)

            kappa = fleiss_kappa(table)
            self.logger.info(f"Inter-annotator agreement (Fleiss' Kappa): {kappa:.3f}")

            return kappa

        def majority_vote_consensus(
            self,
            annotations: List[int]
        ) -> Tuple[int, float]:
            """
            Get consensus label via majority vote

            Returns: (consensus_label, confidence)
            """
            vote_counts = Counter(annotations)
            consensus_label = vote_counts.most_common(1)[0][0]
            confidence = vote_counts[consensus_label] / len(annotations)

            return consensus_label, confidence

        def dawid_skene_consensus(
            self,
            annotations: np.ndarray,
            max_iter: int = 100
        ) -> np.ndarray:
            """
            Probabilistic consensus using Dawid-Skene model

            Accounts for annotator quality/bias
            annotations: (n_samples, n_annotators) matrix
            """
            n_samples, n_annotators = annotations.shape
            n_classes = int(annotations.max()) + 1

            # Initialize: assume all annotators perfect
            annotator_confusion = np.zeros((n_annotators, n_classes, n_classes))
            for a in range(n_annotators):
                annotator_confusion[a] = np.eye(n_classes)

            # E-M algorithm
            for iteration in range(max_iter):
                # E-step: Estimate true labels
                class_probs = np.ones((n_samples, n_classes))
                for i in range(n_samples):
                    for a in range(n_annotators):
                        if not np.isnan(annotations[i, a]):
                            label = int(annotations[i, a])
                            class_probs[i] *= annotator_confusion[a, :, label]

                class_probs /= class_probs.sum(axis=1, keepdims=True)

                # M-step: Update annotator confusion matrices
                for a in range(n_annotators):
                    for j in range(n_classes):
                        for k in range(n_classes):
                            numerator = 0
                            denominator = 0
                            for i in range(n_samples):
                                if not np.isnan(annotations[i, a]) and annotations[i, a] == k:
                                    numerator += class_probs[i, j]
                                    denominator += class_probs[i, j]

                            annotator_confusion[a, j, k] = numerator / (denominator + 1e-10)

            # Final consensus: argmax of class probabilities
            consensus_labels = np.argmax(class_probs, axis=1)
            return consensus_labels

        def evaluate_annotator_quality(
            self,
            annotator_labels: List[int],
            gold_standard: List[int]
        ) -> Dict[str, float]:
            """
            Evaluate individual annotator against gold standard
            """
            accuracy = np.mean(np.array(annotator_labels) == np.array(gold_standard))
            kappa = cohen_kappa_score(gold_standard, annotator_labels)

            return {
                'accuracy': accuracy,
                'kappa': kappa,
                'pass': accuracy >= self.config.min_annotator_accuracy
            }

    # ============= Annotation Task Manager =============
    class AnnotationTaskManager:
        """Manage annotation tasks and assignments"""

        def __init__(self):
            self.tasks = []
            self.assignments = {}
            self.logger = logging.getLogger(__name__)

        def create_tasks(
            self,
            sample_ids: List[str],
            n_annotators_per_sample: int
        ) -> List[Dict]:
            """
            Create annotation tasks with redundancy
            """
            tasks = []
            for sample_id in sample_ids:
                for annotator_round in range(n_annotators_per_sample):
                    task = {
                        'task_id': f"{sample_id}_{annotator_round}",
                        'sample_id': sample_id,
                        'status': 'pending',
                        'annotator_id': None,
                        'label': None,
                        'timestamp': None,
                        'time_spent_seconds': None
                    }
                    tasks.append(task)

            self.tasks.extend(tasks)
            self.logger.info(f"Created {len(tasks)} annotation tasks for {len(sample_ids)} samples")
            return tasks

        def assign_task(
            self,
            annotator_id: str,
            task_filter: Optional[Dict] = None
        ) -> Optional[Dict]:
            """
            Assign next available task to annotator

            Routing strategies:
            - Round-robin
            - Skill-based (match annotator expertise to task difficulty)
            - Load-balancing (distribute evenly)
            """
            # Find pending task
            for task in self.tasks:
                if task['status'] == 'pending':
                    # Avoid self-agreement: don't assign to same annotator
                    sample_tasks = [t for t in self.tasks if t['sample_id'] == task['sample_id']]
                    assigned_annotators = [t['annotator_id'] for t in sample_tasks if t['annotator_id']]

                    if annotator_id not in assigned_annotators:
                        task['status'] = 'assigned'
                        task['annotator_id'] = annotator_id
                        self.logger.info(f"Assigned task {task['task_id']} to {annotator_id}")
                        return task

            return None

        def submit_annotation(
            self,
            task_id: str,
            label: Any,
            time_spent_seconds: float
        ):
            """Submit completed annotation"""
            for task in self.tasks:
                if task['task_id'] == task_id:
                    task['status'] = 'completed'
                    task['label'] = label
                    task['timestamp'] = datetime.now()
                    task['time_spent_seconds'] = time_spent_seconds
                    break

        def get_completed_annotations(self, sample_id: str) -> List[Any]:
            """Get all completed annotations for a sample"""
            return [
                task['label'] for task in self.tasks
                if task['sample_id'] == sample_id and task['status'] == 'completed'
            ]

    # ============= Label Version Control =============
    class LabelVersionControl:
        """Track label changes and versions"""

        def __init__(self):
            self.versions = []
            self.label_history = {}

        def create_version(
            self,
            version_name: str,
            labels: Dict[str, Any],
            metadata: Dict
        ):
            """
            Create a new label dataset version
            """
            version = {
                'version': version_name,
                'timestamp': datetime.now(),
                'num_labels': len(labels),
                'metadata': metadata,
                'labels': labels.copy()
            }
            self.versions.append(version)

        def track_label_change(
            self,
            sample_id: str,
            old_label: Any,
            new_label: Any,
            reason: str
        ):
            """Track individual label corrections"""
            if sample_id not in self.label_history:
                self.label_history[sample_id] = []

            self.label_history[sample_id].append({
                'timestamp': datetime.now(),
                'old_label': old_label,
                'new_label': new_label,
                'reason': reason
            })

        def get_label_statistics(self) -> Dict:
            """Get label dataset statistics"""
            if not self.versions:
                return {}

            latest = self.versions[-1]
            labels = list(latest['labels'].values())

            return {
                'total_samples': len(labels),
                'label_distribution': dict(Counter(labels)),
                'versions': len(self.versions),
                'corrections': len(self.label_history)
            }

    # ============= Example Usage =============
    def example_labeling_pipeline():
        """Example: Active learning + quality assurance pipeline"""
        from sklearn.ensemble import RandomForestClassifier

        # 1. Initialize components
        model = RandomForestClassifier()
        sampler = ActiveLearningSampler(model)
        qa = QualityAssurance(config)
        task_manager = AnnotationTaskManager()
        version_control = LabelVersionControl()

        # 2. Simulate unlabeled data
        unlabeled_data = np.random.randn(10000, 20)

        # 3. Active learning: Select 100 most informative samples
        selected_indices = sampler.uncertainty_sampling(unlabeled_data, n_samples=100)

        # 4. Create annotation tasks (3 annotators per sample)
        sample_ids = [f"sample_{i}" for i in selected_indices]
        tasks = task_manager.create_tasks(sample_ids, n_annotators_per_sample=3)

        # 5. Simulate annotations
        for task in tasks[:9]:  # First 9 tasks (3 samples x 3 annotators)
            task_manager.submit_annotation(
                task_id=task['task_id'],
                label=np.random.randint(0, 3),  # 3 classes
                time_spent_seconds=np.random.uniform(5, 30)
            )

        # 6. Compute consensus for first sample
        sample_0_annotations = task_manager.get_completed_annotations(sample_ids[0])
        consensus_label, confidence = qa.majority_vote_consensus(sample_0_annotations)
        print(f"Sample 0: Consensus = {consensus_label}, Confidence = {confidence:.2f}")

        # 7. Evaluate inter-annotator agreement
        all_annotations = [[np.random.randint(0, 3) for _ in range(100)] for _ in range(3)]
        kappa = qa.compute_inter_annotator_agreement(all_annotations)

        # 8. Create label version
        labels = {sid: np.random.randint(0, 3) for sid in sample_ids}
        version_control.create_version(
            version_name="v1.0",
            labels=labels,
            metadata={'strategy': 'uncertainty_sampling', 'kappa': kappa}
        )

        print(f"Label statistics: {version_control.get_label_statistics()}")
    ```

    ## Quality Metrics

    | Metric | Formula | Good Threshold | Use Case |
    |--------|---------|---------------|----------|
    | **Cohen's Kappa** | (P_o - P_e) / (1 - P_e) | >0.6 | 2 annotators agreement |
    | **Fleiss' Kappa** | Multi-rater version | >0.6 | 3+ annotators agreement |
    | **Accuracy** | Correct / Total | >90% | vs gold standard |
    | **Precision** | TP / (TP + FP) | >85% | Label quality (avoid FP) |
    | **Recall** | TP / (TP + FN) | >85% | Label coverage (avoid FN) |

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Unclear Guidelines** | Low IAA, inconsistent labels | Detailed examples, edge cases, iterative refinement |
    | **Annotator Fatigue** | Quality degrades over time | Break tasks into batches, monitor time-per-label |
    | **Label Imbalance** | Biased model | Stratified sampling, oversampling rare classes |
    | **No Gold Standard** | Can't measure quality | Create expert-labeled test set (1-5% of data) |
    | **Single Annotator** | No consensus, high error rate | 3+ annotators per sample, majority vote |
    | **Ignoring Hard Examples** | Model fails on edge cases | Active learning focuses on uncertain/hard examples |
    | **Static Labeling** | Waste effort on easy examples | Continuous active learning loop |
    | **No Version Control** | Can't reproduce experiments | Track all label changes with timestamps |

    ## Real-World Examples

    **Google's Data Labeling:**
    - **Scale:** 10M+ images labeled for ImageNet, COCO
    - **Quality:** Multiple annotators + expert review
    - **Tools:** Internal tools (Crowdsource, reCAPTCHA for free labels)
    - **Innovation:** Consensus via majority vote + outlier detection
    - **Impact:** Enabled breakthrough in computer vision (AlexNet, ResNet)

    **Tesla's Autopilot Labeling:**
    - **Scale:** Billions of video frames
    - **Strategy:** Active learning (corner cases from fleet)
    - **Process:** Auto-labeling + human review for uncertain cases
    - **Quality:** 99.9%+ accuracy via multi-stage QA
    - **Impact:** Continuous improvement from real-world data

    **Scale AI:**
    - **Business:** Labeling-as-a-Service
    - **Scale:** 1M+ labeled samples/month for customers
    - **Quality:** Consensus (3-5 labelers) + expert review
    - **Tools:** Task-specific UIs, quality dashboards
    - **Customers:** OpenAI (RLHF for ChatGPT), autonomous vehicle companies

    !!! tip "Interviewer's Insight"
        Emphasizes active learning to reduce labeling cost by 50-70%, multi-annotator consensus for quality (Fleiss' Kappa >0.6), and gold standard test sets for ongoing quality monitoring. Discusses trade-offs between labeling cost and model performance, and can explain how Google/Tesla use active learning at scale.

---

### Design a Neural Network Optimizer - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Deep Learning` | **Asked by:** Google, Meta, OpenAI

??? success "View Answer"

    ## Scale Requirements

    - **Search Space:** 10-100 hyperparameters
    - **Trials:** 100-10K training runs
    - **Parallel Trials:** 10-1K concurrent workers
    - **Cost:** $1K-$1M compute budget
    - **Time:** Hours to weeks
    - **Improvement:** 5-30% accuracy gain vs random
    - **GPUs:** 10-1000 GPUs/TPUs

    ## Architecture

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                  Hyperparameter Search Space                     │
    │                                                                  │
    │  Model Architecture:          Training:                          │
    │  - num_layers: [2, 3, 4, 5]  - learning_rate: [1e-5, 1e-1]     │
    │  - hidden_size: [64, 512]    - batch_size: [16, 32, 64, 128]   │
    │  - activation: [relu, gelu]  - optimizer: [adam, sgd, adamw]   │
    │  - dropout: [0.0, 0.5]       - weight_decay: [0, 1e-4]         │
    │                                                                  │
    │  Data Aug: [cutout, mixup, randaugment]                         │
    │  Scheduler: [cosine, step, exponential]                         │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │              Optimization Strategy Selector                      │
    │                                                                  │
    │  Stage 1: Random/Grid (baseline, 10-20 trials)                  │
    │  Stage 2: Bayesian Optimization (100-500 trials)                │
    │  Stage 3: Hyperband/ASHA (early stopping, 1K+ trials)           │
    │  Stage 4: Neural Architecture Search (if needed)                │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │            Bayesian Optimization (Primary Method)                │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │         Surrogate Model (Gaussian Process)                │  │
    │  │                                                            │  │
    │  │  P(accuracy | hyperparameters)                            │  │
    │  │  - Mean: expected accuracy                                │  │
    │  │  - Variance: uncertainty                                  │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │       Acquisition Function (Next Trial Selector)          │  │
    │  │                                                            │  │
    │  │  Methods:                                                 │  │
    │  │  - Expected Improvement (EI)                              │  │
    │  │  - Upper Confidence Bound (UCB)                           │  │
    │  │  - Probability of Improvement (PI)                        │  │
    │  │                                                            │  │
    │  │  Balance: Exploitation (high mean) vs                     │  │
    │  │            Exploration (high variance)                    │  │
    │  └──────────────────────────────────────────────────────────┘  │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │              Early Stopping (Hyperband/ASHA)                     │
    │                                                                  │
    │  Idea: Stop unpromising trials early to save compute            │
    │                                                                  │
    │  ASHA (Asynchronous Successive Halving):                        │
    │  - Start 1000 trials with 1 epoch each                          │
    │  - Keep top 50% → train 2 epochs                                │
    │  - Keep top 50% → train 4 epochs                                │
    │  - Keep top 50% → train 8 epochs                                │
    │  - ...until 1 winner at 64 epochs                               │
    │                                                                  │
    │  Savings: ~10x less compute vs full training                    │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │              Distributed Trial Execution (Ray)                   │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │               Scheduler (Ray Tune)                        │  │
    │  │                                                            │  │
    │  │  - Generates hyperparameter configs                       │  │
    │  │  - Dispatches to workers                                  │  │
    │  │  - Collects results                                       │  │
    │  │  - Updates Bayesian model                                 │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │           Workers (100+ GPUs in parallel)                 │  │
    │  │                                                            │  │
    │  │  Worker 1: lr=1e-3, batch=64 → val_acc=0.85              │  │
    │  │  Worker 2: lr=1e-4, batch=32 → val_acc=0.87              │  │
    │  │  ...                                                       │  │
    │  │  Worker N: lr=3e-4, batch=128 → val_acc=0.91             │  │
    │  └──────────────────────────────────────────────────────────┘  │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                   Best Configuration                             │
    │                                                                  │
    │  {lr: 3e-4, batch_size: 128, hidden_size: 512,                  │
    │   dropout: 0.2, optimizer: 'adamw', ...}                        │
    │                                                                  │
    │  Final validation: 92.3% accuracy                               │
    └─────────────────────────────────────────────────────────────────┘
    ```

    ## Production Implementation (260 lines)

    ```python
    # hyperparameter_optimization.py
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.search import ConcurrencyLimiter
    import optuna
    import numpy as np
    from typing import Dict, Any, Optional
    from dataclasses import dataclass
    import torch
    import torch.nn as nn
    import logging

    # ============= Configuration =============
    @dataclass
    class OptimizationConfig:
        """Hyperparameter optimization configuration"""
        num_samples: int = 100  # Number of trials
        max_concurrent: int = 10  # Parallel trials
        max_epochs_per_trial: int = 64
        grace_period: int = 4  # Min epochs before early stopping
        reduction_factor: int = 2  # For ASHA
        gpus_per_trial: float = 0.5

    config = OptimizationConfig()

    # ============= Search Space Definition =============
    def get_search_space() -> Dict:
        """
        Define hyperparameter search space

        Ray Tune supports:
        - tune.choice() for categorical
        - tune.uniform() for continuous
        - tune.loguniform() for log-scale
        - tune.grid_search() for grid
        """
        return {
            # Model architecture
            'num_layers': tune.choice([2, 3, 4, 5]),
            'hidden_size': tune.choice([128, 256, 512, 1024]),
            'activation': tune.choice(['relu', 'gelu', 'silu']),
            'dropout': tune.uniform(0.0, 0.5),

            # Training hyperparameters
            'learning_rate': tune.loguniform(1e-5, 1e-1),
            'batch_size': tune.choice([16, 32, 64, 128, 256]),
            'optimizer': tune.choice(['adam', 'adamw', 'sgd']),
            'weight_decay': tune.loguniform(1e-6, 1e-2),

            # Scheduler
            'scheduler': tune.choice(['cosine', 'step', 'exponential']),
            'warmup_epochs': tune.choice([0, 5, 10]),

            # Data augmentation
            'mixup_alpha': tune.uniform(0.0, 1.0),
            'label_smoothing': tune.uniform(0.0, 0.2),
        }

    # ============= Training Function =============
    def train_model(config_dict: Dict, checkpoint_dir: Optional[str] = None):
        """
        Training function for a single trial

        Ray Tune will call this function for each hyperparameter config
        """
        import torch.optim as optim
        from torch.utils.data import DataLoader

        # Build model based on config
        model = build_model(config_dict)

        # Setup optimizer
        if config_dict['optimizer'] == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=config_dict['learning_rate'],
                weight_decay=config_dict['weight_decay']
            )
        elif config_dict['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config_dict['learning_rate'],
                weight_decay=config_dict['weight_decay']
            )
        else:  # sgd
            optimizer = optim.SGD(
                model.parameters(),
                lr=config_dict['learning_rate'],
                weight_decay=config_dict['weight_decay'],
                momentum=0.9
            )

        # Setup scheduler
        if config_dict['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.max_epochs_per_trial
            )
        elif config_dict['scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        else:  # exponential
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        # Load checkpoint if resuming
        if checkpoint_dir:
            checkpoint = torch.load(checkpoint_dir + "/checkpoint")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
        else:
            start_epoch = 0

        # Training loop
        for epoch in range(start_epoch, config.max_epochs_per_trial):
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, config_dict)

            # Validate
            val_loss, val_acc = validate(model, val_loader)

            # Scheduler step
            scheduler.step()

            # Report metrics to Ray Tune
            tune.report(
                loss=val_loss,
                accuracy=val_acc,
                epoch=epoch
            )

            # Checkpoint
            if epoch % 10 == 0:
                with tune.checkpoint_dir(epoch) as checkpoint_dir:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_dir + "/checkpoint")

    # ============= Bayesian Optimization =============
    class BayesianOptimizer:
        """Bayesian optimization using Optuna"""

        def __init__(self):
            self.study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(),  # Tree-structured Parzen Estimator
                pruner=optuna.pruners.MedianPruner()   # Early stopping
            )

        def objective(self, trial: optuna.Trial) -> float:
            """
            Objective function for Optuna

            trial.suggest_* methods sample from search space
            """
            # Sample hyperparameters
            config = {
                'num_layers': trial.suggest_int('num_layers', 2, 5),
                'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 512, 1024]),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'dropout': trial.suggest_uniform('dropout', 0.0, 0.5),
                'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd']),
            }

            # Train model with this config
            accuracy = train_and_evaluate(config)

            # Optuna will maximize this value
            return accuracy

        def optimize(self, n_trials: int = 100):
            """Run optimization"""
            self.study.optimize(self.objective, n_trials=n_trials)

            print(f"Best trial: {self.study.best_trial.number}")
            print(f"Best accuracy: {self.study.best_value:.4f}")
            print(f"Best params: {self.study.best_params}")

            return self.study.best_params

    # ============= Ray Tune Orchestration =============
    def run_ray_tune_optimization():
        """
        Main optimization workflow using Ray Tune

        Combines:
        - Optuna for Bayesian search
        - ASHA for early stopping
        - Ray for distributed execution
        """
        # Configure ASHA scheduler for early stopping
        scheduler = ASHAScheduler(
            metric="accuracy",
            mode="max",
            max_t=config.max_epochs_per_trial,
            grace_period=config.grace_period,
            reduction_factor=config.reduction_factor
        )

        # Configure Optuna search algorithm
        search_alg = OptunaSearch(
            metric="accuracy",
            mode="max"
        )

        # Limit concurrent trials
        search_alg = ConcurrencyLimiter(
            search_alg,
            max_concurrent=config.max_concurrent
        )

        # Configure reporting
        reporter = CLIReporter(
            metric_columns=["loss", "accuracy", "epoch"],
            max_report_frequency=60  # seconds
        )

        # Run optimization
        analysis = tune.run(
            train_model,
            resources_per_trial={"gpu": config.gpus_per_trial},
            config=get_search_space(),
            num_samples=config.num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
            progress_reporter=reporter,
            local_dir="./ray_results",
            name="hyperparam_search"
        )

        # Get best config
        best_trial = analysis.get_best_trial("accuracy", "max", "last")
        print(f"Best trial config: {best_trial.config}")
        print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']:.4f}")

        return best_trial.config

    # ============= Neural Architecture Search (NAS) =============
    class NeuralArchitectureSearch:
        """
        Simple NAS using evolutionary algorithm

        More advanced: DARTS, ENAS, NASNet
        """

        def __init__(self, population_size: int = 20, generations: int = 10):
            self.population_size = population_size
            self.generations = generations

        def sample_architecture(self) -> Dict:
            """Sample a random architecture"""
            return {
                'num_layers': np.random.randint(2, 6),
                'layer_configs': [
                    {
                        'type': np.random.choice(['conv', 'depthwise_conv', 'skip']),
                        'channels': np.random.choice([64, 128, 256]),
                        'kernel_size': np.random.choice([3, 5, 7])
                    }
                    for _ in range(np.random.randint(2, 6))
                ]
            }

        def mutate(self, architecture: Dict) -> Dict:
            """Mutate an architecture"""
            mutated = architecture.copy()

            # Random mutation
            if np.random.rand() < 0.3:
                mutated['num_layers'] = np.clip(
                    mutated['num_layers'] + np.random.randint(-1, 2),
                    2, 5
                )

            return mutated

        def search(self) -> Dict:
            """Run evolutionary search"""
            # Initialize population
            population = [self.sample_architecture() for _ in range(self.population_size)]
            fitness = [self.evaluate_architecture(arch) for arch in population]

            for generation in range(self.generations):
                # Selection: keep top 50%
                sorted_indices = np.argsort(fitness)[::-1]
                survivors = [population[i] for i in sorted_indices[:self.population_size // 2]]

                # Crossover & Mutation: create offspring
                offspring = []
                for _ in range(self.population_size // 2):
                    parent = np.random.choice(survivors)
                    child = self.mutate(parent)
                    offspring.append(child)

                # New population
                population = survivors + offspring
                fitness = [self.evaluate_architecture(arch) for arch in population]

                print(f"Generation {generation}: Best fitness = {max(fitness):.4f}")

            # Return best architecture
            best_idx = np.argmax(fitness)
            return population[best_idx]

        def evaluate_architecture(self, architecture: Dict) -> float:
            """Train and evaluate an architecture"""
            # Simplified - in practice, use weight sharing or early stopping
            model = build_model_from_architecture(architecture)
            accuracy = quick_train_and_evaluate(model)
            return accuracy

    # ============= Helper Functions =============
    def build_model(config: Dict) -> nn.Module:
        """Build PyTorch model from config"""
        # Simplified example
        layers = []
        for i in range(config['num_layers']):
            layers.append(nn.Linear(config['hidden_size'], config['hidden_size']))
            if config['activation'] == 'relu':
                layers.append(nn.ReLU())
            elif config['activation'] == 'gelu':
                layers.append(nn.GELU())
            layers.append(nn.Dropout(config['dropout']))

        return nn.Sequential(*layers)

    # ============= Example Usage =============
    if __name__ == "__main__":
        # Option 1: Ray Tune (recommended for scale)
        best_config = run_ray_tune_optimization()

        # Option 2: Pure Optuna
        # optimizer = BayesianOptimizer()
        # best_params = optimizer.optimize(n_trials=100)

        # Option 3: NAS
        # nas = NeuralArchitectureSearch()
        # best_architecture = nas.search()
    ```

    ## Method Comparison

    | Method | Efficiency | Compute Cost | When to Use |
    |--------|-----------|--------------|-------------|
    | **Grid Search** | ❌ Worst | Very High | <5 hyperparams, unlimited budget |
    | **Random Search** | ⭕ Baseline | High | Quick baseline, 10-100 trials |
    | **Bayesian Opt** | ✅ Good | Medium | 10-20 hyperparams, 100-1K trials |
    | **Hyperband/ASHA** | ✅✅ Best | Low | 1K+ trials, early stopping critical |
    | **NAS** | ⭕ Varies | Very High | Architecture matters more than hyperparams |

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **No Early Stopping** | Waste 90%+ compute | Use ASHA/Hyperband (10x speedup) |
    | **Optimizing on Test Set** | Overfitting to test | Use separate validation set |
    | **Ignoring Cost** | Optimize accuracy only | Multi-objective: accuracy vs latency/FLOPs |
    | **Small Search Space** | Miss better configs | Start wide, then refine |
    | **Large Batch Sizes Only** | Miss small batch benefits | Include [16, 32, 64] in search |
    | **No Warmup** | Unstable early training | Add warmup_epochs parameter |
    | **Single Seed** | High variance | Run best config with 3-5 seeds |
    | **Forgetting Checkpoints** | Can't resume | Checkpoint every N epochs |

    ## Real-World Examples

    **Google's AutoML:**
    - **Scale:** 100K+ trials on 800 GPUs for ImageNet
    - **Method:** Neural Architecture Search (NASNet, EfficientNet)
    - **Result:** EfficientNet: 8.4x smaller, 6.1x faster than previous best
    - **Impact:** Achieved SOTA with automated architecture design

    **OpenAI's GPT-3:**
    - **Scale:** Thousands of scaling law experiments
    - **Method:** Grid search over model size, dataset size, compute
    - **Finding:** Predictable scaling laws (power laws)
    - **Impact:** Informed decision to train 175B parameter model

    **DeepMind's AlphaGo:**
    - **Method:** Bayesian optimization for RL hyperparameters
    - **Params:** Learning rate, batch size, exploration constant
    - **Trials:** 100s of full training runs on TPUs
    - **Impact:** Beat world champion with optimized training

    !!! tip "Interviewer's Insight"
        Emphasizes Bayesian optimization for sample efficiency (10x better than random), ASHA/Hyperband for early stopping (10x compute savings), and distributed execution with Ray Tune. Discusses acquisition functions (EI vs UCB), exploration-exploitation trade-off, and can explain how Google's AutoML achieves SOTA through NAS at scale.

---

### Design a Model Retraining System - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `MLOps` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    ## Scale Requirements

    - **Models:** 10-1K models to manage
    - **Retraining Frequency:** Daily to monthly per model
    - **Data Volume:** 1GB-1TB new data per retrain
    - **Training Time:** 1h-24h per model
    - **Deployment:** <1h from trigger to production
    - **Monitoring:** Real-time drift detection
    - **Rollback:** <5min if issues detected

    ## Architecture

    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                   Trigger Detection System                       │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │          1. Scheduled Trigger (Cron-based)                │  │
    │  │                                                            │  │
    │  │  - Daily: 00:00 UTC                                       │  │
    │  │  - Weekly: Sunday midnight                                │  │
    │  │  - Monthly: 1st of month                                  │  │
    │  │  Use: Baseline refresh, seasonal updates                  │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │          2. Data Drift Trigger (Statistical)              │  │
    │  │                                                            │  │
    │  │  Metrics:                                                 │  │
    │  │  - PSI > 0.25 (population stability index)                │  │
    │  │  - KL divergence > threshold                              │  │
    │  │  - Feature distribution shifts (KS test)                  │  │
    │  │                                                            │  │
    │  │  Check: Every hour on recent data vs training data        │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │        3. Performance Degradation Trigger                 │  │
    │  │                                                            │  │
    │  │  Conditions:                                              │  │
    │  │  - Accuracy drop > 5% (e.g., 90% → 85%)                   │  │
    │  │  - Precision/Recall < threshold                           │  │
    │  │  - Business metric impact (e.g., revenue loss)            │  │
    │  │                                                            │  │
    │  │  Alert: Immediate if drop > 10%                           │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │          4. Concept Drift Trigger (Label shift)           │  │
    │  │                                                            │  │
    │  │  - True labels differ from predictions                    │  │
    │  │  - Feedback loop detects pattern changes                  │  │
    │  │  Example: Fraud patterns evolve, user preferences change  │  │
    │  └──────────────────────────────────────────────────────────┘  │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓ (Trigger fired)
    ┌─────────────────────────────────────────────────────────────────┐
    │                  Data Preparation Pipeline                       │
    │                                                                  │
    │  1. Fetch new data (last N days since previous training)        │
    │  2. Combine with historical data (sliding window)               │
    │  3. Data quality checks (schema, nulls, outliers)               │
    │  4. Feature engineering (same transformations as before)        │
    │  5. Train/validation split (temporal split for time-series)     │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                   Training Pipeline (Airflow/Kubeflow)           │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │              Incremental vs Full Retrain                  │  │
    │  │                                                            │  │
    │  │  Incremental (warm start):                                │  │
    │  │  - Load existing model weights                            │  │
    │  │  - Fine-tune on new data                                  │  │
    │  │  - Faster (10x), but risk of catastrophic forgetting      │  │
    │  │                                                            │  │
    │  │  Full retrain (from scratch):                             │  │
    │  │  - Train on all data (new + historical window)            │  │
    │  │  - Slower, but more robust                                │  │
    │  │                                                            │  │
    │  │  Decision: Full if drift severe, else incremental         │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  Training job → GPU cluster → Checkpoints to S3                 │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                  Validation & Testing                            │
    │                                                                  │
    │  1. Offline metrics (validation set):                           │
    │     - Accuracy, precision, recall, AUC                          │
    │     - Must exceed minimum thresholds                            │
    │                                                                  │
    │  2. Backtesting (historical data):                              │
    │     - Test on last 30 days of actual data                       │
    │     - Compare vs old model performance                          │
    │                                                                  │
    │  3. Shadow mode (parallel deployment):                          │
    │     - Run new model alongside old model                         │
    │     - Log predictions, compare results                          │
    │     - Duration: 24-48 hours                                     │
    │                                                                  │
    │  4. Approval gate:                                              │
    │     - Auto-approve if metrics > baseline + 2%                   │
    │     - Human review if 0-2% improvement                          │
    │     - Block if < baseline                                       │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓ (Approved)
    ┌─────────────────────────────────────────────────────────────────┐
    │                  Deployment Strategy                             │
    │                                                                  │
    │  Option 1: Blue-Green Deployment                                │
    │  - Deploy new model to "green" environment                      │
    │  - Switch traffic 100% → green instantly                        │
    │  - Keep "blue" (old) ready for instant rollback                 │
    │                                                                  │
    │  Option 2: Canary Deployment (recommended)                      │
    │  - 5% traffic → new model (1 hour)                              │
    │  - 25% traffic → new model (6 hours)                            │
    │  - 50% traffic → new model (12 hours)                           │
    │  - 100% traffic → new model (24 hours)                          │
    │  - Rollback if error rate spikes or latency > SLA               │
    │                                                                  │
    │  Option 3: A/B Test                                             │
    │  - 50/50 split for 1 week                                       │
    │  - Statistical test for significance                            │
    │  - Gradual rollout after significance                           │
    └────────────────┬────────────────────────────────────────────────┘
                     ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                Production Monitoring                             │
    │                                                                  │
    │  - Model version tracking (e.g., v23 in production)             │
    │  - Performance metrics dashboard                                │
    │  - Automated rollback if:                                       │
    │    * Error rate > 1%                                            │
    │    * Latency p99 > 2x baseline                                  │
    │    * Accuracy drop > 5% (from online feedback)                  │
    │                                                                  │
    │  - Alert on-call team via PagerDuty                             │
    └─────────────────────────────────────────────────────────────────┘
    ```

    ## Production Implementation (240 lines)

    ```python
    # model_retraining.py
    from typing import Dict, Any, Optional, Tuple
    from dataclasses import dataclass
    from datetime import datetime, timedelta
    import numpy as np
    from scipy import stats
    import logging

    # ============= Configuration =============
    @dataclass
    class RetrainingConfig:
        """Model retraining configuration"""
        # Trigger thresholds
        psi_threshold: float = 0.25
        accuracy_drop_threshold: float = 0.05

        # Retraining settings
        retrain_window_days: int = 90  # Use last 90 days of data
        min_new_samples: int = 10000

        # Deployment
        canary_percentages: list = None
        shadow_mode_hours: int = 24

        def __post_init__(self):
            if self.canary_percentages is None:
                self.canary_percentages = [5, 25, 50, 100]

    config = RetrainingConfig()

    # ============= Drift Detection =============
    class DriftDetector:
        """Detect data and concept drift"""

        def __init__(self):
            self.logger = logging.getLogger(__name__)

        def compute_psi(
            self,
            expected: np.ndarray,
            actual: np.ndarray,
            bins: int = 10
        ) -> float:
            """
            Population Stability Index (PSI)

            PSI = Σ (actual% - expected%) * ln(actual% / expected%)

            Thresholds:
            - PSI < 0.1: No change
            - 0.1 < PSI < 0.25: Moderate change
            - PSI > 0.25: Significant change (retrain recommended)
            """
            # Bin the data
            breakpoints = np.linspace(
                min(expected.min(), actual.min()),
                max(expected.max(), actual.max()),
                bins + 1
            )

            expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
            actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

            # Avoid log(0)
            expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
            actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

            psi = np.sum(
                (actual_percents - expected_percents) *
                np.log(actual_percents / expected_percents)
            )

            return psi

        def detect_feature_drift(
            self,
            train_data: Dict[str, np.ndarray],
            production_data: Dict[str, np.ndarray]
        ) -> Dict[str, float]:
            """
            Check drift for all features
            Returns: {feature_name: psi_value}
            """
            drift_scores = {}

            for feature in train_data.keys():
                psi = self.compute_psi(
                    train_data[feature],
                    production_data[feature]
                )
                drift_scores[feature] = psi

                if psi > config.psi_threshold:
                    self.logger.warning(
                        f"Feature '{feature}' has significant drift: PSI={psi:.3f}"
                    )

            return drift_scores

        def kolmogorov_smirnov_test(
            self,
            expected: np.ndarray,
            actual: np.ndarray,
            alpha: float = 0.05
        ) -> Tuple[bool, float]:
            """
            Two-sample KS test for distribution shift

            Returns: (is_different, p_value)
            """
            statistic, p_value = stats.ks_2samp(expected, actual)
            is_different = p_value < alpha

            return is_different, p_value

    # ============= Performance Monitor =============
    class PerformanceMonitor:
        """Monitor model performance in production"""

        def __init__(self, baseline_metrics: Dict[str, float]):
            self.baseline = baseline_metrics
            self.logger = logging.getLogger(__name__)

        def check_performance_degradation(
            self,
            current_metrics: Dict[str, float]
        ) -> Tuple[bool, Dict[str, float]]:
            """
            Check if performance has degraded

            Returns: (should_retrain, degradation_report)
            """
            degradation = {}
            should_retrain = False

            for metric, baseline_value in self.baseline.items():
                if metric in current_metrics:
                    current_value = current_metrics[metric]
                    drop = baseline_value - current_value
                    drop_percent = drop / baseline_value if baseline_value > 0 else 0

                    degradation[metric] = {
                        'baseline': baseline_value,
                        'current': current_value,
                        'drop': drop,
                        'drop_percent': drop_percent
                    }

                    if drop_percent > config.accuracy_drop_threshold:
                        should_retrain = True
                        self.logger.warning(
                            f"{metric} degraded by {drop_percent:.1%}: "
                            f"{baseline_value:.3f} → {current_value:.3f}"
                        )

            return should_retrain, degradation

    # ============= Retraining Orchestrator =============
    class RetrainingOrchestrator:
        """Orchestrate the retraining process"""

        def __init__(self):
            self.drift_detector = DriftDetector()
            self.logger = logging.getLogger(__name__)

        def should_retrain(
            self,
            trigger_type: str,
            **kwargs
        ) -> Tuple[bool, str]:
            """
            Decide whether to trigger retraining

            Returns: (should_retrain, reason)
            """
            if trigger_type == 'scheduled':
                return True, "Scheduled retrain"

            elif trigger_type == 'data_drift':
                drift_scores = kwargs.get('drift_scores', {})
                max_drift = max(drift_scores.values()) if drift_scores else 0

                if max_drift > config.psi_threshold:
                    return True, f"Data drift detected: max PSI={max_drift:.3f}"
                return False, "No significant drift"

            elif trigger_type == 'performance':
                degradation = kwargs.get('degradation', {})

                for metric, info in degradation.items():
                    if info['drop_percent'] > config.accuracy_drop_threshold:
                        return True, f"{metric} degraded by {info['drop_percent']:.1%}"

                return False, "Performance within acceptable range"

            else:
                return False, "Unknown trigger type"

        def execute_retraining(
            self,
            model_id: str,
            retrain_type: str = 'full'  # 'full' or 'incremental'
        ) -> Dict[str, Any]:
            """
            Execute the retraining pipeline

            Returns: training metadata
            """
            self.logger.info(f"Starting {retrain_type} retraining for model {model_id}")

            # 1. Data preparation
            data = self._prepare_data(model_id)

            # 2. Training
            if retrain_type == 'incremental':
                new_model = self._incremental_train(model_id, data)
            else:
                new_model = self._full_retrain(model_id, data)

            # 3. Validation
            validation_metrics = self._validate_model(new_model, data['validation'])

            # 4. Decision: deploy or reject
            if self._should_deploy(validation_metrics):
                deployment_info = self._deploy_model(model_id, new_model)

                return {
                    'status': 'deployed',
                    'model_version': deployment_info['version'],
                    'metrics': validation_metrics,
                    'deployment': deployment_info
                }
            else:
                return {
                    'status': 'rejected',
                    'reason': 'Failed validation checks',
                    'metrics': validation_metrics
                }

        def _prepare_data(self, model_id: str) -> Dict:
            """Fetch and prepare training data"""
            # Fetch new data from last N days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=config.retrain_window_days)

            # Placeholder - actual implementation would query database
            return {
                'train': None,
                'validation': None,
                'metadata': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'num_samples': 100000
                }
            }

        def _incremental_train(self, model_id: str, data: Dict) -> Any:
            """Incremental training (warm start)"""
            # Load existing model
            # Fine-tune on new data
            # Return updated model
            pass

        def _full_retrain(self, model_id: str, data: Dict) -> Any:
            """Full retraining from scratch"""
            # Train new model on all data
            # Return new model
            pass

        def _validate_model(self, model: Any, validation_data: Any) -> Dict:
            """Validate new model"""
            # Compute metrics on validation set
            return {
                'accuracy': 0.92,
                'precision': 0.90,
                'recall': 0.88,
                'f1': 0.89
            }

        def _should_deploy(self, metrics: Dict) -> bool:
            """Decide if model should be deployed"""
            # Check against minimum thresholds
            min_thresholds = {
                'accuracy': 0.85,
                'precision': 0.80,
                'recall': 0.80
            }

            for metric, min_value in min_thresholds.items():
                if metrics.get(metric, 0) < min_value:
                    self.logger.warning(
                        f"{metric}={metrics[metric]:.3f} below threshold {min_value}"
                    )
                    return False

            return True

        def _deploy_model(self, model_id: str, model: Any) -> Dict:
            """Deploy model with canary strategy"""
            version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            self.logger.info(f"Deploying {model_id} {version} with canary rollout")

            # Canary deployment
            for percentage in config.canary_percentages:
                self.logger.info(f"Routing {percentage}% traffic to new model")
                # Monitor for 1 hour at each stage
                # If errors spike, rollback

            return {
                'version': version,
                'strategy': 'canary',
                'deployed_at': datetime.now()
            }

    # ============= Example Usage =============
    def example_retraining_workflow():
        """Example: Complete retraining workflow"""

        # Initialize
        orchestrator = RetrainingOrchestrator()
        drift_detector = DriftDetector()

        # Simulate training and production data
        train_features = {'age': np.random.normal(35, 10, 10000)}
        prod_features = {'age': np.random.normal(40, 10, 1000)}  # Distribution shifted

        # Check for drift
        drift_scores = drift_detector.detect_feature_drift(train_features, prod_features)
        print(f"Drift scores: {drift_scores}")

        # Decide if retraining needed
        should_retrain, reason = orchestrator.should_retrain(
            trigger_type='data_drift',
            drift_scores=drift_scores
        )

        if should_retrain:
            print(f"Retraining triggered: {reason}")
            result = orchestrator.execute_retraining(
                model_id='fraud_detector',
                retrain_type='full'
            )
            print(f"Retraining result: {result}")
        else:
            print(f"No retraining needed: {reason}")
    ```

    ## Trigger Strategy Comparison

    | Trigger Type | Frequency | Pros | Cons | Best For |
    |--------------|-----------|------|------|----------|
    | **Scheduled** | Fixed (daily/weekly) | Predictable, simple | May retrain unnecessarily | Stable models, routine refresh |
    | **Data Drift** | Event-driven | Adaptive, efficient | Requires monitoring | Models sensitive to distribution shifts |
    | **Performance** | Event-driven | Directly targets quality | Reactive (damage done) | Critical models, fast feedback |
    | **Hybrid** | Combines above | Best of all worlds | More complex | Production systems |

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Catastrophic Forgetting** | Incremental training loses old knowledge | Full retrain or experience replay |
    | **No Validation** | Deploy broken models | Multi-stage validation + shadow mode |
    | **Training-Serving Skew** | Features differ train vs prod | Feature store with consistency |
    | **Instant Rollout** | Risk of widespread failure | Canary deployment (5%→25%→100%) |
    | **No Rollback Plan** | Stuck with bad model | Keep old model live, instant rollback |
    | **Stale Data** | Train on old data | Real-time data pipeline, short windows |
    | **Too Frequent Retraining** | Waste compute, instability | Set minimum intervals (e.g., 1 day) |
    | **Ignoring Business Impact** | Optimize wrong metrics | Monitor business KPIs (revenue, churn) |

    ## Real-World Examples

    **Uber's Michelangelo:**
    - **Retraining:** Daily for surge pricing models
    - **Trigger:** Scheduled + performance monitoring
    - **Strategy:** Canary deployment with 5%→100% rollout
    - **Rollback:** Automated if latency > 10ms or errors > 0.1%
    - **Impact:** Keep pricing models current with demand patterns

    **Netflix's Recommendation:**
    - **Retraining:** Weekly for personalization models
    - **Trigger:** A/B test performance + scheduled
    - **Data:** Last 90 days of viewing history
    - **Validation:** Shadow mode for 24h before full rollout
    - **Impact:** Maintains 80% engagement from recommendations

    **Airbnb's Pricing Model:**
    - **Retraining:** Daily updates for dynamic pricing
    - **Trigger:** Scheduled + market events (holidays, etc.)
    - **Strategy:** Blue-green deployment
    - **Monitoring:** Revenue impact, booking rate
    - **Impact:** $1B+ annual revenue from optimized pricing

    !!! tip "Interviewer's Insight"
        Emphasizes drift detection (PSI, KS test) to trigger smart retraining rather than blind scheduling, canary deployment for safe rollout (5%→100%), and shadow mode for validation. Discusses trade-offs between incremental vs full retraining (speed vs quality), and can explain how Uber/Netflix/Airbnb implement continuous retraining at scale.

---

### Design a Vector Search System - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Embeddings`, `Search` | **Asked by:** Google, Meta, OpenAI

??? success "View Answer"

    ## Scale Requirements

    - **Index Size:** 1 billion+ vectors (768-dim embeddings)
    - **QPS:** 10,000+ queries/second
    - **Latency:** p50 < 20ms, p99 < 100ms
    - **Recall@10:** > 95% (vs brute-force)
    - **Throughput:** 50K+ inserts/second
    - **Availability:** 99.99% uptime

    ## Architecture

    ```
    ┌─────────────────────────────────────────────────────────┐
    │                    Vector Search System                  │
    ├─────────────────────────────────────────────────────────┤
    │                                                           │
    │  Query Vector (768-dim)                                  │
    │         ↓                                                 │
    │  ┌─────────────┐                                         │
    │  │ Query API   │ ← Rate limiting, auth                   │
    │  └─────┬───────┘                                         │
    │        │                                                  │
    │  ┌─────▼────────────────────────────────────────┐       │
    │  │        Embedding Normalization                │       │
    │  │  (L2 normalize, dimension check)              │       │
    │  └─────┬────────────────────────────────────────┘       │
    │        │                                                  │
    │  ┌─────▼─────────────────────────────────────────┐      │
    │  │           Index Router (Sharding)              │      │
    │  │  - Hash-based sharding (1B vectors → 10 shards)│     │
    │  │  - Replicas for availability (3x replication)  │      │
    │  └─────┬────────────────────────────────────────┘       │
    │        │                                                  │
    │        ├──────┬──────┬──────┬──────┬──────┐            │
    │        ↓      ↓      ↓      ↓      ↓      ↓             │
    │   ┌────────────────────────────────────────┐            │
    │   │   HNSW Index Shards (In-Memory)        │            │
    │   │                                          │            │
    │   │  Shard 1    Shard 2    ...   Shard 10   │           │
    │   │  100M vec  100M vec         100M vec    │           │
    │   │                                          │            │
    │   │  Layer 0: Full graph (ef=200)           │            │
    │   │  Layer 1: Skip connections (ef=100)     │            │
    │   │  Layer 2-N: Hierarchical shortcuts      │            │
    │   └────────┬──────────────────────────────┘             │
    │            │                                              │
    │     ┌──────▼──────────┐                                 │
    │     │  Result Merger  │                                  │
    │     │  (Top-k heap)   │                                  │
    │     └──────┬──────────┘                                  │
    │            │                                              │
    │     ┌──────▼──────────────────────────┐                 │
    │     │   Post-processing & Filtering   │                 │
    │     │  - Deduplication                 │                 │
    │     │  - Metadata filtering            │                 │
    │     │  - Re-ranking (optional)         │                 │
    │     └──────┬──────────────────────────┘                  │
    │            │                                              │
    │     ┌──────▼──────────┐                                 │
    │     │  Response (k=10)│                                  │
    │     │  [{id, score}]  │                                  │
    │     └─────────────────┘                                  │
    │                                                           │
    │  ┌────────────────────────────────────────────┐         │
    │  │        Background Services                  │         │
    │  ├────────────────────────────────────────────┤         │
    │  │  • Index Builder (batch inserts)           │         │
    │  │  • Compaction (merge segments)             │         │
    │  │  • Snapshot & Backup (hourly)              │         │
    │  │  • Monitoring (latency, recall, QPS)       │         │
    │  └────────────────────────────────────────────┘         │
    └─────────────────────────────────────────────────────────┘

    Storage Layer:
    ┌──────────────────────────────────────┐
    │  Index Storage (SSD)                 │
    │  - HNSW graph snapshots              │
    │  - Metadata (filters, timestamps)    │
    │  - Write-ahead log (WAL)             │
    └──────────────────────────────────────┘
    ```

    ## Production Implementation (280 lines)

    ```python
    # vector_search.py
    from typing import List, Tuple, Dict, Any, Optional
    from dataclasses import dataclass
    import numpy as np
    from scipy.spatial.distance import cosine
    import heapq
    from collections import defaultdict
    import pickle
    import logging

    @dataclass
    class SearchConfig:
        """Vector search configuration"""
        dimension: int = 768
        index_type: str = "hnsw"  # hnsw, ivf, pq
        ef_construction: int = 200  # HNSW: connections during build
        ef_search: int = 100  # HNSW: connections during search
        M: int = 16  # HNSW: max connections per node
        num_clusters: int = 1000  # IVF: number of clusters
        num_subvectors: int = 8  # PQ: subvector count
        metric: str = "cosine"  # cosine, euclidean, dot_product

    class HNSWIndex:
        """
        Hierarchical Navigable Small World (HNSW) index

        Time Complexity:
        - Insert: O(M * log(N))
        - Search: O(ef * log(N))

        Space: O(N * M * d) where d=dimension

        Best for: High recall, fast search (< 10ms)
        """

        def __init__(self, config: SearchConfig):
            self.config = config
            self.dimension = config.dimension
            self.M = config.M  # Max edges per node
            self.ef_construction = config.ef_construction
            self.ef_search = config.ef_search
            self.metric = config.metric

            # Graph structure: level -> node_id -> neighbors
            self.graph: Dict[int, Dict[int, set]] = defaultdict(lambda: defaultdict(set))
            self.vectors: Dict[int, np.ndarray] = {}
            self.metadata: Dict[int, Dict] = {}
            self.entry_point = None
            self.max_level = 0

        def _distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
            """Compute distance between vectors"""
            if self.metric == "cosine":
                return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            elif self.metric == "euclidean":
                return np.linalg.norm(v1 - v2)
            elif self.metric == "dot_product":
                return -np.dot(v1, v2)  # Negative for max-heap

        def _get_random_level(self) -> int:
            """Probabilistically assign level (exponential decay)"""
            ml = 1.0 / np.log(2.0)
            level = int(-np.log(np.random.uniform(0, 1)) * ml)
            return min(level, 16)  # Cap at 16 levels

        def _search_layer(
            self,
            query: np.ndarray,
            entry_points: set,
            num_to_return: int,
            level: int
        ) -> set:
            """Search at a specific layer using greedy best-first"""
            visited = set()
            candidates = []
            w = set()

            # Initialize with entry points
            for point in entry_points:
                dist = self._distance(query, self.vectors[point])
                heapq.heappush(candidates, (-dist, point))
                visited.add(point)

            while candidates:
                current_dist, current = heapq.heappop(candidates)
                current_dist = -current_dist

                # Check if we should continue
                if len(w) >= num_to_return:
                    furthest_dist = max(
                        self._distance(query, self.vectors[p]) for p in w
                    )
                    if current_dist > furthest_dist:
                        break

                w.add(current)

                # Explore neighbors
                for neighbor in self.graph[level].get(current, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        dist = self._distance(query, self.vectors[neighbor])
                        heapq.heappush(candidates, (-dist, neighbor))

            return w

        def insert(
            self,
            vector_id: int,
            vector: np.ndarray,
            metadata: Optional[Dict] = None
        ) -> None:
            """Insert vector into HNSW index"""
            if vector.shape[0] != self.dimension:
                raise ValueError(f"Vector dimension {vector.shape[0]} != {self.dimension}")

            # Normalize for cosine similarity
            if self.metric == "cosine":
                vector = vector / np.linalg.norm(vector)

            self.vectors[vector_id] = vector
            self.metadata[vector_id] = metadata or {}

            # Assign level
            level = self._get_random_level()

            if self.entry_point is None:
                self.entry_point = vector_id
                self.max_level = level
                return

            # Search for nearest neighbors at each level
            nearest = {self.entry_point}
            for lc in range(self.max_level, level, -1):
                nearest = self._search_layer(vector, nearest, 1, lc)

            # Insert at all levels from level down to 0
            for lc in range(level, -1, -1):
                candidates = self._search_layer(
                    vector, nearest, self.ef_construction, lc
                )

                # Select M nearest neighbors
                M = self.M if lc > 0 else 2 * self.M
                neighbors = sorted(
                    candidates,
                    key=lambda x: self._distance(vector, self.vectors[x])
                )[:M]

                # Add bidirectional edges
                self.graph[lc][vector_id] = set(neighbors)
                for neighbor in neighbors:
                    self.graph[lc][neighbor].add(vector_id)

                    # Prune if exceeds M
                    if len(self.graph[lc][neighbor]) > M:
                        pruned = sorted(
                            self.graph[lc][neighbor],
                            key=lambda x: self._distance(
                                self.vectors[neighbor], self.vectors[x]
                            )
                        )[:M]
                        self.graph[lc][neighbor] = set(pruned)

                nearest = candidates

            # Update entry point if new level is higher
            if level > self.max_level:
                self.max_level = level
                self.entry_point = vector_id

        def search(
            self,
            query: np.ndarray,
            k: int = 10,
            ef: Optional[int] = None
        ) -> List[Tuple[int, float]]:
            """
            Search for k nearest neighbors

            Args:
                query: Query vector (dimension d)
                k: Number of results
                ef: Search width (default: self.ef_search)

            Returns:
                List of (vector_id, distance) tuples
            """
            if ef is None:
                ef = self.ef_search

            if self.entry_point is None:
                return []

            # Normalize query
            if self.metric == "cosine":
                query = query / np.linalg.norm(query)

            # Search from top layer down to layer 0
            nearest = {self.entry_point}
            for level in range(self.max_level, 0, -1):
                nearest = self._search_layer(query, nearest, 1, level)

            # Search layer 0 with larger ef
            candidates = self._search_layer(query, nearest, max(ef, k), 0)

            # Return top k with distances
            results = [
                (vid, self._distance(query, self.vectors[vid]))
                for vid in candidates
            ]
            results.sort(key=lambda x: x[1])
            return results[:k]

    class VectorSearchSystem:
        """Production vector search system with sharding and filtering"""

        def __init__(self, config: SearchConfig, num_shards: int = 10):
            self.config = config
            self.num_shards = num_shards
            self.shards = [HNSWIndex(config) for _ in range(num_shards)]
            self.total_vectors = 0

        def _get_shard(self, vector_id: int) -> int:
            """Hash-based sharding"""
            return vector_id % self.num_shards

        def insert(
            self,
            vector_id: int,
            vector: np.ndarray,
            metadata: Optional[Dict] = None
        ) -> None:
            """Insert vector into appropriate shard"""
            shard_idx = self._get_shard(vector_id)
            self.shards[shard_idx].insert(vector_id, vector, metadata)
            self.total_vectors += 1

        def batch_insert(
            self,
            vectors: List[Tuple[int, np.ndarray, Dict]]
        ) -> None:
            """Batch insert for efficiency"""
            for vector_id, vector, metadata in vectors:
                self.insert(vector_id, vector, metadata)

        def search(
            self,
            query: np.ndarray,
            k: int = 10,
            filters: Optional[Dict[str, Any]] = None
        ) -> List[Tuple[int, float, Dict]]:
            """
            Search across all shards and merge results

            Args:
                query: Query vector
                k: Number of results
                filters: Metadata filters (e.g., {"category": "sports"})

            Returns:
                List of (vector_id, distance, metadata)
            """
            # Search each shard in parallel (simplified here)
            all_results = []
            for shard in self.shards:
                shard_results = shard.search(query, k * 2)  # Over-fetch
                all_results.extend(shard_results)

            # Apply metadata filters
            if filters:
                filtered = []
                for vid, dist in all_results:
                    shard_idx = self._get_shard(vid)
                    metadata = self.shards[shard_idx].metadata.get(vid, {})

                    # Check all filter conditions
                    match = all(
                        metadata.get(key) == value
                        for key, value in filters.items()
                    )
                    if match:
                        filtered.append((vid, dist, metadata))
            else:
                filtered = [
                    (vid, dist, self.shards[self._get_shard(vid)].metadata.get(vid, {}))
                    for vid, dist in all_results
                ]

            # Merge and return top k
            filtered.sort(key=lambda x: x[1])
            return filtered[:k]

        def save(self, path: str) -> None:
            """Save index to disk"""
            with open(path, 'wb') as f:
                pickle.dump({
                    'config': self.config,
                    'shards': self.shards,
                    'total_vectors': self.total_vectors
                }, f)

        @classmethod
        def load(cls, path: str) -> 'VectorSearchSystem':
            """Load index from disk"""
            with open(path, 'rb') as f:
                data = pickle.load(f)

            system = cls(data['config'], len(data['shards']))
            system.shards = data['shards']
            system.total_vectors = data['total_vectors']
            return system

    # Example usage
    if __name__ == "__main__":
        # Initialize system
        config = SearchConfig(
            dimension=768,
            index_type="hnsw",
            ef_construction=200,
            ef_search=100,
            M=16,
            metric="cosine"
        )

        search_system = VectorSearchSystem(config, num_shards=10)

        # Insert 1M vectors (simulated)
        print("Inserting vectors...")
        for i in range(1_000_000):
            vector = np.random.randn(768).astype(np.float32)
            metadata = {
                "category": np.random.choice(["tech", "sports", "news"]),
                "timestamp": "2025-01-15"
            }
            search_system.insert(i, vector, metadata)

            if (i + 1) % 100_000 == 0:
                print(f"Inserted {i + 1} vectors")

        # Search
        print("\nSearching...")
        query = np.random.randn(768).astype(np.float32)
        results = search_system.search(
            query,
            k=10,
            filters={"category": "tech"}
        )

        print(f"\nTop 10 results:")
        for vid, dist, metadata in results:
            print(f"  ID: {vid}, Distance: {dist:.4f}, Metadata: {metadata}")
    ```

    ## ANN Algorithm Comparison

    | Algorithm | Build Time | Search Latency | Memory | Recall@10 | Best For |
    |-----------|------------|----------------|--------|-----------|----------|
    | **HNSW** | O(N log N) | **5-20ms** | **High** (2-4x vectors) | **95-99%** | Low latency, high recall |
    | **IVF** | O(N) | 20-50ms | Medium (1.5x) | 90-95% | Large-scale, cost-sensitive |
    | **PQ** | O(N) | 10-30ms | **Low** (0.5x) | 85-92% | Memory-constrained |
    | **LSH** | **O(N)** | 50-100ms | Low | 80-90% | Streaming inserts |
    | **ScaNN** | O(N log N) | 10-25ms | Medium | 92-97% | Google-scale (1B+ vectors) |

    **Hybrid Approach (Best for Production):**
    - **IVF + PQ:** Cluster with IVF, compress with PQ → 10x memory reduction
    - **HNSW + PQ:** Fast search + compression → 3x memory reduction

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Cold Start** | First queries slow (loading index) | Pre-warm cache, keep index in memory |
    | **High-Dimensional Curse** | Distances become similar | Dimensionality reduction (PCA, UMAP) |
    | **Unbalanced Shards** | Some shards overloaded | Consistent hashing, dynamic rebalancing |
    | **Stale Vectors** | Old embeddings don't match new model | Versioning, incremental re-embedding |
    | **No Filtering** | Post-filtering slow | Pre-filtering with inverted index |
    | **Single Index** | No A/B testing of embeddings | Multi-index support, traffic splitting |
    | **Ignoring Quantization** | 4x memory overhead (float32) | Use float16 or int8 (minimal quality loss) |
    | **Sequential Inserts** | Slow indexing | Batch inserts (10K-100K at a time) |

    ## Real-World Examples

    **Google Vertex AI Matching Engine:**
    - **Scale:** 10 billion+ vectors, 768-1536 dimensions
    - **Algorithm:** ScaNN (Google's HNSW variant)
    - **Latency:** p50 < 10ms, p99 < 50ms at 10K QPS
    - **Features:** Auto-sharding, streaming updates, metadata filtering
    - **Use Cases:** YouTube recommendations, Google Shopping

    **Meta FAISS:**
    - **Scale:** 1 billion vectors, 512-2048 dimensions
    - **Algorithm:** IVF + PQ (memory-optimized)
    - **Throughput:** 100K QPS on single server
    - **Optimization:** GPU acceleration (10x faster than CPU)
    - **Use Cases:** Instagram Explore, Facebook Search

    **OpenAI Vector Search:**
    - **Scale:** 100M+ embeddings (text-embedding-ada-002)
    - **Algorithm:** Custom HNSW with caching
    - **Latency:** < 20ms p99 for GPT retrieval
    - **Features:** Hybrid search (dense + sparse), re-ranking
    - **Use Cases:** ChatGPT memory, code search

    **Pinecone (SaaS):**
    - **Scale:** Multi-tenant, 10B+ vectors across customers
    - **Algorithm:** Proprietary (HNSW-based)
    - **Latency:** p50 < 30ms globally
    - **Features:** Serverless, auto-scaling, namespaces
    - **Customers:** Shopify, Gong, Jasper

    ## Key Metrics to Monitor

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Search Latency (p99)** | < 100ms | > 150ms |
    | **Recall@10** | > 95% | < 90% |
    | **QPS** | 10K+ | Capacity planning at 80% |
    | **Index Build Time** | < 1 hour (10M vectors) | > 2 hours |
    | **Memory Usage** | < 80% of available | > 90% |
    | **Error Rate** | < 0.1% | > 1% |

    !!! tip "Interviewer's Insight"
        Discusses HNSW for low-latency search (< 20ms p99) with 95%+ recall, explains sharding strategy for billion-scale indexes, and understands trade-offs between memory (HNSW > IVF > PQ), speed (HNSW > PQ > IVF), and recall (HNSW > IVF > PQ). Can explain how Google/Meta/OpenAI use hybrid approaches (IVF+PQ) for 10x memory reduction while maintaining 90%+ recall.

---

### Design an Embedding Service - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Embeddings` | **Asked by:** Google, Meta, OpenAI

??? success "View Answer"

    ## Scale Requirements

    - **QPS:** 50,000+ requests/second
    - **Latency:** p50 < 20ms, p99 < 50ms (single request)
    - **Batch Latency:** p99 < 100ms (batch of 100)
    - **Throughput:** 5M+ embeddings/second
    - **Model Size:** BERT-base (110M params), Sentence-BERT (330M params)
    - **GPU Utilization:** > 80% (cost efficiency)
    - **Cache Hit Rate:** > 70% (for repeat queries)

    ## Architecture

    ```
    ┌────────────────────────────────────────────────────────┐
    │               Embedding Service Architecture            │
    ├────────────────────────────────────────────────────────┤
    │                                                          │
    │  Client Requests (text → embeddings)                    │
    │         ↓                                                │
    │  ┌──────────────────┐                                   │
    │  │   Load Balancer  │ ← Rate limiting (per-user)        │
    │  │   (NGINX/Envoy)  │                                   │
    │  └────────┬─────────┘                                   │
    │           │                                              │
    │           ├──────────┬──────────┬──────────┐           │
    │           ↓          ↓          ↓          ↓            │
    │      ┌─────────────────────────────────────────┐       │
    │      │       API Servers (FastAPI/gRPC)        │       │
    │      │  - Request validation                   │       │
    │      │  - Input preprocessing                  │       │
    │      │  - Cache lookup (Redis)                 │       │
    │      └─────────┬───────────────────────────────┘       │
    │                │                                         │
    │         ┌──────▼──────────────┐                        │
    │         │  Cache Layer (Redis)│                        │
    │         │  - LRU eviction     │                        │
    │         │  - TTL: 24h         │                        │
    │         │  - Key: hash(text)  │                        │
    │         └──────┬──────────────┘                        │
    │                │ (cache miss)                           │
    │         ┌──────▼──────────────────────────┐            │
    │         │  Dynamic Batch Collector         │           │
    │         │  - Max wait: 10ms                │           │
    │         │  - Max batch: 128                │           │
    │         │  - Timeout: adaptive             │           │
    │         └──────┬──────────────────────────┘            │
    │                │                                         │
    │                ↓                                         │
    │         ┌──────────────────────────────────┐           │
    │         │  Model Inference Servers (GPU)   │           │
    │         │                                   │           │
    │         │  ┌─────────────────────────────┐ │           │
    │         │  │  GPU 1: BERT (TensorRT)     │ │           │
    │         │  │  - Mixed precision (FP16)   │ │           │
    │         │  │  - Batch size: 128          │ │           │
    │         │  │  - Throughput: 2K req/s     │ │           │
    │         │  └─────────────────────────────┘ │           │
    │         │                                   │           │
    │         │  ┌─────────────────────────────┐ │           │
    │         │  │  GPU 2-N: Replicas          │ │           │
    │         │  │  - Auto-scaling (K8s HPA)   │ │           │
    │         │  │  - GPU utilization > 80%    │ │           │
    │         │  └─────────────────────────────┘ │           │
    │         └──────┬──────────────────────────┘            │
    │                │                                         │
    │         ┌──────▼──────────────────────────┐            │
    │         │  Response Aggregator             │           │
    │         │  - Unbatch results               │           │
    │         │  - Update cache (async)          │           │
    │         │  - Logging & metrics             │           │
    │         └──────┬──────────────────────────┘            │
    │                │                                         │
    │         ┌──────▼──────────────┐                        │
    │         │  Response            │                        │
    │         │  {embedding: [768]}  │                        │
    │         └─────────────────────┘                         │
    │                                                          │
    │  ┌──────────────────────────────────────────┐          │
    │  │       Background Services                 │          │
    │  ├──────────────────────────────────────────┤          │
    │  │  • Model Warmup (preload GPU)            │          │
    │  │  • Metrics Export (Prometheus)           │          │
    │  │  • Health Checks (liveness/readiness)    │          │
    │  │  • A/B Testing (model versions)          │          │
    │  └──────────────────────────────────────────┘          │
    └────────────────────────────────────────────────────────┘

    Model Storage:
    ┌──────────────────────────────────┐
    │  S3 / GCS                        │
    │  - Model weights (versioned)     │
    │  - TensorRT engines              │
    │  - Tokenizer configs             │
    └──────────────────────────────────┘
    ```

    ## Production Implementation (290 lines)

    ```python
    # embedding_service.py
    from typing import List, Dict, Optional, Tuple
    from dataclasses import dataclass
    import asyncio
    import time
    import hashlib
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModel
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import redis
    import logging
    from prometheus_client import Counter, Histogram, Gauge

    # Metrics
    REQUESTS = Counter('embedding_requests_total', 'Total requests')
    LATENCY = Histogram('embedding_latency_seconds', 'Request latency')
    CACHE_HITS = Counter('cache_hits_total', 'Cache hits')
    CACHE_MISSES = Counter('cache_misses_total', 'Cache misses')
    BATCH_SIZE = Histogram('batch_size', 'Batch size distribution')
    GPU_UTIL = Gauge('gpu_utilization', 'GPU utilization %')

    @dataclass
    class EmbeddingConfig:
        """Embedding service configuration"""
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
        max_batch_size: int = 128
        max_batch_wait_ms: int = 10
        cache_ttl_hours: int = 24
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        use_fp16: bool = True  # Mixed precision
        max_seq_length: int = 512

    class EmbeddingRequest(BaseModel):
        """API request schema"""
        texts: List[str]
        normalize: bool = True

    class EmbeddingResponse(BaseModel):
        """API response schema"""
        embeddings: List[List[float]]
        cached: List[bool]
        latency_ms: float

    class DynamicBatcher:
        """
        Dynamic batching for GPU efficiency

        Accumulates requests until:
        1. Batch size reaches max_batch_size, OR
        2. Wait time exceeds max_batch_wait_ms

        This increases GPU utilization from ~30% → 80%+
        """

        def __init__(self, config: EmbeddingConfig):
            self.config = config
            self.queue: List[Tuple[str, asyncio.Future]] = []
            self.lock = asyncio.Lock()
            self.timer_task = None

        async def add_request(self, text: str) -> np.ndarray:
            """Add request to batch queue"""
            future = asyncio.Future()

            async with self.lock:
                self.queue.append((text, future))

                # Start timer on first request in batch
                if len(self.queue) == 1:
                    self.timer_task = asyncio.create_task(
                        self._wait_and_flush()
                    )

                # Flush if batch is full
                if len(self.queue) >= self.config.max_batch_size:
                    if self.timer_task:
                        self.timer_task.cancel()
                    await self._flush_batch()

            # Wait for batch to complete
            embedding = await future
            return embedding

        async def _wait_and_flush(self):
            """Wait for max_batch_wait_ms, then flush"""
            await asyncio.sleep(self.config.max_batch_wait_ms / 1000.0)
            async with self.lock:
                await self._flush_batch()

        async def _flush_batch(self):
            """Process accumulated batch"""
            if not self.queue:
                return

            batch = self.queue
            self.queue = []

            # Record batch size
            BATCH_SIZE.observe(len(batch))

            # This will be filled by the inference engine
            # For now, we just signal the batch is ready
            # (actual inference handled by EmbeddingModel)
            pass

    class EmbeddingModel:
        """
        GPU-optimized embedding model

        Optimizations:
        - TorchScript compilation
        - Mixed precision (FP16)
        - Dynamic batching
        - Model warmup
        """

        def __init__(self, config: EmbeddingConfig):
            self.config = config
            self.device = torch.device(config.device)

            logging.info(f"Loading model {config.model_name} on {self.device}")

            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModel.from_pretrained(config.model_name)
            self.model.to(self.device)
            self.model.eval()

            # Enable mixed precision (FP16) for 2x speedup
            if config.use_fp16 and config.device == "cuda":
                self.model.half()

            # Warmup (avoid cold start latency)
            self._warmup()

        def _warmup(self):
            """Warmup model with dummy inputs"""
            logging.info("Warming up model...")
            dummy_texts = ["hello world"] * 32
            with torch.no_grad():
                self.encode(dummy_texts)
            logging.info("Warmup complete")

        @torch.no_grad()
        def encode(
            self,
            texts: List[str],
            normalize: bool = True
        ) -> np.ndarray:
            """
            Encode texts to embeddings

            Args:
                texts: List of text strings
                normalize: L2 normalize embeddings

            Returns:
                np.ndarray of shape (len(texts), embedding_dim)
            """
            # Tokenize
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors='pt'
            )

            # Move to GPU
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Forward pass
            outputs = self.model(**encoded)

            # Mean pooling (use attention mask for proper averaging)
            attention_mask = encoded['attention_mask']
            token_embeddings = outputs.last_hidden_state

            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
            )

            embeddings = torch.sum(
                token_embeddings * input_mask_expanded, dim=1
            ) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

            # L2 normalization (for cosine similarity)
            if normalize:
                embeddings = torch.nn.functional.normalize(
                    embeddings, p=2, dim=1
                )

            # Move to CPU and convert to numpy
            embeddings = embeddings.cpu().numpy()

            return embeddings

    class EmbeddingCache:
        """Redis-based embedding cache"""

        def __init__(self, redis_client: redis.Redis, ttl_hours: int = 24):
            self.redis = redis_client
            self.ttl_seconds = ttl_hours * 3600

        def _get_key(self, text: str) -> str:
            """Generate cache key from text hash"""
            return f"emb:{hashlib.md5(text.encode()).hexdigest()}"

        def get(self, text: str) -> Optional[np.ndarray]:
            """Get cached embedding"""
            key = self._get_key(text)
            cached = self.redis.get(key)

            if cached:
                CACHE_HITS.inc()
                # Deserialize numpy array
                return np.frombuffer(cached, dtype=np.float32)
            else:
                CACHE_MISSES.inc()
                return None

        def set(self, text: str, embedding: np.ndarray):
            """Cache embedding"""
            key = self._get_key(text)
            # Serialize numpy array
            self.redis.setex(
                key,
                self.ttl_seconds,
                embedding.astype(np.float32).tobytes()
            )

        def get_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
            """Batch get for efficiency"""
            keys = [self._get_key(text) for text in texts]
            cached = self.redis.mget(keys)

            results = []
            for c in cached:
                if c:
                    CACHE_HITS.inc()
                    results.append(np.frombuffer(c, dtype=np.float32))
                else:
                    CACHE_MISSES.inc()
                    results.append(None)

            return results

    class EmbeddingService:
        """Production embedding service"""

        def __init__(self, config: EmbeddingConfig):
            self.config = config
            self.model = EmbeddingModel(config)
            self.cache = EmbeddingCache(
                redis.Redis(host='localhost', port=6379, db=0),
                ttl_hours=config.cache_ttl_hours
            )
            self.batcher = DynamicBatcher(config)

        async def embed(
            self,
            texts: List[str],
            normalize: bool = True
        ) -> Tuple[List[np.ndarray], List[bool]]:
            """
            Get embeddings for texts (with caching)

            Returns:
                (embeddings, cached_flags)
            """
            # Check cache first
            cached_embeddings = self.cache.get_batch(texts)

            # Separate cached vs uncached
            uncached_indices = [
                i for i, emb in enumerate(cached_embeddings) if emb is None
            ]
            uncached_texts = [texts[i] for i in uncached_indices]

            # Compute embeddings for uncached texts
            if uncached_texts:
                new_embeddings = self.model.encode(uncached_texts, normalize)

                # Update cache (async)
                for text, emb in zip(uncached_texts, new_embeddings):
                    self.cache.set(text, emb)

                # Merge cached + new embeddings
                result_embeddings = []
                new_idx = 0
                for i, cached in enumerate(cached_embeddings):
                    if cached is not None:
                        result_embeddings.append(cached)
                    else:
                        result_embeddings.append(new_embeddings[new_idx])
                        new_idx += 1
            else:
                result_embeddings = cached_embeddings

            # Cached flags
            cached_flags = [emb is not None for emb in cached_embeddings]

            return result_embeddings, cached_flags

    # FastAPI application
    app = FastAPI(title="Embedding Service")

    # Global service instance
    config = EmbeddingConfig()
    service = EmbeddingService(config)

    @app.post("/embed", response_model=EmbeddingResponse)
    async def embed_endpoint(request: EmbeddingRequest):
        """Embed texts endpoint"""
        REQUESTS.inc()

        start_time = time.time()

        try:
            embeddings, cached = await service.embed(
                request.texts,
                normalize=request.normalize
            )

            latency_ms = (time.time() - start_time) * 1000
            LATENCY.observe(latency_ms / 1000)

            return EmbeddingResponse(
                embeddings=[emb.tolist() for emb in embeddings],
                cached=cached,
                latency_ms=latency_ms
            )

        except Exception as e:
            logging.error(f"Embedding failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "device": str(service.model.device)}

    # Example usage
    if __name__ == "__main__":
        import uvicorn

        # Start service
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            workers=4,  # Multi-process
            log_level="info"
        )
    ```

    ## Optimization Strategies Comparison

    | Strategy | Latency Improvement | Throughput Improvement | Implementation Cost |
    |----------|---------------------|------------------------|---------------------|
    | **Dynamic Batching** | 1.5x (amortize overhead) | **10x** | Low |
    | **Mixed Precision (FP16)** | **2x** | 2x | Very Low |
    | **TensorRT Optimization** | **3x** | 3x | High |
    | **Quantization (INT8)** | 4x | 4x | Medium (1-2% quality loss) |
    | **Model Distillation** | 5x | 5x | Very High (retrain smaller model) |
    | **Caching (70% hit rate)** | **5x** (cached requests) | 3x | Low |
    | **GPU vs CPU** | **10x** | 10x | Medium (infrastructure) |

    **Best Combo for Production:**
    - Dynamic batching + FP16 + Caching → **20-30x improvement** over naive CPU implementation

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **Cold Start** | First request 5-10s slow | Model warmup on startup |
    | **Small Batches** | GPU utilization < 30% | Dynamic batching (wait 10ms) |
    | **OOM Errors** | Large batches crash GPU | Max batch size + gradient checkpointing |
    | **Stale Cache** | Serve old embeddings after model update | Version cache keys with model version |
    | **No Rate Limiting** | Abuse/DDoS | Per-user rate limits (1K/min) |
    | **Blocking I/O** | Slow cache lookups block service | Async Redis client |
    | **No Monitoring** | Silent failures | Prometheus metrics + alerting |
    | **Single GPU** | No redundancy | Multi-GPU with load balancing |

    ## Real-World Examples

    **OpenAI Embedding API:**
    - **Scale:** Billions of requests/month
    - **Model:** text-embedding-ada-002 (1536-dim)
    - **Latency:** p50 < 100ms, p99 < 500ms
    - **Pricing:** $0.0001 per 1K tokens (~750 words)
    - **Optimization:** TensorRT, multi-GPU, aggressive caching
    - **Throughput:** 100K+ embeddings/second per region

    **Google Vertex AI Embeddings:**
    - **Models:** textembedding-gecko (768-dim)
    - **Latency:** p50 < 50ms, p99 < 200ms
    - **Features:** Multi-lingual, batch API (up to 250 texts)
    - **Optimization:** TPU acceleration, dynamic batching
    - **SLA:** 99.9% uptime

    **Cohere Embed:**
    - **Models:** embed-english-v3.0 (1024-dim)
    - **Latency:** p50 < 30ms, p99 < 100ms
    - **Features:** Compression (256-dim), semantic search mode
    - **Optimization:** Custom CUDA kernels, quantization
    - **Throughput:** 10K+ req/s per instance

    **HuggingFace Inference API:**
    - **Scale:** 1M+ models served
    - **Infrastructure:** AWS Inferentia, NVIDIA GPUs
    - **Latency:** p99 < 500ms (shared), < 50ms (dedicated)
    - **Features:** Auto-scaling, cold start optimization
    - **Pricing:** $0.60/hour (dedicated GPU)

    ## Key Metrics to Monitor

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Latency (p99)** | < 50ms (single), < 100ms (batch) | > 100ms |
    | **Throughput** | > 5K req/s per GPU | < 2K req/s |
    | **GPU Utilization** | > 80% | < 50% (under-utilized) |
    | **Cache Hit Rate** | > 70% | < 50% |
    | **Error Rate** | < 0.1% | > 1% |
    | **Model Load Time** | < 10s | > 30s |

    !!! tip "Interviewer's Insight"
        Explains dynamic batching to increase GPU utilization from 30% → 80%+ (10x throughput gain), discusses FP16 mixed precision for 2x speedup with minimal quality loss, and implements Redis caching with 70%+ hit rate for 5x latency improvement on repeat queries. Understands trade-offs between batch size (throughput) vs latency, and can explain how OpenAI/Cohere/Google optimize embedding services at billion-request scale.

---

### Design a Content Moderation System - Meta, YouTube Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Trust & Safety` | **Asked by:** Meta, YouTube, TikTok

??? success "View Answer"

    ## Scale Requirements

    - **Volume:** 500M+ pieces of content/day
    - **Latency:** < 100ms (fast filters), < 1s (ML), < 24h (human review)
    - **Precision:** > 95% (minimize false positives - bad UX)
    - **Recall:** > 90% (catch most violations)
    - **Human Review Capacity:** 10K+ moderators globally
    - **Appeals:** Process 1M+ appeals/day
    - **Languages:** 100+ languages supported

    ## Architecture

    ```
    ┌──────────────────────────────────────────────────────────┐
    │           Content Moderation System (Multi-Layer)         │
    ├──────────────────────────────────────────────────────────┤
    │                                                            │
    │  User-Generated Content (text, image, video)              │
    │         ↓                                                  │
    │  ┌────────────────────────────────────────────┐          │
    │  │     Layer 1: Fast Filters (< 10ms)         │          │
    │  ├────────────────────────────────────────────┤          │
    │  │  • Hash matching (PhotoDNA, PDQ)           │          │
    │  │  • Blocklist (profanity, known bad actors) │          │
    │  │  • Rate limiting (spam detection)          │          │
    │  │  • Metadata checks (file size, format)     │          │
    │  └────────┬───────────────────────────────────┘          │
    │           │ (90% of violations caught here)               │
    │           ↓                                                │
    │  ┌────────────────────────────────────────────┐          │
    │  │   Layer 2: ML Classifiers (< 1s)           │          │
    │  ├────────────────────────────────────────────┤          │
    │  │                                             │          │
    │  │  Text: BERT/RoBERTa                        │          │
    │  │  - Hate speech: toxicity score             │          │
    │  │  - Spam: promotional content               │          │
    │  │  - Misinformation: fact-check needed       │          │
    │  │                                             │          │
    │  │  Image: ResNet/EfficientNet                │          │
    │  │  - NSFW detection (nudity, gore)           │          │
    │  │  - Violence detection                      │          │
    │  │  - Logo/trademark infringement             │          │
    │  │                                             │          │
    │  │  Video: 3D CNN + temporal models           │          │
    │  │  - Frame sampling (1 fps)                  │          │
    │  │  - Audio analysis (ASR + toxicity)         │          │
    │  │  - Scene detection                         │          │
    │  │                                             │          │
    │  │  Multi-modal: CLIP/ALIGN                   │          │
    │  │  - Text-image consistency                  │          │
    │  │  - Context-aware moderation                │          │
    │  └────────┬───────────────────────────────────┘          │
    │           │ (confidence < 0.8 → human review)            │
    │           ↓                                                │
    │  ┌────────────────────────────────────────────┐          │
    │  │   Layer 3: Human Review Queue              │          │
    │  ├────────────────────────────────────────────┤          │
    │  │  • Priority scoring (viral content first)  │          │
    │  │  • Workload balancing (round-robin)        │          │
    │  │  • Moderator specialization (NSFW, hate)   │          │
    │  │  • Quality control (double-review)         │          │
    │  │  • Moderator wellness (rotation, breaks)   │          │
    │  └────────┬───────────────────────────────────┘          │
    │           │                                                │
    │           ↓                                                │
    │  ┌────────────────────────────────────────────┐          │
    │  │      Action Taken                           │          │
    │  ├────────────────────────────────────────────┤          │
    │  │  • Remove: Delete content                  │          │
    │  │  • Restrict: Reduce distribution           │          │
    │  │  • Warn: User notification                 │          │
    │  │  • Ban: Suspend account (temp/permanent)   │          │
    │  │  • Approve: Mark as safe                   │          │
    │  └────────┬───────────────────────────────────┘          │
    │           │                                                │
    │           ↓                                                │
    │  ┌────────────────────────────────────────────┐          │
    │  │   Appeals System                            │          │
    │  ├────────────────────────────────────────────┤          │
    │  │  • User submits appeal                     │          │
    │  │  • Senior moderator review                 │          │
    │  │  • Overturn decision if error              │          │
    │  │  • Feedback loop to ML models              │          │
    │  └────────────────────────────────────────────┘          │
    │                                                            │
    │  ┌────────────────────────────────────────────┐          │
    │  │   Feedback & Model Improvement              │          │
    │  ├────────────────────────────────────────────┤          │
    │  │  • Log all decisions                       │          │
    │  │  • Disagreements → training data           │          │
    │  │  • Retrain models weekly                   │          │
    │  │  • A/B test new models (shadow mode)       │          │
    │  └────────────────────────────────────────────┘          │
    └──────────────────────────────────────────────────────────┘

    Monitoring & Analytics:
    ┌────────────────────────────────────┐
    │  • Violation rate by category      │
    │  • False positive rate (appeals)   │
    │  • Moderator throughput & accuracy │
    │  • Model performance drift         │
    │  • SLA compliance (< 24h review)   │
    └────────────────────────────────────┘
    ```

    ## Production Implementation (270 lines)

    ```python
    # content_moderation.py
    from typing import List, Dict, Optional, Tuple
    from dataclasses import dataclass
    from enum import Enum
    import hashlib
    import numpy as np
    from datetime import datetime
    import logging

    class ViolationType(Enum):
        """Content violation types"""
        HATE_SPEECH = "hate_speech"
        HARASSMENT = "harassment"
        NSFW = "nsfw"
        VIOLENCE = "violence"
        SPAM = "spam"
        MISINFORMATION = "misinformation"
        COPYRIGHT = "copyright"
        SAFE = "safe"

    class ModerationAction(Enum):
        """Actions taken on violating content"""
        REMOVE = "remove"
        RESTRICT = "restrict"  # Reduce distribution
        WARN = "warn"
        BAN_USER = "ban_user"
        APPROVE = "approve"
        NEEDS_REVIEW = "needs_review"

    @dataclass
    class ModerationResult:
        """Result of moderation check"""
        violation_type: ViolationType
        confidence: float
        action: ModerationAction
        explanation: str
        model_version: str
        flagged_by: str  # "hash", "ml", "human"

    class HashMatcher:
        """
        Fast hash-based matching (Layer 1)

        Uses perceptual hashing (PhotoDNA, PDQ) to match
        against known violating content database

        Time: O(1) lookup
        """

        def __init__(self):
            # Database of known violation hashes
            self.violation_hashes: set = set()
            self.load_violation_database()

        def load_violation_database(self):
            """Load known violation hashes (from NCMEC, industry partners)"""
            # In production, load from secure database
            logging.info("Loading violation hash database...")

        def compute_pdq_hash(self, image_bytes: bytes) -> str:
            """
            Compute PDQ (Perceptual Detection Quality) hash

            PDQ is Meta's open-source perceptual hash for images
            - Robust to minor edits (resize, crop, filter)
            - 256-bit hash
            """
            # Simplified - actual PDQ uses DCT + quantization
            return hashlib.md5(image_bytes).hexdigest()

        def check(self, content_hash: str) -> Optional[ModerationResult]:
            """Check if content matches known violations"""
            if content_hash in self.violation_hashes:
                return ModerationResult(
                    violation_type=ViolationType.NSFW,
                    confidence=1.0,
                    action=ModerationAction.REMOVE,
                    explanation="Matches known violating content",
                    model_version="hash_v1",
                    flagged_by="hash"
                )
            return None

    class TextClassifier:
        """ML-based text moderation (Layer 2)"""

        def __init__(self):
            # In production: Load BERT/RoBERTa model
            self.model = None
            self.thresholds = {
                ViolationType.HATE_SPEECH: 0.7,
                ViolationType.HARASSMENT: 0.75,
                ViolationType.SPAM: 0.8,
                ViolationType.MISINFORMATION: 0.6,
            }

        def predict(self, text: str) -> Dict[ViolationType, float]:
            """
            Predict violation probabilities

            Returns:
                {ViolationType: probability}
            """
            # Simplified - actual would use transformer model
            scores = {
                ViolationType.HATE_SPEECH: 0.15,
                ViolationType.HARASSMENT: 0.05,
                ViolationType.SPAM: 0.1,
                ViolationType.MISINFORMATION: 0.02,
            }

            # Check for blocklisted terms
            blocklist = ["badword1", "badword2"]
            if any(term in text.lower() for term in blocklist):
                scores[ViolationType.HATE_SPEECH] = 0.95

            return scores

        def check(self, text: str) -> Optional[ModerationResult]:
            """Check text for violations"""
            scores = self.predict(text)

            # Find highest scoring violation
            max_violation = max(scores.items(), key=lambda x: x[1])
            violation_type, score = max_violation

            if score > self.thresholds.get(violation_type, 0.8):
                return ModerationResult(
                    violation_type=violation_type,
                    confidence=score,
                    action=self._get_action(score),
                    explanation=f"{violation_type.value} detected",
                    model_version="text_bert_v2",
                    flagged_by="ml"
                )

            return None

        def _get_action(self, confidence: float) -> ModerationAction:
            """Determine action based on confidence"""
            if confidence > 0.95:
                return ModerationAction.REMOVE
            elif confidence > 0.8:
                return ModerationAction.RESTRICT
            else:
                return ModerationAction.NEEDS_REVIEW

    class ImageClassifier:
        """ML-based image moderation (Layer 2)"""

        def __init__(self):
            # In production: Load ResNet/EfficientNet
            self.nsfw_model = None
            self.violence_model = None

        def predict_nsfw(self, image_bytes: bytes) -> float:
            """NSFW detection score"""
            # Simplified - actual would use CNN
            return 0.3

        def predict_violence(self, image_bytes: bytes) -> float:
            """Violence detection score"""
            return 0.1

        def check(self, image_bytes: bytes) -> Optional[ModerationResult]:
            """Check image for violations"""
            nsfw_score = self.predict_nsfw(image_bytes)
            violence_score = self.predict_violence(image_bytes)

            if nsfw_score > 0.8:
                return ModerationResult(
                    violation_type=ViolationType.NSFW,
                    confidence=nsfw_score,
                    action=ModerationAction.REMOVE,
                    explanation="NSFW content detected",
                    model_version="image_resnet_v3",
                    flagged_by="ml"
                )

            if violence_score > 0.85:
                return ModerationResult(
                    violation_type=ViolationType.VIOLENCE,
                    confidence=violence_score,
                    action=ModerationAction.REMOVE,
                    explanation="Violent content detected",
                    model_version="image_resnet_v3",
                    flagged_by="ml"
                )

            return None

    class HumanReviewQueue:
        """Human review queue management (Layer 3)"""

        def __init__(self):
            self.queue: List[Tuple[str, ModerationResult, float]] = []
            # content_id, result, priority_score

        def add(
            self,
            content_id: str,
            result: ModerationResult,
            metadata: Dict
        ):
            """Add content to human review queue"""
            # Calculate priority score
            priority = self._calculate_priority(result, metadata)

            self.queue.append((content_id, result, priority))
            self.queue.sort(key=lambda x: x[2], reverse=True)

        def _calculate_priority(
            self,
            result: ModerationResult,
            metadata: Dict
        ) -> float:
            """
            Priority scoring for review queue

            Higher priority:
            - Viral content (high engagement)
            - Low confidence (borderline cases)
            - Sensitive categories (NSFW, violence)
            """
            priority = 0.0

            # Virality score
            views = metadata.get("views", 0)
            priority += min(views / 1000, 100)  # Cap at 100

            # Confidence (lower = higher priority)
            priority += (1 - result.confidence) * 50

            # Category severity
            severity_weights = {
                ViolationType.NSFW: 2.0,
                ViolationType.VIOLENCE: 2.0,
                ViolationType.HATE_SPEECH: 1.5,
                ViolationType.HARASSMENT: 1.3,
                ViolationType.SPAM: 0.5,
            }
            priority *= severity_weights.get(result.violation_type, 1.0)

            return priority

        def get_next(self, moderator_specialization: str) -> Optional[Tuple]:
            """Get next item for moderator"""
            # Filter by specialization
            for i, (content_id, result, priority) in enumerate(self.queue):
                if moderator_specialization == "all" or \
                   result.violation_type.value == moderator_specialization:
                    return self.queue.pop(i)

            return None

    class ModerationPipeline:
        """End-to-end moderation pipeline"""

        def __init__(self):
            self.hash_matcher = HashMatcher()
            self.text_classifier = TextClassifier()
            self.image_classifier = ImageClassifier()
            self.review_queue = HumanReviewQueue()

        def moderate_text(
            self,
            content_id: str,
            text: str,
            metadata: Dict
        ) -> ModerationResult:
            """Moderate text content"""
            # Layer 1: Fast filters (blocklists, rate limits)
            # Skipped for brevity

            # Layer 2: ML classifier
            result = self.text_classifier.check(text)

            if result:
                # Low confidence → human review
                if result.confidence < 0.8:
                    result.action = ModerationAction.NEEDS_REVIEW
                    self.review_queue.add(content_id, result, metadata)

                return result

            # No violation detected
            return ModerationResult(
                violation_type=ViolationType.SAFE,
                confidence=0.95,
                action=ModerationAction.APPROVE,
                explanation="No violations detected",
                model_version="text_bert_v2",
                flagged_by="ml"
            )

        def moderate_image(
            self,
            content_id: str,
            image_bytes: bytes,
            metadata: Dict
        ) -> ModerationResult:
            """Moderate image content"""
            # Layer 1: Hash matching
            image_hash = self.hash_matcher.compute_pdq_hash(image_bytes)
            hash_result = self.hash_matcher.check(image_hash)

            if hash_result:
                return hash_result  # Immediate removal

            # Layer 2: ML classifier
            ml_result = self.image_classifier.check(image_bytes)

            if ml_result:
                # Low confidence → human review
                if ml_result.confidence < 0.85:
                    ml_result.action = ModerationAction.NEEDS_REVIEW
                    self.review_queue.add(content_id, ml_result, metadata)

                return ml_result

            # No violation detected
            return ModerationResult(
                violation_type=ViolationType.SAFE,
                confidence=0.9,
                action=ModerationAction.APPROVE,
                explanation="No violations detected",
                model_version="image_resnet_v3",
                flagged_by="ml"
            )

    # Example usage
    if __name__ == "__main__":
        pipeline = ModerationPipeline()

        # Moderate text
        result = pipeline.moderate_text(
            content_id="post_123",
            text="This is a test post",
            metadata={"views": 1000, "user_id": "user_456"}
        )

        print(f"Violation: {result.violation_type.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Action: {result.action.value}")
        print(f"Explanation: {result.explanation}")

        # Moderate image
        image_data = b"fake_image_bytes"
        result = pipeline.moderate_image(
            content_id="img_789",
            image_bytes=image_data,
            metadata={"views": 5000, "user_id": "user_456"}
        )

        print(f"\nImage moderation:")
        print(f"Violation: {result.violation_type.value}")
        print(f"Action: {result.action.value}")
    ```

    ## Moderation Strategy Comparison

    | Approach | Precision | Recall | Latency | Cost | Best For |
    |----------|-----------|--------|---------|------|----------|
    | **Hash Matching** | **99%** | 30% | **< 10ms** | Very Low | Known violations (CSAM, terrorist content) |
    | **Blocklists** | 95% | 40% | < 1ms | Very Low | Profanity, spam keywords |
    | **ML Classifiers** | 90% | **85%** | < 1s | Medium | New/unknown violations |
    | **Human Review** | **98%** | **95%** | Hours-Days | **Very High** | Edge cases, context-dependent |
    | **Hybrid (All Layers)** | **96%** | **92%** | Varies | Medium | Production systems |

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **High False Positives** | Users frustrated, appeals spike | Lower thresholds, human review for borderline |
    | **Context Ignorance** | Ban satire, educational content | Multi-modal models, context understanding |
    | **Model Bias** | Over-moderate minorities | Diverse training data, fairness metrics |
    | **Moderator Burnout** | High turnover, PTSD | Rotation, wellness programs, AI pre-filtering |
    | **No Feedback Loop** | Models stagnate | Log all decisions, retrain weekly |
    | **Single Model** | Brittle, fails on new attacks | Ensemble models, defense in depth |
    | **Slow Review** | Violations go viral | Priority queue (viral content first) |
    | **No Appeals** | Erode user trust | Transparent appeals process |

    ## Real-World Examples

    **Meta Content Moderation:**
    - **Scale:** 3 billion posts/day across Facebook, Instagram
    - **Team:** 40K+ human moderators globally
    - **Proactive Rate:** 97% of hate speech caught before user reports
    - **Technology:** PhotoDNA (hash), PyTorch models (ML), human review
    - **Latency:** < 1s (automated), < 24h (human review)
    - **Challenges:** 100+ languages, cultural context

    **YouTube Trust & Safety:**
    - **Scale:** 500 hours of video uploaded/minute
    - **Removals:** 6M+ videos/quarter for violations
    - **Automation:** 95% of removed content flagged by ML
    - **Technology:** TensorFlow (video classification), ASR (audio)
    - **Human Review:** 10K+ moderators + community flagging
    - **Appeals:** 50% overturn rate on appeals

    **TikTok Moderation:**
    - **Scale:** 1B+ daily active users
    - **Speed:** Real-time moderation (< 5s before going live)
    - **Technology:** ByteDance's Douyin moderation stack
    - **Multi-modal:** Text, video, audio, music analysis
    - **Human Review:** 24/7 operations, 18-hour shifts
    - **Challenges:** Short-form video (15-60s), trends/memes

    **Reddit Community Moderation:**
    - **Hybrid Approach:** AutoModerator (rules) + volunteer mods
    - **Scale:** 100K+ active communities
    - **Automation:** Keyword filters, karma thresholds, spam detection
    - **Human:** 140K+ volunteer moderators
    - **Transparency:** Public mod logs, appeal to admins

    ## Key Metrics to Monitor

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Precision** | > 95% | < 90% (too many false positives) |
    | **Recall** | > 90% | < 80% (missing violations) |
    | **Proactive Rate** | > 95% (catch before reports) | < 85% |
    | **False Positive Rate** | < 5% | > 10% |
    | **Review SLA** | < 24h | > 48h |
    | **Appeal Overturn Rate** | 10-20% (balanced) | > 30% (models too aggressive) |
    | **Moderator Accuracy** | > 95% | < 90% |

    !!! tip "Interviewer's Insight"
        Designs multi-layer defense (hash matching → ML → human review) to balance precision (95%+), recall (90%+), and latency (< 1s automated, < 24h human). Understands trade-offs between false positives (bad UX) vs false negatives (safety risk), explains how Meta/YouTube handle 3B+ posts/day with 95%+ proactive detection rate, and discusses moderator wellness (rotation, AI pre-filtering) to prevent burnout. Can explain feedback loops (appeals → retraining) for continuous model improvement.

---

### Design a Notification System - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `System Design` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    ## Scale Requirements

    - **Volume:** 100M+ notifications/day
    - **Throughput:** 10K+ events/second (peak)
    - **Latency:** < 1s (real-time), < 5min (batch)
    - **Channels:** Push (mobile/web), Email, SMS, In-app
    - **User Base:** 50M+ active users
    - **Delivery Rate:** > 95% success rate
    - **Opt-out Compliance:** GDPR, CAN-SPAM compliant

    ## Architecture

    ```
    ┌────────────────────────────────────────────────────────────┐
    │              Notification System Architecture               │
    ├────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Trigger Events (user actions, system events)               │
    │         ↓                                                    │
    │  ┌──────────────────────────────────────────┐              │
    │  │   Event Ingestion (Kafka)                │              │
    │  │  Topics:                                  │              │
    │  │  - user_actions (likes, comments, etc.)  │              │
    │  │  - system_events (job complete, etc.)    │              │
    │  │  - marketing_campaigns                   │              │
    │  └────────┬─────────────────────────────────┘              │
    │           │                                                  │
    │           ↓                                                  │
    │  ┌──────────────────────────────────────────┐              │
    │  │  Notification Service (Consumers)         │              │
    │  │                                            │              │
    │  │  ┌────────────────────────────────────┐  │              │
    │  │  │  1. Event Processing                │  │              │
    │  │  │  - Parse event                      │  │              │
    │  │  │  - Extract user IDs                 │  │              │
    │  │  │  - Determine notification type      │  │              │
    │  │  └─────────┬──────────────────────────┘  │              │
    │  │            │                               │              │
    │  │  ┌─────────▼──────────────────────────┐  │              │
    │  │  │  2. User Preference Check           │  │              │
    │  │  │  - Query preferences DB             │  │              │
    │  │  │  - Check opt-out status             │  │              │
    │  │  │  - Get channel preferences          │  │              │
    │  │  │  - Filter muted/blocked users       │  │              │
    │  │  └─────────┬──────────────────────────┘  │              │
    │  │            │                               │              │
    │  │  ┌─────────▼──────────────────────────┐  │              │
    │  │  │  3. Rate Limiting                   │  │              │
    │  │  │  - Per-user limits (10/hour)        │  │              │
    │  │  │  - Per-channel limits               │  │              │
    │  │  │  - Global throttling                │  │              │
    │  │  └─────────┬──────────────────────────┘  │              │
    │  │            │                               │              │
    │  │  ┌─────────▼──────────────────────────┐  │              │
    │  │  │  4. ML Optimization (Optional)      │  │              │
    │  │  │  - Relevance scoring                │  │              │
    │  │  │  - Send time optimization           │  │              │
    │  │  │  - Channel selection                │  │              │
    │  │  └─────────┬──────────────────────────┘  │              │
    │  │            │                               │              │
    │  │  ┌─────────▼──────────────────────────┐  │              │
    │  │  │  5. Notification Rendering          │  │              │
    │  │  │  - Template engine (Jinja2)         │  │              │
    │  │  │  - Personalization                  │  │              │
    │  │  │  - Localization (i18n)              │  │              │
    │  │  └─────────┬──────────────────────────┘  │              │
    │  └────────────┼───────────────────────────┘  │              │
    │               │                                              │
    │        ┌──────┴───────┬────────┬────────┬──────┐          │
    │        ↓              ↓        ↓        ↓      ↓           │
    │  ┌──────────────────────────────────────────────────┐     │
    │  │         Multi-Channel Delivery                    │     │
    │  ├──────────────────────────────────────────────────┤     │
    │  │                                                    │     │
    │  │  ┌──────────────┐  ┌──────────────┐             │     │
    │  │  │  Push (FCM)  │  │  Email (SES) │             │     │
    │  │  │  - iOS/APNS  │  │  - SMTP      │             │     │
    │  │  │  - Android   │  │  - Templates │             │     │
    │  │  │  - Web       │  │  - Tracking  │             │     │
    │  │  └──────┬───────┘  └──────┬───────┘             │     │
    │  │         │                  │                      │     │
    │  │  ┌──────▼──────┐  ┌───────▼──────┐             │     │
    │  │  │  SMS (Twilio)│  │  In-App      │             │     │
    │  │  │  - Shortcode │  │  - WebSocket │             │     │
    │  │  │  - 2FA       │  │  - Badge     │             │     │
    │  │  └─────────────┘  └──────────────┘              │     │
    │  └──────────┬───────────────────────────────────────┘     │
    │             │                                               │
    │             ↓                                               │
    │  ┌──────────────────────────────────────────┐             │
    │  │   Delivery Tracking & Analytics           │             │
    │  │  - Sent, delivered, opened, clicked       │             │
    │  │  - Bounce handling (email/SMS)            │             │
    │  │  - Unsubscribe handling                   │             │
    │  │  - A/B test results                       │             │
    │  └──────────────────────────────────────────┘             │
    │                                                              │
    └────────────────────────────────────────────────────────────┘

    Data Stores:
    ┌──────────────────────────────────┐
    │  User Preferences DB (PostgreSQL)│
    │  - Channel preferences           │
    │  - Quiet hours (9pm-8am)         │
    │  - Opt-out lists                 │
    │  - Frequency caps                │
    └──────────────────────────────────┘

    ┌──────────────────────────────────┐
    │  Notification Log (Cassandra)    │
    │  - Delivery status per user      │
    │  - Deduplication (24h window)    │
    │  - Analytics (open/click rates)  │
    └──────────────────────────────────┘
    ```

    ## Production Implementation (260 lines)

    ```python
    # notification_system.py
    from typing import List, Dict, Optional, Set
    from dataclasses import dataclass
    from enum import Enum
    from datetime import datetime, timedelta
    import hashlib
    import logging

    class NotificationChannel(Enum):
        """Notification delivery channels"""
        PUSH = "push"
        EMAIL = "email"
        SMS = "sms"
        IN_APP = "in_app"

    class NotificationPriority(Enum):
        """Notification priority levels"""
        CRITICAL = "critical"  # Immediate delivery (2FA, security)
        HIGH = "high"  # Real-time (< 1s)
        MEDIUM = "medium"  # Near real-time (< 1min)
        LOW = "low"  # Batch (< 1 hour)

    @dataclass
    class NotificationEvent:
        """Incoming notification event"""
        event_type: str  # "new_comment", "friend_request", etc.
        user_id: str
        data: Dict  # Event-specific data
        priority: NotificationPriority = NotificationPriority.MEDIUM
        timestamp: datetime = None

    @dataclass
    class UserPreferences:
        """User notification preferences"""
        user_id: str
        enabled_channels: Set[NotificationChannel]
        muted_types: Set[str]  # Muted notification types
        quiet_hours_start: int = 22  # 10 PM
        quiet_hours_end: int = 8  # 8 AM
        timezone: str = "UTC"
        max_per_hour: int = 10

    @dataclass
    class Notification:
        """Rendered notification"""
        notification_id: str
        user_id: str
        channel: NotificationChannel
        title: str
        body: str
        data: Dict
        priority: NotificationPriority
        created_at: datetime

    class UserPreferenceStore:
        """User preferences storage"""

        def __init__(self):
            # In production: PostgreSQL/DynamoDB
            self.preferences: Dict[str, UserPreferences] = {}

        def get_preferences(self, user_id: str) -> UserPreferences:
            """Get user preferences (with defaults)"""
            if user_id not in self.preferences:
                # Default preferences
                return UserPreferences(
                    user_id=user_id,
                    enabled_channels={
                        NotificationChannel.PUSH,
                        NotificationChannel.EMAIL,
                        NotificationChannel.IN_APP
                    },
                    muted_types=set(),
                    quiet_hours_start=22,
                    quiet_hours_end=8,
                    timezone="UTC",
                    max_per_hour=10
                )
            return self.preferences[user_id]

        def update_preferences(
            self,
            user_id: str,
            preferences: UserPreferences
        ):
            """Update user preferences"""
            self.preferences[user_id] = preferences

    class RateLimiter:
        """
        Rate limiting for notifications

        Prevents notification fatigue by:
        - Per-user limits (10/hour default)
        - Per-channel limits
        - Global throttling
        """

        def __init__(self):
            # user_id -> [(timestamp, channel), ...]
            self.notification_log: Dict[str, List[tuple]] = {}
            self.window_hours = 1

        def can_send(
            self,
            user_id: str,
            channel: NotificationChannel,
            max_per_hour: int = 10
        ) -> bool:
            """Check if notification can be sent"""
            now = datetime.utcnow()
            cutoff = now - timedelta(hours=self.window_hours)

            # Clean old entries
            if user_id in self.notification_log:
                self.notification_log[user_id] = [
                    (ts, ch) for ts, ch in self.notification_log[user_id]
                    if ts > cutoff
                ]

                # Count notifications in window
                count = len(self.notification_log[user_id])
                if count >= max_per_hour:
                    logging.warning(
                        f"Rate limit exceeded for user {user_id}: "
                        f"{count}/{max_per_hour}"
                    )
                    return False

            return True

        def record_send(
            self,
            user_id: str,
            channel: NotificationChannel
        ):
            """Record sent notification"""
            if user_id not in self.notification_log:
                self.notification_log[user_id] = []

            self.notification_log[user_id].append(
                (datetime.utcnow(), channel)
            )

    class NotificationDeduplicator:
        """
        Deduplication to prevent duplicate notifications

        Uses sliding window (24h) to track sent notifications
        """

        def __init__(self, window_hours: int = 24):
            self.sent_hashes: Dict[str, datetime] = {}
            self.window_hours = window_hours

        def _compute_hash(
            self,
            user_id: str,
            event_type: str,
            data: Dict
        ) -> str:
            """Compute notification hash for deduplication"""
            # Use stable fields only
            key = f"{user_id}:{event_type}:{data.get('entity_id', '')}"
            return hashlib.md5(key.encode()).hexdigest()

        def is_duplicate(
            self,
            user_id: str,
            event_type: str,
            data: Dict
        ) -> bool:
            """Check if notification is duplicate"""
            hash_key = self._compute_hash(user_id, event_type, data)
            now = datetime.utcnow()

            # Clean expired hashes
            expired = [
                k for k, ts in self.sent_hashes.items()
                if ts < now - timedelta(hours=self.window_hours)
            ]
            for k in expired:
                del self.sent_hashes[k]

            # Check duplicate
            if hash_key in self.sent_hashes:
                logging.info(f"Duplicate notification detected: {hash_key}")
                return True

            return False

        def mark_sent(
            self,
            user_id: str,
            event_type: str,
            data: Dict
        ):
            """Mark notification as sent"""
            hash_key = self._compute_hash(user_id, event_type, data)
            self.sent_hashes[hash_key] = datetime.utcnow()

    class SendTimeOptimizer:
        """
        ML-based send time optimization

        Predicts best time to send notification for max engagement
        """

        def __init__(self):
            # In production: Load ML model
            self.model = None

        def get_optimal_send_time(
            self,
            user_id: str,
            notification: Notification
        ) -> datetime:
            """
            Predict optimal send time for user

            Features:
            - Historical open rates by hour
            - User timezone
            - Notification type
            - Day of week

            Returns:
                Optimal send timestamp
            """
            # Simplified - actual would use ML model
            # For now, return immediate for high priority
            if notification.priority in [
                NotificationPriority.CRITICAL,
                NotificationPriority.HIGH
            ]:
                return datetime.utcnow()

            # For low priority, delay to next active hour
            # (e.g., 9 AM in user's timezone)
            return datetime.utcnow() + timedelta(hours=1)

    class NotificationRenderer:
        """Render notifications from templates"""

        def __init__(self):
            # In production: Load Jinja2 templates
            self.templates = {
                "new_comment": {
                    "title": "New comment on your post",
                    "body": "{user_name} commented: {comment_text}"
                },
                "friend_request": {
                    "title": "New friend request",
                    "body": "{user_name} sent you a friend request"
                }
            }

        def render(
            self,
            event: NotificationEvent,
            channel: NotificationChannel
        ) -> Notification:
            """Render notification from event"""
            template = self.templates.get(event.event_type, {})

            # Format with event data
            title = template.get("title", "Notification")
            body = template.get("body", "").format(**event.data)

            return Notification(
                notification_id=f"notif_{event.user_id}_{event.timestamp}",
                user_id=event.user_id,
                channel=channel,
                title=title,
                body=body,
                data=event.data,
                priority=event.priority,
                created_at=event.timestamp or datetime.utcnow()
            )

    class NotificationService:
        """Main notification orchestration service"""

        def __init__(self):
            self.preference_store = UserPreferenceStore()
            self.rate_limiter = RateLimiter()
            self.deduplicator = NotificationDeduplicator()
            self.send_time_optimizer = SendTimeOptimizer()
            self.renderer = NotificationRenderer()

        def process_event(self, event: NotificationEvent):
            """Process incoming notification event"""
            # 1. Get user preferences
            prefs = self.preference_store.get_preferences(event.user_id)

            # 2. Check if event type is muted
            if event.event_type in prefs.muted_types:
                logging.info(f"Event {event.event_type} muted for user {event.user_id}")
                return

            # 3. Check deduplication
            if self.deduplicator.is_duplicate(
                event.user_id, event.event_type, event.data
            ):
                return

            # 4. Determine channels to send
            for channel in prefs.enabled_channels:
                # 5. Check rate limits
                if not self.rate_limiter.can_send(
                    event.user_id, channel, prefs.max_per_hour
                ):
                    logging.warning(f"Rate limit exceeded for {event.user_id}")
                    continue

                # 6. Check quiet hours (for non-critical)
                if event.priority != NotificationPriority.CRITICAL:
                    if self._is_quiet_hours(prefs):
                        logging.info(f"Quiet hours for user {event.user_id}")
                        # Schedule for later
                        continue

                # 7. Render notification
                notification = self.renderer.render(event, channel)

                # 8. Optimize send time (for low priority)
                send_time = self.send_time_optimizer.get_optimal_send_time(
                    event.user_id, notification
                )

                # 9. Send notification
                self._send(notification, send_time)

                # 10. Record send
                self.rate_limiter.record_send(event.user_id, channel)
                self.deduplicator.mark_sent(
                    event.user_id, event.event_type, event.data
                )

        def _is_quiet_hours(self, prefs: UserPreferences) -> bool:
            """Check if current time is in quiet hours"""
            # Simplified - should use user's timezone
            current_hour = datetime.utcnow().hour

            if prefs.quiet_hours_start > prefs.quiet_hours_end:
                # Wraps midnight (e.g., 22:00 - 08:00)
                return (
                    current_hour >= prefs.quiet_hours_start or
                    current_hour < prefs.quiet_hours_end
                )
            else:
                return (
                    prefs.quiet_hours_start <= current_hour < prefs.quiet_hours_end
                )

        def _send(self, notification: Notification, send_time: datetime):
            """Send notification via appropriate channel"""
            if send_time > datetime.utcnow():
                logging.info(f"Scheduling notification for {send_time}")
                # In production: Queue in delayed queue
                return

            logging.info(
                f"Sending {notification.channel.value} notification to "
                f"{notification.user_id}: {notification.title}"
            )

            # In production: Call channel-specific API
            if notification.channel == NotificationChannel.PUSH:
                self._send_push(notification)
            elif notification.channel == NotificationChannel.EMAIL:
                self._send_email(notification)
            elif notification.channel == NotificationChannel.SMS:
                self._send_sms(notification)

        def _send_push(self, notification: Notification):
            """Send push notification (FCM, APNS)"""
            # Call FCM/APNS API
            pass

        def _send_email(self, notification: Notification):
            """Send email (SES, SendGrid)"""
            # Call email provider API
            pass

        def _send_sms(self, notification: Notification):
            """Send SMS (Twilio)"""
            # Call Twilio API
            pass

    # Example usage
    if __name__ == "__main__":
        service = NotificationService()

        # Process incoming event
        event = NotificationEvent(
            event_type="new_comment",
            user_id="user_123",
            data={
                "user_name": "Alice",
                "comment_text": "Great post!",
                "post_id": "post_456"
            },
            priority=NotificationPriority.HIGH
        )

        service.process_event(event)
    ```

    ## Notification Strategy Comparison

    | Channel | Latency | Cost | Open Rate | Best For |
    |---------|---------|------|-----------|----------|
    | **Push** | **< 1s** | Very Low | **40-60%** | Real-time engagement, time-sensitive |
    | **Email** | < 1min | Low | 15-25% | Detailed info, marketing, digests |
    | **SMS** | < 5s | **High** ($0.01/msg) | **90%+** | Critical alerts, 2FA, OTP |
    | **In-App** | Real-time | Very Low | 80%+ (if app open) | Contextual, non-urgent |

    **Best Practices:**
    - **Multi-channel:** Send critical alerts via Push + SMS for redundancy
    - **Fallback:** Email if push token invalid
    - **Personalization:** Use name, timezone, language

    ## Common Pitfalls & Solutions

    | Pitfall | Impact | Solution |
    |---------|--------|----------|
    | **No Rate Limiting** | Notification fatigue, uninstalls | Per-user limits (10/hour) |
    | **Ignoring Preferences** | GDPR violations, user frustration | Respect opt-outs, channel preferences |
    | **No Deduplication** | Duplicate notifications | Hash-based dedup (24h window) |
    | **Wrong Timing** | Low engagement | ML send time optimization |
    | **No Tracking** | Can't measure effectiveness | Track sent/delivered/opened/clicked |
    | **Single Channel** | Miss users if one fails | Multi-channel with fallback |
    | **No Quiet Hours** | Wake users at night | Respect quiet hours (default 10pm-8am) |
    | **Poor Templates** | Low click-through | A/B test copy, personalize |

    ## Real-World Examples

    **Slack Notifications:**
    - **Channels:** Push, email, desktop, mobile
    - **Intelligence:** Smart batching (5 messages → 1 notification)
    - **Preferences:** Per-channel, per-workspace, keyword alerts
    - **Delivery:** < 1s for @mentions, batched for channel messages
    - **Engagement:** 90%+ open rate for @mentions

    **LinkedIn Notifications:**
    - **Volume:** 1B+ notifications/day
    - **ML:** Send time optimization (increase open rate by 30%)
    - **Channels:** Push, email, in-app
    - **Digests:** Weekly summary for low-engagement users
    - **Personalization:** Job alerts, connection suggestions

    **Amazon Notifications:**
    - **Scale:** 100M+ notifications/day
    - **Types:** Order updates, delivery, deals, recommendations
    - **Timing:** Real-time for deliveries, batched for deals
    - **Channels:** Push, email, SMS (critical only)
    - **Optimization:** A/B test send times (2x engagement)

    **Gmail Smart Notifications:**
    - **ML:** Only notify for "important" emails (95% accuracy)
    - **Bundling:** Group emails by thread
    - **Quiet Hours:** Auto-detect sleep schedule
    - **Snooze:** Let users delay notifications

    ## Key Metrics to Monitor

    | Metric | Target | Alert Threshold |
    |--------|--------|-----------------|
    | **Delivery Rate** | > 95% | < 90% |
    | **Open Rate** | Push: 40%+, Email: 20%+ | Drop > 10% |
    | **Click-Through Rate** | > 10% | < 5% |
    | **Unsubscribe Rate** | < 2% | > 5% |
    | **Latency (High Priority)** | < 1s | > 5s |
    | **Bounce Rate** | < 5% | > 10% |
    | **Opt-out Rate** | < 1%/month | > 3%/month |

    !!! tip "Interviewer's Insight"
        Designs multi-channel system (push/email/SMS/in-app) with rate limiting (10/hour per user) to prevent fatigue, deduplication (24h window) to avoid duplicates, and quiet hours respect (10pm-8am). Discusses ML send time optimization to increase engagement 30%+, explains how Slack/LinkedIn handle 1B+ notifications/day with smart batching, and understands trade-offs between channels (push: fast but low open rate vs SMS: expensive but 90%+ open rate).

---

### Design a Cache Invalidation Strategy - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Caching` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Strategies:**
    
    | Strategy | Use Case |
    |----------|----------|
    | TTL | Time-based expiry |
    | Write-through | Consistent, slower writes |
    | Write-behind | Fast writes, eventual consistency |
    | Event-based | Data change triggers |
    
    **ML Context:** Model version changes, feature updates.

    !!! tip "Interviewer's Insight"
        Chooses strategy based on consistency needs.

---

### Design a Feature Flag System - Netflix, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `DevOps` | **Asked by:** Netflix, Meta, Uber

??? success "View Answer"

    **Capabilities:**
    - User targeting (percentage, segments)
    - Kill switches
    - Experiment integration
    - Audit logging
    
    **ML Use Cases:** Model rollouts, shadow testing.

    !!! tip "Interviewer's Insight"
        Integrates with experiment platform.

---

### Design a Rate Limiter - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `System Design` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **Algorithms:**
    - Token bucket
    - Sliding window
    - Fixed window counter
    
    **ML API Context:**
    - Per-user limits
    - Tiered pricing
    - Burst handling

    !!! tip "Interviewer's Insight"
        Uses sliding window for smooth limiting.

---

### Design a Batch Prediction System - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Inference` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Architecture:**
    ```
    [Scheduler] → [Data Fetch] → [Batch Inference] → [Store Results]
    ```
    
    **Considerations:**
    - Parallelization
    - Checkpointing
    - Error handling
    - Result storage (BigQuery, S3)

    !!! tip "Interviewer's Insight"
        Designs for resumability and monitoring.

---

### Design a CI/CD Pipeline for ML - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `MLOps` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **Stages:**
    1. Code/data validation
    2. Unit tests + integration tests
    3. Model training
    4. Evaluation against holdout
    5. Shadow deployment
    6. Canary rollout
    
    **Tools:** GitHub Actions, MLflow, Kubeflow.

    !!! tip "Interviewer's Insight"
        Includes model evaluation in pipeline.

---

### Design a Time Series Forecasting System - Amazon, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Forecasting`, `Time Series` | **Asked by:** Amazon, Google, Uber

??? success "View Answer"

    **Architecture:**
    ```
    [Historical Data] → [Feature Engineering] → [Model] → [Forecast] → [Monitoring]
           ↓
    [Seasonality Detection]
    ```

    **Key Components:**

    | Component | Techniques |
    |-----------|------------|
    | Feature Engineering | Lags, rolling stats, seasonality |
    | Models | ARIMA, Prophet, LSTM, Transformers |
    | Validation | Time-based cross-validation |
    | Monitoring | Forecast accuracy, drift detection |

    **Scale Considerations:**
    - Hierarchical forecasting (product → category → total)
    - Parallel training for multiple series
    - Cold-start handling for new products

    ```python
    from prophet import Prophet

    # Hierarchical forecasting
    def forecast_hierarchy(data):
        # Bottom-up: sum leaf forecasts
        # Top-down: distribute total forecast
        # Middle-out: reconciliation
        return reconciled_forecasts
    ```

    !!! tip "Interviewer's Insight"
        Discusses backtesting strategy and handling seasonality at scale.

---

### Design a Computer Vision Pipeline - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Computer Vision`, `Deep Learning` | **Asked by:** Google, Meta, Tesla

??? success "View Answer"

    **End-to-End Pipeline:**
    ```
    [Image/Video] → [Preprocessing] → [Model Inference] → [Post-processing] → [Results]
                         ↓
                   [Data Augmentation]
    ```

    **Components:**
    1. **Data Ingestion:** Handle images, videos, streams
    2. **Preprocessing:** Resize, normalize, batch
    3. **Model:** ResNet, EfficientNet, ViT
    4. **Post-processing:** NMS, filtering, tracking

    **Optimization:**
    - TensorRT for GPU inference
    - ONNX for portability
    - Quantization (INT8) for edge devices

    **Scale:** Process 1M+ images/day with <100ms latency.

    !!! tip "Interviewer's Insight"
        Discusses model selection based on accuracy vs latency tradeoffs.

---

### Design an NLP Pipeline for Production - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `NLP`, `Transformers` | **Asked by:** Google, Amazon, OpenAI

??? success "View Answer"

    **Architecture:**
    ```
    [Text] → [Tokenization] → [Embedding] → [Model] → [Post-process] → [Output]
                  ↓
            [Text Cleaning]
    ```

    **Key Decisions:**

    | Stage | Options |
    |-------|---------|
    | Tokenization | BPE, WordPiece, SentencePiece |
    | Model | BERT, RoBERTa, GPT, T5 |
    | Serving | ONNX, TorchServe, Triton |
    | Latency | Distillation, quantization |

    **Challenges:**
    - Long context handling (16K+ tokens)
    - Multi-lingual support
    - Domain adaptation

    ```python
    # Model distillation for faster inference
    student_model = distill(teacher_model, alpha=0.5)
    # 10x faster, 95% accuracy retained
    ```

    !!! tip "Interviewer's Insight"
        Knows when to use fine-tuning vs prompt engineering.

---

### Design a Graph Neural Network System - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Graph ML`, `GNN` | **Asked by:** Google, Meta, LinkedIn

??? success "View Answer"

    **Use Cases:**
    - Social network analysis
    - Fraud detection (transaction graphs)
    - Recommendation (user-item graphs)
    - Knowledge graphs

    **Architecture:**
    ```
    [Graph Data] → [Graph Construction] → [GNN] → [Node/Edge Predictions]
                          ↓
                  [Sampling Strategy]
    ```

    **Key Components:**
    - Graph sampling (GraphSAGE, neighbor sampling)
    - Message passing (GCN, GAT, GraphTransformer)
    - Distributed training (DGL, PyG)

    **Scale:** Billion-node graphs with mini-batch training.

    !!! tip "Interviewer's Insight"
        Discusses sampling strategies for large-scale graphs.

---

### Design a Reinforcement Learning System - Google, DeepMind Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `RL`, `Online Learning` | **Asked by:** Google, DeepMind, OpenAI

??? success "View Answer"

    **Components:**
    1. **Environment:** Simulator or real-world
    2. **Agent:** Policy network
    3. **Experience Replay:** Store (s, a, r, s')
    4. **Training:** Off-policy or on-policy

    **Architecture:**
    ```
    [Agent] ↔ [Environment]
       ↓
    [Replay Buffer] → [Training] → [Updated Policy]
    ```

    **Algorithms:**
    - DQN, A3C, PPO, SAC
    - Model-based RL for sample efficiency

    **Challenges:**
    - Exploration vs exploitation
    - Reward shaping
    - Sim-to-real transfer

    !!! tip "Interviewer's Insight"
        Discusses reward engineering and safety constraints.

---

### Design a Model Explainability System - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Interpretability`, `XAI` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Techniques:**

    | Method | Use Case | Complexity |
    |--------|----------|------------|
    | SHAP | Feature importance | Medium |
    | LIME | Local explanations | Low |
    | Attention Viz | Transformers | Low |
    | Counterfactuals | What-if analysis | High |

    **Architecture:**
    ```
    [Prediction] → [Explanation Generator] → [Visualization] → [User]
                          ↓
                  [Explanation Store]
    ```

    **Requirements:**
    - Real-time explanations (<100ms)
    - Human-readable outputs
    - Regulatory compliance (GDPR, FCRA)

    !!! tip "Interviewer's Insight"
        Balances explanation quality with computational cost.

---

### Design a Federated Learning System - Google, Apple Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Privacy`, `Distributed ML` | **Asked by:** Google, Apple, Meta

??? success "View Answer"

    **Privacy-Preserving ML:**
    ```
    [Edge Devices] → [Local Training] → [Encrypted Updates] → [Central Server]
                                              ↓
                                    [Aggregation (FedAvg)]
    ```

    **Key Concepts:**
    1. **Local Training:** Data never leaves device
    2. **Secure Aggregation:** Encrypted model updates
    3. **Differential Privacy:** Add noise to updates
    4. **Communication Efficiency:** Compression, quantization

    **Challenges:**
    - Non-IID data distribution
    - Stragglers (slow devices)
    - Byzantine attacks

    **Tools:** TensorFlow Federated, PySyft.

    !!! tip "Interviewer's Insight"
        Discusses communication efficiency and privacy guarantees.

---

### Design a Multi-Tenant ML Platform - Amazon, Microsoft Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Platform`, `Multi-tenancy` | **Asked by:** Amazon, Microsoft, Google

??? success "View Answer"

    **Requirements:**
    - Isolation (data, compute, models)
    - Resource quotas
    - Cost tracking per tenant
    - Shared infrastructure efficiency

    **Architecture:**
    ```
    [API Gateway] → [Tenant Router] → [Isolated Namespaces]
                          ↓
                    [Shared Resources]
    ```

    **Implementation:**
    - Kubernetes namespaces
    - Resource limits (CPU, GPU, memory)
    - Data encryption at rest/transit
    - Audit logging

    **Scaling:** Support 1000+ tenants efficiently.

    !!! tip "Interviewer's Insight"
        Balances isolation with resource efficiency.

---

### Design a Cost Optimization System for ML - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Cost Optimization`, `FinOps` | **Asked by:** Amazon, Google, Microsoft

??? success "View Answer"

    **Cost Levers:**

    | Component | Optimization |
    |-----------|--------------|
    | Compute | Spot instances, right-sizing |
    | Storage | Data lifecycle, compression |
    | Inference | Batching, autoscaling |
    | Training | Early stopping, efficient architectures |

    **Monitoring:**
    ```
    [Usage Metrics] → [Cost Analysis] → [Recommendations] → [Auto-actions]
    ```

    **Strategies:**
    - Schedule training during off-peak hours
    - Use cheaper storage tiers for old data
    - Implement model caching
    - Optimize batch sizes for GPU utilization

    !!! tip "Interviewer's Insight"
        Provides cost breakdown by experiment/model/team.

---

### Design an AutoML System - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `AutoML`, `Meta-learning` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **Components:**
    1. **Data Preprocessing:** Auto feature engineering
    2. **Model Selection:** Search over architectures
    3. **Hyperparameter Optimization:** Bayesian optimization
    4. **Ensemble:** Combine top models

    **Architecture:**
    ```
    [Dataset] → [AutoML Engine] → [Model Zoo] → [Best Model]
                     ↓
              [Search Space]
    ```

    **Techniques:**
    - Neural Architecture Search (NAS)
    - Meta-learning for warm starts
    - Progressive training (ASHA)

    **Tools:** Google AutoML, H2O.ai, Auto-sklearn.

    !!! tip "Interviewer's Insight"
        Discusses search space design and computational budget.

---

### Design an Active Learning System - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Active Learning`, `Data Efficiency` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Goal:** Minimize labeling cost by selecting most informative samples.

    **Strategies:**

    | Strategy | When to Use |
    |----------|-------------|
    | Uncertainty Sampling | Classification confidence |
    | Query-by-Committee | Ensemble disagreement |
    | Expected Model Change | Impact on model |
    | Diversity Sampling | Cover feature space |

    **Pipeline:**
    ```
    [Model] → [Uncertainty Estimation] → [Sample Selection] → [Labeling] → [Retrain]
    ```

    **Metrics:** Accuracy vs number of labeled samples.

    !!! tip "Interviewer's Insight"
        Combines uncertainty with diversity for better coverage.

---

### Design an Online Learning System - Netflix, Uber Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Online Learning`, `Streaming` | **Asked by:** Netflix, Uber, LinkedIn

??? success "View Answer"

    **Characteristics:**
    - Learn from streaming data
    - Update model incrementally
    - Adapt to changing distributions

    **Architecture:**
    ```
    [Stream] → [Feature Extraction] → [Online Model] → [Prediction]
                     ↓                      ↓
              [Feature Store]        [Model Update]
    ```

    **Algorithms:**
    - Stochastic Gradient Descent (SGD)
    - Online gradient descent
    - Vowpal Wabbit, River

    **Challenges:**
    - Concept drift detection
    - Catastrophic forgetting
    - Model stability

    !!! tip "Interviewer's Insight"
        Discusses when online learning is preferred over batch retraining.

---

### Design a Knowledge Graph for ML - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Knowledge Graphs`, `Graph ML` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **Use Cases:**
    - Enhanced search (semantic understanding)
    - Recommendation (entity relationships)
    - Question answering
    - Feature enrichment for ML

    **Architecture:**
    ```
    [Data Sources] → [Entity Extraction] → [Knowledge Graph] → [Graph Embeddings]
                           ↓
                    [Relation Extraction]
    ```

    **Components:**
    - Entity resolution and linking
    - Relation extraction (distant supervision)
    - Graph storage (Neo4j, Neptune)
    - Embedding (TransE, ComplEx, RotatE)

    **Scale:** Billions of entities and relations.

    !!! tip "Interviewer's Insight"
        Discusses entity disambiguation and knowledge graph completion.

---

### Design an ML System for Edge Devices - Apple, Tesla Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Edge Computing`, `Mobile ML` | **Asked by:** Apple, Tesla, Google

??? success "View Answer"

    **Constraints:**
    - Limited compute (mobile CPU/GPU)
    - Memory constraints (<100MB models)
    - Battery efficiency
    - No/intermittent connectivity

    **Optimization Techniques:**

    | Technique | Benefit | Trade-off |
    |-----------|---------|-----------|
    | Quantization | 4x smaller | Slight accuracy drop |
    | Pruning | Faster inference | More training needed |
    | Knowledge Distillation | Smaller model | Requires teacher |
    | Mobile architectures | Optimized for edge | Different training |

    **Tools:** TensorFlow Lite, Core ML, ONNX Runtime Mobile.

    !!! tip "Interviewer's Insight"
        Balances model size, accuracy, and latency for edge constraints.

---

### Design a Containerization Strategy for ML - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `DevOps`, `Containers` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **Architecture:**
    ```
    [Model Code] → [Dockerfile] → [Container Image] → [Container Registry]
                        ↓
                  [Orchestration (K8s)]
    ```

    **Best Practices:**
    1. **Reproducibility:** Pin all dependencies
    2. **Caching:** Layer Docker images efficiently
    3. **Security:** Scan for vulnerabilities
    4. **Size:** Multi-stage builds to reduce size

    ```dockerfile
    FROM python:3.9-slim
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY model.pkl app.py ./
    CMD ["python", "app.py"]
    ```

    **Tools:** Docker, Kubernetes, Helm.

    !!! tip "Interviewer's Insight"
        Uses multi-stage builds and proper dependency management.

---

### Design a Data Quality Framework - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Data Quality`, `Data Engineering` | **Asked by:** Amazon, Google, Microsoft

??? success "View Answer"

    **Quality Dimensions:**

    | Dimension | Checks |
    |-----------|--------|
    | Completeness | Missing values, null rates |
    | Consistency | Schema validation, referential integrity |
    | Accuracy | Statistical tests, anomaly detection |
    | Timeliness | Data freshness, SLA compliance |

    **Architecture:**
    ```
    [Data Pipeline] → [Quality Checks] → [Alerts] → [Dashboard]
                           ↓
                    [Remediation]
    ```

    **Tools:** Great Expectations, Deequ, Monte Carlo.

    !!! tip "Interviewer's Insight"
        Implements automated data quality checks in pipeline.

---

### Design a Model Compression System - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Model Compression`, `Optimization` | **Asked by:** Google, Meta, Apple

??? success "View Answer"

    **Techniques:**

    | Method | Compression Ratio | Accuracy Impact |
    |--------|-------------------|-----------------|
    | Quantization (INT8) | 4x | <1% drop |
    | Pruning | 2-5x | 1-3% drop |
    | Knowledge Distillation | 10x | 2-5% drop |
    | Low-rank Factorization | 2-3x | <1% drop |

    **Pipeline:**
    ```
    [Trained Model] → [Compression] → [Fine-tuning] → [Validation] → [Deployment]
    ```

    **Workflow:**
    1. Quantization-aware training
    2. Structured pruning
    3. Distillation with teacher-student
    4. Validation on representative data

    !!! tip "Interviewer's Insight"
        Combines multiple compression techniques for maximum efficiency.

---

### Design a Transfer Learning System - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Transfer Learning`, `Fine-tuning` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Strategy:**
    1. **Pretrain:** Large dataset (ImageNet, WebText)
    2. **Fine-tune:** Target domain with smaller dataset
    3. **Adapt:** Layer freezing, learning rate scheduling

    **Architecture:**
    ```
    [Pretrained Model] → [Feature Extractor] → [Task-specific Head] → [Fine-tune]
    ```

    **Best Practices:**
    - Freeze early layers, fine-tune later layers
    - Use lower learning rate for pretrained weights
    - Data augmentation for small datasets
    - Regularization to prevent overfitting

    **Domain Adaptation:** Handle distribution shift between source and target.

    !!! tip "Interviewer's Insight"
        Discusses layer-wise learning rates and progressive unfreezing.

---

### Design a Model Ensembling System - Netflix, Uber Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Ensemble Learning` | **Asked by:** Netflix, Uber, Airbnb

??? success "View Answer"

    **Ensemble Methods:**

    | Method | Approach | Benefit |
    |--------|----------|---------|
    | Bagging | Bootstrap samples | Reduce variance |
    | Boosting | Sequential learning | Reduce bias |
    | Stacking | Meta-model | Best of both |
    | Voting | Majority/average | Simple, effective |

    **Architecture:**
    ```
    [Input] → [Model 1, Model 2, ..., Model N] → [Aggregation] → [Final Prediction]
    ```

    **Considerations:**
    - Model diversity (different architectures, features)
    - Calibration for probability outputs
    - Computational cost vs accuracy gain

    **Netflix example:** Ensembles 100+ models for recommendations.

    !!! tip "Interviewer's Insight"
        Ensures diversity in base models for effective ensembling.

---

### Design a Synthetic Data Generation System - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Data Augmentation`, `Synthetic Data` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Use Cases:**
    - Privacy-preserving ML (replace sensitive data)
    - Rare event augmentation
    - Testing and validation
    - Cold-start problems

    **Techniques:**

    | Method | Use Case |
    |--------|----------|
    | GANs | Image/video generation |
    | VAEs | Controlled generation |
    | SMOTE | Imbalanced classification |
    | Statistical sampling | Tabular data |

    **Pipeline:**
    ```
    [Real Data] → [Generative Model] → [Synthetic Data] → [Quality Checks] → [Mix with Real]
    ```

    **Validation:** Statistical similarity, downstream task performance.

    !!! tip "Interviewer's Insight"
        Validates synthetic data quality with statistical tests and model performance.

---

### Design a Data Augmentation Pipeline - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Data Augmentation`, `Training` | **Asked by:** Google, Meta, Tesla

??? success "View Answer"

    **Image Augmentation:**
    - Geometric: Rotation, flip, crop, resize
    - Color: Brightness, contrast, saturation
    - Advanced: Mixup, CutMix, AutoAugment

    **Text Augmentation:**
    - Synonym replacement
    - Back-translation
    - Paraphrasing with LLMs

    **Architecture:**
    ```
    [Training Data] → [Augmentation Pipeline] → [Augmented Batch] → [Model]
    ```

    **Best Practices:**
    - Apply augmentation on-the-fly during training
    - Use task-specific augmentations
    - Test time augmentation (TTA) for inference

    !!! tip "Interviewer's Insight"
        Discusses domain-specific augmentation strategies and AutoAugment.

---

### Design a Model Testing Framework - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Testing`, `QA` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Testing Levels:**

    | Level | Focus | Examples |
    |-------|-------|----------|
    | Unit | Individual functions | Data preprocessing logic |
    | Integration | Component interactions | Feature pipeline → model |
    | System | End-to-end | Full prediction pipeline |
    | Performance | Model quality | Accuracy, latency, fairness |

    **Architecture:**
    ```
    [Code] → [Unit Tests] → [Integration Tests] → [Model Tests] → [CI/CD]
    ```

    **ML-Specific Tests:**
    - Data validation tests
    - Model performance tests (accuracy, bias)
    - Invariance tests (predictions shouldn't change for certain inputs)
    - Metamorphic testing

    !!! tip "Interviewer's Insight"
        Includes behavioral testing and model-specific test cases.

---

### Design a Shadow Testing System - Netflix, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Testing`, `Deployment` | **Asked by:** Netflix, Amazon, Uber

??? success "View Answer"

    **Concept:** Run new model in parallel with production model without affecting users.

    **Architecture:**
    ```
    [User Request] → [Production Model] → [Response to User]
                          ↓
                    [Shadow Model] → [Logging & Analysis]
    ```

    **Benefits:**
    - Compare model performance in production traffic
    - Detect issues before full rollout
    - A/B test without risk

    **Metrics to Compare:**
    - Prediction differences
    - Latency
    - Error rates
    - Business metrics

    !!! tip "Interviewer's Insight"
        Uses shadow mode before canary deployment for risk mitigation.

---

### Design a Blue-Green Deployment for ML - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Deployment`, `DevOps` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **Strategy:**
    - Blue: Current production model
    - Green: New model version
    - Switch traffic from blue to green after validation
    - Keep blue as rollback option

    **Architecture:**
    ```
    [Load Balancer] → [Blue Environment (v1)]
                  ↘   [Green Environment (v2)]
    ```

    **Deployment Steps:**
    1. Deploy new model to green environment
    2. Run smoke tests on green
    3. Route small % of traffic to green
    4. Monitor metrics
    5. Full cutover if successful
    6. Keep blue for 24h, then decommission

    **Rollback:** Instant by switching load balancer back to blue.

    !!! tip "Interviewer's Insight"
        Combines blue-green with canary for gradual rollout.

---

### Design a Model Governance System - Google, Microsoft Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Governance`, `Compliance` | **Asked by:** Google, Microsoft, Amazon

??? success "View Answer"

    **Governance Requirements:**

    | Aspect | Implementation |
    |--------|----------------|
    | Audit Trail | Track all model changes |
    | Access Control | RBAC for models/data |
    | Compliance | GDPR, CCPA, industry regulations |
    | Risk Assessment | Model risk tiering |

    **Architecture:**
    ```
    [Model Registry] → [Governance Layer] → [Compliance Checks] → [Approval Workflow]
                              ↓
                        [Audit Logs]
    ```

    **Key Features:**
    - Model approval workflows
    - Automated compliance checks
    - Lineage tracking (data → features → model → predictions)
    - Documentation requirements

    !!! tip "Interviewer's Insight"
        Implements automated compliance checks and approval workflows.

---

### Design an Experiment Tracking System - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `MLOps`, `Experiment Management` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **Requirements:**
    - Track hyperparameters, metrics, artifacts
    - Compare experiments
    - Reproducibility
    - Collaboration

    **Architecture:**
    ```
    [Experiment] → [Logging] → [Tracking Server] → [UI Dashboard]
                       ↓
                [Artifact Store]
    ```

    **Track:**
    - Code version (git commit)
    - Data version
    - Hyperparameters
    - Metrics (training + validation)
    - Model artifacts
    - Environment (dependencies)

    **Tools:** MLflow, Weights & Biases, Neptune.

    !!! tip "Interviewer's Insight"
        Ensures reproducibility by tracking all experiment components.

---

### Design a Hyperparameter Optimization Service - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Optimization`, `AutoML` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **Algorithms:**

    | Method | Efficiency | Use Case |
    |--------|-----------|----------|
    | Grid Search | Low | Small spaces |
    | Random Search | Medium | Baseline |
    | Bayesian Optimization | High | Expensive evaluations |
    | Hyperband/ASHA | Very High | Large-scale |

    **Architecture:**
    ```
    [Search Space] → [Optimization Algorithm] → [Trial Scheduler] → [Best Config]
                            ↓
                    [Resource Manager]
    ```

    **Key Features:**
    - Parallel trial execution
    - Early stopping of poor trials
    - Resource allocation optimization
    - Warm start from previous runs

    **Scale:** 1000s of parallel trials.

    !!! tip "Interviewer's Insight"
        Uses multi-fidelity optimization (ASHA) for efficiency.

---

### Design a Feature Selection System - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Feature Engineering`, `Model Optimization` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Methods:**

    | Category | Techniques | When to Use |
    |----------|-----------|-------------|
    | Filter | Correlation, mutual information | Fast, model-agnostic |
    | Wrapper | Forward/backward selection | Accurate, expensive |
    | Embedded | L1 regularization, tree importance | Model-specific |

    **Pipeline:**
    ```
    [All Features] → [Feature Selection] → [Reduced Features] → [Model Training]
                            ↓
                    [Validation Score]
    ```

    **Benefits:**
    - Reduce overfitting
    - Faster training and inference
    - Better interpretability
    - Lower costs

    !!! tip "Interviewer's Insight"
        Combines multiple methods and validates on holdout set.

---

### Design a Data Drift Detection System - Netflix, Uber Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Monitoring`, `Drift Detection` | **Asked by:** Netflix, Uber, Airbnb

??? success "View Answer"

    **Drift Types:**

    | Type | Description | Detection |
    |------|-------------|-----------|
    | Covariate Shift | Input distribution changes | PSI, KS test |
    | Concept Drift | Input-output relationship changes | Model performance drop |
    | Label Drift | Output distribution changes | Label statistics |

    **Architecture:**
    ```
    [Production Data] → [Drift Detector] → [Alert] → [Retrain Trigger]
                              ↓
                    [Reference Distribution]
    ```

    **Metrics:**
    - Population Stability Index (PSI)
    - Kolmogorov-Smirnov test
    - KL divergence

    **Action:** Trigger model retraining when drift detected.

    !!! tip "Interviewer's Insight"
        Sets appropriate thresholds and monitors both data and model drift.

---

### Design a Model Performance Degradation Detection System - Amazon, Google Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Monitoring`, `Performance` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Monitoring:**

    | Metric Type | Examples |
    |-------------|----------|
    | Model Metrics | Accuracy, AUC, precision, recall |
    | Business Metrics | Revenue, conversion, engagement |
    | Operational | Latency, error rate, throughput |

    **Architecture:**
    ```
    [Predictions] → [Ground Truth (delayed)] → [Metric Calculation] → [Alerting]
                                                      ↓
                                              [Historical Baseline]
    ```

    **Challenges:**
    - Delayed ground truth labels
    - Seasonality in metrics
    - Statistical significance testing

    **Proxy Metrics:** Use prediction confidence, data drift as early signals.

    !!! tip "Interviewer's Insight"
        Uses proxy metrics when ground truth is delayed.

---

### Design a Real-Time Analytics Dashboard - Netflix, Uber Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Analytics`, `Visualization` | **Asked by:** Netflix, Uber, Airbnb

??? success "View Answer"

    **Requirements:**
    - Real-time data ingestion
    - Interactive visualizations
    - Drill-down capabilities
    - Alerting

    **Architecture:**
    ```
    [Events] → [Stream Processing] → [Aggregation] → [Time-series DB] → [Dashboard]
                                                            ↓
                                                    [Materialized Views]
    ```

    **Components:**
    - Data ingestion: Kafka, Kinesis
    - Processing: Flink, Spark Streaming
    - Storage: InfluxDB, TimescaleDB
    - Visualization: Grafana, Tableau, Custom UI

    **Optimizations:** Pre-aggregation, caching, sampling for scale.

    !!! tip "Interviewer's Insight"
        Uses materialized views and caching for low-latency queries.

---

### Design an ML Model Marketplace - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Platform`, `Marketplace` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **Features:**
    - Model discovery and search
    - Model versioning and hosting
    - API access with rate limiting
    - Usage tracking and billing
    - Model quality indicators

    **Architecture:**
    ```
    [Model Provider] → [Upload] → [Model Registry] → [API Gateway] → [Consumers]
                                        ↓
                                  [Hosting Service]
    ```

    **Challenges:**
    - Model evaluation and benchmarking
    - Licensing and IP protection
    - Fair pricing models
    - Quality assurance

    **Examples:** Hugging Face Hub, AWS Marketplace, Replicate.

    !!! tip "Interviewer's Insight"
        Includes standardized evaluation benchmarks and clear licensing.

---

### Design a Neural Architecture Search System - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `AutoML`, `NAS` | **Asked by:** Google, Meta, OpenAI

??? success "View Answer"

    **Approaches:**

    | Method | Search Strategy | Efficiency |
    |--------|----------------|------------|
    | Random Search | Random sampling | Baseline |
    | Reinforcement Learning | Controller RNN | Medium |
    | Evolutionary | Genetic algorithms | Medium |
    | Gradient-based | DARTS | High |

    **Architecture:**
    ```
    [Search Space] → [NAS Algorithm] → [Architecture] → [Train & Evaluate]
                          ↑                                    ↓
                          └────────────[Feedback]─────────────┘
    ```

    **Optimizations:**
    - Weight sharing (ENAS)
    - Early stopping
    - Proxy tasks (train on subset)
    - Transfer from related tasks

    !!! tip "Interviewer's Insight"
        Uses efficient methods like DARTS or weight sharing to reduce search cost.

---

### Design a Model Debugging System - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Debugging`, `Interpretability` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Debugging Tools:**

    | Tool | Purpose |
    |------|---------|
    | Error Analysis | Identify failure modes |
    | Slice Analysis | Performance by subgroups |
    | Visualization | Attention maps, embeddings |
    | Counterfactuals | What-if scenarios |

    **Architecture:**
    ```
    [Model] → [Predictions] → [Debug Tools] → [Insights] → [Model Improvements]
                  ↓
            [Error Cases]
    ```

    **Workflow:**
    1. Identify systematic errors
    2. Analyze error patterns
    3. Generate hypotheses
    4. Test fixes (more data, features, architecture)

    !!! tip "Interviewer's Insight"
        Systematically analyzes errors by slice and creates targeted improvements.

---

### Design an ML Observability Platform - Netflix, Uber Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Observability`, `Monitoring` | **Asked by:** Netflix, Uber, Airbnb

??? success "View Answer"

    **Three Pillars:**
    1. **Metrics:** Model performance, system health
    2. **Logs:** Prediction logs, error logs
    3. **Traces:** Request flow through system

    **Architecture:**
    ```
    [ML Services] → [Telemetry] → [Observability Platform] → [Dashboards/Alerts]
                        ↓
                [Time-series DB]
    ```

    **Key Features:**
    - Distributed tracing (OpenTelemetry)
    - Anomaly detection on metrics
    - Log aggregation and search
    - SLI/SLO tracking
    - Root cause analysis

    **Tools:** Prometheus, Grafana, ELK stack, Jaeger.

    !!! tip "Interviewer's Insight"
        Correlates metrics, logs, and traces for effective debugging.

---

### Design a Data Catalog System - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Data Discovery`, `Metadata` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **Capabilities:**
    - Data discovery and search
    - Metadata management
    - Data lineage
    - Schema evolution tracking
    - Access control information

    **Architecture:**
    ```
    [Data Sources] → [Metadata Extraction] → [Catalog] → [Search/Browse UI]
                            ↓
                    [Lineage Tracker]
    ```

    **Metadata:**
    - Technical: Schema, size, location
    - Business: Ownership, description, tags
    - Operational: Freshness, quality scores
    - Lineage: Upstream/downstream dependencies

    **Tools:** DataHub, Amundsen, Apache Atlas.

    !!! tip "Interviewer's Insight"
        Includes automated metadata extraction and lineage tracking.

---

### Design a Metadata Management System - Amazon, Microsoft Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Metadata`, `Governance` | **Asked by:** Amazon, Microsoft, Google

??? success "View Answer"

    **Metadata Types:**

    | Type | Examples |
    |------|----------|
    | Business | Glossary, ownership, definitions |
    | Technical | Schema, types, constraints |
    | Operational | SLAs, quality metrics, usage stats |
    | Lineage | Data flow, transformations |

    **Architecture:**
    ```
    [Systems] → [Metadata Extraction] → [Central Repository] → [APIs/UI]
                        ↓
                [Lineage Graph]
    ```

    **Features:**
    - Automated discovery
    - Impact analysis
    - Search and recommendations
    - Change management

    !!! tip "Interviewer's Insight"
        Automates metadata collection and maintains lineage graph.

---

### Design an ML Platform for Multi-Cloud - Amazon, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Multi-Cloud`, `Platform` | **Asked by:** Amazon, Google, Microsoft

??? success "View Answer"

    **Requirements:**
    - Cloud-agnostic APIs
    - Cost optimization across clouds
    - Data portability
    - Vendor lock-in avoidance

    **Architecture:**
    ```
    [Abstraction Layer] → [Cloud Provider A]
                      → [Cloud Provider B]
                      → [Cloud Provider C]
    ```

    **Components:**
    - Unified ML APIs (training, serving, monitoring)
    - Cross-cloud data transfer
    - Workload placement optimization
    - Centralized monitoring

    **Challenges:**
    - Network latency between clouds
    - Data gravity
    - Different service capabilities

    !!! tip "Interviewer's Insight"
        Uses abstraction layer but allows cloud-specific optimizations.

---

### Design a Disaster Recovery System for ML - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Reliability`, `DR` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **Requirements:**
    - Recovery Time Objective (RTO): < 1 hour
    - Recovery Point Objective (RPO): < 15 minutes
    - Multi-region deployment
    - Automated failover

    **Architecture:**
    ```
    [Primary Region] ←→ [Replication] ←→ [DR Region]
          ↓                                    ↓
    [Data Backup]                        [Data Backup]
    ```

    **Components:**
    - Model replication to DR region
    - Data replication (async/sync)
    - Health checks and failover logic
    - Regular DR testing

    **Scenarios:** Region outage, data corruption, security incident.

    !!! tip "Interviewer's Insight"
        Regularly tests DR procedures and monitors replication lag.

---

### Design a Model Security and Adversarial Robustness System - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Security`, `Adversarial ML` | **Asked by:** Google, Meta, OpenAI

??? success "View Answer"

    **Threats:**

    | Attack Type | Description | Defense |
    |-------------|-------------|---------|
    | Evasion | Adversarial examples | Adversarial training |
    | Poisoning | Corrupt training data | Data validation |
    | Model Stealing | Extract model via queries | Rate limiting, watermarking |
    | Backdoors | Trigger malicious behavior | Input sanitization |

    **Architecture:**
    ```
    [Input] → [Validation] → [Adversarial Detection] → [Model] → [Output Sanitization]
    ```

    **Defenses:**
    - Adversarial training (PGD, FGSM)
    - Input sanitization and validation
    - Model watermarking
    - Anomaly detection on queries

    !!! tip "Interviewer's Insight"
        Combines multiple defense layers and monitors for attacks.

---

### Design an ML Compliance and Audit System - Microsoft, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Compliance`, `Audit` | **Asked by:** Microsoft, Amazon, Google

??? success "View Answer"

    **Regulatory Requirements:**
    - GDPR: Right to explanation, data deletion
    - CCPA: Data access and deletion
    - Industry-specific: HIPAA, SOC 2, PCI-DSS

    **Architecture:**
    ```
    [ML System] → [Audit Logger] → [Audit Trail] → [Compliance Dashboard]
                        ↓
                [Policy Engine]
    ```

    **Audit Trail:**
    - All data access events
    - Model training and deployment
    - Predictions and explanations
    - Data deletions

    **Features:**
    - Immutable audit logs
    - Retention policies
    - Compliance reporting
    - Automated alerts for violations

    !!! tip "Interviewer's Insight"
        Implements privacy-by-design and maintains comprehensive audit trails.

---

### Design a Real-Time Feature Computation System - Netflix, Uber Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Real-Time`, `Feature Engineering` | **Asked by:** Netflix, Uber, LinkedIn

??? success "View Answer"

    **Requirements:**
    - <10ms feature computation
    - Handle 100K+ QPS
    - Consistent with training features

    **Architecture:**
    ```
    [Events] → [Stream Processing] → [Feature Store] → [Model Serving]
                     ↓
              [Windowed Aggregations]
    ```

    **Features:**
    - Real-time aggregations (last 5 min, 1 hour, 1 day)
    - User/item embeddings
    - Context features

    **Challenges:**
    - Training/serving skew
    - Low-latency requirements
    - State management for aggregations

    **Tools:** Flink, ksqlDB, Materialize.

    !!! tip "Interviewer's Insight"
        Ensures feature consistency between training and serving.

---

### Design a Streaming Feature Engineering System - Uber, Netflix Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Streaming`, `Feature Engineering` | **Asked by:** Uber, Netflix, LinkedIn

??? success "View Answer"

    **Architecture:**
    ```
    [Event Stream] → [Stateful Processing] → [Feature Store] → [Online Serving]
                           ↓
                  [Tumbling/Sliding Windows]
    ```

    **Features to Compute:**
    - Count/sum over time windows
    - Average, percentiles
    - Distinct counts (HyperLogLog)
    - Session-based features

    **Challenges:**
    - Late-arriving data (watermarks)
    - State management at scale
    - Exactly-once semantics
    - Feature freshness vs latency

    **Example:**
    ```sql
    -- ksqlDB example
    CREATE TABLE user_clicks_5min AS
    SELECT user_id, COUNT(*) as click_count
    FROM clicks_stream
    WINDOW TUMBLING (SIZE 5 MINUTES)
    GROUP BY user_id;
    ```

    !!! tip "Interviewer's Insight"
        Uses watermarks for late data and manages state efficiently.

---

### Design a Model Lifecycle Management System - Amazon, Microsoft Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `MLOps`, `Lifecycle` | **Asked by:** Amazon, Microsoft, Google

??? success "View Answer"

    **Lifecycle Stages:**

    | Stage | Activities |
    |-------|-----------|
    | Development | Experimentation, prototyping |
    | Staging | Validation, integration testing |
    | Production | Serving, monitoring |
    | Retired | Archival, decommissioning |

    **Architecture:**
    ```
    [Development] → [Staging] → [Production] → [Monitoring] → [Retrain/Retire]
                        ↓
                [Model Registry]
    ```

    **Key Features:**
    - Stage promotion workflows
    - Approval gates
    - Automated testing between stages
    - Rollback capabilities
    - Sunset policies for old models

    **Tools:** MLflow, Kubeflow, SageMaker.

    !!! tip "Interviewer's Insight"
        Implements automated testing and approval workflows between stages.

---

### Design a Chatbot/Conversational AI System - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `NLP`, `Dialogue Systems` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Components:**
    1. **Intent Classification:** Identify user intent
    2. **Entity Extraction:** Extract key information
    3. **Dialogue Management:** Track conversation state
    4. **Response Generation:** Generate or retrieve response
    5. **Context Management:** Multi-turn understanding

    **Architecture:**
    ```
    [User Input] → [NLU] → [Dialogue Manager] → [Response Gen] → [User]
                      ↓
                [Context Store]
    ```

    **Techniques:**
    - Transformer-based models (BERT, GPT)
    - Retrieval-augmented generation (RAG)
    - Reinforcement learning for policy
    - Personalization layer

    **Scale:** Handle 1M+ conversations/day with <500ms latency.

    !!! tip "Interviewer's Insight"
        Discusses context management and handling multi-turn conversations.

---

### Design a Document Processing System - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `OCR`, `Document AI` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **Pipeline:**
    ```
    [Document] → [OCR] → [Layout Analysis] → [Entity Extraction] → [Structured Output]
                   ↓
            [Document Classification]
    ```

    **Components:**
    - **OCR:** Tesseract, Cloud Vision API
    - **Layout:** Detect tables, forms, sections
    - **NER:** Extract names, dates, amounts
    - **Classification:** Invoice, receipt, contract

    **Challenges:**
    - Multiple languages
    - Poor quality scans
    - Complex layouts
    - Privacy (PII redaction)

    **Tools:** AWS Textract, Google Document AI, Azure Form Recognizer.

    !!! tip "Interviewer's Insight"
        Handles multi-modal inputs (text, tables, images) and ensures PII compliance.

---

### Design a Video Understanding System - Google, YouTube Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Computer Vision`, `Video` | **Asked by:** Google, YouTube, Meta

??? success "View Answer"

    **Tasks:**
    - Video classification
    - Action recognition
    - Object tracking
    - Scene understanding
    - Content moderation

    **Architecture:**
    ```
    [Video] → [Frame Sampling] → [Feature Extraction] → [Temporal Model] → [Output]
                    ↓
              [Optical Flow]
    ```

    **Models:**
    - 3D CNNs (C3D, I3D)
    - Two-stream networks
    - Transformers (TimeSformer, ViViT)

    **Optimization:**
    - Keyframe extraction to reduce compute
    - Efficient architectures (MobileNet-based)
    - Distributed processing

    !!! tip "Interviewer's Insight"
        Discusses temporal modeling and efficient video processing at scale.

---

### Design an Audio/Speech Processing System - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Speech Recognition`, `Audio` | **Asked by:** Google, Amazon, Apple

??? success "View Answer"

    **Use Cases:**
    - Speech-to-text (ASR)
    - Speaker identification
    - Emotion recognition
    - Audio classification

    **Architecture:**
    ```
    [Audio] → [Preprocessing] → [Feature Extraction] → [Model] → [Post-process] → [Text]
                  ↓
            [Mel Spectrogram]
    ```

    **Models:**
    - RNN/LSTM, Transformers (Wav2Vec, Whisper)
    - CTC loss for sequence alignment
    - Language models for correction

    **Challenges:**
    - Noisy environments
    - Accents and dialects
    - Real-time processing
    - Speaker diarization

    !!! tip "Interviewer's Insight"
        Discusses handling accents, noise, and real-time constraints.

---

### Design a Multimodal Fusion System - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Multimodal`, `Fusion` | **Asked by:** Google, Meta, OpenAI

??? success "View Answer"

    **Use Cases:**
    - Visual question answering
    - Image captioning
    - Video + text understanding
    - Audio-visual learning

    **Fusion Strategies:**

    | Strategy | When | Complexity |
    |----------|------|------------|
    | Early Fusion | Concat inputs | Low |
    | Late Fusion | Concat outputs | Low |
    | Cross-attention | Learn interactions | High |

    **Architecture:**
    ```
    [Image] → [Vision Encoder] ↘
                                 [Fusion Layer] → [Output]
    [Text] → [Text Encoder]    ↗
    ```

    **Models:** CLIP, ALIGN, Flamingo, GPT-4V.

    !!! tip "Interviewer's Insight"
        Discusses cross-modal attention and alignment between modalities.

---

### Design a Few-Shot Learning System - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Few-Shot`, `Meta-Learning` | **Asked by:** Google, Meta, DeepMind

??? success "View Answer"

    **Goal:** Learn from few labeled examples (1-10 per class).

    **Approaches:**

    | Method | Strategy |
    |--------|----------|
    | Meta-learning | MAML, Prototypical Networks |
    | Transfer Learning | Fine-tune pretrained models |
    | Data Augmentation | Synthesize more examples |
    | Prompt Engineering | For LLMs |

    **Architecture:**
    ```
    [Support Set] → [Meta-Learner] → [Adapted Model] → [Query Prediction]
    ```

    **Applications:** New product categories, rare diseases, personalization.

    !!! tip "Interviewer's Insight"
        Discusses when few-shot learning is preferred over traditional supervised learning.

---

### Design a Zero-Shot Learning System - Google, OpenAI Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Zero-Shot`, `Generalization` | **Asked by:** Google, OpenAI, Meta

??? success "View Answer"

    **Goal:** Classify unseen classes without training examples.

    **Approaches:**
    1. **Semantic Embeddings:** Map classes to embedding space
    2. **Attribute-based:** Describe classes by attributes
    3. **Prompt-based:** Use LLMs with natural language descriptions

    **Architecture:**
    ```
    [Input] → [Encoder] → [Embedding Space] → [Similarity] → [Class]
                              ↓
                      [Class Descriptions]
    ```

    **Example:** CLIP for zero-shot image classification.

    **Challenges:** Requires good semantic representations.

    !!! tip "Interviewer's Insight"
        Discusses using semantic embeddings and language models for zero-shot tasks.

---

### Design a Continual Learning System - Google, DeepMind Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Continual Learning`, `Lifelong Learning` | **Asked by:** Google, DeepMind, Meta

??? success "View Answer"

    **Goal:** Learn new tasks without forgetting old ones (avoid catastrophic forgetting).

    **Strategies:**

    | Approach | Method |
    |----------|--------|
    | Regularization | EWC (Elastic Weight Consolidation) |
    | Replay | Store examples from old tasks |
    | Dynamic Architectures | Add capacity for new tasks |
    | Meta-learning | Learn to learn continually |

    **Architecture:**
    ```
    [Task 1] → [Model] → [Task 2] → [Updated Model] → [Task 3]
                  ↓
            [Memory Buffer]
    ```

    **Evaluation:** Average accuracy across all tasks over time.

    !!! tip "Interviewer's Insight"
        Discusses strategies to prevent catastrophic forgetting.

---

### Design a Model Fairness and Bias Detection System - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Fairness`, `Bias` | **Asked by:** Google, Meta, Microsoft

??? success "View Answer"

    **Fairness Metrics:**

    | Metric | Definition |
    |--------|------------|
    | Demographic Parity | Equal positive rate across groups |
    | Equal Opportunity | Equal TPR across groups |
    | Equalized Odds | Equal TPR and FPR across groups |
    | Calibration | Predicted probabilities match actual rates |

    **Architecture:**
    ```
    [Model] → [Predictions] → [Bias Detection] → [Mitigation] → [Fair Model]
                                    ↓
                            [Protected Attributes]
    ```

    **Mitigation:**
    - Pre-processing: Balance training data
    - In-processing: Fairness constraints during training
    - Post-processing: Adjust thresholds per group

    **Tools:** Fairlearn, AI Fairness 360.

    !!! tip "Interviewer's Insight"
        Discusses trade-offs between different fairness metrics.

---

### Design a Model Watermarking System - Google, Meta Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Security`, `IP Protection` | **Asked by:** Google, Meta, OpenAI

??? success "View Answer"

    **Goal:** Embed verifiable signature in model to prove ownership.

    **Techniques:**
    1. **Backdoor Watermarking:** Train model to output specific pattern for trigger inputs
    2. **Parameter Watermarking:** Encode signature in model weights
    3. **Output-based:** Statistical properties of outputs

    **Architecture:**
    ```
    [Model Training] → [Watermark Embedding] → [Watermarked Model]
                              ↓
                    [Verification Trigger Set]
    ```

    **Requirements:**
    - Undetectable (doesn't degrade performance)
    - Robust (survives fine-tuning, pruning)
    - Verifiable (can prove ownership)

    !!! tip "Interviewer's Insight"
        Discusses robustness to model extraction and fine-tuning attacks.

---

### Design a Cross-Lingual ML System - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Multilingual`, `NLP` | **Asked by:** Google, Meta, Amazon

??? success "View Answer"

    **Challenges:**
    - Limited labeled data for low-resource languages
    - Different scripts and tokenization
    - Cultural context differences

    **Approaches:**

    | Method | Strategy |
    |--------|----------|
    | Multilingual Models | Train on many languages jointly (mBERT, XLM-R) |
    | Cross-lingual Transfer | Train on high-resource, transfer to low-resource |
    | Machine Translation | Translate to English, process, translate back |
    | Zero-shot | Use multilingual embeddings |

    **Architecture:**
    ```
    [Text (any language)] → [Multilingual Encoder] → [Task Head] → [Output]
    ```

    **Best Practices:**
    - Use language-agnostic tokenization (SentencePiece)
    - Balance training data across languages
    - Evaluate on diverse language families

    !!! tip "Interviewer's Insight"
        Discusses handling low-resource languages and script variations.

---

## Quick Reference: 30 System Design Questions

| Sno | Question Title                                                                                      | Practice Links                                                                                                                                   | Companies Asking                          | Difficulty | Topics                                    |
|-----|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|------------|-------------------------------------------|
| 1   | Design an End-to-End Machine Learning Pipeline                                                      | [Towards Data Science](https://towardsdatascience.com/designing-end-to-end-machine-learning-pipelines-3d2a5eabc123)                                | Google, Amazon, Facebook                  | Medium     | ML Pipeline, MLOps                        |
| 2   | Design a Scalable Data Ingestion & Processing System for ML                                          | [Medium](https://medium.com/@example/scalable-data-ingestion-for-ml-abc123)                                                                      | Amazon, Google, Microsoft                 | Hard       | Data Engineering, Scalability             |
| 3   | Design a Recommendation System                                                                      | [Towards Data Science](https://towardsdatascience.com/building-recommendation-systems-456def)                                                    | Google, Amazon, Facebook                  | Medium     | Recommender Systems, Personalization      |
| 4   | Design a Fraud Detection System                                                                     | [Medium](https://medium.com/@example/fraud-detection-system-design-789ghi)                                                                      | Amazon, Facebook, PayPal                  | Hard       | Real-Time Analytics, Anomaly Detection     |
| 5   | Design a Feature Store for Machine Learning                                                         | [Towards Data Science](https://towardsdatascience.com/feature-stores-in-machine-learning-123jkl)                                                  | Google, Amazon, Microsoft                 | Medium     | Data Preprocessing, Feature Engineering    |
| 6   | Design an Online ML Model Serving Architecture                                                      | [Towards Data Science](https://towardsdatascience.com/deploying-machine-learning-models-987mno)                                                   | Google, Amazon, Facebook                  | Hard       | Model Deployment, Real-Time Serving        |
| 7   | Design a Continuous Model Retraining and Monitoring System                                          | [Medium](https://medium.com/@example/continuous-training-and-monitoring-for-ml-456stu)                                                           | Google, Microsoft, Amazon                 | Hard       | MLOps, Automation                         |
| 8   | Design an A/B Testing Framework for ML Models                                                       | [Towards Data Science](https://towardsdatascience.com/ab-testing-for-machine-learning-789pqr)                                                     | Google, Facebook, Amazon                  | Medium     | Experimentation, Evaluation                |
| 9   | Design a Distributed ML Training System                                                             | [Towards Data Science](https://towardsdatascience.com/distributed-training-for-deep-learning-234vwx)                                              | Google, Amazon, Microsoft                 | Hard       | Distributed Systems, Deep Learning         |
| 10  | Design a Real-Time Prediction Serving System                                                        | [Towards Data Science](https://towardsdatascience.com/real-time-ml-model-serving-123abc)                                                         | Amazon, Google, Facebook                  | Hard       | Model Serving, Real-Time Processing        |
| 11  | Design a System for Anomaly Detection in Streaming Data                                             | [Medium](https://medium.com/@example/anomaly-detection-in-streaming-data-567def)                                                                 | Amazon, Google, Facebook                  | Hard       | Streaming Data, Anomaly Detection          |
| 12  | Design a Real-Time Personalization System for E-Commerce                                            | [Medium](https://medium.com/@example/designing-real-time-personalization-890ghi)                                                                 | Amazon, Facebook, Uber                    | Medium     | Personalization, Real-Time Analytics       |
| 13  | Design a Data Versioning and Model Versioning System                                                | [Towards Data Science](https://towardsdatascience.com/data-and-model-versioning-456jkl)                                                           | Google, Amazon, Microsoft                 | Medium     | MLOps, Version Control                     |
| 14  | Design a System to Ensure Fairness and Transparency in ML Predictions                               | [Medium](https://medium.com/@example/fairness-transparency-in-ml-system-design-123stu)                                                           | Google, Facebook, Amazon                  | Hard       | Ethics, Model Interpretability             |
| 15  | Design a Data Governance and Compliance System for ML                                               | [Towards Data Science](https://towardsdatascience.com/data-governance-for-machine-learning-789mno)                                                 | Microsoft, Google, Amazon                 | Hard       | Data Governance, Compliance                |
| 16  | Design an MLOps Pipeline for End-to-End Automation                                                  | [Towards Data Science](https://towardsdatascience.com/mlops-pipelines-design-234vwx)                                                             | Google, Amazon, Facebook                  | Hard       | MLOps, Automation                          |
| 17  | Design a System for Real-Time Prediction Serving with Low Latency                                   | [Medium](https://medium.com/@example/real-time-low-latency-model-serving-567def)                                                                 | Google, Amazon, Microsoft                 | Hard       | Model Serving, Scalability                 |
| 18  | Design a Scalable Data Warehouse for ML-Driven Analytics                                            | [Towards Data Science](https://towardsdatascience.com/designing-data-warehouses-for-ml-345abc)                                                    | Google, Amazon, Facebook                  | Medium     | Data Warehousing, Analytics                |
| 19  | Design a System for Hyperparameter Tuning at Scale                                                  | [Medium](https://medium.com/@example/hyperparameter-tuning-system-design-789ghi)                                                                 | Google, Amazon, Microsoft                 | Hard       | Optimization, Automation                   |
| 20  | Design an Event-Driven Architecture for ML Pipelines                                                | [Towards Data Science](https://towardsdatascience.com/event-driven-architecture-for-ml-567jkl)                                                     | Amazon, Google, Facebook                  | Medium     | Event-Driven, Real-Time Processing         |
| 21  | Design a System for Multimodal Data Processing in Machine Learning                                  | [Towards Data Science](https://towardsdatascience.com/multimodal-data-processing-for-ml-123stu)                                                    | Google, Amazon, Facebook                  | Hard       | Data Integration, Deep Learning            |
| 22  | Design a System to Handle High-Volume Streaming Data for ML                                          | [Towards Data Science](https://towardsdatascience.com/high-volume-streaming-data-for-ml-456vwx)                                                     | Amazon, Google, Microsoft                 | Hard       | Streaming, Scalability                     |
| 23  | Design a Secure and Scalable ML Infrastructure                                                      | [Towards Data Science](https://towardsdatascience.com/secure-scalable-ml-infrastructure-789pqr)                                                     | Google, Amazon, Facebook                  | Hard       | Security, Scalability                      |
| 24  | Design a Scalable Feature Engineering Pipeline                                                      | [Towards Data Science](https://towardsdatascience.com/scalable-feature-engineering-345abc)                                                        | Google, Amazon, Microsoft                 | Medium     | Feature Engineering, Scalability           |
| 25  | Design a System for Experimentation and A/B Testing in Data Science                                 | [Towards Data Science](https://towardsdatascience.com/ab-testing-for-data-science-567jkl)                                                          | Google, Amazon, Facebook                  | Medium     | Experimentation, Analytics                 |
| 26  | Design an Architecture for a Data Lake Tailored for ML Applications                                  | [Towards Data Science](https://towardsdatascience.com/data-lakes-for-ml-123abc)                                                                   | Amazon, Google, Microsoft                 | Medium     | Data Lakes, Data Engineering               |
| 27  | Design a Fault-Tolerant Machine Learning System                                                     | [Medium](https://medium.com/@example/fault-tolerant-ml-system-design-890ghi)                                                                     | Google, Amazon, Facebook                  | Hard       | Reliability, Distributed Systems           |
| 28  | Design a System for Scalable Deep Learning Inference                                                | [Towards Data Science](https://towardsdatascience.com/scalable-deep-learning-inference-234vwx)                                                    | Google, Amazon, Microsoft                 | Hard       | Deep Learning, Inference                   |
| 29  | Design a Collaborative Platform for Data Science Projects                                           | [Towards Data Science](https://towardsdatascience.com/collaborative-platforms-for-data-science-456def)                                             | Google, Amazon, Facebook                  | Medium     | Collaboration, Platform Design             |
| 30  | Design a System for Model Monitoring and Logging                                                  | [Towards Data Science](https://towardsdatascience.com/model-monitoring-for-machine-learning-789mno)                                                | Google, Amazon, Microsoft                 | Medium     | MLOps, Monitoring                          |

---

## Questions asked in Google interview
- Design an End-to-End Machine Learning Pipeline  
- Design a Real-Time Prediction Serving System  
- Design a Continuous Model Retraining and Monitoring System  
- Design a System for Hyperparameter Tuning at Scale  
- Design a Secure and Scalable ML Infrastructure  

## Questions asked in Amazon interview
- Design a Scalable Data Ingestion & Processing System for ML  
- Design a Recommendation System  
- Design a Fraud Detection System  
- Design an MLOps Pipeline for End-to-End Automation  
- Design a System to Handle High-Volume Streaming Data for ML  

## Questions asked in Facebook interview
- Design an End-to-End Machine Learning Pipeline  
- Design an Online ML Model Serving Architecture  
- Design a Real-Time Personalization System for E-Commerce  
- Design a System for Model Monitoring and Logging  
- Design a System for Multimodal Data Processing in ML  

## Questions asked in Microsoft interview
- Design a Data Versioning and Model Versioning System  
- Design a Scalable Data Warehouse for ML-Driven Analytics  
- Design a Distributed ML Training System  
- Design a System for Real-Time Prediction Serving with Low Latency  
- Design a System for Secure and Scalable ML Infrastructure  

---

