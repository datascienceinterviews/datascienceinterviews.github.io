---
title: System Design Interview Questions (DS & ML)
description: A curated list of system design questions for Data Science and Machine Learning interviews
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

    **What to Monitor:**
    
    | Type | Metrics |
    |------|---------|
    | Data Quality | Missing values, schema drift |
    | Data Drift | PSI, KL divergence |
    | Model Performance | Accuracy, latency, throughput |
    | Business Metrics | Revenue impact, user engagement |
    
    **Alert Thresholds:**
    - PSI > 0.2: Significant drift
    - Latency p99 > SLA: Performance issue
    - Accuracy drop > 5%: Model degradation

    !!! tip "Interviewer's Insight"
        Monitors both technical and business metrics.

---

### Design a Distributed Training System - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Deep Learning`, `Scale` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Strategies:**
    
    | Strategy | Use Case |
    |----------|----------|
    | Data Parallel | Same model, different data |
    | Model Parallel | Large models (split layers) |
    | Pipeline Parallel | Very large models |
    
    ```python
    # PyTorch DistributedDataParallel
    model = DDP(model, device_ids=[local_rank])
    
    # Gradient synchronization
    # All-reduce across workers
    ```
    
    **Optimizations:** Gradient compression, async SGD, ZeRO.

    !!! tip "Interviewer's Insight"
        Knows when to use each parallelism strategy.

---

### Design an A/B Testing Platform - Netflix, Airbnb Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Experimentation` | **Asked by:** Netflix, Airbnb, Uber

??? success "View Answer"

    **Components:**
    
    1. **Assignment Service:** Consistent hashing
    2. **Event Logging:** Kafka → DataWarehouse
    3. **Stats Engine:** Automated analysis
    4. **Dashboard:** Results, SRM checks
    
    **Scale:** Netflix runs 100s of concurrent experiments.
    
    **Key Features:**
    - Experiment isolation
    - Automatic SRM detection
    - Variance reduction (CUPED)

    !!! tip "Interviewer's Insight"
        Handles interaction effects between experiments.

---

### Design a Data Pipeline for ML - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Data Engineering` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Architecture:**
    
    ```
    [Sources] → [Ingestion] → [Processing] → [Feature Store] → [Training]
        ↓
    [Data Lake] → [Quality Checks] → [Versioning]
    ```
    
    **Tools:**
    - Orchestration: Airflow, Prefect
    - Processing: Spark, Dask
    - Storage: S3, BigQuery
    - Versioning: DVC, Delta Lake

    !!! tip "Interviewer's Insight"
        Includes data quality checks and lineage tracking.

---

### Design a Model Registry - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `MLOps` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **Capabilities:**
    
    | Feature | Purpose |
    |---------|---------|
    | Model Versioning | Track all versions |
    | Metadata | Metrics, hyperparameters |
    | Stage Management | Dev → Staging → Prod |
    | Lineage | Data and code provenance |
    
    **Tools:** MLflow, Weights & Biases, SageMaker.

    !!! tip "Interviewer's Insight"
        Uses model registry for reproducibility.

---

### Design a Low-Latency Inference Service - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Performance` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Optimization Strategies:**
    
    1. **Model:** Quantization, distillation, pruning
    2. **Serving:** Batching, caching
    3. **Infrastructure:** GPU, Triton, TensorRT
    
    **Latency Budget:**
    ```
    Total: 50ms
    - Network: 5ms
    - Feature Lookup: 10ms
    - Inference: 30ms
    - Post-processing: 5ms
    ```

    !!! tip "Interviewer's Insight"
        Breaks down latency budget by component.

---

### Design a Search System - Google, Amazon Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Search`, `Information Retrieval` | **Asked by:** Google, Amazon, LinkedIn

??? success "View Answer"

    **Architecture:**
    ```
    [Query] → [Query Understanding] → [Retrieval] → [Ranking] → [Results]
                    ↓                       ↓
            [Spell Check]          [Inverted Index]
    ```
    
    **Components:**
    - Query parsing, spell correction
    - Inverted index (Elasticsearch, Solr)
    - Two-stage ranking (BM25 → neural)
    - Personalization layer

    !!! tip "Interviewer's Insight"
        Discusses query understanding and learning-to-rank.

---

### Design a Data Warehouse - Amazon, Google Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Data Engineering` | **Asked by:** Amazon, Google, Meta

??? success "View Answer"

    **Schema Design:**
    - Star schema (fact + dimension tables)
    - Slowly changing dimensions (SCD Type 1/2)
    
    **Technology Stack:**
    - Storage: S3, GCS
    - Processing: Spark, DBT
    - Query: BigQuery, Snowflake, Redshift
    
    **Partitioning:** By date for time-series data.

    !!! tip "Interviewer's Insight"
        Knows star vs snowflake schema and partitioning.

---

### Design a Stream Processing System - Uber, Netflix Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Streaming` | **Asked by:** Uber, Netflix, LinkedIn

??? success "View Answer"

    ```
    [Events] → [Kafka] → [Flink/Spark] → [Feature Store] → [Model]
                                ↓
                         [Aggregations]
    ```
    
    **Key Concepts:**
    - Windowing (tumbling, sliding, session)
    - Watermarks for late data
    - Exactly-once semantics
    - State management

    !!! tip "Interviewer's Insight"
        Handles late data and stateful processing.

---

### Design an ML Labeling Pipeline - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `Data Quality` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Components:**
    1. **Label UI:** Annotation interface
    2. **Quality assurance:** Multiple annotators, consensus
    3. **Active learning:** Prioritize uncertain samples
    4. **Version control:** Track label changes
    
    **Tools:** Label Studio, Scale AI, Labelbox.

    !!! tip "Interviewer's Insight"
        Includes quality control and active learning.

---

### Design a Neural Network Optimizer - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Deep Learning` | **Asked by:** Google, Meta, OpenAI

??? success "View Answer"

    **Hyperparameter Search:**
    - Grid search → Random search → Bayesian
    - Neural Architecture Search (NAS)
    
    **Infrastructure:**
    - Ray Tune, Optuna
    - Distributed trials
    - Early stopping
    - Checkpoint management

    !!! tip "Interviewer's Insight"
        Uses Bayesian optimization for efficiency.

---

### Design a Model Retraining System - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `MLOps` | **Asked by:** Google, Amazon, Microsoft

??? success "View Answer"

    **Triggers:**
    - Scheduled (daily/weekly)
    - Drift-based (data/concept drift)
    - Performance-based (accuracy drop)
    
    **Pipeline:**
    ```
    [Trigger] → [Data] → [Train] → [Validate] → [Deploy]
                                        ↓
                              [Shadow Mode/Canary]
    ```

    !!! tip "Interviewer's Insight"
        Uses drift detection for smart retraining.

---

### Design a Vector Search System - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Embeddings`, `Search` | **Asked by:** Google, Meta, OpenAI

??? success "View Answer"

    **ANN (Approximate Nearest Neighbor) Options:**
    
    | Algorithm | Pros | Cons |
    |-----------|------|------|
    | HNSW | Fast, good recall | Memory |
    | IVF | Scalable | Slower |
    | PQ | Memory efficient | Lower recall |
    
    **Systems:** Faiss, Pinecone, Weaviate, Milvus.

    !!! tip "Interviewer's Insight"
        Knows HNSW vs IVF tradeoffs.

---

### Design an Embedding Service - Google, Meta Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Embeddings` | **Asked by:** Google, Meta, OpenAI

??? success "View Answer"

    **Requirements:**
    - Low latency (< 50ms)
    - High throughput
    - Batching for efficiency
    
    **Architecture:**
    ```
    [Request] → [Batch Collector] → [GPU Inference] → [Cache]
    ```
    
    **Optimization:** Model quantization, TensorRT.

    !!! tip "Interviewer's Insight"
        Uses batching and caching for efficiency.

---

### Design a Content Moderation System - Meta, YouTube Interview Question

**Difficulty:** 🔴 Hard | **Tags:** `Trust & Safety` | **Asked by:** Meta, YouTube, TikTok

??? success "View Answer"

    **Multi-stage Pipeline:**
    1. **Fast filters:** Hashes, blocklists
    2. **ML classifiers:** Text, image, video
    3. **Human review:** Edge cases
    4. **Appeals:** User feedback loop
    
    **Metrics:** Precision (avoid false positives), latency.

    !!! tip "Interviewer's Insight"
        Balances automation with human review.

---

### Design a Notification System - Google, Amazon Interview Question

**Difficulty:** 🟡 Medium | **Tags:** `System Design` | **Asked by:** Google, Amazon, Meta

??? success "View Answer"

    **Components:**
    - Event ingestion (Kafka)
    - User preferences store
    - Rate limiting
    - Multi-channel delivery (push, email, SMS)
    
    **ML Integration:** Optimal send time, relevance scoring.

    !!! tip "Interviewer's Insight"
        Uses ML for send time optimization.

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

