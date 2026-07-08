---
title: Suggested Learning Paths
description: "Structured learning paths for data science interview preparation: from beginner fundamentals to ML engineering and GenAI, using the free resources on this site."
---

# 📅 Suggested Learning Paths

Not sure where to start? These four paths turn the scattered resources on this site into a clear, ordered route. Pick the path that matches your target role, follow the steps top to bottom, and use the "You are ready when..." checklist at the end of each path to confirm you have really absorbed the material before moving on.

Each path builds on the one before it, so if you are aiming for a Machine Learning Engineer or GenAI role, work through the earlier paths first and treat the later steps as your specialization.

## 🌱 Path 1: Foundations First (Beginner)

**Suggested duration:** 4 to 6 weeks

Start here if you are new to programming or data science. This path gives you the core toolkit that every later path assumes you already have: Python, SQL, statistics, and the two libraries you will use every single day.

1. Learn Python syntax and core data types with the [Python Cheat Sheet](Cheat-Sheets/Python.md).
2. Practice the language deeper through the [Python Interview Questions](Interview-Questions/Python.md).
3. Get comfortable querying data with the [SQL Cheat Sheet](Cheat-Sheets/SQL.md), then test yourself on the [SQL Interview Questions](Interview-Questions/SQL-Interview-Questions.md).
4. Build a statistics and probability base with the [Probability Interview Questions](Interview-Questions/Probability.md).
5. Reinforce a key concept most interviews probe by reading [Normal Distribution](Machine-Learning/Normal%20Distribution.md).
6. Learn numerical computing with the [NumPy Cheat Sheet](Cheat-Sheets/NumPy.ipynb) and the [NumPy Interview Questions](Interview-Questions/NumPy.md).
7. Learn data wrangling with the [Pandas Cheat Sheet](Cheat-Sheets/Pandas.ipynb) and the [Pandas Interview Questions](Interview-Questions/Pandas.md).
8. Bookmark the broader [Online Material for Learning](Online-Material/Online-Material-for-Learning.md) for anything you want to explore further.

**You are ready when...**

- [ ] You can write a Python function with loops, conditionals, and comprehensions without looking things up.
- [ ] You can write a SQL query with joins, aggregations, and a GROUP BY from memory.
- [ ] You can explain mean, variance, and a probability distribution in plain language.
- [ ] You can load a CSV into a pandas DataFrame and compute grouped summary statistics.

## 📊 Path 2: Data Scientist

**Suggested duration:** 6 to 8 weeks (after Path 1)

Now that the foundations are solid, this path turns you into someone who can frame a problem, train a model, evaluate it honestly, and reason about experiments the way a data scientist is expected to in interviews.

1. Build core intuition with the [Machine Learning Interview Questions](Interview-Questions/Machine-Learning.md).
2. Learn to train models in code with the [scikit-learn Cheat Sheet](Cheat-Sheets/Sk-learn.md) and the [Scikit-Learn Interview Questions](Interview-Questions/Scikit-Learn.md).
3. Master the workhorse algorithms through targeted deep dives: [Linear Regression](Machine-Learning/Linear%20Regression.md), [Logistic Regression](Machine-Learning/Logistic%20Regression.md), [Decision Trees](Machine-Learning/Decision%20Trees.md), and [Random Forest](Machine-Learning/Random%20Forest.md).
4. Understand model quality and its pitfalls with [Confusion Matrix](Machine-Learning/Confusion%20Matrix.md), [Overfitting, Underfitting](Machine-Learning/Overfitting,%20Underfitting.md), and [Normalization Regularisation](Machine-Learning/Normalization%20Regularisation.md).
5. Handle messy real data using [Unbalanced, Skewed data](Machine-Learning/Unbalanced,%20Skewed%20data.md).
6. Learn dimensionality reduction and clustering with [PCA](Machine-Learning/PCA.md), [K-means clustering](Machine-Learning/K-means%20clustering.md), and [DBSCAN](Machine-Learning/DBSCAN.md).
7. Learn experimentation with the [A/B Testing Interview Questions](Interview-Questions/AB-testing.md) and the [Hypothesis Tests Cheat Sheet](Cheat-Sheets/Hypothesis-Tests.md).
8. Drill with the flashcards on the [Flashcards](flashcards.md) page and the curated [Interview Question Resources](Interview-Questions/Interview-Question-Resources.md).

**You are ready when...**

- [ ] You can pick an appropriate algorithm for a problem and justify the choice.
- [ ] You can explain the bias-variance tradeoff and how regularization helps.
- [ ] You can design a clean A/B test and describe how you would read the results.
- [ ] You can interpret a confusion matrix and choose the right metric for the business goal.

## ⚙️ Path 3: Machine Learning Engineer

**Suggested duration:** 8 to 10 weeks (after Path 2)

This path adds the engineering muscle: writing efficient code, designing systems, scaling to big data, and shipping a model into production so it actually serves predictions.

1. Sharpen your coding with the [Data Structures and Algorithms Interview Questions](Interview-Questions/data-structures-algorithms.md).
2. Learn to reason about large systems with the [System Design Interview Questions](Interview-Questions/System-design.md).
3. Study the theory behind modern models with [Neural Networks](Machine-Learning/Neural%20Networks.md), [Activation functions](Machine-Learning/Activation%20functions.md), [Gradient Boosting](Machine-Learning/Gradient%20Boosting.md), and [Loss Function MAE, RMSE](Machine-Learning/Loss%20Function%20MAE,%20RMSE.md).
4. Learn deep learning frameworks with the [TensorFlow Interview Questions](Interview-Questions/TensorFlow.md) and [PyTorch Interview Questions](Interview-Questions/PyTorch.md), backed by the [TensorFlow Cheat Sheet](Cheat-Sheets/tensorflow.md), [Keras Cheat Sheet](Cheat-Sheets/Keras.md), and [PyTorch Cheat Sheet](Cheat-Sheets/PyTorch.md).
5. Scale to big data with the [PySpark Cheat Sheet](Cheat-Sheets/PySpark.md).
6. Learn to serve models behind an API with the [Flask Cheat Sheet](Cheat-Sheets/Flask.md) and the [Django Cheat Sheet](Cheat-Sheets/Django.md).
7. Put it all together by [Deploying ML Models](Deploying-ML-models/deploying-ml-models.md).

**You are ready when...**

- [ ] You can solve a medium-difficulty coding problem and analyze its time and space complexity.
- [ ] You can sketch an end-to-end ML system and discuss its bottlenecks and tradeoffs.
- [ ] You can train a model in TensorFlow or PyTorch and explain the training loop.
- [ ] You can wrap a trained model in a web service and deploy it to serve live predictions.

## 🤖 Path 4: GenAI / LLM Engineer

**Suggested duration:** 6 to 8 weeks (after Path 3)

The newest and fastest-moving specialization. This path assumes you already know deep learning and focuses on the transformer architecture and the tooling used to build applications on top of large language models.

1. Understand the architecture that powers modern LLMs with the [Transformers Interview Questions](Interview-Questions/Transformers.md).
2. Ground yourself in language tasks with the [Natural Language Processing Interview Questions](Interview-Questions/Natural-Language-Processing.md).
3. Learn to build LLM applications with the [LangChain Interview Questions](Interview-Questions/LangChain.md).
4. Learn to orchestrate stateful, multi-step agents with the [LangGraph Interview Questions](Interview-Questions/LangGraph.md).
5. Keep both frameworks handy with the [LangChain and LangGraph Cheat Sheet](Cheat-Sheets/LangChain-LangGraph.md).
6. Stay current with the community picks on the [Popular Resources](Online-Material/popular-resources.md) page.

**You are ready when...**

- [ ] You can explain self-attention and why transformers replaced recurrent models.
- [ ] You can describe a retrieval-augmented generation pipeline end to end.
- [ ] You can build a multi-step agent with LangChain or LangGraph and reason about its state.
- [ ] You can discuss the tradeoffs of prompting, fine-tuning, and retrieval for a given task.
