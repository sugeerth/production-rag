# Machine Learning Fundamentals

## Introduction to Machine Learning

Machine Learning (ML) is a subset of Artificial Intelligence that enables systems to learn and improve from experience without being explicitly programmed. ML algorithms build mathematical models based on training data to make predictions or decisions.

## Types of Machine Learning

### Supervised Learning
In supervised learning, the model learns from labeled training data. Each training example has an input and a known correct output. Common algorithms include:
- **Linear Regression**: Predicts continuous values
- **Logistic Regression**: Binary classification
- **Decision Trees**: Hierarchical decision rules
- **Random Forests**: Ensemble of decision trees
- **Support Vector Machines (SVM)**: Maximum margin classifiers
- **Neural Networks**: Multi-layer perceptrons

### Unsupervised Learning
Unsupervised learning finds patterns in unlabeled data:
- **K-Means Clustering**: Groups data into k clusters
- **DBSCAN**: Density-based clustering
- **PCA**: Dimensionality reduction
- **Autoencoders**: Neural network-based compression

### Reinforcement Learning
An agent learns by interacting with an environment:
- Takes actions and receives rewards
- Goal: maximize cumulative reward
- Applications: game playing, robotics, recommendation systems

## Deep Learning

### Neural Network Architecture
A neural network consists of:
- **Input layer**: Receives raw features
- **Hidden layers**: Transform features through learned weights
- **Output layer**: Produces predictions
- **Activation functions**: ReLU, sigmoid, tanh, softmax

### Transformer Architecture
Transformers revolutionized NLP and beyond:
- **Self-attention mechanism**: Relates all positions in a sequence
- **Multi-head attention**: Parallel attention with different representations
- **Positional encoding**: Injects sequence order information
- **Applications**: GPT, BERT, T5, Vision Transformers

## Large Language Models (LLMs)

### How LLMs Work
LLMs are large transformer models trained on massive text corpora:
- **Pre-training**: Learn language patterns from billions of tokens
- **Fine-tuning**: Adapt to specific tasks with smaller datasets
- **In-context learning**: Few-shot learning through prompting
- **RLHF**: Reinforcement Learning from Human Feedback for alignment

### Key LLM Capabilities
- Text generation and completion
- Question answering
- Summarization
- Translation
- Code generation
- Reasoning and analysis

## Embeddings

### What are Embeddings?
Embeddings are dense vector representations of data (text, images, etc.) in a continuous vector space. Similar items have similar embeddings (close in vector space).

### Text Embeddings
- Word2Vec, GloVe: Word-level embeddings
- Sentence-BERT: Sentence-level embeddings
- OpenAI embeddings, Cohere embeddings: API-based

### Applications of Embeddings
- Semantic search
- Clustering and classification
- Recommendation systems
- Anomaly detection
- RAG systems (document retrieval)

## Model Evaluation

### Classification Metrics
- **Accuracy**: Correct predictions / total predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

### Regression Metrics
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R-squared**: Proportion of variance explained

### Information Retrieval Metrics
- **Recall@K**: Fraction of relevant items in top-k results
- **MRR**: Mean Reciprocal Rank
- **nDCG**: Normalized Discounted Cumulative Gain
- **MAP**: Mean Average Precision
