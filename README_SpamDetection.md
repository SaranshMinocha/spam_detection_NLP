# SMS Spam Detection — Naive Bayes + TF-IDF

## What this project does
Classifies SMS messages as spam or ham (not spam) using Natural Language Processing (NLP).
Demonstrates how text data can be converted into numerical features and used to train a
classical ML classifier that achieves 96.68% accuracy in under 1 second of training time.

## Dataset
- **SMS Spam Collection Dataset** — 5,572 real SMS messages
- 4,825 ham messages (86.6%)
- 747 spam messages (13.4%)
- Source: UCI Machine Learning Repository

## Pipeline
```
Raw Text Messages
→ Label Encoding (ham=0, spam=1)
→ Train/Test Split (80/20)
→ TF-IDF Vectorization (7,702 unique words)
→ Multinomial Naive Bayes Classifier
→ Evaluation (Accuracy, Precision, Recall, F1)
→ Custom Message Prediction
```

## Results
| Metric | Ham | Spam |
|---|---|---|
| Precision | 0.96 | 1.00 |
| Recall | 1.00 | 0.75 |
| F1-Score | 0.98 | 0.86 |
| **Overall Accuracy** | **96.68%** | |

**Training time: 0.7 seconds**

## Confusion Matrix Results
- True Negatives (Ham correctly identified): 966
- True Positives (Spam correctly caught): 112
- False Positives (Legitimate emails blocked): 0
- False Negatives (Spam slipped through): 37

## Key Concepts Applied

### TF-IDF (Term Frequency-Inverse Document Frequency)
Converts raw text into numerical vectors while:
- Giving higher weight to words unique to specific messages
- Reducing weight of common words appearing everywhere
- Producing sparse matrices (mostly zeros) stored efficiently

### Multinomial Naive Bayes
Probability-based classifier that:
- Learns the likelihood of each word appearing in spam vs ham
- Multiplies word probabilities to classify new messages
- Assumes word independence (naive assumption) but works remarkably well for text
- Trains in milliseconds vs hours for neural networks

### Precision vs Recall Tradeoff
For spam detection, precision is prioritized over recall:
- Precision 1.00 = zero legitimate emails ever blocked (critical)
- Recall 0.75 = 25% of spam slips through (acceptable tradeoff)
- Better to see some spam than miss important emails

### Distribution Shift
Model performs well on SMS-style spam patterns from training data but struggles with
modern phishing language not present in the original dataset. Real-world models require
regular retraining to adapt to evolving spam tactics.

## What I Learned
- How to convert text data into numerical features using TF-IDF
- Why classical ML (Naive Bayes) often outperforms deep learning for simple NLP tasks
- The difference between precision and recall and how to choose which to optimize
- How class imbalance affects model evaluation — accuracy alone is misleading
- What distribution shift means and why it matters in production ML systems
- How to build an end-to-end NLP pipeline from raw text to predictions

## Comparison: Classical ML vs Deep Learning
| | Naive Bayes (This Project) | Neural Network |
|---|---|---|
| Training Time | 0.7 seconds | Minutes to hours |
| Accuracy | 96.68% | Similar or slightly better |
| Interpretability | High | Low |
| Data Required | Small | Large |
| Best For | Text classification | Complex patterns |

## Tech Stack
- Python
- Pandas
- Scikit-learn (TfidfVectorizer, MultinomialNB)
- Matplotlib
- Seaborn
