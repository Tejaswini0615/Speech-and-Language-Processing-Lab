EXERCISE 1
Problem Statement
An electronics retailer wants to automatically process customer reviews to:
Determine overall sentiment (positive/negative/neutral)
Identify frequently mentioned product features
Detect key entities (brands, product models, components)
Surface common complaints/praises about specific features
Sentiment Analysis:
Uses NLTK's VADER for sentiment scoring
Classifies reviews as positive/negative/neutral
Shows overall sentiment distribution as percentage
Feature Extraction:
Identifies product features through nouns and proper nouns
Tracks feature frequency across all reviews
Calculates positive/negative mentions per feature
Entity Recognition:
Uses spaCy's NER to detect:
Brands (ORG/PRODUCT)
Origins (GPE/LOC)
Components (other entities)
Shows most frequent entities in each category
Aspect-Based Feedback:
Identifies adjective-noun pairs (e.g., "excellent battery")
Extracts common praises and complaints per feature
Shows most frequent positive/negative phrases
1.Initialize Resources
Prepare any resources required for:
Splitting text into sentences.
Identifying words, their types (like noun, adjective), and their base forms.
Recognizing named entities (such as brands, products, locations).
Assessing the sentiment (positive, negative, neutral) of a given text.
2.Set Up Data Structures
Create storage for:
Overall sentiment counts (positive, negative, neutral).
Product features and how often they are mentioned, along with counts of
positive and negative mentions.
Entities (like brands, origins, components) and their frequencies.
Lists of phrases that praise or complain about each feature.
3.Process Each Review
For every review in the dataset:
a. Combine Review Text
Gather all sentences of the review into a single text block.
b. Determine Overall Sentiment
Evaluate the overall sentiment of the review text.
Classify as positive, negative, or neutral based on the sentiment score.
Update the corresponding sentiment count.
c. Analyze Review Content
Break the review text into sentences.
For each sentence:
Identify and extract all nouns and proper nouns (potential product features),
ignoring common stopwords and very short words.
For each feature found, increment its mention count.
d. Extract Named Entities
Identify named entities in the sentence.
Categorize each entity as a brand, origin, or component based on its type.
Update the count for each entity.
e. Feature-Specific Sentiment
For each noun phrase (a noun and its modifiers) in the sentence:
Identify the main noun (feature) of the phrase.
Find adjectives directly describing this noun.
If adjectives are present:
Combine the adjectives and the noun phrase into a descriptive phrase.
Assess the sentiment of this phrase.
If the phrase sentiment is clearly positive and the feature is among those
extracted, increment the positive count for that feature and record the
phrase as praise.
If the phrase sentiment is clearly negative and the feature is among those
extracted, increment the negative count for that feature and record the
phrase as a complaint.
4.Summarize and Display Results
a. Overall Sentiment
Show the distribution of positive, negative, and neutral reviews.
b. Top Features
List the most frequently mentioned product features, along with their positive
and negative mention counts.
c. Key Entities
For each entity type (brand, origin, component), list the most frequently
mentioned entities.
d. Common Praises and Complaints
For each feature, display the most common praise and complaint phrases.
1.Initialize Resources
NLTK:
Download required corpora (product_reviews_1,vader_lexicon, punkt).
Initialize the VADER sentiment analyzer.
spaCy:
Load the English language model (en_core_web_sm) for linguistic analysis
(tokenization, POS tagging, NER, etc.)
2.Set Up Data Structures
Use standard Python data structures (such as defaultdict,Counter, and
dictionaries) to store sentiment counts, feature mentions, entities, and
collected praise/complaint phrases.
3.Process Each Review
a.Combine Review Text
For each review, concatenate all sentences into a single string.
If using NLTK’s corpus, iterate through review lines and join the sentences
b.Determine Overall Sentiment
Use NLTK’s SentimentIntensityAnalyzer to compute sentiment scores for the
review text.
Classify the score as positive, negative, or neutral based on the compound
value
c.Analyze Review Content
Pass the review text to spaCy’s NLP pipeline:
Sentence Segmentation: Use doc.sents to split into sentences
Tokenization & POS Tagging: For each sentence, iterate through tokens.
Identifynouns and proper nouns(token.pos_ in ['NOUN', 'PROPN']) as potential
features.
Use token.lemma_ for the base form and filter out stopwords and short words
Count feature mentions.
d.Extract Named Entities
For each sentence, usespaCy’s NER (ent.label_) to identify entities.
Categorize as brand, origin, or component based on the entity type
(ORG,PRODUCT, GPE,LOC, etc.).
Count entity mentions.
e.Feature-Specific Sentiment
For each noun chunk in the sentence (sent_doc.noun_chunks):
Extract the main noun (chunk.root.lemma_).
Find adjectives modifying the noun by checking children with dependency
labels amod or acomp and POSADJ.
If modifiers are found:
Combine them with the noun chunk to form a descriptive phrase.
Use VADER to analyze the phrase sentiment.
If the phrase is positive and the feature was extracted, increment the
positive count and record the phrase as praise.
If negative, increment the negative count and record as a complaint.
4.Summarize and Display Results
Calculate overall sentiment distribution.
Sort and display top features by mention count, with positive/negative
breakdown.
Show most frequent entities for each type.
For each feature, display the most common praise and complaint phrase.
EXERCISE 2Title: Implementation of BOW and Topic Models for Text Representation and
Classification
Objective:
Develop a pipeline to preprocess text data, generate two distinct feature
representations (Bag-of-Words and Topic Models), and evaluate their impact
on classification performance.
Problem Description:
You are given a dataset of 20 Newsgroups articles. Each article belongs to one
of multiple predefined categories (e.g., "sports," "politics," "technology"). Your
task is to:
1. Preprocess the text data and convert it into two feature representations:
Bag-of-Words (BOW) with TF-IDF weighting.
Topic distributions derived from an LDA (Latent Dirichlet Allocation)
model.
2. Train classifiers using each representation.
3. Compare how these representations affect classification accuracy,
efficiency, and interpretability
Steps to be followed
1. Data Preprocessing:
➢ Load the dataset and perform text cleaning (lowercasing, removing stop
words/punctuation, lemmatization).
➢ Split data into training/testing sets (e.g., 80:20).
2. Feature Representation:
BOW/TF-IDF Representation:
Vectorize the text using CountVectorizer followed by TfidfTransformer.
Limit vocabulary to the top 1,000 most frequent terms.
Topic Model Representation:
Train an LDA model on the training set
Transform both training and testing sets into topic probability
distributions (each document → 10-dimensional vector).
3. Classification
Train two classifiers (e.g., Logistic Regression and Naive Bayes) using:
The BOW/TF-IDF features.
The LDA topic distributions.
Evaluate models using accuracy, F1-score, and a confusion matrix
4. Analysis:
Compare classification performance between BOW and LDA
representations.
Interpret the LDA topics (e.g., show top 5 keywords per topic).
EXERCISE 3
Implementing Word embedding based text classification
1. Dataset Preparation:
Uses NLTK's product_reviews_1 corpus (261 sentences)
Converts sentiment labels to numerical values (positive=0, negative=1,
neutral=2)
Applies one-hot encoding for classification
2. Text Preprocessing:
Tokenization with 10,000 word vocabulary
Sequence padding to uniform length (100 tokens)
Handles out-of-vocabulary words with <OOV> token
3. LSTM Model Architecture:
Embedding Layer (128 dimensions)
Global Average Pooling for dimensionality reduction
Dense layers with ReLU activation and Dropout
Softmax output
4. Training:
30 epochs with batch size 32
Adam optimizer with learning rate 0.001
80-20 train-test split with stratified sampling
5. Analysis Features:
Sentiment prediction with confidence scores
Feature importance extraction using embedding magnitudes
Visualization of training history
EXERCISE 4
Implement the Skip-Gram variant of the Word2Vec algorithm from scratch to
generate high-quality word embeddings. The goal is to capture semantic and
syntactic relationships between words in a large corpus, enabling
downstream NLP tasks (e.g., sentiment analysis, machine translation) to
leverage these distributed representations.
Guidelines:
Architecture: Predict context words given a target word.
Training: Use negative sampling (5-20 negative samples per positive pair) for
efficient optimization.
Hyperparameters:
Embedding dimensionality: 300 dimensions.
Context window size: ±5 words (dynamic window shrinking).
Subsampling threshold: Discard words with frequency > 1e-5 (to mitigate bias
toward frequent words).
Learning rate: 0.025 (decayed linearly during training).
Optimization: Stochastic Gradient Descent (SGD) without external libraries
(e.g., gensim).
Dataset
Primary Dataset: Text8 Corpus (preprocessed subset of Wikipedia).
Justification:
Size: ~17 million words (100MB uncompressed), balancing computational
feasibility and linguistic richness.
Content: Cleaned English text (lowercase, alphanumeric-only, no
markup/punctuation).
Diversity: Covers science, history, technology, and culture.
EXERCISE 5
Design and implement a Convolutional Neural Network (CNN) to classify
sentiment in movie reviews as positive or negative, using the IMDB dataset. The
model must achieve at least 85% accuracy on the test set while efficiently
handling variable-length text sequences.
Guidelines:
1. Data Preparation
Load Dataset: Use the IMDB dataset from Keras (tensorflow.keras.datasets.imdb),
which contains 50,000 labeled reviews (25k train, 25k test).
Preprocess Text:
Tokenize reviews and map words to integers.
Pad/truncate sequences to a fixed length (e.g., 500 words) using pad_sequences.
Split Data: Reserve 20% of the training data for validation.
2. Model Architecture
Embedding Layer: Convert word indices into dense vectors (e.g., 100-dimensional
embeddings).
Convolutional Layers:
Apply 1D convolutions with multiple filter sizes (e.g., 3, 4, 5) to capture n-gram
features.
Use 128 filters per convolution, followed by ReLU activation.
Pooling: Global max-pooling to extract key features from each filter.
Dropout: Add dropout (e.g., 0.5) to prevent overfitting.
Dense Layers:
Fully connected layer (128 units, ReLU).
Output layer (1 unit, sigmoid activation for binary classification).
3. Model Training
Compile Model:
Optimizer: Adam (learning rate = 0.001).
Loss: Binary cross-entropy.
Metrics: Accuracy.
Train:
Batch size: 64 (change as required)
Epochs: 10 (with early stopping if validation loss plateaus).
Use validation data to monitor overfitting.
4. Evaluation
Test Accuracy: Evaluate model performance on the held-out test set.
Confusion Matrix: Analyze false positives/negatives.
Sample Predictions: Test on custom reviews (e.g., "This movie was a captivating
masterpiece!" → Positive).
5. Optimization
Hyperparameter Tuning: Adjust embedding dimensions, filter sizes, dropout rate,
and dense units.
Pre-trained Embeddings: Experiment with GloVe or Word2Vec embeddings.
Regularization: Add L2 regularization or increase dropout if overfitting occurs.
EXERCISE 6
Design and implement a Bidirectional LSTM model for Named Entity
Recognition (NER) using the CoNLL-2003 dataset. The model should classify
each word in a sentence into one of nine entity categories (e.g., person,
organization, location) or "non-entity." Try to achieve at least 85% F1-score on
the test set while handling variable-length sequences.
1. Data Preparation
Load Dataset: Use the CoNLL-2003 dataset (available via nltk or Hugging
Face datasets).
Preprocess Data:
Tokenize sentences and extract POS tags, entity labels.
Map words and entity labels to integer indices. Add <PAD> and <UNK> tokens.
Pad sequences to a fixed length (e.g., 100 tokens) using pad_sequences.
Split Data: Use built-in train/dev/test splits.
2. Model Architecture
Embedding Layer: Convert word indices to dense vectors (e.g., 100D).
Bidirectional LSTM Layer:
1–2 LSTM layers (128 units each) to capture contextual dependencies.
Output Layer:
Time-distributed dense layer with softmax activation (output: num_entity_classes).
3. Model Training
Compile Model:
Loss: SparseCategoricalCrossentropy (supports integer labels).
Optimizer: Adam (lr=0.001).
Metrics: Accuracy.
Train:
Batch size: 32, epochs: 10.
Use 20% validation split.
4. Evaluation
Test Metrics:
Report per-token accuracy and entity-level F1-score (use seqeval library).
Visualize Predictions:
Run inference on sample sentences (e.g., "Apple launched iPhone in
California").
5. Optimization
Hyperparameter Tuning:
Adjust LSTM units, dropout rate, or embedding dimensions.
Advanced Add-ons
Add a CRF layer to model label transitions.
Use pre-trained word embeddings (e.g., GloVe).
EXERCISE 7
Design and abstractive text summarization system using a deep learning
sequence-to-sequence (seq2seq) model with attention mechanisms. The
model should generate concise, meaningful summaries of news articles
from the CNN/DailyMail dataset, achieving a ROUGE-1 score of at least 0.35
on the test set.
Guidelines to Solve the Problem
1. Data Preparation: Dataset: Use the CNN/DailyMail dataset (available via
Hugging Face datasets or TensorFlow Datasets).
Preprocessing:
Clean text: Remove special characters, lowercase, and tokenize.
Split into articles (input) and highlights (target summaries).
Truncate/pad sequences to fixed lengths (e.g., 400 tokens for input, 100 for
output).
Vocabulary: Build word-to-index mappings for input and target,
with <PAD>, <UNK>, <SOS>, and <EOS> tokens.
2. Model Architecture
Encoder:
Embedding layer (e.g., 256D).
Bidirectional LSTM/GRU layers (e.g., 300 units) to process input articles.
Decoder:
LSTM/GRU layer (e.g., 300 units) with attention mechanism (e.g., Bahdanau
attention).
Time-distributed dense layer with softmax for word prediction.
Alternate Approach: Use a Transformer-based architecture (e.g., T5-small or
BART) for better performance.
3. Training
Teacher Forcing: Use target sequences as decoder input during training.
Loss Function: Cross-entropy loss (ignore padded tokens).
Optimizer: Adam (learning rate = 0.001).
Batch Size: 16–32 (due to memory constraints).
Epochs: 10–15 (use early stopping if validation loss plateaus).
4. Inference
Beam Search: Use beam search (width=3–5) during decoding to generate
better summaries.
Decoding Loop: Feed previous predictions as input until <EOS> is generated or
max length is reached.
5. Evaluation
ROUGE Metrics: Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores using
the rouge-score library.
Human Evaluation: Assess readability and relevance on a sample of
summaries.
6. Optimization
Pre-trained Embeddings: Use GloVe or Word2Vec embeddings.
Hyperparameter Tuning: Adjust hidden units, learning rate, and beam width.
Pointer-Generator: Add copy mechanism to handle out-of-vocabulary words.
EXERCISE 8
Problem Statement
Build a neural machine translation (NMT) system using an encoder-decoder
architecture with attention to translate sentences from English to French. The
model should achieve a BLEU score of at least 0.25 on the test set using the
Tatoeba sentence pairs dataset.
Guidelines to Solve the Problem
1. Data Preparation
Dataset: Use the English-French sentence pairs from the Tatoeba dataset
(available via TensorFlow Datasets or opus_books from Hugging Face).
Preprocessing:
Clean text: Remove special characters, lowercase, and tokenize.
Add <start> and <end> tokens to target sequences.
Truncate/pad sequences to fixed lengths (e.g., 20 tokens for input, 20 for
output).
Vocabulary: Build separate word-to-index mappings for source (English) and
target (French). Include <pad>, <unk>, <start>, and <end> tokens.
2. Model Architecture
Encoder:
Embedding layer (e.g., 256D) for source language.
Bidirectional LSTM/GRU layers (e.g., 512 units) to process input sentences.
Decoder:
Embedding layer (e.g., 256D) for target language.
LSTM/GRU layer (e.g., 512 units) with attention mechanism (e.g., Bahdanau
attention).
Dense layer with softmax for word prediction over target vocabulary.
Alternate Approach: Use a Transformer architecture for better performance.
3. Training
Teacher Forcing: Use target sequences as decoder input during training.
Loss Function: Sparse categorical cross-entropy (ignore padded tokens).
Optimizer: Adam (learning rate = 0.001).
Batch Size: 64–128.
Epochs: 10–20 (use early stopping if validation loss plateaus).
4. Inference
Greedy Decoding: Generate tokens one by one, taking the argmax at each
step.
Beam Search: Use beam search (width=3–5) for better translations.
Decoding Loop: Feed previous predictions as input until <end> is generated
or max length is reached.
5. Evaluation
BLEU Score: Calculate the BLEU score using
the nltk.translate.bleu_score module.
Human Evaluation: Assess translation quality on a sample of sentences.
6. Optimization
Pre-trained Embeddings: Use aligned word embeddings for both languages.
Hyperparameter Tuning: Adjust hidden units, embedding dimensions, and
learning rate.
Regularization: Use dropout in encoder and decoder to prevent overfitting.
EXERCISE 9
Implementing a Generative Chatbot using the Chitchat Dataset
Building an Open-Domain Chatbot with a Seq2Seq Model on Social
Conversation Data.
The objective of this lab is to train a generative chatbot on a dataset of realworld,
open-domain chitchat conversations. You will learn to handle the
nuances of social dialogue and implement a full deep learning pipeline. By the
end, you will be able to:
Preprocess and structure a social conversation dataset stored in JSON
format.
Implement and train a Sequence-to-Sequence (Seq2Seq) model with an
attention mechanism.
Understand the challenges of training on noisy, real-world text data.
Evaluate the chatbot's ability to generate coherent and contextually relevant
social responses.
Dataset
https://github.com/BYU-PCCL/chitchat-dataset
EXERCISE 10
Developing a Speech Recognition
System for Voice Commands
Design, implement, and evaluate a deep learning-based speech recognition system
capable of classifying voice commands
use the Speech Commands Dataset by Google, a widely-used benchmark for
voice command recognition.
Dataset Details:
Source: TensorFlow Datasets - Speech Commands
Size: ~2.5 GB, 105,000+ audio recordings
Commands: 35 words including "yes", "no", "up", "down", "left", "right", "on", "off",
"stop", "go"
Format: 1-second WAV files at 16kHz sampling rate
Background Noise: Includes background noise samples for robust training
Steps
1. Implement Audio Preprocessing and Feature Extraction
2. Implement a CNN-based and CRNN Model (CNN + RNN for sequence modeling)
3. Use the Evaluation Metrics: Accuracy, confusion matrix, classification report
