# Emotion Detection from Text Using NLP, ML & Deep Learning

## Project Overview  
This project develops a robust emotion classification system that identifies emotions from short text inputs. Using a combination of classical Machine Learning and Deep Learning techniques, it classifies text into six emotion categories: joy, sadness, anger, fear, love, and surprise.

The system compares three modeling approaches—TF-IDF with Naive Bayes, Bidirectional LSTM (BiLSTM), and Convolutional Neural Networks (CNN)—to determine the most effective method for text-based emotion recognition.

## Problem Statement  
Emotion detection from short texts is a critical task in applications such as social media analysis, customer feedback, chatbots, and mental health monitoring. Challenges include ambiguity, similarity between emotions, and noisy or sarcastic language.

This project aims to accurately classify short text sentences into their respective emotional categories by evaluating different NLP-based models and identifying the best-performing solution.

## Dataset  
- **Name:** Emotions Dataset for NLP  
- **Source:** Kaggle  
- **Files:** train.txt, test.txt, val.txt  
- **Format:** Each file contains two columns separated by a semicolon (`;`):  
  `text ; emotion`  

### Sample:  
- i am so happy ; joy
- i feel scared ; fear
- thank you so much ; love

## Tools & Libraries  
- Python, Pandas for data handling  
- Matplotlib, Seaborn for visualization  
- Scikit-learn for TF-IDF and Naive Bayes  
- Keras & TensorFlow for BiLSTM and CNN models  
- Google Colab and Drive for dataset storage and GPU-enabled computation  

## Methodology  

### Data Preparation  
- Mounted Google Drive to access dataset files  
- Loaded and combined train, test, and validation sets for unified processing  
- Visualized emotion class distribution to identify class imbalance  
- Preprocessed text: TF-IDF vectorization for ML model; tokenization and padding for DL models  
- Encoded emotion labels into numerical format  

### Modeling Approaches  
- **TF-IDF + Naive Bayes:** Simple and fast baseline with 75.25% accuracy  
- **BiLSTM:** Captures sequence and context with 89.03% accuracy  
- **CNN:** Learns local text features with best performance at 92.25% accuracy  

### Evaluation  
- Compared models using accuracy, classification reports, and confusion matrices  
- Developed reusable prediction functions for real-time emotion detection on custom input text  

## Results Summary  

| Model               | Accuracy | Highlights                   |
|---------------------|----------|------------------------------|
| TF-IDF + Naive Bayes | 75.25%   | Fast baseline, interpretable  |
| BiLSTM              | 89.03%   | Good at capturing context     |
| CNN                 | 92.25%   | Best accuracy, detects n-grams|

## Challenges  
- Imbalanced emotion classes affecting minority categories  
- Ambiguity and shortness of text inputs reduce context clues  
- Sarcasm and negations reduce ML model effectiveness  

## Business Impact  
This model can be integrated into customer service tools, social media monitoring platforms, and mental health applications to better understand user sentiment and emotion, enabling personalized and responsive interactions.

## What I Learned  
- Practical NLP preprocessing techniques for different model types  
- Strengths and limitations of classical ML versus deep learning for text  
- How to design, train, and compare multiple models on a common task  
- End-to-end NLP project workflow: data loading → preprocessing → modeling → evaluation → deployment-ready prediction functions  

## How to Run  
1. Clone this repository  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook to preprocess data, train models, and test predictions.

## Author
Gowtham Reddy S