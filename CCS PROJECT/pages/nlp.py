import streamlit as st

def train_naive_bayes_classifier(sentences, tags):
    vocabulary = {}
    total_words_positive = 0
    total_words_negative = 0

    for idx, sentence in enumerate(sentences):
        for word in sentence.split():
            tag = tags[idx]

            if word in vocabulary:
                if tag == 'Positive':
                    vocabulary[word]['Positive'] += 1
                    total_words_positive += 1
                else:
                    vocabulary[word]['Negative'] += 1
                    total_words_negative += 1
            else:
                if tag == 'Positive':
                    vocabulary[word] = {'Positive': 1, 'Negative': 0}
                    total_words_positive += 1
                else:
                    vocabulary[word] = {'Positive': 0, 'Negative': 1}
                    total_words_negative += 1

    return vocabulary, total_words_positive, total_words_negative

def predict_sentiment(new_sentence, vocabulary, total_words_positive, total_words_negative):
    overall_positive_prob = 1
    overall_negative_prob = 1

    for word in new_sentence.split():
        if word in vocabulary:
            positive_prob = (vocabulary[word]['Positive'] + 1) / (total_words_positive + len(vocabulary))
            negative_prob = (vocabulary[word]['Negative'] + 1) / (total_words_negative + len(vocabulary))

            overall_positive_prob *= positive_prob
            overall_negative_prob *= negative_prob

    total_prob = overall_positive_prob + overall_negative_prob

    positive_ratio = overall_positive_prob / total_prob
    negative_ratio = overall_negative_prob / total_prob

    return positive_ratio, negative_ratio

# Streamlit GUI
st.title("Sentiment Analysis App")

# User input for training data
num_samples = st.number_input("Enter the number of training samples:", min_value=1, step=1)
sentences = []
tags = []

for i in range(num_samples):
    sentence = st.text_input(f"Enter sentence {i + 1}:", key=f"sentence{i}")
    tag = st.text_input(f"Enter sentiment tag for sentence {i + 1} (e.g., Positive, Negative):", key=f"tag{i}")

    sentences.append(sentence)
    tags.append(tag)

# Train Naive Bayes classifier
vocabulary, total_words_positive, total_words_negative = train_naive_bayes_classifier(sentences, tags)

# User input for a new sentence
new_sentence = st.text_input("Enter a new sentence:")

if new_sentence:
    positive_ratio, negative_ratio = predict_sentiment(new_sentence, vocabulary, total_words_positive, total_words_negative)

    st.subheader("Prediction Result:")
    st.write(f"Positive Probability Ratio: {positive_ratio:.4f}")
    st.write(f"Negative Probability Ratio: {negative_ratio:.4f}")

    # Make a prediction based on probabilities
    if positive_ratio > negative_ratio:
        st.write("The sentence is predicted to be Positive")
    else:
        st.write("The sentence is predicted to be Negative")
