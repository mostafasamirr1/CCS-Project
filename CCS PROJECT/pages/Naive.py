import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def naive_bayes_classifier(X_train, y_train, X):
    total_samples = len(y_train)
    prior_probability_class_1 = sum(1 for label in y_train if label == 1) / total_samples
    prior_probability_class_0 = 1 - prior_probability_class_1

    likelihood_class_1 = 1.0
    likelihood_class_0 = 1.0

    for i in range(len(X)):
        feature = X[i]
        likelihood_class_1 *= sum(1 for j in range(len(X_train))
                                  if X_train[j][i] == feature and y_train[j] == 1) / sum(1 for label in y_train if label == 1)
        likelihood_class_0 *= sum(1 for j in range(len(X_train))
                                  if X_train[j][i] == feature and y_train[j] == 0) / sum(1 for label in y_train if label == 0)

    evidence = likelihood_class_1 * prior_probability_class_1 + likelihood_class_0 * prior_probability_class_0
    posterior_probability_class_1 = (likelihood_class_1 * prior_probability_class_1) / evidence
    posterior_probability_class_0 = (likelihood_class_0 * prior_probability_class_0) / evidence
    final_class = 0 if posterior_probability_class_0 > posterior_probability_class_1 else 1

    return posterior_probability_class_1, posterior_probability_class_0, final_class

def plot_training_points(X_train, y_train, X, final_class):
    X_train_array = np.array(X_train)
    fig, ax = plt.subplots()
    ax.scatter(X_train_array[y_train == 0][:, 0], X_train_array[y_train == 0][:, 1], label='Class 0', marker='o')
    ax.scatter(X_train_array[y_train == 1][:, 0], X_train_array[y_train == 1][:, 1], label='Class 1', marker='x')
    ax.scatter(X[0], X[1], marker='s', color='red', label='New Point')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    ax.set_title('Training Points and New Point')
    st.pyplot(fig)

def main():
    st.title("Naive Bayes Classifier - Streamlit App")

    # Training dataset
    X_train = [
        [1, 1, 1],
        [1, 1, 0],
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ]
    y_train = [0, 0, 0, 1, 1, 1]

    # Input X for prediction
    X = [
        st.number_input("Enter Feature 1 for the new point:", value=0.0, step=0.1),
        st.number_input("Enter Feature 2 for the new point:", value=0.0, step=0.1),
        st.number_input("Enter Feature 3 for the new point:", value=0.0, step=0.1),
    ]

    posterior_class_1, posterior_class_0, final_class = naive_bayes_classifier(X_train, y_train, X)

    st.subheader("Classification Result:")
    st.write(f"Posterior probability for class 1 given X = {X}: {posterior_class_1}")
    st.write(f"Posterior probability for class 0 given X = {X}: {posterior_class_0}")
    st.write(f"The point {X} is classified to class: {final_class}")

    st.subheader("Training Points and New Point:")
    plot_training_points(X_train, y_train, X, final_class)

if __name__ == "__main__":
    main()
