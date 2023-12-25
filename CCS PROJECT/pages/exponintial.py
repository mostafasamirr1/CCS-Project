import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

def generate_exponential_samples(size, class_label, lambda_):
    np.random.seed(42)
    return np.random.exponential(scale=1/lambda_, size=size), np.full(size, class_label)

# Streamlit app
def main():
    st.title("Exponential Distribution and Classification")

    # Get user input for data sizes and lambda values
    train_data_size = st.number_input("Enter the size of the training data per class:", min_value=1, step=1)
    test_data_size = st.number_input("Enter the size of the testing data per class:", min_value=1, step=1)
    lambda_class_0 = st.number_input("Enter lambda for Class 0:", value=1.0)
    lambda_class_1 = st.number_input("Enter lambda for Class 1:", value=1.0)

    # Generate sample data
    samples_class_0, labels_class_0 = generate_exponential_samples(train_data_size, 0, lambda_class_0)
    samples_class_1, labels_class_1 = generate_exponential_samples(train_data_size, 1, lambda_class_1)

    # Generate test data
    samples_class_0_test, _ = generate_exponential_samples(test_data_size, 0, lambda_class_0)
    samples_class_1_test, _ = generate_exponential_samples(test_data_size, 1, lambda_class_1)

    X_test = np.concatenate([samples_class_0_test, samples_class_1_test])
    y_test = np.concatenate([np.zeros(test_data_size), np.ones(test_data_size)])

    # Classify a new instance
    new_instance = st.number_input("Enter a new instance for classification:")

    # Calculate PDF for visualization
    x_values = np.linspace(0, max(max(samples_class_0), max(samples_class_1)), 1000)
    pdf_class_0 = expon.pdf(x_values, scale=1/lambda_class_0)
    pdf_class_1 = expon.pdf(x_values, scale=1/lambda_class_1)

    # Calculate the probability of the new instance
    prob_class_0 = expon.pdf(new_instance, scale=1/lambda_class_0)
    prob_class_1 = expon.pdf(new_instance, scale=1/lambda_class_1)

    # Plot the distribution
    st.pyplot(plot_distribution(x_values, pdf_class_0, pdf_class_1, X_test, y_test, new_instance,
                                 lambda_class_0, lambda_class_1))

    st.write('Probability of Class 0: {:.4f}'.format(prob_class_0))
    st.write('Probability of Class 1: {:.4f}'.format(prob_class_1))

# Function to create the distribution plot
def plot_distribution(x_values, pdf_class_0, pdf_class_1, X_test, y_test, new_instance, lambda_class_0, lambda_class_1):
    fig, ax = plt.subplots()
    ax.plot(x_values, pdf_class_0, label=f'Class 0 (True: {lambda_class_0:.2f})')
    ax.plot(x_values, pdf_class_1, label=f'Class 1 (True: {lambda_class_1:.2f})')
    ax.scatter(X_test, np.zeros_like(X_test), c=y_test, marker='x', label='Test Data')
    ax.scatter([new_instance], [0], c='red', marker='o', label='New Instance')

    ax.set_xlabel('X Values')
    ax.set_ylabel('Probability Density')
    ax.set_title('Exponential Distribution and Classification')
    ax.legend()

    return fig

if __name__ == "__main__":
    main()
