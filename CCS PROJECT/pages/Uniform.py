import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def uniform_distribution(x, alpha, beta):
    pdf = np.zeros_like(x, dtype=float)
    pdf[(x >= alpha) & (x <= beta)] = 1 / (beta - alpha)
    return pdf

def calculate_mean(alpha, beta):
    return (alpha + beta) / 2

def calculate_std_dev(alpha, beta):
    return np.sqrt(((beta - alpha)**2) / 12)

def predict_class(x1, x2, alpha, beta):
    return np.where((x1 + x2) / 2 > (alpha + beta) / 2, 1, 0)

def main():
    st.title("Two-Dimensional Uniform Distribution with Training Data and Classes")

    # Get user input for alpha and beta
    alpha = st.number_input("Enter the lower bound (alpha) of the uniform distribution:", value=0.0)
    beta = st.number_input("Enter the upper bound (beta) of the uniform distribution:", value=1.0)

    # Get user input for the training data
    num_samples = st.number_input("Enter the number of training samples:", min_value=1, value=5, step=1)

    x1_values = np.array([st.number_input(f"Enter x1 for sample {i+1}:") for i in range(num_samples)])
    x2_values = np.array([st.number_input(f"Enter x2 for sample {i+1}:") for i in range(num_samples)])
    classes = np.array([st.selectbox(f"Select class for sample {i+1}:", [0, 1]) for i in range(num_samples)])

    x_values = np.linspace(alpha - 1, beta + 1, 1000)
    pdf_values = uniform_distribution(x_values, alpha, beta)
    mean_value = calculate_mean(alpha, beta)
    std_dev_value = calculate_std_dev(alpha, beta)

    plt.plot(x_values, pdf_values, label=f'Uniform Distribution\n(alpha={alpha}, beta={beta})')
    plt.scatter(x1_values[classes == 0], x2_values[classes == 0], color='blue', label='Class 0')
    plt.scatter(x1_values[classes == 1], x2_values[classes == 1], color='red', label='Class 1')

    # Display the training data
    plt.scatter(x1_values, x2_values, c=classes, cmap=plt.cm.Paired, edgecolor='k', marker='o', label='Training Data')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.axvline(x=mean_value, color='r', linestyle='--', label=f'Mean (μ) = {mean_value:.2f}')
    plt.axvline(x=mean_value + std_dev_value, color='g', linestyle='--', label=f'Standard Deviation (σ) = {std_dev_value:.2f}')
    plt.axvline(x=mean_value - std_dev_value, color='g', linestyle='--')

    new_sample_x1 = st.number_input("Enter x1 value for the new sample point:")
    new_sample_x2 = st.number_input("Enter x2 value for the new sample point:")
    plt.scatter(new_sample_x1, new_sample_x2, marker='x', s=100, color='green', label='New Sample Point')

    predicted_class = predict_class(new_sample_x1, new_sample_x2, alpha, beta)
    st.write(f"Predicted Class for the New Sample: {predicted_class}")

    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

if __name__ == "__main__":
    main()
