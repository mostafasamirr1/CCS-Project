import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Function to calculate Gaussian probability density function
def gaussian_pdf(x, mean, std_dev):
    return (1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

# Streamlit app
def main():
    st.title("Gaussian Distribution and Classification")

    # Get user input for the dataset
    num_samples = st.number_input("Enter the number of samples in the dataset:", min_value=1, step=1)
    data = []

    for _ in range(num_samples):
        x1 = st.number_input("Enter x1 value:", key=f"x1_{_}")
        x2 = st.number_input("Enter x2 value:", key=f"x2_{_}")
        label = st.selectbox("Enter class (0 or 1):", options=[0, 1], key=f"label_{_}")
        data.append([x1, x2, label])

    # Convert the data list to a NumPy array
    data = np.array(data)

    # Separate the dataset into features (X) and labels (Y)
    X_train = data[:, :2]
    y_train = data[:, 2]

    # Assume standard deviation for both classes
    std_dev = st.number_input("Standard deviation:", value=1.0)

    # Get user input for the new data point
    x1_new = st.number_input("Enter x1 value for the new data point:")
    x2_new = st.number_input("Enter x2 value for the new data point:")
    X_new = np.array([x1_new, x2_new])

    # Create a meshgrid for contour plot
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Calculate Gaussian distributions for both classes
    pdf_class_0 = gaussian_pdf(xx, np.mean(X_train[y_train == 0, 0]), std_dev) * \
                  gaussian_pdf(yy, np.mean(X_train[y_train == 0, 1]), std_dev)
    pdf_class_1 = gaussian_pdf(xx, np.mean(X_train[y_train == 1, 0]), std_dev) * \
                  gaussian_pdf(yy, np.mean(X_train[y_train == 1, 1]), std_dev)

    # Create a Matplotlib figure
    fig, ax = plt.subplots()
    
    # Plot the Gaussian distributions
    ax.contour(xx, yy, pdf_class_0, cmap='Blues', alpha=0.5)
    ax.contour(xx, yy, pdf_class_1, cmap='Reds', alpha=0.5)

    # Plot the old data points
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='blue', label='Class 0')
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='red', label='Class 1')

    # Plot the new data point
    ax.scatter(X_new[0], X_new[1], marker='x', s=100, color='green', label='New Data Point')

    # Set plot labels and legend
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Gaussian Distributions and Data Points')

    # Display plot as an image using st.image
    st.image(fig_to_image(fig), caption='Gaussian Distributions and Data Points', use_column_width=True)

    # Calculate the likelihoods for the new data point
    likelihood_class_0 = gaussian_pdf(X_new[0], np.mean(X_train[y_train == 0, 0]), std_dev) * \
                         gaussian_pdf(X_new[1], np.mean(X_train[y_train == 0, 1]), std_dev)
    likelihood_class_1 = gaussian_pdf(X_new[0], np.mean(X_train[y_train == 1, 0]), std_dev) * \
                         gaussian_pdf(X_new[1], np.mean(X_train[y_train == 1, 1]), std_dev)

    # Classify the new data point
    predicted_class = 0 if likelihood_class_0 > likelihood_class_1 else 1

    st.write(f"MAP(y = 0) = {likelihood_class_0}")
    st.write(f"MAP(y = 1) = {likelihood_class_1}")
    st.write(f"X = {X_new} can be classified as:")
    st.write(f"Class {predicted_class}")
    st.write(f'Probability of class 0: {likelihood_class_0 / (likelihood_class_1 + likelihood_class_0)}')
    st.write(f'Probability of class 1: {likelihood_class_1 / (likelihood_class_1 + likelihood_class_0)}')
    st.write(data)

# Function to convert Matplotlib figure to PNG image
def fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf

if __name__ == "__main__":
    main()

