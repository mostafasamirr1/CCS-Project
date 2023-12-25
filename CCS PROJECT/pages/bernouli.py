import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

class Bernoulli:
    @staticmethod
    def pmf(x, p):
        f = p**x * (1 - p)**(1 - x)
        return f

    @staticmethod
    def mean(p):
        return p

    @staticmethod
    def var(p):
        return p * (1 - p)

    @staticmethod
    def std(p):
        return Bernoulli.var(p)**(1/2)

    @staticmethod
    def rvs(p, size=1):
        rvs = np.random.choice([0, 1], size=size, p=[1-p, p])
        return rvs

# Streamlit app
def main():
    st.title("Bernoulli Distribution Generator")

    # Get user input for probability_of_success and sample_size
    probability_of_success = st.slider("Enter the probability of success:", min_value=0.0, max_value=1.0, step=0.01)
    sample_size = st.slider("Enter the number of samples to generate:", min_value=1, max_value=1000, step=1)

    # Create an instance of the Bernoulli class
    bernoulli_dist = Bernoulli()

    # Test the random variates method
    rvs_samples = bernoulli_dist.rvs(probability_of_success, sample_size)
    st.write(f"Generated Bernoulli samples: {rvs_samples}")

    # Print mean, variance, and standard deviation
    st.write(f"Mean: {bernoulli_dist.mean(probability_of_success)}")
    st.write(f"Variance: {bernoulli_dist.var(probability_of_success)}")
    st.write(f"Standard Deviation: {bernoulli_dist.std(probability_of_success)}")

    # Plotting the histogram
    st.pyplot(plot_histogram(rvs_samples))

# Function to create the histogram plot
def plot_histogram(samples):
    fig, ax = plt.subplots()
    ax.hist(samples, bins=[-0.5, 0.5, 1.5], align='mid', rwidth=0.3, color='blue', alpha=0.7, label='Generated Samples')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Bernoulli Distribution of Generated Samples')
    ax.legend()
    return fig

if __name__ == "__main__":
    main()

