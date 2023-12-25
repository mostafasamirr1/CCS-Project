import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

def plot_binomial_distribution_from_input(n, p, k):
    """
    Calculate the probability mass function (PMF) for each possible number of successes,
    calculate the binomial probability for the specified number of successes,
    and plot a bar chart to visualize the distribution.

    Parameters:
    - n (int): Number of trials
    - p (float): Probability of success
    - k (int): Number of successes

    Returns: None
    """
    # Generate the probability mass function (PMF) for each possible number of successes
    x = np.arange(0, n + 1)
    pmf = binom.pmf(x, n, p)

    # Calculate the binomial probability for the specified number of successes
    binomial_probability = binom.pmf(k, n, p)

    # Plot a bar chart of the probability mass function
    plt.bar(x, pmf, alpha=0.7, align='center')
    plt.xlabel('Number of Successes')
    plt.ylabel('Probability')
    plt.title('Binomial Distribution (n={}, p={}, k={})'.format(n, p, k))
    
    # Display the calculated binomial probability
    plt.text(k, binomial_probability, f'Binomial Probability: {binomial_probability:.4f}', fontsize=10,
             va='bottom', ha='center', color='red')

    st.pyplot(plt)

# Streamlit app
def main():
    st.title("Binomial Distribution Plotter")

    # Get user input for the number of trials
    n = st.slider("Enter the number of trials (n):", min_value=1, max_value=100, value=10)

    # Get user input for the probability of success
    p = st.slider("Enter the probability of success (p):", min_value=0.0, max_value=1.0, step=0.01, value=0.5)

    # Get user input for the number of successes
    k = st.slider("Enter the number of successes (k):", min_value=0, max_value=n, value=n//2)

    # Button to plot the distribution
    if st.button("Plot Distribution"):
        plot_binomial_distribution_from_input(n, p, k)

if __name__ == "__main__":
    main()
