import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate Root Mean Squared Deviation (RMSD)
def rmsd(observed, predicted):
    return np.sqrt(np.mean((observed - predicted) ** 2))

# Function to calculate 2-Chi-squared discrepancy for count data
def chi_squared(observed, predicted):
    return np.sum((observed - predicted) ** 2 / predicted)

def main():
    st.title("RMSD and 2-Chi-squared Discrepancy Calculator")

    # Get user input for observed and predicted counts
    observed_input = st.text_input("Enter observed counts (comma-separated):")
    predicted_input = st.text_input("Enter predicted counts (comma-separated):")

    if st.button("Calculate"):
        try:
            # Convert the input strings to arrays
            observed_counts = np.array([float(val) for val in observed_input.split(',')])
            predicted_counts = np.array([float(val) for val in predicted_input.split(',')])

            # Calculate discrepancies
            rmsd_value = rmsd(observed_counts, predicted_counts)
            chi_squared_value = chi_squared(observed_counts, predicted_counts)

            # Display discrepancies
            st.write(f"Root Mean Squared Deviation (RMSD): {rmsd_value:.4f}")
            st.write(f"2-Chi-squared Discrepancy: {chi_squared_value:.4f}")

            # Plotting
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # Bar plot for observed and predicted counts
            axes[0].bar(np.arange(len(observed_counts)), observed_counts, color='blue', alpha=0.6, label='Observed')
            axes[0].bar(np.arange(len(predicted_counts)), predicted_counts, color='red', alpha=0.6, label='Predicted')
            axes[0].set_xlabel('Categories')
            axes[0].set_ylabel('Counts')
            axes[0].set_title('Observed vs Predicted Counts')
            axes[0].legend()

            # Bar plot for discrepancies
            discrepancies = [rmsd_value, chi_squared_value]
            labels = ['RMSD', '2-Chi-squared']
            axes[1].bar(labels, discrepancies, color='green', alpha=0.6)
            axes[1].set_ylabel('Discrepancy')
            axes[1].set_title('Discrepancy Comparison')

            # Show plots
            st.pyplot(fig)

        except ValueError:
            st.error("Invalid input. Please enter valid numeric values separated by commas.")

if __name__ == '__main__':
    main()
