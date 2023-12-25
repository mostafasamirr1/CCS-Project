import streamlit as st
import numpy as np

def calculate_posterior(prior_probabilities, likelihoods, evidences):
    prior_probabilities = np.array(prior_probabilities)
    likelihoods = np.array(likelihoods)
    evidences = np.array(evidences)

    evidence_likelihood = np.prod(likelihoods ** evidences, axis=1) * prior_probabilities
    posterior_probabilities = evidence_likelihood / np.sum(evidence_likelihood)

    return posterior_probabilities

def main():
    st.title("Bayesian Inference - Streamlit App")

    prior_probabilities = []
    likelihoods = []
    evidences = []

    # Input prior probabilities
    st.header("Enter prior probabilities:")
    for i in range(3):
        prior = st.number_input(f"Prior probability for H{i+1}:", key=f"prior{i+1}")
        prior_probabilities.append(prior)

    # Input likelihoods
    st.header("Enter likelihoods:")
    for i in range(3):
        likelihoods_row = []
        for j in range(3):
            likelihood = st.number_input(f"Likelihood of E{j+1} given H{i+1}:", key=f"likelihood{i+1}_{j+1}")
            likelihoods_row.append(likelihood)
        likelihoods.append(likelihoods_row)

    # Input evidences
    st.header("Enter evidences (0 or 1):")
    for i in range(3):
        evidence = st.selectbox(f"Evidence E{i+1}:", [0, 1], key=f"evidence{i+1}")
        evidences.append(evidence)

    # Calculate posterior probabilities
    posterior_probabilities = calculate_posterior(prior_probabilities, likelihoods, evidences)

    # Display posterior probabilities for each hypothesis
    st.header("Posterior probabilities:")
    for i, posterior in enumerate(posterior_probabilities):
        st.write(f"H{i+1}: {posterior:.4f}")

if __name__ == "__main__":
    main()
