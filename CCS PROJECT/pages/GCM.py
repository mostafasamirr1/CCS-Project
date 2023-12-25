import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

class GeneralizedContextModel:
    def __init__(self, num_variables, c):
        self.num_variables = num_variables
        self.data = []
        self.classes = []
        self.c = c

    def add_instance(self, values, class_label):
        self.data.append(values)
        self.classes.append(class_label)

    def calculate_distance(self, instance):
        distances = np.linalg.norm(np.array(self.data) - instance, axis=1)
        return distances

    def calculate_similarity(self, instance):
        distances = self.calculate_distance(instance)
        similarity = np.exp(-self.c * distances)
        return similarity

    def calculate_probability(self, instance):
        similarity = self.calculate_similarity(instance)
        total_similarity = np.sum(similarity)
        probabilities = similarity / total_similarity
        return probabilities

    def predict_class(self, instance):
        probabilities = self.calculate_probability(instance)
        predicted_class_index = np.argmax(probabilities)
        predicted_class = self.classes[predicted_class_index]
        return predicted_class

# ... (previous code)

def plot_all_data(gcm, new_instance):
    # Create a mapping from class labels to numeric values
    class_mapping = {label: i for i, label in enumerate(set(gcm.classes))}

    # Convert class labels to numeric values
    numeric_classes = [class_mapping[label] for label in gcm.classes]

    data_array = np.array(gcm.data)

    if gcm.num_variables == 1:
        fig, ax = plt.subplots()
        ax.scatter(data_array[:, 0], np.zeros_like(data_array[:, 0]), c=numeric_classes, label='Old Instances', cmap='viridis')
        ax.scatter(new_instance[0], 0, marker='x', s=100, c='red', label='New Instance')
        ax.set_xlabel('Variable 1')
        ax.legend()
        ax.set_title('1D Visualization of Data with All Points and Classes')
    elif gcm.num_variables == 2:
        fig, ax = plt.subplots()
        # Ensure that data_array has at least 2 columns before indexing [:, 1]
        if data_array.shape[1] > 1:
            ax.scatter(data_array[:, 0], data_array[:, 1], c=numeric_classes, label='Old Instances', cmap='viridis')
            ax.scatter(new_instance[0], new_instance[1], marker='x', s=100, c='red', label='New Instance')
            ax.set_xlabel('Variable 1')
            ax.set_ylabel('Variable 2')
            ax.legend()
            ax.set_title('2D Visualization of Data with All Points and Classes')
        else:
            st.warning("Insufficient data for 2 variables. Cannot create 2D plot.")
            return None
    else:
        st.warning("Plotting is currently supported for 1 or 2 variables only.")
        return None

    return fig


# ... (remaining code)

def main():
    st.title("Generalized Context Model (GCM) - Streamlit App")

    num_variables = st.number_input("Enter the number of variables:", min_value=1, step=1)

    c = st.number_input("Enter the value of c for the similarity formula:", value=1.0, step=0.1)

    gcm = GeneralizedContextModel(num_variables, c)

    num_instances = st.number_input("Enter the number of training instances:", min_value=1, step=1, value=1)
    for instance_idx in range(num_instances):
        # Use a loop to get values for each variable
        values = []
        for i in range(num_variables):
            # Include instance_idx in the key to ensure uniqueness
            variable_value = st.number_input(f"Enter value for variable {i + 1} of instance {instance_idx + 1}:", key=f"value{i+1}_{instance_idx}")
            values.append(variable_value)

        # Include instance_idx in the key to ensure uniqueness
        class_label = st.text_input(f"Enter the class label for instance {instance_idx + 1}:", key=f"class_label{instance_idx}")
        gcm.add_instance(values, class_label)

    new_instance = [st.number_input(f"Enter value for variable {i + 1} of the new instance:") for i in range(num_variables)]

    fig = plot_all_data(gcm, np.array(new_instance))

    if fig is not None:
        # Show the plot on Streamlit
        st.pyplot(fig)

    predicted_class = gcm.predict_class(np.array(new_instance))

    st.subheader("Prediction Result:")
    st.write("Predicted Class:", predicted_class)

# ... (remaining code)


if __name__ == "__main__":
    main()

