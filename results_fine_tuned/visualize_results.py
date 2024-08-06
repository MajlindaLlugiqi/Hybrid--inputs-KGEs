import pandas as pd
import os
import matplotlib.pyplot as plt

# Parameters
# onto_names = ["heart_small", "heart_extended", "heart_snomed"]
onto_names = ["heart_extended"]
domain = "heart"
vector_sizes = [100]
score_cols = ['F2_Score']

# Read and process the data
for onto_name in onto_names:
    temp_data_list = []
    for vector_size in vector_sizes:
        file_path = f'{domain}/{vector_size}/{onto_name}/evaluation_results.csv'
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            temp_data_list.append(data)

    if temp_data_list:
        concatenated_data = pd.concat(temp_data_list)

        # Filter data for KNN model
        knn_df = concatenated_data[concatenated_data['Model'] == 'KNN']
        knn_df['n_neighbors'] = knn_df['Settings'].apply(lambda x: int(eval(x)['n_neighbors']))

        # Plotting the accuracy for different n_neighbors
        plt.figure(figsize=(10, 6))
        for scenario, group in knn_df.groupby('Scenario'):
            plt.plot(group['n_neighbors'], group['F2_Score'], marker='o', label=scenario)
        plt.xlabel('Number of Neighbors')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Change for Different n_neighbors for KNN Across Different Scenarios ({onto_name})')
        plt.legend()
        plt.grid(True)
        plt.show()
