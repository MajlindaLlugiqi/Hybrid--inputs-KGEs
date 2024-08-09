import pandas as pd
import os
import matplotlib.pyplot as plt

# Parameters
onto_names = ["heart_snomed"]
domain = "heart"
vector_sizes = [64]
score_cols = ['F2_Score']
# score_cols = ['Accuracy']

# scenarios_to_include = ['Baseline', 'TransH Only', 'TransH + Tabular', 'TransH Tabular + FE Only', 'DistMult Only', 'DistMult + Tabular', 'DistMult Tabular + FE Only', 'Autoencoder Only', 'Autoencoder + Tabular', 'Autoencoder Tabular + FE Only'
#                         , 'RDF2Vec Only', 'RDF2Vec + Tabular', 'RDF2Vec Tabular + FE Only', 'Node2Vec Only', 'Node2Vec + Tabular', 'Node2Vec Tabular + FE Only']


# scenarios_to_include = ['Baseline', 'TransH Tabular + FE Only', 'DistMult Tabular + FE Only', 'Autoencoder Tabular + FE Only'
#                         ,'RDF2Vec Tabular + FE Only',  'Node2Vec Tabular + FE Only']

# scenarios_to_include = ['Baseline', 'TransH + Tabular + FE', 'TransH Clustered', 'TransH Interaction', 'DistMult + Tabular + FE', 'DistMult Clustered', 'DistMult Interaction'
#                         , 'Autoencoder + Tabular + FE', 'Autoencoder Clustered', 'Autoencoder Interaction', 'RDF2Vec + Tabular + FE', 'RDF2Vec Clustered', 'RDF2Vec Interaction','Node2Vec + Tabular + FE', 'Node2Vec Clustered', 'Node2Vec Interaction']

# scenarios_to_include = ['Baseline', 'TransH Clustered', 'TransH Interaction',  'DistMult Clustered', 'DistMult Interaction'
#                         , 'Autoencoder Clustered', 'Autoencoder Interaction','RDF2Vec Clustered', 'RDF2Vec Interaction','Node2Vec Clustered', 'Node2Vec Interaction']

scenarios_to_include = ['Baseline', 'TransH Clustered', 'TransH Interaction',  'DistMult Clustered', 'DistMult Interaction']


# Read and process the data
for onto_name in onto_names:
    temp_data_list = []
    for vector_size in vector_sizes:
        file_path = f'{domain}/{vector_size}/checkpoints/heart_snomed_intermediate_results.csv'
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            temp_data_list.append(data)

    if temp_data_list:
        concatenated_data = pd.concat(temp_data_list)

        # Filter data for KNN model
        knn_df = concatenated_data[concatenated_data['Model'] == 'KNN']
        knn_df['n_neighbors'] = knn_df['Settings'].apply(lambda x: int(eval(x)['n_neighbors']))

        # Filter for the specified scenarios
        knn_df = knn_df[knn_df['Scenario'].isin(scenarios_to_include)]

        # Calculate the average F2_Score for each n_neighbors per scenario
        avg_knn_df = knn_df.groupby(['Scenario', 'n_neighbors'])[score_cols].mean().reset_index()

        # Plotting the average F2_Score for different n_neighbors
        plt.figure(figsize=(10, 6))
        for scenario in scenarios_to_include:
            group = avg_knn_df[avg_knn_df['Scenario'] == scenario]
            plt.plot(group['n_neighbors'], group['F2_Score'], marker='o', label=scenario)
        plt.xlabel('Number of Neighbors')
        plt.ylabel('F2_Score')
        plt.title(f'Average F2_Score for Different n_neighbors for KNN Across Different Scenarios ({onto_name})')
        plt.legend()
        plt.grid(True)
        plt.show()