import pandas as pd
import os

def calculate_averages(data, score_cols):
    """
    Calculate averages for given score columns, excluding rows where F2 score is 0.
    """
    average_scores = data.groupby(['Model', "Scenario"])[score_cols].mean().reset_index()
    return average_scores

# List of ontology names
onto_names = ["heart_snomed"]

domain = "heart"
vector_sizes = [64]
score_cols = ['Accuracy', 'F1_Score', 'Precision', 'Recall', 'F2_Score']
aggregated_data = {onto_name: pd.DataFrame() for onto_name in onto_names}
transpose_table = False  # Set this to False to keep the table as required

method_names = ["TransH", "DistMult", "Autoencoder", "RDF2Vec", "Node2Vec"]

feature_set_order = ['Baseline']

# Loop through each method name and add the corresponding feature sets
for method_name in method_names:
    feature_set_order.append(f'{method_name} Only')
    feature_set_order.append(f'{method_name} + Tabular')
    feature_set_order.append(f'{method_name} FE Only')
    feature_set_order.append(f'{method_name} + Tabular + FE')

print(feature_set_order)

for onto_name in onto_names:
    temp_data_list = []
    for vector_size in vector_sizes:
        file_path = f'{domain}/{vector_size}/checkpoints/heart_snomed_intermediate_results.csv'
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            average_scores = calculate_averages(data, score_cols)
            temp_data_list.append(average_scores)
            print(temp_data_list)

    if temp_data_list:
        concatenated_data = pd.concat(temp_data_list)
        aggregated_average_scores = concatenated_data.groupby(['Model', 'Scenario'])[score_cols].mean().reset_index()
        for col in score_cols:
            aggregated_average_scores[col] = aggregated_average_scores[col] * 100
        aggregated_data[onto_name] = aggregated_average_scores

for onto_name, data in aggregated_data.items():
    if not data.empty:
        dir_path = f'latex_tables/{onto_name}/'
        os.makedirs(dir_path, exist_ok=True)

        for score in score_cols:
            if transpose_table:
                pivot_df = data.pivot(index='Scenario', columns='Model', values=score).T.reset_index()
                column_order = ['Model'] + feature_set_order
            else:
                pivot_df = data.pivot(index='Scenario', columns='Model', values=score).reset_index()
                column_order = ['Scenario'] + list(data['Model'].unique())

            # Ensure all columns in your order exist in the DataFrame, to avoid KeyError
            column_order = [col for col in column_order if col in pivot_df.columns]

            pivot_df = pivot_df.reindex(columns=column_order)

            for col in pivot_df.columns[1:]:
                pivot_df[col] = pivot_df[col].apply(lambda x: f'{x:.2f}' if not pd.isnull(x) else '-')

            latex_table = pivot_df.to_latex(index=False, na_rep='-')
            print(f"dir {dir_path}")
            with open(f'{dir_path}latex_table_averages_{score}_{vector_sizes[0]}_new.tex', 'w') as file:
                file.write(latex_table)
