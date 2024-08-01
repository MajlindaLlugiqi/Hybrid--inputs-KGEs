import pandas as pd
from owlready2 import *
import os

def populate_ontology_from_csv(ontology_path, csv_file, mapping_file,output_file_path):
    # Load the ontology
    onto = get_ontology(ontology_path).load()

    print(onto)

    # Load the CSV file containing the dataset
    df = pd.read_csv(csv_file)

    #pre-process dataset
    # Delete 'id' columns
    df.drop('id', axis=1, inplace=True)

    # df[[col for col in df.columns if df[col].dtype == 'object']]

    columns_to_convert = ['pcv', 'wc', 'rc']
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')



    #List of boolean columns
    columns_to_convert = ['htn', 'dm', 'cad', 'pe', 'ane']  # Replace 'column1', 'column2' with your actual column names

    # Convert 'yes'/'no' to True/False for each specified column
    for column in columns_to_convert:
        df[column] = df[column].replace({'yes': True, 'no': False})

    df['classification'].replace(to_replace={'ckd\t': True, 'ckd': True, 'notckd': False}, inplace=True)

    # separating object & numeric columns
    print(df.head(10))

    # Match classes:
    # Replace values in 'rbc' and 'pc' columns
    df['rbc'] = df['rbc'].replace({'normal': 'Normal', 'abnormal': 'Abnormal'})
    df['pc'] = df['pc'].replace({'normal': 'PC_Normal', 'abnormal': 'PC_Abnormal'})

    # Replace values in 'pcc' and 'ba' columns
    df['pcc'] = df['pcc'].replace({'notpresent': 'PCC_Notpresent', 'present': 'PCC_Present'})
    df['ba'] = df['ba'].replace({'notpresent': 'BA_Notpresent', 'present': 'BA_Present'})

    # Replace values in 'appet' column
    df['appet'] = df['appet'].replace({'good': 'Good_Appetite', 'poor': 'Poor_Appetite'})

    print(df.head(10))

    # Load the mapping file
    mappings = {}
    with open(mapping_file, 'r') as f:
        for line in f.readlines():
            attr, prop = line.strip().split(',')
            mappings[attr] = prop

    def get_or_create_individual(class_name, identifier):
        # print(f"----{class_name}")
        # print(f"----{identifier}")
        # Search for the class in the ontology using the class name
        class_ref = onto.search_one(iri=f"*{class_name}")

        if not class_ref:
            raise ValueError(f"Class {class_name} not found in ontology")

        # Search for an existing individual with the given identifier
        individual = onto.search_one(iri=f"*{identifier}")

        if not individual:
            # If not found, create a new individual of the specified class
            individual = class_ref(identifier)

        return individual

    # # Function to create or get an individual for object properties
    # def get_or_create_individual(class_name, identifier):
    #     # Search for an existing individual with the given identifier
    #     individual = onto.search_one(iri=f"*{identifier}")
    #     if not individual:
    #         # If not found, create a new individual of the specified class
    #         class_ref = onto[class_name]
    #         individual = class_ref(identifier)
    #     return individual

    # Function to create an individual from a CSV row
    def create_individual(row, idx):
        # Create a new individual of the Patient class
        patient_class = onto.search_one(iri="*116154003")
        # print(patient_class)
        individual = patient_class(f"Patient_{idx}")

        # Set properties for the individual based on the mappings and the row data_heart_303
        for column, value in row.items():
            if pd.notna(value):  # Check if the value is not NaN
                prop_name = mappings.get(column)
                # print(prop_name)
                if prop_name:
                    prop = onto.search_one(iri=f"*{prop_name}")
                    print(prop)
                    if prop:
                        if pd.isna(value):
                            value = ''  # Set value to empty string if it is NaN
                        if isinstance(prop, DataPropertyClass):
                            # print(f'DataProp:  {prop_name}')
                            setattr(individual, prop_name, value)
                        elif isinstance(prop, ObjectPropertyClass):
                            # print(f'ObjProp:  {prop_name}')
                            # Use the value as the class name for object properties
                            related_individual = get_or_create_individual(value, f"{value}")
                            # setattr(individual, prop_name, [related_individual])
                            setattr(individual, prop_name, related_individual)

                        # elif isinstance(prop, ObjectPropertyClass):
                        #     print(f'ObjProp:  {prop_name}')
                        #     print(prop)
                        #     # For object properties, create or fetch the related individual
                        #     related_individual = get_or_create_individual(prop_name, f"{prop_name}_{value}")
                        #     print(related_individual)
                        #     setattr(individual, prop_name, [related_individual])

    # Iterate over the rows in the dataframe and create an individual for each row
    for idx, row in df.iterrows():
        create_individual(row, idx)

    # Save the updated ontology
    onto.save(file=output_file_path)


csv_dir = "data/"
output_dir = "knowledge_graphs/kidney"
domain = 'kidney'
# Set the paths for the ontology, dataset CSV, and the mapping file
ontology_path = f"ontologies/{domain}/{domain}_dataset_ontology.owl"
csv_file = f"{csv_dir}/{domain}/{domain}_disease.csv"

mapping_file = "mappings/kidney_snomed.attr.values"

output_file_path = os.path.join(output_dir, 'kidney_snomed.owl')

# Populate the ontology with instances from the CSV file
populate_ontology_from_csv(ontology_path, csv_file, mapping_file, output_file_path)

print("Ontology has been populated with instances from the dataset.")
