import pandas as pd
from owlready2 import *

def populate_ontology_from_csv(ontology_name, csv_file, output_dir):
    if domain == 'heart':
        onto_path = f"ontologies/{domain}_{ontology_name}.owl"
    elif domain == 'kidney':
        onto_path = f"ontologies/{domain}/{domain}_dataset_ontology.owl"
        # http: // snomed.info / id / 116154003
    onto = get_ontology(onto_path).load()
    print(onto)
    try:
        Patient = onto["Patient"]
    except KeyError:
        print(f"Patient class not found in {ontology_name} ontology.")


    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Load the mapping file
    with open(f'mappings/{domain}_{ontology_name}.attr.values', 'r') as f:
        mappings = {}
        for line in f.readlines():
            attr, prop = line.strip().split(',')
            mappings[attr] = prop
        # print(mappings)

    # Function to create an individual from a CSV row
    def create_individual(row, idx):
        individual = Patient(f"Patient_{idx}")
        # individual = Patient(f"{ontology_name}_Patient_{idx}")

        print(ontology_name)
        print(individual)
        for column, value in row.items():
            if column in mappings:
                prop_name = mappings[column]
                prop = onto.search_one(iri=f"*#{prop_name}")

                if prop is not None:
                    if isinstance(prop, DataPropertyClass):
                        setattr(individual, prop_name, value)
                    elif isinstance(prop, ObjectPropertyClass):
                        # create or fetch object individual based on the value, if needed
                        if domain == 'heart':
                            if prop_name == "hasGender":
                                gender_individual = onto.search_one(iri=f"*#{value}")
                                if gender_individual:
                                    setattr(individual, prop_name, gender_individual)

    # Iterate over the rows in the dataframe
    for idx, row in df.iterrows():
        create_individual(row, idx)

    # Save the modified ontology
    onto.save(file=f"{output_dir}/{domain}_{ontology_name}.owl")

# Directory paths
csv_dir = "data/"
output_dir = "knowledge_graphs"

# Ontology names
# ontology_names = ["heart_small", "extended", "snomed"]
ontology_names = ["snomed"]
# domain = 'heart'
domain = 'kidney'

# Populate and save ontologies for each ontology name
for ontology_name in ontology_names:
    if domain == 'heart':
        csv_file = f"{csv_dir}/{domain}_dataset.csv"
    elif domain == 'kidney':
        csv_file = f"{csv_dir}/{domain}/{domain}_disease.csv"
    print(ontology_name)
    populate_ontology_from_csv(ontology_name, csv_file, output_dir)

print("Ontologies updated successfully for all ontologies!")
