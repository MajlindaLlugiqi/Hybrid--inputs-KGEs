import os
import networkx as nx
from rdflib import Graph, RDF
from sklearn.preprocessing import LabelEncoder
from node2vec import Node2Vec
import json

# Parameters
current_path = os.getcwd()
# domain = "kidney"
domain = "heart"
onto_names = ["heart_small", "heart_extended", "heart_snomed"]  # List of ontologies
# onto_names = ["kidney_snomed"]  # List of ontologies

vector_sizes = [64, 128, 100]   # Desired vector sizes for embeddings
base_knowledge_graphs_dir = os.path.join(current_path, f"knowledge_graphs/{domain}")

# Node2Vec parameter settings for each KG
params_1 = {
    "heart_small": {"dimensions": 64, "walk_length": 40, "num_walks": 10, "window": 5},
    "heart_extended": {"dimensions": 128, "walk_length": 60, "num_walks": 20, "window": 10},
    "heart_snomed": {"dimensions": 100, "walk_length": 50, "num_walks": 15, "window": 7}
}

params_2 = {
    "heart_small": {"dimensions": 64, "walk_length": 40, "num_walks": 200, "window": 5},
    "heart_extended": {"dimensions": 128, "walk_length": 60, "num_walks": 200, "window": 10},
    "heart_snomed": {"dimensions": 100, "walk_length": 50, "num_walks": 200, "window": 7}
}
# #TODO: CHECK FOR PARAMETERS
# params_1 = {
#     "kidney_snomed":{"dimensions": 100, "walk_length": 50, "num_walks": 15, "window": 8},
# }
#
# params_2 = {
#     "kidney_snomed": {"dimensions": 100, "walk_length": 50, "num_walks": 200, "window": 7}
# }

# Function to create a graph from RDF data_heart_303
def create_graph(onto_name, g):
    G = nx.Graph()

    le = LabelEncoder()

    for s, p, o in g:
        # Apply condition based on ontology name
        if "heart_" in onto_name and 'Patient_' in str(s) and 'hasHeartDisease' in str(p):
            continue
        elif "hepatitis_" in onto_name and 'Patient_' in str(s) and 'hasLifeState' in str(p):
            continue
        elif "kidney_" in onto_name and 'Patient_' in str(s) and 'hasKidneyDisease' in str(p):
            continue

        G.add_node(str(s))
        G.add_node(str(o))
        G.add_edge(str(s), str(o), property=str(p))

        if not (o, RDF.type, None) in g:
            G.nodes[str(s)][str(p)] = le.fit_transform([str(o)])[0]

    return G

# Process each ontology

# Process each ontology and parameter set
for param_set_name, param_set in [("params_1", params_1), ("params_2", params_2)]:
    for ontology in onto_names:
        # Parse RDF data into a graph
        # if "heart_" in ontology:
        #     rdf_file_path = os.path.join(base_knowledge_graphs_dir, f'{ontology}.owl')
        # elif 'kidney_' in ontology:
        #     rdf_file_path = os.path.join(base_knowledge_graphs_dir, f'{ontology}.owl')
        rdf_file_path = os.path.join(base_knowledge_graphs_dir, f'{ontology}.owl')
        g = Graph().parse(rdf_file_path, format='xml')
        G = create_graph(ontology, g)

        current_params = param_set[ontology]
        for vector_size in vector_sizes:
            embeddings_dir = os.path.join(current_path, "data", ontology, param_set_name, f"vector_size_{vector_size}")
            if not os.path.exists(embeddings_dir):
                os.makedirs(embeddings_dir)
            node2vec = Node2Vec(G, dimensions=vector_size, walk_length=current_params["walk_length"],
                                num_walks=current_params["num_walks"], workers=4)
            model = node2vec.fit(window=current_params["window"], min_count=1, batch_words=4)
            embeddings = {node: model.wv[node].tolist() for node in G.nodes()}
            embeddings_file_path = os.path.join(embeddings_dir, f'x_node2vec_{ontology}_v{vector_size}.json')
            with open(embeddings_file_path, 'w') as f:
                json.dump(embeddings, f)
            print(
                f"Node2Vec embeddings for ontology {ontology} with vector size {vector_size} under {param_set_name} have been saved to {embeddings_file_path}")

            # if vector_size == current_params["dimensions"]:


