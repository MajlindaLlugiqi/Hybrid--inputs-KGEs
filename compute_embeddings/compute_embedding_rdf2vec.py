import os
import json
from rdflib import Graph
from gensim.models import Word2Vec
import random

# Get the current working directory
current_path = os.getcwd()
domain = 'kidney'
# Parameters
# onto_names = ["heart_small", "heart_extended", "heart_snomed"]  # List of ontologies
onto_names = ["kidney_snomed"]
vector_sizes = [64, 128, 100]  # Desired vector sizes for embeddings
base_knowledge_graphs_dir = os.path.join(current_path, f"knowledge_graphs/{domain}")

# params_1 = {
#     "heart_small": {"depth": 4, "walks_per_node": 10, "window": 5},
#     "heart_extended": {"depth": 6, "walks_per_node": 20, "window": 10},
#     "heart_snomed": {"depth": 5, "walks_per_node": 15, "window": 7}
# }
# params_2 = {
#     "heart_small": {"depth": 4, "walks_per_node": 100, "window": 5},
#     "heart_extended": {"depth": 6, "walks_per_node": 150, "window": 10},
#     "heart_snomed": {"depth": 5, "walks_per_node": 100, "window": 7}
# }

# #TODO: CHECK FOR PARAMETERS
params_1 = {
    "kidney_snomed": {"depth": 5, "walks_per_node": 15, "window": 7}
}
params_2 = {
    "kidney_snomed": {"depth": 10, "walks_per_node": 100, "window": 7}
}

# Function to generate walks
def generate_walks(graph, depth, walks_per_node, onto_name=""):
    walks = []
    condition = None
    if "heart_" in onto_name:
        condition = "hasHeartDisease"
    elif "hepatitis_" in onto_name:
        condition = "hasLifeState"
    elif 'kidney_' in onto_name:
        condition = "hasKidneyDisease"

    for s in graph.subjects():
        for _ in range(walks_per_node):
            walk = [str(s)]
            current_node = s
            for _ in range(depth):
                if condition:
                    # print("lala")
                    neighbors = [o for p in graph.predicates(subject=current_node) if condition not in str(p) or 'Patient_' not in str(current_node) for o in graph.objects(subject=current_node, predicate=p)]
                else:
                    neighbors = [o for p in graph.predicates(subject=current_node) for o in graph.objects(subject=current_node, predicate=p)]

                if not neighbors:
                    break

                # next_node = str(neighbors[0])
                next_node = str(random.choice(neighbors))
                walk.append(next_node)
                current_node = next_node

            walks.append(walk)

    return walks

# Process each ontology
# Process each ontology and parameter set
for param_set_name, param_set in [("params_1", params_1), ("params_2", params_2)]:
    for ontology in onto_names:
        rdf_file_path = os.path.join(base_knowledge_graphs_dir, f'{ontology}.owl')
        g = Graph()
        g.parse(rdf_file_path, format='xml')

        params = param_set[ontology]

        walks = generate_walks(g, depth=params["depth"], walks_per_node=params["walks_per_node"], onto_name=ontology)

        for vector_size in vector_sizes:
            embeddings_dir = os.path.join(current_path, "data", ontology, param_set_name, f"vector_size_{vector_size}")
            if not os.path.exists(embeddings_dir):
                os.makedirs(embeddings_dir)

            model = Word2Vec(sentences=walks, vector_size=vector_size, window=params["window"], min_count=1, sg=1)
            embeddings_dict = {word: model.wv[word].tolist() for word in model.wv.index_to_key}

            embeddings_file_path = os.path.join(embeddings_dir, f'x_rdf2vec_{ontology}_v{vector_size}.json')
            with open(embeddings_file_path, 'w') as f:
                json.dump(embeddings_dict, f, indent=4)

            print(f"RDF2Vec embeddings for ontology {ontology} with vector size {vector_size} under {param_set_name} have been stored to {embeddings_file_path}")