import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from pykeen.pipeline import pipeline
from pykeen.models import TransH, DistMult
from pykeen.triples import TriplesFactory
import rdflib
# import networkx as nx
# from node2vec import Node2Vec as N2V
# from pyrdf2vec.graphs import KG
# from pyrdf2vec.walkers import RandomWalker
# from pyrdf2vec.embedders import RDF2Vec
# from gensim.models import Word2Vec
# import torch

# Set random seeds for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# ontologies = ["heart_small", "heart_extended", "heart_snomed"]
domain="kidney"
ontologies = ["kidney_snomed"]

# Set paths and parameters
current_path = os.getcwd()
base_data_dir = os.path.join(current_path, "data")
kg_base_dir = os.path.join(current_path, "knowledge_graphs")

# dataset_path = os.path.join(base_data_dir, 'heart_dataset.csv')

dataset_path = os.path.join(base_data_dir, 'kidney_disease.csv')

uri_mapping = {
    "heart_small": "http://www.semanticweb.org/heart_ontology",
    "heart_extended": "http://www.bmi.utah.edu/ontologies/2015/hfo",
    "heart_snomed": "owlapi:ontology",
    "kidney_snomed": "http://snomed.info/id/"
}
# vector_sizes = [64, 100]
vector_sizes = [64, 128, 100]

def load_and_preprocess_dataset(df):
    categorical_columns = []
    numerical_columns = []
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = pd.Categorical(df[column]).codes
            categorical_columns.append(column)
        else:
            numerical_columns.append(column)

    numerical_imputer = SimpleImputer(strategy='mean')
    df[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns])

    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

    y = df['label']
    X_dataset = df.drop('label', axis=1)
    scaler = StandardScaler()
    X_dataset_scaled = scaler.fit_transform(X_dataset)
    return X_dataset_scaled, y

# df = pd.read_csv(dataset_path, delimiter=';')
df = pd.read_csv(dataset_path, delimiter=',')
X_dataset, y = load_and_preprocess_dataset(df)

def load_ontology(ontology_path):
    if not os.path.exists(ontology_path):
        raise FileNotFoundError(f"Ontology file not found: {ontology_path}")

    g = rdflib.Graph()
    g.parse(ontology_path, format='xml')

    triples = []
    for subj, pred, obj in g:
        triples.append((str(subj), str(pred), str(obj)))

    print(f"Total triples loaded: {len(triples)}")

    triples = np.array(triples)
    return triples

def pykeen_embeddings(model, dimension, triples_factory_train, triples_factory_test):
    train_result = pipeline(
        model=model,
        training=triples_factory_train,
        testing=triples_factory_test,
        training_loop='slcwa',
        model_kwargs=dict(embedding_dim=dimension),
        training_kwargs=dict(num_epochs=50, batch_size=128),
    )

    entity_embeddings = train_result.model.entity_representations[0](indices=None).detach().numpy()
    entity_to_id = triples_factory_train.entity_to_id

    return entity_embeddings, entity_to_id
#
# def rdf2vec_embeddings(triples, dimension):
#     kg = KG(location="temp_kg.nt", file_type="ntriples")
#     kg.add_triples(triples.tolist())
#     walker = RandomWalker(2, 4)
#     embedder = RDF2Vec(Word2Vec(size=dimension, sg=1, hs=0, negative=5, min_count=1, workers=4))
#     embeddings = embedder.fit_transform(kg, walker)
#     entity_to_id = {entity: idx for idx, entity in enumerate(kg._entities)}
#     return embeddings, entity_to_id

# def rdf2vec_embeddings(triples, params, dimension):
#     kg = KG(location="temp_kg.nt", file_type="ntriples")
#     kg.add_triples(triples.tolist())
#     walker = RandomWalker(params['depth'], params['walks_per_node'])
#     embedder = RDF2Vec(Word2Vec(vector_size=dimension, window=params['window'], sg=1, hs=0, negative=5, min_count=1, workers=4))
#     embeddings = embedder.fit_transform(kg, walker)
#     entity_to_id = {entity: idx for idx, entity in enumerate(kg._entities)}
#     return embeddings, entity_to_id
#
# # Parameters for RDF2Vec
# # params_1 = {
# #     "kidney_snomed": {"depth": 5, "walks_per_node": 15, "window": 7}
# # }
# params_kidney = {
#     "kidney_snomed": {"depth": 10, "walks_per_node": 100, "window": 7}
# }
# params_heart_rdf2vec = {
#     "heart_small": {"depth": 4, "walks_per_node": 100, "window": 5},
#     "heart_extended": {"depth": 6, "walks_per_node": 150, "window": 10},
#     "heart_snomed": {"depth": 5, "walks_per_node": 100, "window": 7}
# }
#
# params_heart_node2vec = {
#     "heart_small": {"dimensions": 64, "walk_length": 40, "num_walks": 200, "window": 5},
#     "heart_extended": {"dimensions": 128, "walk_length": 60, "num_walks": 200, "window": 10},
#     "heart_snomed": {"dimensions": 100, "walk_length": 50, "num_walks": 200, "window": 7}
# }
#
# #
# def node2vec_embeddings(triples, params, dimension):
#     G = nx.Graph()
#     for subj, pred, obj in triples:
#         G.add_edge(subj, obj)
#     n2v = N2V(G, dimensions=dimension, walk_length=10, num_walks=100, workers=4)
#     model = n2v.fit(window=10, min_count=1, batch_words=4)
#     embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
#     entity_to_id = {str(node): idx for idx, node in enumerate(G.nodes())}
#     return embeddings, entity_to_id


def create_model(input_dim):
    model = Sequential()
    model.add(Dense(32, input_dim=input_dim, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def initialize_models_with_params():
    models_with_params = []
    knn_params = [{'n_neighbors': k} for k in range(5, 51, 5)]
    models_with_params.extend([('KNN', KNeighborsClassifier(), param) for param in knn_params])
    svm_params = [{'C': C, 'kernel': 'rbf', 'probability': True} for C in np.arange(0.9, 2.5, 0.1)]
    models_with_params.extend([('SVM', SVC(random_state=42), param) for param in svm_params])
    xgb_params = [{'learning_rate': lr} for lr in np.arange(0.01, 0.21, 0.01)]
    models_with_params.extend([('XGBoost', GradientBoostingClassifier(random_state=42), param) for param in xgb_params])
    models_with_params.extend([('NN', None, {'input_dim': None})])
    return models_with_params

def add_distance_features(dataframe, embeddings, entity_to_id, feature_prefix, y_train):
    default_embedding = np.zeros(embeddings.shape[1])
    valid_indices = dataframe['patient_key'].apply(lambda k: k in entity_to_id)
    valid_dataframe = dataframe[valid_indices].reset_index(drop=True)
    if valid_dataframe.empty:
        print(f"Warning: Valid dataframe is empty for {feature_prefix}")
        return valid_dataframe

    y_train_filtered = y_train[valid_indices.values].reset_index(drop=True)

    valid_embeddings = [embeddings[entity_to_id.get(str(k), -1)] if str(k) in entity_to_id else default_embedding for k
                        in valid_dataframe['patient_key']]

    class_yes_embeddings = np.mean([emb for emb, label in zip(valid_embeddings, y_train_filtered) if label == 1],
                                   axis=0)
    class_no_embeddings = np.mean([emb for emb, label in zip(valid_embeddings, y_train_filtered) if label == 0], axis=0)

    valid_dataframe[f'{feature_prefix}_distance_to_class_yes'] = [distance.euclidean(emb, class_yes_embeddings) for emb
                                                                  in valid_embeddings]
    valid_dataframe[f'{feature_prefix}_distance_to_class_no'] = [distance.euclidean(emb, class_no_embeddings) for emb in
                                                                 valid_embeddings]

    return valid_dataframe

def map_indices_to_uris(indices, base_uri):
    return [f"{base_uri}#Patient_{i}" for i in indices]

models_with_params = initialize_models_with_params()

def create_combined_triples_factory(triples, train_uris, test_uris):
    combined_uris = np.unique(np.concatenate((train_uris, test_uris)))
    combined_triples = triples[np.isin(triples[:, 0], combined_uris)]
    combined_triples_factory = TriplesFactory.from_labeled_triples(combined_triples)
    return combined_triples_factory

def filter_triples_test(triples, patient_prefix, predicate_to_match):
    filtered_triples = []
    for subj, pred, obj in triples:
        if not (subj.startswith(patient_prefix) and pred == predicate_to_match):
            filtered_triples.append((subj, pred, obj))
    return np.array(filtered_triples, dtype=object)

def train_and_evaluate(models_with_params, Xs, y, ontology, triples, base_uri, checkpoint_dir, vector_size):
    results_list = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model, param in models_with_params:
        for X_name, X in Xs.items():
            accuracies, f1s, precisions, recalls, f2s = [], [], [], [], []
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                train_uris = map_indices_to_uris(y_train.index, base_uri)
                test_uris = map_indices_to_uris(y_test.index, base_uri)
                combined_triples_factory = create_combined_triples_factory(triples, train_uris, test_uris)
                entity_to_id = combined_triples_factory.entity_to_id

                train_triples = combined_triples_factory.triples[
                    np.isin(combined_triples_factory.triples[:, 0], train_uris)]
                test_triples = combined_triples_factory.triples[
                    np.isin(combined_triples_factory.triples[:, 0], test_uris)]
                train_triples_factory = TriplesFactory.from_labeled_triples(train_triples)

                patient_prefix = f"{base_uri}#Patient_"
                # predicate_to_match = f"{base_uri}#hasHeartDisease"
                predicate_to_match = f"{base_uri}#hasKidneyDisease"

                test_triples_factory = TriplesFactory.from_labeled_triples(test_triples)
                filtered_test_triples = filter_triples_test(test_triples, patient_prefix, predicate_to_match)
                filtered_test_triples_factory = TriplesFactory.from_labeled_triples(filtered_test_triples)

                embeddings_transh_train, entity_to_id_transh_train = pykeen_embeddings(
                    TransH, dimension=vector_size, triples_factory_train=train_triples_factory,
                    triples_factory_test=filtered_test_triples_factory)
                embeddings_distmult_train, entity_to_id_distmult_train = pykeen_embeddings(
                    DistMult, dimension=vector_size, triples_factory_train=train_triples_factory,
                    triples_factory_test=filtered_test_triples_factory)

                embeddings_transh_test, entity_to_id_transh_test = pykeen_embeddings(
                    TransH, dimension=vector_size, triples_factory_train=test_triples_factory,
                    triples_factory_test=filtered_test_triples_factory)
                embeddings_distmult_test, entity_to_id_distmult_test = pykeen_embeddings(
                    DistMult, dimension=vector_size, triples_factory_train=test_triples_factory,
                    triples_factory_test=filtered_test_triples_factory)

                # # Generate RDF2Vec and Node2Vec embeddings
                # embeddings_rdf2vec_train, entity_to_id_rdf2vec_train = rdf2vec_embeddings(train_triples, param=params_heart_rdf2vec, dimension=vector_size)
                # embeddings_rdf2vec_test, entity_to_id_rdf2vec_test = rdf2vec_embeddings(test_triples, param=params_heart_rdf2vec, dimension=vector_size)
                # embeddings_node2vec_train, entity_to_id_node2vec_train = node2vec_embeddings(train_triples, params_heart_node2vec,dimension=vector_size)
                # embeddings_node2vec_test, entity_to_id_node2vec_test = node2vec_embeddings(test_triples, param=params_heart_node2vec, dimension=128)

                embedding_methods = {
                    'TransH': (embeddings_transh_train, entity_to_id_transh_train, embeddings_transh_test,
                               entity_to_id_transh_test),
                    'DistMult': (embeddings_distmult_train, entity_to_id_distmult_train, embeddings_distmult_test,
                                 entity_to_id_distmult_test),
                    # 'RDF2Vec': (embeddings_rdf2vec_train, entity_to_id_rdf2vec_train, embeddings_rdf2vec_test,
                    #             entity_to_id_rdf2vec_test),
                    # 'Node2Vec': (embeddings_node2vec_train, entity_to_id_node2vec_train, embeddings_node2vec_test,
                    #              entity_to_id_node2vec_test)
                }

            for method_name, (embeddings_train, entity_to_id_train, embeddings_test, entity_to_id_test) in embedding_methods.items():
                    train_indices = [entity_to_id_train[uri] for uri in train_uris if uri in entity_to_id_train]
                    test_indices = [entity_to_id_test[uri] for uri in test_uris if uri in entity_to_id_test]

                    train_embeddings = embeddings_train[train_indices]
                    test_embeddings = embeddings_test[test_indices]

                    valid_train_indices = [i for i, uri in zip(train_index, train_uris) if uri in entity_to_id]
                    valid_test_indices = [i for i, uri in zip(test_index, test_uris) if uri in entity_to_id]

                    X_train_filtered = X[valid_train_indices]
                    X_test_filtered = X[valid_test_indices]
                    y_train_filtered = y.iloc[valid_train_indices]
                    y_test_filtered = y.iloc[valid_test_indices]

                    X_train_combined = np.concatenate((X_train_filtered, train_embeddings), axis=1)
                    X_test_combined = np.concatenate((X_test_filtered, test_embeddings), axis=1)

                    df_train = pd.DataFrame(X_train_filtered, columns=df.columns[:-1])
                    df_test = pd.DataFrame(X_test_filtered, columns=df.columns[:-1])

                    df_train['patient_key'] = map_indices_to_uris(valid_train_indices, base_uri)
                    df_test['patient_key'] = map_indices_to_uris(valid_test_indices, base_uri)

                    df_train = add_distance_features(df_train, embeddings_train, entity_to_id_train, method_name, y_train_filtered)
                    df_test = add_distance_features(df_test, embeddings_test, entity_to_id_test, method_name, y_test_filtered)

                    X_train_fe = df_train[[f'{method_name}_distance_to_class_yes', f'{method_name}_distance_to_class_no']].values
                    X_test_fe = df_test[[f'{method_name}_distance_to_class_yes', f'{method_name}_distance_to_class_no']].values

                    X_train_combined_fe = np.concatenate((X_train_combined, X_train_fe), axis=1)
                    X_test_combined_fe = np.concatenate((X_test_combined, X_test_fe), axis=1)

                    scenarios = {
                        'Baseline': (X_train, X_test),
                        f'{method_name} Only': (train_embeddings, test_embeddings),
                        f'{method_name} + Tabular': (X_train_combined, X_test_combined),
                        f'{method_name} FE Only': (X_train_fe, X_test_fe),
                        f'{method_name} + Tabular + FE': (X_train_combined_fe, X_test_combined_fe),
                    }

                    for scenario_name, (X_train_final, X_test_final) in scenarios.items():
                        if model_name == 'NN':
                            model_instance = create_model(X_train_final.shape[1])
                            model_instance.fit(X_train_final, y_train_filtered, epochs=50, verbose=0)
                        else:
                            model_instance = model
                            model_instance.set_params(**param)
                            model_instance.fit(X_train_final, y_train_filtered)

                        y_pred = model_instance.predict(X_test_final)
                        if model_name == 'NN':
                            y_pred = (y_pred > 0.5).astype(int).flatten()

                        accuracy = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, zero_division=0)
                        recall = recall_score(y_test, y_pred, zero_division=0)

                        if precision + recall == 0:
                            f2 = 0.0
                        else:
                            f2 = (5 * precision * recall) / (4 * precision + recall)

                        accuracies.append(accuracy)
                        f1s.append(f1)
                        precisions.append(precision)
                        recalls.append(recall)
                        f2s.append(f2)
                        results_list.append({
                            'Model': model_name,
                            'Settings': str(param),
                            'Embedding Method': method_name,
                            'Scenario': scenario_name,
                            'Accuracy': np.mean(accuracies),
                            'F1_Score': np.mean(f1s),
                            'Precision': np.mean(precisions),
                            'Recall': np.mean(recalls),
                            'F2_Score': np.mean(f2s)
                        })
                    # Save results after each fold
                    results_df = pd.DataFrame(results_list)
                    results_df.to_csv(os.path.join(checkpoint_dir, f'{ontology}_intermediate_results.csv'),index=False)

    return results_list

all_results_list = []

for ontology in ontologies:
    ontology_path = os.path.join(kg_base_dir, f'{domain}/{ontology}.owl')
    base_uri = uri_mapping[ontology]
    triples = load_ontology(ontology_path)

    for vector_size in vector_sizes:
        triples_factory = TriplesFactory.from_labeled_triples(triples)
        checkpoint_dir = os.path.join('results_fine_tuned', domain, str(vector_size), 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        results_list = train_and_evaluate(models_with_params, {'Original Dataset': X_dataset}, y, ontology, triples,
                                          base_uri, checkpoint_dir, vector_size)
        all_results_list.extend(results_list)
        results_df = pd.DataFrame(results_list)
        results_dir = os.path.join('results_fine_tuned', domain, str(vector_size))
        os.makedirs(results_dir, exist_ok=True)
        results_file_path = os.path.join(results_dir, f'{ontology}/evaluation_results.csv')
        if not os.path.exists(results_file_path):
            with open(results_file_path, 'w') as f:
                pass  # Create the file if it does not exist
        results_df.to_csv(os.path.join(results_dir, f'{ontology}/evaluation_results.csv'), index=False)
    #
    # triples_factory = TriplesFactory.from_labeled_triples(triples)
    # checkpoint_dir = os.path.join('results_fine_tuned', 'heart', '128', 'checkpoints')
    # os.makedirs(checkpoint_dir, exist_ok=True)
    # results_list = train_and_evaluate(models_with_params, {'Original Dataset': X_dataset}, y, ontology, triples,  
    #                                   base_uri, checkpoint_dir, vector_size)
    # all_results_list.extend(results_list)
    # results_df = pd.DataFrame(results_list)
    # results_dir = os.path.join('results_fine_tuned', 'heart', vector_size)
    # os.makedirs(results_dir, exist_ok=True)
    # results_df.to_csv(os.path.join(results_dir, f'{ontology}/evaluation_results.csv'), index=False)

# Save all results
#         all_results_df = pd.DataFrame(all_results_list)
#         all_results_df.to_csv(os.path.join('results_fine_tuned', 'heart', vector_size, 'all_ontologies_evaluation_results.csv'), index=False)
