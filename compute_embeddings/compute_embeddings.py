from pykeen.pipeline import pipeline


def generate_pykeen_embeddings(model_name, train_entities, kg_path):
    result = pipeline(
        model=model_name,
        dataset='YourDataset',  # Replace with your dataset if available
        training=kg_path,
        testing=kg_path,
    )

    entity_embeddings = result.model.entity_representations[0]().cpu().detach().numpy()
    entity_to_embedding = {entity: entity_embeddings[i] for i, entity in enumerate(train_entities)}

    return entity_to_embedding


compgcn_train_embeddings = generate_pykeen_embeddings('CompGCN', train_entities, kg_path)
compgcn_test_embeddings = generate_pykeen_embeddings('CompGCN', test_entities, kg_path)

transh_train_embeddings = generate_pykeen_embeddings('TransH', train_entities, kg_path)
transh_test_embeddings = generate_pykeen_embeddings('TransH', test_entities, kg_path)

distmult_train_embeddings = generate_pykeen_embeddings('DistMult', train_entities, kg_path)
distmult_test_embeddings = generate_pykeen_embeddings('DistMult', test_entities, kg_path)