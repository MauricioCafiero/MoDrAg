from sentence_transformers import SentenceTransformer
import numpy as np

# Tool Descriptions
tool_descriptions = {
    'uniprot_node': 'Find the UNIPROT Accession codes (IDs) for this DNA Gyrase. Report the organisms and gene names.',
    'listbioactives_node': 'Find the Chembl IDs for the protein with UNIPROT Accession code P091H7 and report the number of bioactive molecules for each Chembl ID.',
    'getbioactives_node': 'Find all of the bioactives molecule SMILES and IC50s for the Chembl ID CHEMBL8999.',
    # Add other tools similarly
}

tool_descriptions_keys = list(tool_descriptions.keys())
tool_descriptions_values = list(tool_descriptions.values())

def start_embedding(tool_descriptions_values):
    """
    Starts the embedding model and encodes the tool descriptions.
    """
    embed_model = SentenceTransformer("google/embeddinggemma-300m")
    document_embeddings = embed_model.encode(tool_descriptions_values)
    return document_embeddings, embed_model

def rank_tools_for_query(query, document_embeddings, embed_model):
    """
    Accepts a query string, computes its embedding, and ranks tool descriptions based on similarity.

    Args:
        query: str, the input query from the user.
        document_embeddings: np.array, the precomputed embeddings of all tool descriptions.
        embed_model: SentenceTransformer object.

    Returns:
        Ranked list of tools.
    """
    query_embedding = embed_model.encode(query)
    scores = np.dot(query_embedding, document_embeddings.T)
    ranking = np.argsort(scores)[::-1]  # Sort scores in descending order
    ranked_tools = [tool_descriptions_keys[idx] for idx in ranking]
    return ranked_tools, scores

if __name__ == "__main__":
    # Initialize embedding model and compute tool embeddings
    document_embeddings, embed_model = start_embedding(tool_descriptions_values)

    # Example testing: Accept a query from the user
    user_query = input("Enter your query: ")
    ranked_tools, scores = rank_tools_for_query(user_query, document_embeddings, embed_model)

    # Display the results
    print("\nRanked Tools (Best Matches First):")
    for i, (tool, score) in enumerate(zip(ranked_tools, scores)):
        print(f"{i + 1}. {tool} (Score: {score:.4f})")