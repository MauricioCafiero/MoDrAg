from sentence_transformers import SentenceTransformer
import numpy as np

# Tool Descriptions
tool_descriptions = {
    # modrag_protein_functions.py
    'uniprot_node': 'Find the UNIPROT Accession codes (IDs) for this DNA Gyrase. Report the organisms and gene names.',
    'listbioactives_node': 'Find the Chembl IDs for the protein with UNIPROT Accession code P091H7 and \
report the number of bioactive molecules for each Chembl ID.',
    'getbioactives_node': 'Find all of the bioactives molecule SMILES and IC50s for the Chembl ID CHEMBL8999.',
    'predict_node': 'Predict the IC50 for dopamine based on the chembl ID chembl908564.',
    'gpt_node': 'Use the Chembl dataset chembl98775 to generate novel molecules; do this by trainig a GPT.',
    'pdb_node': 'Find the protein sequence and and ligands (small molecules) present in the crystal structure \
represented by the PDB ID 6YT5.',
    'find_node': 'Find all the PDB IDs in the protein databank for DNA gyrase.',
    'docking_node': 'Find the docking scores for c1cccc1 and CCCCC=O in DNA gyrase. Dock c1cccc1 and CCCCC=O in the protein DNA gyrase.',
    'target_node': 'Find possible protein targets for the disease phenylketonuria.',
    # modrag_property_functions.py
    'substitution_node': 'Generate analogues of O=C([O-])CCc1ccc(O)cc1 by substitution of different groups. Report the QED values as well.',
    'lipinski_node': 'Find the Lipinski properties for c1cccc1 and CCCCC=O; report the\
QED, LogP, number of hydrogen bond donors and acceptors, molar mass, and polar surface area.',
    'pharmfeature_node': 'Find the similarity in the pharmacophores between c1cccc1 and CCCCC=O. \
Find the similarity in the pharmacophores between ibuprofen and aspirin.',
    
    # modrag_molecule_functions.py
    'name_node': 'Find the name of this molecule c1cc(O)ccc1',
    'smiles_node': 'Finds SMILES strings for cyclohexane and aspirin',
    'related_node': 'Find molecules similar to c1cc(O)ccc1',
    'structure_node': 'Find the structure of the molecule with SMILES string c1cc(O)ccc1, or the name Aspirin.',
    # modrag_task_graphs.py
    'get_actives_for_protein': 'Find the bioactive molecules for the protein DNA gyrase.',
    'get_predictions_for_protein': 'Predict the IC50 value for c1cc(O)ccc1 in the protein DNA gyrase.',
    'dock_from_names': 'Dock 1,3-butadiene and levadopa in the protein MAOB.'
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
    scores = embed_model.similarity(query_embedding, document_embeddings)
    best_tools = []
    best_scores = []
    for i in range(6):
        try:
            best_idx = np.argmax(scores[0])
            this_tool = tool_descriptions_keys[best_idx]
            best_scores.append(scores[0][best_idx].item())
            scores[0][best_idx] = -1
        except:
            this_tool = 'None'
            best_scores.append(999)
        best_tools.append(this_tool)

    print(f"Chosen tool is: {best_tools[0]} for query: {query}")
    print(f"Second choSice is: {best_tools[1]}")
    print(f"Third choice is: {best_tools[2]}")
    return best_tools, best_scores

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