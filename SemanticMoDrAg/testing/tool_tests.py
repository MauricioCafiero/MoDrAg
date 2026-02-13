from modrag_molecule_functions import *
from modrag_property_functions import *
from modrag_protein_functions import *
from modrag_task_graphs import *

tools_list = [
    # From modrag_molecule_functions.py
    name_node,
    smiles_node,
    related_node,
    structure_node,
    # From modrag_property_functions.py
    substitution_node,
    lipinski_node,
    pharmfeature_node,
    # From modrag_protein_functions.py
    uniprot_node,
    listbioactives_node,
    getbioactives_node,
    predict_node,
    gpt_node,
    pdb_node,
    find_node,
    docking_node,
    target_node,
    # From modrag_task_graphs.py
    get_actives_for_protein,
    get_predictions_for_protein,    
    dock_from_names
]

prompts_list = [
    # From modrag_molecule_functions.py
    [["CCO", "c1ccccc1"]],  # name_node(smiles_list)
    [["aspirin", "caffeine"]],  # smiles_node(names_list)
    [["CCO"]],  # related_node(smiles_list)
    [["CCO", "c1ccccc1"]],  # structure_node(smiles_list)
    # From modrag_property_functions.py
    [["c1cc(O)ccc1"]],  # substitution_node(smiles_list)
    [["CCO", "c1ccccc1"]],  # lipinski_node(smiles_list)
    ["CCO", ["c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]],  # pharmfeature_node(known_smiles, test_smiles)
    # From modrag_protein_functions.py
    [["DNA gyrase"], False],  # uniprot_node(protein_names, human_flag)
    [["P0AES4"]],  # listbioactives_node(up_ids_list)
    [["CHEMBL1909140"]],  # getbioactives_node(chembl_ids_list)
    [["CCO"], "CHEMBL2039"],  # predict_node(smiles_list_in, chembl_id)
    ["CHEMBL2039"],  # gpt_node(chembl_id)
    [["6YT5"]],  # pdb_node(test_pdb_list)
    [["DNA gyrase"]],  # find_node(test_protein_list)
    [["CCO"], "EGFR"],  # docking_node(smiles_list, query_protein)
    [["phenylketonuria"]],  # target_node(search_descriptors)
    # From modrag_task_graphs.py
    ["MAOB"],  # get_actives_for_protein(query_protein)
    [["CCO"], "MAOB"],  # get_predictions_for_protein(smiles_list, query_protein)
    [["aspirin", "caffeine"], "DRD2"],  # dock_from_names(names_list, protein)
]