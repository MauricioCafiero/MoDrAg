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
    #gpt_node,
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
    [["P27338"]],  # listbioactives_node(up_ids_list)
    [["CHEMBL2039"]],  # getbioactives_node(chembl_ids_list)
    [["[NH3+]CCc1ccc(O)cc1"], "CHEMBL2039"],  # predict_node(smiles_list_in, chembl_id)
    #["CHEMBL2039"],  # gpt_node(chembl_id)
    [["6YT5"]],  # pdb_node(test_pdb_list)
    [["DNA gyrase"]],  # find_node(test_protein_list)
    [["[NH3+]CCc1ccc(O)cc1"], "DRD2"],  # docking_node(smiles_list, query_protein)
    [["phenylketonuria"]],  # target_node(search_descriptors)
    # From modrag_task_graphs.py
    ["MAOB"],  # get_actives_for_protein(query_protein)
    [["CCO"], "MAOB"],  # get_predictions_for_protein(smiles_list, query_protein)
    [["aspirin", "caffeine"], "DRD2"],  # dock_from_names(names_list, protein)
]

truths_list = [
    # From modrag_molecule_functions.py
    ['ethanol', 'benzene'],  # name_node
    ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'],  # smiles_node
    [['CCO', '[2H]C([2H])([2H])C([2H])([2H])O[2H]', '[2H]OCC', 'CCO.[Tl]', '[2H]C([2H])(C)O']],  # related_node
    2,  # structure_node
    # From modrag_property_functions.py
    [['c1ccc(Cl)cc1', 'c1ccc(F)cc1', 'c1ccc(C)cc1']],  # substitution_node
    [[0.4068079656553945, 46.069], [0.4426283718993647, 78.11399999999999]],  # lipinski_node
    [0.0, 0.4698023538175749],  # pharmfeature_node
    # From modrag_protein_functions.py
    [['Q5SHZ4', 'P0AES4', 'Q08582', 'P33012', 'P0AES6']],  # uniprot_node
    [[5518, 51]],  # listbioactives_node
    50,  # getbioactives_node
    10995.537239000963,  # predict_node
    #None,  # gpt_node (commented out)
    12,  # pdb_node
    [['3QTD', '1VL4', '1VPB', '5NJ5', '3NUH']],  # find_node
    [-6.3],  # docking_node
    [['PAH', 'DNAJC12', 'NSUN2', 'COL1A1', 'PI4KA']],  # target_node
    # From modrag_task_graphs.py
    50,  # get_actives_for_protein
    15228.239232991393,  # get_predictions_for_protein
    [-6.4, -5.6],  # dock_from_names
]

def test_results(result, truth, tool_name):
    """Test tool results against expected truth values."""
    try:
        if isinstance(truth, list):
            if len(truth) == 0:
                return True
            if isinstance(result, tuple):
                result_data = result[0]
            else:
                result_data = result
            if isinstance(truth[0], list):
                if isinstance(result_data[0], list):
                    matches = sum(1 for t in truth[0] if t in result_data[0][:len(truth[0])*2])
                    passed = matches >= len(truth[0]) * 0.8
                    print(f"✓ {tool_name}: Matched {matches}/{len(truth[0])} items" if passed 
                          else f"✗ {tool_name}: Only matched {matches}/{len(truth[0])} items")
                    return passed
            if isinstance(result_data, list):
                if all(isinstance(x, list) for x in truth):
                    matches = sum(1 for t in truth[0] if t in result_data[0])
                    passed = matches >= len(truth[0]) * 0.8
                    print(f"✓ {tool_name}: Matched {matches}/{len(truth[0])} items" if passed 
                          else f"✗ {tool_name}: Only matched {matches}/{len(truth[0])} items")
                    return passed
                else:
                    matches = sum(1 for t in truth if t in result_data)
                    passed = matches == len(truth)
                    print(f"✓ {tool_name}: All {len(truth)} items matched" if passed 
                          else f"✗ {tool_name}: Only matched {matches}/{len(truth)} items")
                    return passed
        elif isinstance(truth, int):
            if isinstance(result, tuple):
                result_data = result[0]
            else:
                result_data = result
            if isinstance(result_data, list):
                count = len(result_data)
                passed = count >= truth
                print(f"✓ {tool_name}: Returned {count} items (expected >= {truth})" if passed 
                      else f"✗ {tool_name}: Returned {count} items (expected >= {truth})")
                return passed
        elif isinstance(truth, float):
            if isinstance(result, tuple):
                result_data = result[0]
            else:
                result_data = result
            if isinstance(result_data, list) and len(result_data) > 0:
                result_val = float(result_data[0])
            else:
                result_val = float(result_data)
            tolerance = abs(truth * 0.1)
            passed = abs(result_val - truth) <= tolerance
            print(f"✓ {tool_name}: {result_val:.2f} ≈ {truth:.2f}" if passed 
                  else f"✗ {tool_name}: {result_val:.2f} != {truth:.2f}")
            return passed
        return False
    except Exception as e:
        print(f"✗ {tool_name}: Test error - {e}")
        return False

def run_tests():
    '''Run tests for all tools and summarize results.'''

    results = []
    test_results_summary = []

    for tool, prompt, truth in zip(tools_list, prompts_list, truths_list):
        try:
            print(f"\nTesting {tool.__name__} with prompt: {prompt}")
            result = tool(*prompt)
            results.append(result)
            passed = test_results(result, truth, tool.__name__)
            test_results_summary.append((tool.__name__, passed))
        except Exception as e:
            print(f"✗ Error in {tool.__name__}: {e}")
            test_results_summary.append((tool.__name__, False))
            print()

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed_count = sum(1 for _, passed in test_results_summary if passed)
    total_count = len(test_results_summary)
    print(f"Passed: {passed_count}/{total_count} ({100*passed_count/total_count:.1f}%)")
    print()
    for tool_name, passed in test_results_summary:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {tool_name}")

