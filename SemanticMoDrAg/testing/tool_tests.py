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
    [["2A3R"]],  # pdb_node(test_pdb_list)
    [["DNA gyrase"]],  # find_node(test_protein_list)
    [["[NH3+]CCc1ccc(O)cc1"], "DRD2"],  # docking_node(smiles_list, query_protein)
    [["phenylketonuria"]],  # target_node(search_descriptors)
    # From modrag_task_graphs.py
    ["MAOB"],  # get_actives_for_protein(query_protein)
    [["[NH3+]CCc1ccc(O)cc1"], "MAOB"],  # get_predictions_for_protein(smiles_list, query_protein)
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
    [[('Cc1cc2c(nn1)-c1cc(OCCCC(F)(F)F)ccc1C2=O', 0.014), ('C#CCN(C)[C@H](C)Cc1ccccc1', 0.017), 
      ('C#CCN(C)[C@H](C)Cc1ccccc1', 0.017), ('C[C@H](NCc1ccc(OCc2cccc(F)c2)cc1)C(N)=O', 0.023), 
      ('O=C(O)c1coc2ccccc2c1=O', 0.048), ('C#CCN[C@@H]1CCc2ccccc21', 0.05), 
      ('Cc1ccc2oc(=O)c(-c3cccc(Br)c3)cc2c1', 0.134), ('O=C(Nc1ccc(Cl)c(Cl)c1)c1ccc2[nH]ccc2c1', 0.227), 
      ('Cc1ccc(-c2cc3cc(C)ccc3oc2=O)cc1', 0.308), ('Cc1ccc(-c2cc3cc(C)ccc3oc2=O)cc1', 0.3085), 
      ('Cc1ccc(-c2cc3cc(C)ccc3oc2=O)cc1', 0.31), ('O=C1CCc2cc(OCCCC(F)(F)F)ccc21', 0.318), 
      ('C#CCNC1CCc2c(SCCc3cccc(C)c3)cccc21', 0.35), ('O=C1CCc2ccc(OCc3ccccc3F)cc2O1', 0.37), 
      ('O=c1cc(CCCl)c2ccc(OCc3cccc(Cl)c3)cc2o1', 0.37), ('Cn1ncc2cc(C(=O)Nc3ccc(Cl)c(Cl)c3)ccc21', 0.386), 
      ('Cn1ncc2cc(C(=O)Nc3ccc(Cl)c(Cl)c3)ccc21', 0.386), ('O=C(Nc1cccc(Cl)c1)c1coc2ccccc2c1=O', 0.4), 
      ('O=c1ccc2ccc(OCc3ccc(Br)cc3)cc2o1', 0.5), ('O=C(Nc1ccc(Cl)c(Cl)c1)c1ccc2[nH]ncc2c1', 0.586), 
      ('O=C(Nc1ccc(Cl)c(Cl)c1)c1ccc2[nH]ncc2c1', 0.586), ('O=C(Nc1ccc(Cl)c(Cl)c1)c1ccc2[nH]ncc2c1', 0.586), 
      ('COC(=O)c1cc2ccc(OCc3cccc(F)c3)cc2oc1=O', 0.588), ('Clc1ccc(/N=C/c2ccc3[nH]ncc3c2)cc1Cl', 0.612), 
      ('O=C1Nc2ccc(CCCCc3ccccc3)cc2C1=O', 0.66), ('Cn1ncc2cc(C(=O)Nc3ccc(F)c(Cl)c3)ccc21', 0.662), 
      ('O=C(Nc1ccc(Cl)c(F)c1)c1ccc2[nH]ncc2c1', 0.668), ('Cc1ccc(NC(=O)c2coc3ccccc3c2=O)cc1C', 0.67), 
      ('O=C(Nc1ccc(F)c(Cl)c1)c1ccc2[nH]ncc2c1', 0.679), ('O=C(Nc1ccc(F)c(Cl)c1)c1ccc2[nH]ncc2c1', 0.679), 
      ('COc1cccc(Cn2ccc3c(N4CCC(OC)CC4)nc4ccccc4c32)c1', 0.7), ('COc1ccc(-c2cc3cc(C)ccc3oc2=O)cc1Br', 0.74), 
      ('C#CCCCCOc1ccc2c(C)cc(=O)oc2c1', 0.77), ('COc1cccc(-c2cc3cc(C)ccc3oc2=O)c1', 0.8), 
      ('Cc1c(Cl)c(=O)oc2cc(OCc3ccc(Br)cc3)ccc12', 0.8), ('COC1CCN(c2nc3ccccc3c3c2ccn3Cc2cccc(Cl)c2)CC1', 0.8), 
      ('COc1cccc(-c2cc3cc(C)ccc3oc2=O)c1', 0.8026), ('CCCCCCOc1ccc2cc(C(=O)OC)c(=O)oc2c1', 0.875), 
      ('Cc1c(C)c2ccc(OCc3nnc(C(C)C)s3)cc2oc1=O', 0.9), ('Cc1cc(=O)oc2cc(OCc3ccc(Br)cc3)ccc12', 0.9), 
      ('COc1cccc(-c2cc3cc(Cl)ccc3oc2=O)c1', 1.0), ('Cn1ncc2cc(/C=N/c3ccc(Cl)c(Cl)c3)ccc21', 1.03), 
      ('O=C(Nc1ccccc1Br)c1coc2ccccc2c1=O', 1.03), ('COCCn1ncc2cc(C(=O)Nc3ccc(Cl)c(Cl)c3)ccc21', 1.08), 
      ('Cc1c(C)c2ccc(OCc3ccc(F)c(F)c3)cc2oc1=O', 1.1), ('O=C1CCCc2ccc(OCc3ccc(Cl)cc3)cc21', 1.1), 
      ('O=C(Nc1ccc2[nH]ccc2n1)c1ccc(Cl)c(Cl)c1', 1.11), ('Cc1c(C)c2ccc(OCc3ccc(F)c(F)c3)cc2oc1=O', 1.14), 
      ('O=C(Nc1ccc(Br)cc1)c1coc2ccccc2c1=O', 1.15), ('C=C(Br)COc1ccc2c3c(c(=O)oc2c1C)CCC3', 1.18)]],  # getbioactives_node
    [10995.537239000963],  # predict_node
    #None,  # gpt_node (commented out)
    [['MELIQDTSRPPLEYVKGVPLIKYFAEALGPLQSFQARPDDLLINTYPKSGTTWVSQILDMIYQGGDLEKCNRAPIYVRVPFLEVNDPGEPSGLETLKD\
    TPPPRLIKSHLPLALLPQTLLDQKVKVVYVARNPKDVAVSYYHFHRMEKAHPEPGTWDSFLEKFMAGEVSYGSWYQHVQEWWELSRTHPVLYLFYEDMKE\
    NPKREIQKILEFVGRSLPEETMDFMVQHTSFKEMKKNPMTNYTTVPQELMDHSISPFMRKGMAGDWKTTFTVAQNERFDADYAEKMAGCSLSFRSEL',
     'MELIQDTSRPPLEYVKGVPLIKYFAEALGPLQSFQARPDDLLINTYPKSGTTWVSQILDMIYQGGDLEKCNRAPIYVRVPFLEVNDPGEPSGLETLKD\
    TPPPRLIKSHLPLALLPQTLLDQKVKVVYARNPKDVAVSYYHFHRMEKAHPEPGTWDSFLEKFMAGEVSYGSWYQHVQEWWELSRTHPVLYLFYEDMKE\
    NPKREIQKILEFVGRSLPEETMDFMVQHTSFKEMKKNPMTNYTTVPQELMDHSISPFMRKGMAGDWKTTFTVAQNERFDADYAEKMAGCSLSFRSEL']],  # pdb_node
    [['3QTD', '1VL4', '1VPB', '5NJ5', '3NUH']],  # find_node
    [-6.3],  # docking_node
    [['PAH', 'DNAJC12', 'NSUN2', 'COL1A1', 'PI4KA']],  # target_node
    # From modrag_task_graphs.py
    [[('Cc1cc2c(nn1)-c1cc(OCCCC(F)(F)F)ccc1C2=O', 0.014), ('C#CCN(C)[C@H](C)Cc1ccccc1', 0.017), 
      ('C#CCN(C)[C@H](C)Cc1ccccc1', 0.017), ('C[C@H](NCc1ccc(OCc2cccc(F)c2)cc1)C(N)=O', 0.023), 
      ('O=C(O)c1coc2ccccc2c1=O', 0.048), ('C#CCN[C@@H]1CCc2ccccc21', 0.05), 
      ('Cc1ccc2oc(=O)c(-c3cccc(Br)c3)cc2c1', 0.134), ('O=C(Nc1ccc(Cl)c(Cl)c1)c1ccc2[nH]ccc2c1', 0.227), 
      ('Cc1ccc(-c2cc3cc(C)ccc3oc2=O)cc1', 0.308), ('Cc1ccc(-c2cc3cc(C)ccc3oc2=O)cc1', 0.3085), 
      ('Cc1ccc(-c2cc3cc(C)ccc3oc2=O)cc1', 0.31), ('O=C1CCc2cc(OCCCC(F)(F)F)ccc21', 0.318), 
      ('C#CCNC1CCc2c(SCCc3cccc(C)c3)cccc21', 0.35), ('O=C1CCc2ccc(OCc3ccccc3F)cc2O1', 0.37), 
      ('O=c1cc(CCCl)c2ccc(OCc3cccc(Cl)c3)cc2o1', 0.37), ('Cn1ncc2cc(C(=O)Nc3ccc(Cl)c(Cl)c3)ccc21', 0.386), 
      ('Cn1ncc2cc(C(=O)Nc3ccc(Cl)c(Cl)c3)ccc21', 0.386), ('O=C(Nc1cccc(Cl)c1)c1coc2ccccc2c1=O', 0.4), 
      ('O=c1ccc2ccc(OCc3ccc(Br)cc3)cc2o1', 0.5), ('O=C(Nc1ccc(Cl)c(Cl)c1)c1ccc2[nH]ncc2c1', 0.586), 
      ('O=C(Nc1ccc(Cl)c(Cl)c1)c1ccc2[nH]ncc2c1', 0.586), ('O=C(Nc1ccc(Cl)c(Cl)c1)c1ccc2[nH]ncc2c1', 0.586), 
      ('COC(=O)c1cc2ccc(OCc3cccc(F)c3)cc2oc1=O', 0.588), ('Clc1ccc(/N=C/c2ccc3[nH]ncc3c2)cc1Cl', 0.612), 
      ('O=C1Nc2ccc(CCCCc3ccccc3)cc2C1=O', 0.66), ('Cn1ncc2cc(C(=O)Nc3ccc(F)c(Cl)c3)ccc21', 0.662), 
      ('O=C(Nc1ccc(Cl)c(F)c1)c1ccc2[nH]ncc2c1', 0.668), ('Cc1ccc(NC(=O)c2coc3ccccc3c2=O)cc1C', 0.67), 
      ('O=C(Nc1ccc(F)c(Cl)c1)c1ccc2[nH]ncc2c1', 0.679), ('O=C(Nc1ccc(F)c(Cl)c1)c1ccc2[nH]ncc2c1', 0.679), 
      ('COc1cccc(Cn2ccc3c(N4CCC(OC)CC4)nc4ccccc4c32)c1', 0.7), ('COc1ccc(-c2cc3cc(C)ccc3oc2=O)cc1Br', 0.74), 
      ('C#CCCCCOc1ccc2c(C)cc(=O)oc2c1', 0.77), ('COc1cccc(-c2cc3cc(C)ccc3oc2=O)c1', 0.8), 
      ('Cc1c(Cl)c(=O)oc2cc(OCc3ccc(Br)cc3)ccc12', 0.8), ('COC1CCN(c2nc3ccccc3c3c2ccn3Cc2cccc(Cl)c2)CC1', 0.8), 
      ('COc1cccc(-c2cc3cc(C)ccc3oc2=O)c1', 0.8026), ('CCCCCCOc1ccc2cc(C(=O)OC)c(=O)oc2c1', 0.875), 
      ('Cc1c(C)c2ccc(OCc3nnc(C(C)C)s3)cc2oc1=O', 0.9), ('Cc1cc(=O)oc2cc(OCc3ccc(Br)cc3)ccc12', 0.9), 
      ('COc1cccc(-c2cc3cc(Cl)ccc3oc2=O)c1', 1.0), ('Cn1ncc2cc(/C=N/c3ccc(Cl)c(Cl)c3)ccc21', 1.03), 
      ('O=C(Nc1ccccc1Br)c1coc2ccccc2c1=O', 1.03), ('COCCn1ncc2cc(C(=O)Nc3ccc(Cl)c(Cl)c3)ccc21', 1.08), 
      ('Cc1c(C)c2ccc(OCc3ccc(F)c(F)c3)cc2oc1=O', 1.1), ('O=C1CCCc2ccc(OCc3ccc(Cl)cc3)cc21', 1.1), 
      ('O=C(Nc1ccc2[nH]ccc2n1)c1ccc(Cl)c(Cl)c1', 1.11), ('Cc1c(C)c2ccc(OCc3ccc(F)c(F)c3)cc2oc1=O', 1.14), 
      ('O=C(Nc1ccc(Br)cc1)c1coc2ccccc2c1=O', 1.15), ('C=C(Br)COc1ccc2c3c(c(=O)oc2c1C)CCC3', 1.18)]],   # get_actives_for_protein
    [10995.537239000963],  # get_predictions_for_protein
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

def run_tests(prompts_list = prompts_list, truths_list = truths_list, tools_list = tools_list): 
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

