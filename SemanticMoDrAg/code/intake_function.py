from input_parsing import start_ner, start_embedding, intake, define_tool_hash, tool_descriptions_values

def start_models():
    parse_model = start_ner()
    document_embeddings, embed_model = start_embedding(tool_descriptions_values)
    return parse_model, document_embeddings, embed_model


def query_to_context(query: str, parse_model, embed_model, document_embeddings):
    '''
    '''

    best_tools, present, proteins_list, names_list, smiles_list, uniprot_list, pdb_list, chembl_list = intake(query, parse_model, embed_model, document_embeddings)
    tool_function_hash = define_tool_hash(best_tools[0], proteins_list, names_list, smiles_list, uniprot_list, pdb_list, chembl_list)

    args_list = tool_function_hash[best_tools[0]][1]
    results_tuple  = tool_function_hash[best_tools[0]][0](*args_list)

    i=1
    while results_tuple[0] == [] :
      tool_function_hash = define_tool_hash(best_tools[i], proteins_list, names_list, smiles_list, uniprot_list, pdb_list, chembl_list)
      args_list = tool_function_hash[best_tools[i]][1]
      results_tuple  = tool_function_hash[best_tools[i]][0](*args_list)
      i+=1
      if i == 3:
        break

    
    results_list, results_string, results_images = results_tuple
    
    return results_list, results_string, results_images