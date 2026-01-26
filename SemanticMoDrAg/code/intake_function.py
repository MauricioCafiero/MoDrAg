from input_parsing import start_ner, start_embedding, intake, define_tool_hash, tool_descriptions_values, second_intake, parse_input
from input_parsing import define_tool_reqs  
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from PIL import Image
import io

def start_models():
    '''
    Starts the necessary models for processing.
    Returns:
        parse_model: The NER model.
        document_embeddings: The encoded document embeddings.
        embed_model: The embedding model.
    '''
    parse_model = start_ner()
    document_embeddings, embed_model = start_embedding(tool_descriptions_values)
    return parse_model, document_embeddings, embed_model

class chat_manager():
  '''
  '''
  def __init__(self, model_id: str = 'google/gemma-3-1b-it', device: str = 'cuda'):
    '''
    '''
    self.model_id = model_id
    self.device = device
    self.chat_idx = 0
    self.chat_history = []

  def start_model_tokenizer(self):
    '''
      Downloads and loads the model and tokenizer.

      Args:
        None
      Returns:
        None
      Also defines:
        model: The model to use.
        tokenizer: The tokenizer to use.
    '''
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    self.llm_model = AutoModelForCausalLM.from_pretrained(
        self.model_id, quantization_config=quantization_config).eval()

    self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    print(f"Model loaded on {self.device}")

  def start_support_models(self):
    '''
    '''

    self.parse_model, self.document_embeddings, self.embed_model = start_models()
  
  def reset_chat(self):
    '''
    '''
    self.chat_idx = 0
    self.best_tools = []
    self.proteins_list = []
    self.names_list = []
    self.smiles_list = []
    self.uniprot_list = []
    self.pdb_list = []
    self.chembl_list = []
    self.query = ''
    self.present = []

  def chat(self, query: str, ai_flag: str = 'AI'):
    '''
      Chats with the model.

      Args:
        query: The prompt to send to the model.
        ai_flag: whether to use AI or not.
      Returns:
        chat_history: The chat history.
    '''
    if self.chat_idx == 0:
      #self.chat_history = []
      local_chat_history = []
      local_chat_history.append(query)

      self.query = query
      self.best_tools, self.present, self.proteins_list, self.names_list, self.smiles_list, self.uniprot_list, self.pdb_list, self.chembl_list = \
      intake(self.query, self.parse_model, self.embed_model, self.document_embeddings)

      response = 'The tools chosen based on your query are:'
      for i,tool in enumerate(self.best_tools):
        response += '\n' + f'{i+1}. {tool} : {full_tool_descriptions[tool]}'

      response += ' \n\n And the following information was found in your query:\n'
      for (entity_type, entity_list) in zip(self.present, [self.proteins_list, self.names_list, self.smiles_list, self.uniprot_list, self.pdb_list, self.chembl_list]):
        if self.present[entity_type] > 0:
          response += f'{entity_type}: {self.present[entity_type]}\n'
          for entity in entity_list:
            response += f'{entity_type}: {entity}\n'
      response += '\n To accept the #1 tool choice, hit enter; to choose 2 or 3, enter that number.'
      response += '\n To start over, click the clear button and enter a new query.' 
      self.chat_idx += 1

      local_chat_history.append(response)
      self.chat_history.append(local_chat_history)

      return '', self.chat_history, None

    elif self.chat_idx == 1:
      local_chat_history = []
      local_chat_history.append(query)

      if query == '':
        self.tool_choice = 0
      else:
        self.tool_choice = int(query) - 1
      
      ''' Check that the necessary data is present for the chosen tool'''
      tool_function_reqs = define_tool_reqs(self.best_tools[self.tool_choice], self.proteins_list,
                                            self.names_list, self.smiles_list, self.uniprot_list, self.pdb_list, self.chembl_list)
      data_request = f'The necessary data was not found for tool {self.best_tools[self.tool_choice]}.\n'
      missing_data = False
      reqs_list = tool_function_reqs[self.best_tools[self.tool_choice]][0]
      list_names = tool_function_reqs[self.best_tools[self.tool_choice]][1]

      for sub_list, list_name in zip(reqs_list, list_names):
        if len(sub_list) == 0:
          data_request += f'Missing information for: {list_name}.\n'
          missing_data = True
      data_request += 'Please provide the necessary information to proceed.'
      if missing_data:
        local_chat_history.append(data_request)
        self.chat_history.append(local_chat_history)
        self.chat_idx = 999
        return '', self.chat_history, None
      ''' End data check: if not missing data, call tool function '''

      tool_function_hash = define_tool_hash(self.best_tools[self.tool_choice], self.proteins_list,
                                            self.names_list, self.smiles_list, self.uniprot_list, self.pdb_list, self.chembl_list)

      args_list = tool_function_hash[self.best_tools[self.tool_choice]][1]
      results_tuple  = tool_function_hash[self.best_tools[self.tool_choice]][0](*args_list)

      results_list, results_string, self.results_images = results_tuple

      if ai_flag == 'Manual':
        local_chat_history.append(results_string)
        self.chat_history.append(local_chat_history)
        try:
          img = Image.open(io.BytesIO(self.results_images[0].data))
        except:
          img = None
        
        self.reset_chat()

        return '', self.chat_history, img

      role_text = "Answer the query using the information in the context. Add explanations \
or enriching information where appropriate."

      prompt = f'Query: {self.query}.\n Context: {results_string}'

      messages = [[{
                  "role": "system",
                  "content": [{"type": "text", "text": role_text},]
              },{
                  "role": "user",
                  "content": [{"type": "text", "text": prompt},]
              }]]

      inputs = self.tokenizer.apply_chat_template(
          messages,
          add_generation_prompt=True,
          tokenize=True,
          return_dict=True,
          return_tensors="pt",
      ).to(self.llm_model.device) #.to(torch.bfloat16)

      with torch.inference_mode():
          outputs = self.llm_model.generate(**inputs, max_new_tokens=500, do_sample=True, temperature=0.5)

      outputs = self.tokenizer.batch_decode(outputs, )

      parts = outputs[0].split('<start_of_turn>model')
      response = parts[1].strip('\n').strip('<end_of_turn>')
      local_chat_history.append(response)
      self.chat_history.append(local_chat_history)
      self.chat_idx += 1

      # self.reset_chat()

      #convert self.results_images[0] from ipython display to an image for gradio
      try:
        img = Image.open(io.BytesIO(self.results_images[0].data))
      except:
        img = None

      if img != None:
        return '', self.chat_history, img
      else:
        return '', self.chat_history, None
    
    #if chat_idx > 1, call just tool embedding and use existing lists
    elif self.chat_idx == 2:
      local_chat_history = []
      local_chat_history.append(query)
      self.query = query

      context = self.chat_history[-1][-1]  # last response from the model
      self.best_tools, self.present, self.proteins_list, self.names_list, self.smiles_list, self.uniprot_list, self.pdb_list, self.chembl_list = \
      second_intake(self.query, context, self.parse_model, self.embed_model, self.document_embeddings)

      response = f'Your new query is: {self.query}\n'
      response += 'The tools chosen based on your query are:'
      for i,tool in enumerate(self.best_tools):
        response += '\n' + f'{i+1}. {tool} : {full_tool_descriptions[tool]}'

      response += ' \n\n And the following information was found in your query:\n'
      for (entity_type, entity_list) in zip(self.present, [self.proteins_list, self.names_list, self.smiles_list, self.uniprot_list, self.pdb_list, self.chembl_list]):
        if self.present[entity_type] > 0:
          response += f'{entity_type}: {self.present[entity_type]}\n'
          for entity in entity_list:
            response += f'{entity_type}: {entity}\n'
      response += '\n To accept the #1 tool choice, hit enter; to choose 2 or 3, enter that number.'
      response += '\n To start over, click the clear button and enter a new query.' 
      self.chat_idx = 1

      local_chat_history.append(response)
      self.chat_history.append(local_chat_history)

      return '', self.chat_history, None
    
    elif self.chat_idx == 999:
      local_chat_history = []
      local_chat_history.append(query)

      present, proteins_list, names_list, smiles_list, uniprot_list, pdb_list, chembl_list = parse_input(query, self.parse_model)

      if len(self.proteins_list) == 0 and len(proteins_list) > 0:
        self.proteins_list = proteins_list
      if len(self.names_list) == 0 and len(names_list) > 0:
        self.names_list = names_list
      if len(self.smiles_list) == 0 and len(smiles_list) > 0:
        self.smiles_list = smiles_list
      if len(self.uniprot_list) == 0 and len(uniprot_list) > 0:
        self.uniprot_list = uniprot_list
      if len(self.pdb_list) == 0 and len(pdb_list) > 0:
        self.pdb_list = pdb_list
      if len(self.chembl_list) == 0 and len(chembl_list) > 0:
        self.chembl_list = chembl_list

      for item in present:
        self.present[item] += present[item]
      
      response = f'Your new query is: {self.query}\n'
      response += 'The tools chosen based on your query are:'
      for i,tool in enumerate(self.best_tools):
        response += '\n' + f'{i+1}. {tool} : {full_tool_descriptions[tool]}'

      response += ' \n\n And the following information was found in your query:\n'
      for (entity_type, entity_list) in zip(self.present, [self.proteins_list, self.names_list, self.smiles_list, self.uniprot_list, self.pdb_list, self.chembl_list]):
        if self.present[entity_type] > 0:
          response += f'{entity_type}: {self.present[entity_type]}\n'
          for entity in entity_list:
            response += f'{entity_type}: {entity}\n'
      response += '\n To accept the #1 tool choice, hit enter; to choose 2 or 3, enter that number.'
      response += '\n To start over, click the clear button and enter a new query.' 
      self.chat_idx = 1

      local_chat_history.append(response)
      self.chat_history.append(local_chat_history)

      return '', self.chat_history, None


full_tool_descriptions = {
  'smiles_node' : 'Queries Pubchem for the smiles string of the molecule based on the name.',
  'name_node' : 'Queries Pubchem for the name of the molecule based on the smiles string.',
  'related_node' : 'Queries Pubchem for similar molecules based on the smiles string or name.',
  'substitution_node' : 'A simple substitution routine that looks for a substituent on a phenyl ring and\
substitutes different fragments in that location. Returns a list of novel molecules and their\
QED score (1 is most drug-like, 0 is least drug-like).',
  'lipinski_node' : 'A tool to calculate QED and other lipinski properties of a molecule.',
  'pharmfeature_node': 'A tool to compare the pharmacophore features of a query molecule against\
those of a reference molecule and report the pharmacophore features of both and the feature\
score of the query molecule.',
  'uniprot_node' : 'This tool takes in the user requested protein and searches UNIPROT for matches.\
It returns a string scontaining the protein ID, gene name, organism, and protein name.',
  'listbioactives_node' : 'Accepts a UNIPROT ID and searches for bioactive molecules. Returns counts of\
the bioactives found and their ChEMBL IDs.',
  'getbioactives_node' : 'Accepts a Chembl ID and get all bioactives molecule SMILES and IC50s for that ID.',
  'predict_node' : 'uses the current_bioactives.csv file from the get_bioactives node to fit the\
Light GBM model and predict the IC50 for the current smiles.',
  'gpt_node' : 'Uses a Chembl dataset, previously stored in a CSV file by the get_bioactives node, to\
finetune a GPT model to generate novel molecules for the target protein.',
  'pdb_node' : 'Accepts a PDB ID and queires the protein databank for the sequence of the protein, \
as well as other information such as ligands.',
  'find_node': 'Accepts a protein name and searches the protein databack for PDB IDs that match along \
with the entry titles.',
  'docking_node' : 'Docking tool: uses dockstring to dock the molecule into the protein binding site and returns \
the docking score and the binding pose.',
  'get_actives_for_protein' : 'Finds Bioactive molecules for a give protein. Uses Uniprot to find chembl IDs \
for the protein, and then queries chembl for bioactive molecules.',
  'get_predictions_for_protein' : 'Uses Uniprot to find chembl IDs for the protein, and then queries chembl \
for bioactive molecules to train a model and predict the activity of the given smiles.',
  'dock_from_names' : 'Accepts names of molecules and docks them in a given protein.'
}

## older functions retained for compatibility


def query_to_context(query: str, parse_model, embed_model, document_embeddings):
    '''
    Processes a query to extract relevant context and information.
    Args:
        query (str): The input query string.
        parse_model: The NER model.
        embed_model: The embedding model.
        document_embeddings: The encoded document embeddings.
    Returns:
        results_list: List of results.
        results_string: String representation of results.
        results_images: Any associated images with the results.
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