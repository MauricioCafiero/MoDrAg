from input_parsing import start_ner, start_embedding, intake, define_tool_hash, tool_descriptions_values, second_intake, parse_input
from input_parsing import define_tool_reqs, smiles_regex
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from PIL import Image
import io, json, pprint as pp
from scholarly import scholarly, ProxyGenerator
import numpy as np
import ast
from gradio_client import Client, handle_file

# imports for HF Spaces
import torch
import gradio as gr
# end imports for HF Spaces

device = "cuda" if torch.cuda.is_available() else "cpu"

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

    #check to see if chat history file exists, if so load it
    try:
      with open('chat_session_history.txt', 'r') as f:
        data_string = f.read()
        self.chat_history = ast.literal_eval(data_string)
    except:
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
    Starts the supporting models for parsing and embedding.
    '''

    self.parse_model, self.document_embeddings, self.embed_model = start_models()
  
  def reset_chat(self):
    '''
    Resets the chat state.
    '''
    self.chat_idx = 0
    self.best_tools = []
    self.proteins_list = []
    self.names_list = []
    self.diseases_list = []
    self.smiles_list = []
    self.uniprot_list = []
    self.pdb_list = []
    self.chembl_list = []
    self.query = ''
    self.present = []
  
  def hard_reset_chat(self):
    '''
    Resets the chat state.
    '''
    self.chat_idx = 0
    self.best_tools = []
    self.proteins_list = []
    self.names_list = []
    self.diseases_list = []
    self.smiles_list = []
    self.uniprot_list = []
    self.pdb_list = []
    self.chembl_list = []
    self.query = ''
    self.present = []
    self.chat_history = []
  
  def uploaded_pic_chat(self, filepath):
    '''
      Chats with the model using an uploaded image.

      Args:
        file: The image file to send to the model.
      Returns:
        chat_history: The chat history.
    '''
    # Process the uploaded image file
    new_img = Image.open(filepath)
    saved_filename = 'saved_input.png'
    new_img.save(saved_filename)

    client = Client("cafierom/ImageToSmiles")
    result = client.predict(
    api_flag = "True",
	  img=handle_file(saved_filename),
	  api_name="/agent_make_smiles")

    nameandsmiles = result[0]
    result_image = Image.open(result[1])

    filename = "chat_image.png"
    result_image.save(filename)
    img = Image.open(filename)

    self.chat_history.append({'role': 'user', 'content': '**User uploaded an image for name/SMILES analysis**'})
    self.chat_history.append({'role': 'assistant', 'content': nameandsmiles})

    return '', self.chat_history, img

  def chat(self, query: str, mode_flag: str = 'AI'):
    '''
      Chats with the model.

      Args:
        query: The prompt to send to the model.
        mode_flag: The mode to use (AI, Manual, Web Search, Chat).
      Returns:
        chat_history: The chat history.
    '''
    ''' ===============================================================================================
    Handle Web Search Mode  - 
    if user chooses web search mode, send query to websearch node and return results
    ==================================================================================================='''
    if mode_flag == 'Web Search':
      self.chat_idx = 0
      local_chat_history = []
      local_chat_history.append(query)

      top_hits, search_string, _ = websearch_node(query, self.embed_model)
      local_chat_history.append(search_string)
      self.chat_history.append(local_chat_history)
      with open('chat_session_history.txt', 'w') as f:
        pp.pp(self.chat_history, stream=f)

      return '', self.chat_history, None

    ''' =============================================================================================
    Handle Chat Mode  -
    if user chooses chat mode, send query to LLM and return response
    ================================================================================================='''
    if mode_flag == 'Chat':
      self.chat_idx  = 0
      local_chat_history = []
      local_chat_history.append(query)
      self.query = query

      context = 'Previous chat history: '
      for chat in self.chat_history:
          for turn in chat:
            context += '\n' + turn


      role_text = f"You are part of a drug design agent. Answer user questions to the best of your ability. \
If the user asks any non-scientific questions, respond with 'I'm sorry, I can't assist with that request.' \
If the user asks for general information, provide a concise and accurate answer. If the user asks about drug design, \
provide detailed and informative answers or refer them to the tools. They can access the tools by switching to AI or \
manual mode. Reference the previous conversation in the context if needed."

      prompt = f'Query: {self.query}; CONTEXT: {context}.'

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
      self.latest_response = response
      local_chat_history.append(response)
      self.chat_history.append(local_chat_history)
      with open('chat_session_history.txt', 'w') as f:
        pp.pp(self.chat_history, stream=f)
      
      return '', self.chat_history, None

    '''=============================================================================================
    AI Mode: not chat or web search - 
    First interaction: get best tools and parse entities
    ============================================================================================='''
    if self.chat_idx == 0:
      #self.chat_history = []
      local_chat_history = []
      local_chat_history.append(query)

      '''sends the query to the intake function to get best tools and parsed entities'''
      self.query = query
      self.best_tools, self.present, self.proteins_list, self.names_list, self.diseases_list, self.smiles_list, self.uniprot_list, self.pdb_list, self.chembl_list = \
      intake(self.query, self.parse_model, self.embed_model, self.document_embeddings)

      response = '## The tools chosen based on your query are:'
      for i,tool in enumerate(self.best_tools):
        response += '\n' + f'{i+1}. {tool} : {full_tool_descriptions[tool]}'

      response += ' \n\n ## And the following information was found in your query:\n'
      for (entity_type, entity_list) in zip(self.present, [self.proteins_list, self.names_list, self.diseases_list, self.smiles_list, self.uniprot_list, self.pdb_list, self.chembl_list]):
        if self.present[entity_type] > 0:
          response += f'**{entity_type}**: {self.present[entity_type]}\n'
          for ent_idx, entity in enumerate(entity_list):
            response += f'- **{ent_idx+1}. {entity_type}**: {entity}\n'
      response += '\n To accept the #1 tool choice, hit enter; to choose 2 or 3, enter that number.'
      response += '\n To edit the items in a list, enter "edit".'
      response += '\n To start over, click the clear button and enter a new query.' 
      self.chat_idx += 1

      matches = smiles_regex(response)
      for m in matches:
        if m in response:
          response = response.replace(m, f'```{m}```')

      local_chat_history.append(response)
      self.chat_history.append(local_chat_history)
      with open('chat_session_history.txt', 'w') as f:
        pp.pp(self.chat_history, stream=f)

      return '', self.chat_history, None

    elif self.chat_idx == 1:
      '''=============================================================================================
      AI Mode: not chat or web search - 
       Second interaction: get tool choice and call tool function, return results
       either directly (manual mode) or via LLM (AI mode)
      ============================================================================================='''
      local_chat_history = []
      local_chat_history.append(query)

      ''' ===============================================================================================
      if the user has chosen to edit a list, go to edit list step 501
      =============================================================================================='''
      if 'edit' in query.lower():
        self.chat_idx = 501

        list_list = ['smiles list', 'names list', 'proteins list', 'diseases list', 'uniprot list', 'pdb list', 'chembl list']
        response = '## Enter the list to edit:\n'
        for i, list_name in enumerate(list_list):
          response += f'**{i+1}**. {list_name}\n'
        
        local_chat_history.append(response)
        self.chat_history.append(local_chat_history)
        with open('chat_session_history.txt', 'w') as f:
          pp.pp(self.chat_history, stream=f)
        return '', self.chat_history, None
      
      elif query == '':
        self.tool_choice = 0
      elif query in ['1','2','3']:
        self.tool_choice = int(query) - 1
      else:
        ''' ===============================================================================================
        In the case that the user enters an invalid tool choice, return to tool choice step
        ==================================================================================================='''
        response = 'Invalid input. Please enter 1, 2, or 3 to choose one of the tools above, or hit enter to accept the #1 tool choice. \
or enter "edit" to edit a list.'
        local_chat_history.append(response)
        self.chat_history.append(local_chat_history)
        with open('chat_session_history.txt', 'w') as f:
          pp.pp(self.chat_history, stream=f)

        self.chat_idx = 1
        return '', self.chat_history, None
      
      ''' ==============================================================================================
      Check that the necessary data is present for the chosen tool
      ask user for missing data if not
      ================================================================================================='''
      tool_function_reqs = define_tool_reqs(self.best_tools[self.tool_choice], self.proteins_list,
                                            self.names_list, self.diseases_list, self.smiles_list, self.uniprot_list, self.pdb_list, self.chembl_list)
      data_request = f'The necessary data was not found for tool {self.best_tools[self.tool_choice]}.\n'
      missing_data = False
      reqs_list = tool_function_reqs[self.best_tools[self.tool_choice]][0]
      list_names = tool_function_reqs[self.best_tools[self.tool_choice]][1]

      for sub_list, list_name in zip(reqs_list, list_names):
        if len(sub_list) == 0:
          data_request += f'Missing information for: *{list_name}*.\n'
          missing_data = True
      data_request += 'Please provide the necessary information to proceed.'
      if missing_data:
        local_chat_history.append(data_request)
        self.chat_history.append(local_chat_history)
        with open('chat_session_history.txt', 'w') as f:
          pp.pp(self.chat_history, stream=f)
        self.chat_idx = 999
        return '', self.chat_history, None
      ''' ====================================================================================================
      End data check: if not missing data, call tool function 
      ====================================================================================================='''

      ''' Get the chosen tool function and args, call it, and get results'''
      tool_function_hash = define_tool_hash(self.best_tools[self.tool_choice], self.proteins_list,
                                            self.names_list, self.diseases_list, self.smiles_list, self.uniprot_list, self.pdb_list, self.chembl_list)

      args_list = tool_function_hash[self.best_tools[self.tool_choice]][1]
      results_tuple  = tool_function_hash[self.best_tools[self.tool_choice]][0](*args_list)

      results_list, self.results_string, self.results_images = results_tuple
      print(self.results_string)

      '''=============================================================================================
      If manual mode, return results directly; if AI mode, send results to LLM for response generation
      ============================================================================================='''
      if mode_flag == 'Manual':
        
        matches = smiles_regex(self.results_string)

        for m in matches:
          if m in self.results_string:
            self.results_string = self.results_string.replace(m, f'```{m}```')

        local_chat_history.append(self.results_string)
        self.chat_history.append(local_chat_history)
        with open('chat_session_history.txt', 'w') as f:
          pp.pp(self.chat_history, stream=f)
        try:
          img = Image.open(io.BytesIO(self.results_images[0].data))
        except:
          img = None
        
        self.reset_chat()

        return '', self.chat_history, img

      ''' ============================================================================================= 
      AI mode: send results to LLM for response generation
      ============================================================================================='''
      role_text = "Answer the query using the information in the context. Add explanations \
or enriching information where appropriate."

      prompt = f'Query: {self.query}.\n Context: {self.results_string}'

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
      self.latest_response = response

      matches = smiles_regex(response)
      for m in matches:
          if m in response:
              response = response.replace(m, f'```{m}```')

      local_chat_history.append(response)
      self.chat_history.append(local_chat_history)
      with open('chat_session_history.txt', 'w') as f:
        pp.pp(self.chat_history, stream=f)
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
    
    elif self.chat_idx == 2:
      '''=============================================================================================
      for every turn after the first tool call:
      if chat_idx  = 2, call second intake to get new tools based on latest response, last results,
      and the new query
      calls chat_idx = 1 again to get tool choice from user
      ============================================================================================='''
      local_chat_history = []
      local_chat_history.append(query)
      self.query = query

      ''' =============================================================================================
      if user wants to review history, set context to full chat history, else set context to latest response,
      last results, and the new query
      ============================================================================================'''
      if mode_flag == 'Review History':
        context = self.query
        for chat in self.chat_history:
          for turn in chat:
            context += '\n' + turn
      else:
        context = self.latest_response + '\n' + self.results_string + '\n' + self.query

      self.best_tools, self.present, self.proteins_list, self.names_list, self.diseases_list, self.smiles_list, self.uniprot_list, self.pdb_list, self.chembl_list = \
      second_intake(self.query, context, self.parse_model, self.embed_model, self.document_embeddings)

      response = f'## Your new query is: {self.query}\n'
      response += '## The tools chosen based on your query are:'
      for i,tool in enumerate(self.best_tools):
        response += '\n' + f'{i+1}. {tool} : {full_tool_descriptions[tool]}'

      response += ' \n\n ## And the following information was found in your query:\n'
      for (entity_type, entity_list) in zip(self.present, [self.proteins_list, self.names_list, self.diseases_list, self.smiles_list, self.uniprot_list, self.pdb_list, self.chembl_list]):
        if self.present[entity_type] > 0:
          response += f'**{entity_type}**: {self.present[entity_type]}\n'
          for ent_idx, entity in enumerate(entity_list):
            response += f'- **{ent_idx+1}. {entity_type}**: {entity}\n'
      response += '\n To accept the #1 tool choice, hit enter; to choose 2 or 3, enter that number.'
      response += '\n To edit the items in a list, enter "edit".'
      response += '\n To start over, click the clear button and enter a new query.' 
      self.chat_idx = 1

      matches = smiles_regex(response)
      for m in matches:
          if m in response:
              response = response.replace(m, f'```{m}```')

      local_chat_history.append(response)
      self.chat_history.append(local_chat_history)
      with open('chat_session_history.txt', 'w') as f:
        pp.pp(self.chat_history, stream=f)

      return '', self.chat_history, None
    
    elif self.chat_idx == 999:
      ''' ============================================================================================== 
      condition for missing data after tool choice; if user was prompted for missing data, parse new input,
      then return to tool choice step (chat_idx = 1)
      ============================================================================================='''
      local_chat_history = []
      local_chat_history.append(query)

      ''' Parse the new input to get missing data '''
      present, proteins_list, names_list, diseases_list, smiles_list, uniprot_list, pdb_list, chembl_list = parse_input(query, self.parse_model)

      ''' Update the existing lists with any new data only if the existing lists are empty'''
      if len(self.proteins_list) == 0 and len(proteins_list) > 0:
        self.proteins_list = proteins_list
      if len(self.names_list) == 0 and len(names_list) > 0:
        self.names_list = names_list
      if len(self.diseases_list) == 0 and len(diseases_list) > 0:
        self.diseases_list = diseases_list
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
      
      response = f'## Your new query is: {self.query}\n'
      response += '## The tools chosen based on your query are:'
      for i,tool in enumerate(self.best_tools):
        response += '\n' + f'{i+1}. {tool} : {full_tool_descriptions[tool]}'

      response += ' \n\n ## And the following information was found in your query:\n'
      for (entity_type, entity_list) in zip(self.present, [self.proteins_list, self.names_list, self.diseases_list, self.smiles_list, self.uniprot_list, self.pdb_list, self.chembl_list]):
        if self.present[entity_type] > 0:
          response += f'**{entity_type}**: {self.present[entity_type]}\n'
          for ent_idx, entity in enumerate(entity_list):
            response += f'- **{ent_idx+1}. {entity_type}**: {entity}\n'
      response += '\n To accept the #1 tool choice, hit enter; to choose 2 or 3, enter that number.'
      response += '\n To edit the items in a list, enter "edit".'
      response += '\n To start over, click the clear button and enter a new query.' 
      self.chat_idx = 1

      matches = smiles_regex(response)
      for m in matches:
          if m in response:
              response = response.replace(m, f'```{m}```')

      local_chat_history.append(response)
      self.chat_history.append(local_chat_history)
      with open('chat_session_history.txt', 'w') as f:
        pp.pp(self.chat_history, stream=f)

      return '', self.chat_history, None
    
    elif self.chat_idx == 501:
      ''' ==============================================================================================
      condition for editing a list; get which list to edit from user
      ============================================================================================='''
      local_chat_history = []
      local_chat_history.append(query)
      self.chat_idx = 502

      list_list = ['smiles list', 'names list', 'proteins list', 'diseases list', 'uniprot list', 'pdb list', 'chembl list']
      try:
        choice_idx = int(query) - 1
        self.list_to_edit = list_list[choice_idx]

        response = f'## You have chosen to edit the {self.list_to_edit}.\n'
        response += 'Enter the numbers for the *items to keep* in the list.'

      except:
        response = 'Invalid input. Please enter the number corresponding to the list you wish to edit.'
        self.chat_idx = 501

      local_chat_history.append(response)
      self.chat_history.append(local_chat_history)
      with open('chat_session_history.txt', 'w') as f:
        pp.pp(self.chat_history, stream=f)
      return '', self.chat_history, None  

    if self.chat_idx == 502:
      ''' ==============================================================================================
      condition for editing a list; get which items to keep from user
      ============================================================================================='''
      local_chat_history = []
      local_chat_history.append(query)
      self.chat_idx = 503

      if ',' in query:
        items_to_keep = query.split(',')
      elif ';' in query:
        items_to_keep = query.split(';')
      else: 
        items_to_keep = query.split()
      
      for item in items_to_keep:
        if not item.isdigit():
          response = 'Invalid input. Please enter the numbers corresponding to the items you wish to keep.'
          self.chat_idx = 502
          local_chat_history.append(response)
          self.chat_history.append(local_chat_history)
          with open('chat_session_history.txt', 'w') as f:
            pp.pp(self.chat_history, stream=f)
          return '', self.chat_history, None
      
      try:
        if self.list_to_edit == 'smiles list':
          current_list = self.smiles_list
        elif self.list_to_edit == 'names list':
          current_list = self.names_list
        elif self.list_to_edit == 'proteins list':
          current_list = self.proteins_list
        elif self.list_to_edit == 'diseases list':
          current_list = self.diseases_list
        elif self.list_to_edit == 'uniprot list':
          current_list = self.uniprot_list
        elif self.list_to_edit == 'pdb list':
          current_list = self.pdb_list
        elif self.list_to_edit == 'chembl list':
          current_list = self.chembl_list
        
        new_list = []
        for item in items_to_keep:
          idx = int(item) - 1
          new_list.append(current_list[idx])
        
        if self.list_to_edit == 'smiles list':
          self.smiles_list = new_list
        elif self.list_to_edit == 'names list': 
          self.names_list = new_list
        elif self.list_to_edit == 'proteins list':
          self.proteins_list = new_list
        elif self.list_to_edit == 'diseases list':
          self.diseases_list = new_list
        elif self.list_to_edit == 'uniprot list':
          self.uniprot_list = new_list
        elif self.list_to_edit == 'pdb list':
          self.pdb_list = new_list
        elif self.list_to_edit == 'chembl list':
          self.chembl_list = new_list
        
        self.present = {
          'proteins': len(self.proteins_list),
          'molecules': len(self.names_list),
          'diseases': len(self.diseases_list),
          'smiles': len(self.smiles_list),
          'uniprot': len(self.uniprot_list),
          'pdb': len(self.pdb_list),
          'chembl': len(self.chembl_list)
        }

        response = '## The tools chosen based on your query are:'
        for i,tool in enumerate(self.best_tools):
          response += '\n' + f'{i+1}. {tool} : {full_tool_descriptions[tool]}'

        response += ' \n\n ## And the following information was found in your query:\n'
        for (entity_type, entity_list) in zip(self.present, [self.proteins_list, self.names_list, self.diseases_list, self.smiles_list, self.uniprot_list, self.pdb_list, self.chembl_list]):
          if self.present[entity_type] > 0:
            response += f'**{entity_type}**: {self.present[entity_type]}\n'
            for ent_idx, entity in enumerate(entity_list):
              response += f'- **{ent_idx+1}. {entity_type}**: {entity}\n'
        response += '\n To accept the #1 tool choice, hit enter; to choose 2 or 3, enter that number.'
        response += '\n To edit the items in a list, enter "edit".'
        response += '\n To start over, click the clear button and enter a new query.' 
        self.chat_idx = 1

        matches = smiles_regex(response)
        for m in matches:
          if m in response:
            response = response.replace(m, f'```{m}```')

      except:
        response = 'An error occurred while processing your input. Please try again.'
        self.chat_idx = 502

      local_chat_history.append(response)
      self.chat_history.append(local_chat_history)
      with open('chat_session_history.txt', 'w') as f:
        pp.pp(self.chat_history, stream=f)
        
      return '', self.chat_history, None
        


full_tool_descriptions = {
  'smiles_node' : 'Queries Pubchem for the smiles string of the molecule based on the name.',
  'name_node' : 'Queries Pubchem for the name of the molecule based on the smiles string.',
  'related_node' : 'Queries Pubchem for similar molecules based on the smiles string or name.',
  'structure_node' : 'Generates the 3D structure and chemical formula of the molecule based on the name or smiles string.',
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
  'target_node' : 'Accepts a disease name and queries Open Targets for target matches.',
  'get_actives_for_protein' : 'Finds Bioactive molecules for a given protein. Uses Uniprot to find chembl IDs \
for the protein, and then queries chembl for bioactive molecules.',
  'get_predictions_for_protein' : 'Uses Uniprot to find chembl IDs for the protein, and then queries chembl \
for bioactive molecules to train a model and predict the activity of the given smiles.',
  'dock_from_names' : 'Accepts names of molecules and docks them in a given protein.'
}

def websearch_node(query: str, embed_model, proxy_flag: bool = True) -> (list[str], str, list):
  '''
  Performs a web search using scholarly and ranks results based on similarity to the query.
  Args:
      query (str): The input query string.
      embed_model: The embedding model.
  Returns:  
      top_hits (list[str]): List of top hit titles and links.
      search_string (str): String representation of the top hits.
      None: Placeholder for images (not used here).
  '''
  try:
    if proxy_flag:
      pg = ProxyGenerator() 
      success = pg.FreeProxies()
      if success:
        pg.FreeProxies()
        scholarly.use_proxy(pg)

    scholarly.set_timeout(15) 

    search_query = scholarly.search_pubs(query)
    print(f'Search generator created for query: {query}')

    titles = []
    links = []
    abstracts = []

    for i in range(10):
      item = next(search_query)
      res_string = json.dumps(item)
      res_dict = json.loads(res_string)
      links.append(res_dict['pub_url'])
      titles.append(res_dict['bib']['title'])
      abstracts.append(res_dict['bib']['abstract'])
      print(f'Found result {i+1}')

    assert(len(titles) == len(links) == len(abstracts))
    print(f'Found {len(titles)} results')
    
    abstract_embeddings = embed_model.encode_document(abstracts)
    query_embeddings = embed_model.encode_query(query)

    scores = embed_model.similarity(query_embeddings, abstract_embeddings)

    max_hits = 10
    if len(scores) < max_hits:
      max_hits = len(scores)
    top_hits = []
    hits_idx = 0
    while hits_idx < max_hits:
      current_hit_idx = np.argmax(scores[0])
      current_score = scores[0][current_hit_idx].item()
      top_hits.append((titles[current_hit_idx], links[current_hit_idx], current_score))
      scores[0][current_hit_idx] = -1
      hits_idx += 1

    search_string = f'The top {max_hits} hits for your query are:\n'
    i = 0
    for title, link, score in top_hits:
      search_string += f'{i}. {title}\nLink: {link}\nScore: {score:.3f}\n\n'
      i += 1
    print('Web search completed successfully.')
  except:
    top_hits = []
    search_string = 'Web search failed. Please try again later.'
    print('Web search failed due to an exception.')

  return top_hits, search_string, None

'''================================================================================================
Functions for use on Hugging Face Spaces
===================================================================================================='''


''' ======================================================================================================
older functions retained for compatibility
======================================================================================================'''


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

    best_tools, present, proteins_list, names_list, diseases_list, smiles_list, uniprot_list, pdb_list, chembl_list = intake(query, parse_model, embed_model, document_embeddings)
    tool_function_hash = define_tool_hash(best_tools[0], proteins_list, names_list, diseases_list, smiles_list, uniprot_list, pdb_list, chembl_list)

    args_list = tool_function_hash[best_tools[0]][1]
    results_tuple  = tool_function_hash[best_tools[0]][0](*args_list)

    i=1
    while results_tuple[0] == [] :
      tool_function_hash = define_tool_hash(best_tools[i], proteins_list, names_list, diseases_list, smiles_list, uniprot_list, pdb_list, chembl_list)
      args_list = tool_function_hash[best_tools[i]][1]
      results_tuple  = tool_function_hash[best_tools[i]][0](*args_list)
      i+=1
      if i == 3:
        break

    results_list, results_string, results_images = results_tuple
    
    return results_list, results_string, results_images

