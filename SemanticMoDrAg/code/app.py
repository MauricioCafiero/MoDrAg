import torch
import os, re
import gradio as gr
import numpy as np
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from intake_function import intake, second_intake, start_models, full_tool_descriptions
from input_parsing import define_tool_hash


device = "cuda" if torch.cuda.is_available() else "cpu"

global model_id
model_id = 'google/gemma-3-1b-it'
global chat_idx
chat_idx = 0
global chat_history
chat_history = []
global best_tools
best_tools = []
global proteins_list
proteins_list = []
global names_list
names_list = []
global smiles_list
smiles_list = []
global uniprot_list
uniprot_list = []
global pdb_list
pdb_list = []
global chembl_list
chembl_list = []
global present
present = []
global original_query
original_query = ''


quantization_config = BitsAndBytesConfig(load_in_8bit=True)
llm_model = AutoModelForCausalLM.from_pretrained(
model_id, quantization_config=quantization_config).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)
parse_model, document_embeddings, embed_model = start_models()

print(f"Model loaded on {device}")

def reset_chat(self):
  '''
  '''
  global chat_idx
  chat_idx = 0
  global best_tools
  best_tools = []
  global proteins_list
  proteins_list = []
  global names_list
  names_list = []
  global smiles_list
  smiles_list = []
  global uniprot_list
  uniprot_list = []
  global pdb_list
  pdb_list = []
  global chembl_list
  chembl_list = []
  global original_query
  original_query = ''
  global present
  present = []

@spaces.GPU
def new_chat(query: str, ai_flag: str = 'AI'):
  '''
    Chats with the model.

    Args:
      query: The prompt to send to the model.
      ai_flag: whether to use AI or not.
    Returns:
      chat_history: The chat history.
  '''
  global chat_idx
  global chat_history
  global best_tools
  global proteins_list
  global names_list
  global smiles_list
  global uniprot_list
  global pdb_list
  global chembl_list
  global present
  global original_query

  if chat_idx == 0:
    #self.chat_history = []
    local_chat_history = []
    local_chat_history.append(query)

    original_query = query
    best_tools, present, proteins_list, names_list, smiles_list, uniprot_list, pdb_list, chembl_list = \
    intake(query, parse_model, embed_model, document_embeddings)

    response = 'The tools chosen based on your query are:'
    for i,tool in enumerate(best_tools):
      response += '\n' + f'{i+1}. {tool} : {full_tool_descriptions[tool]}'

    response += ' \n\n And the following information was found in your query:\n'
    for (entity_type, entity_list) in zip(present, [proteins_list, names_list, smiles_list, uniprot_list, pdb_list, chembl_list]):
      if present[entity_type] > 0:
        response += f'{entity_type}: {present[entity_type]}\n'
        for entity in entity_list:
          response += f'{entity_type}: {entity}\n'
    response += '\n To accept the #1 tool choice, hit enter; to choose 2 or 3, enter that number.'
    response += '\n To start over, click the clear button and enter a new query.' 
    chat_idx += 1

    local_chat_history.append(response)
    chat_history.append(local_chat_history)

    return '', chat_history, None

  elif chat_idx == 1:
    local_chat_history = []
    local_chat_history.append(query)

    if query == '':
      tool_choice = 0
    else:
      tool_choice = int(query) - 1

    tool_function_hash = define_tool_hash(best_tools[tool_choice], proteins_list, 
                                          names_list, smiles_list, uniprot_list, pdb_list, chembl_list)

    args_list = tool_function_hash[best_tools[tool_choice]][1]
    results_tuple  = tool_function_hash[best_tools[tool_choice]][0](*args_list)

    results_list, results_string, results_images = results_tuple

    if ai_flag == 'Manual':
      local_chat_history.append(results_string)
      chat_history.append(local_chat_history)
      try:
        img = Image.open(io.BytesIO(results_images[0].data))
      except:
        img = None

      reset_chat()

      return '', chat_history, img

    role_text = "Answer the query using the information in the context. Add explanations \
or enriching information where appropriate."

    prompt = f'Query: {original_query}.\n Context: {results_string}'

    messages = [[{
                "role": "system",
                "content": [{"type": "text", "text": role_text},]
            },{
                "role": "user",
                "content": [{"type": "text", "text": prompt},]
            }]]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(llm_model.device) #.to(torch.bfloat16)

    with torch.inference_mode():
        outputs = llm_model.generate(**inputs, max_new_tokens=500, do_sample=True, temperature=0.5)

    outputs = tokenizer.batch_decode(outputs, )

    parts = outputs[0].split('<start_of_turn>model')
    response = parts[1].strip('\n').strip('<end_of_turn>')
    local_chat_history.append(response)
    chat_history.append(local_chat_history)
    chat_idx += 1

    # self.reset_chat()

    #convert results_images[0] from ipython display to an image for gradio
    try:
      img = Image.open(io.BytesIO(results_images[0].data))
    except:
      img = None

    if img != None:
      return '', chat_history, img
    else:
      return '', chat_history, None

  #if chat_idx > 1, call just tool embedding and use existing lists
  elif chat_idx > 1:
    local_chat_history = []
    local_chat_history.append(query)
    original_query = query

    context = chat_history[-1][-1]  # last response from the model
    best_tools, present, proteins_list, names_list, smiles_list, uniprot_list, pdb_list, chembl_list = \
    second_intake(original_query, context, parse_model, embed_model, document_embeddings)

    response = f'Your new query is: {original_query}\n'
    response += 'The tools chosen based on your query are:'
    for i,tool in enumerate(best_tools):
      response += '\n' + f'{i+1}. {tool} : {full_tool_descriptions[tool]}'

    response += ' \n\n And the following information was found in your query:\n'
    for (entity_type, entity_list) in zip(present, [proteins_list, names_list, smiles_list, uniprot_list, pdb_list, chembl_list]):
      if present[entity_type] > 0:
        response += f'{entity_type}: {present[entity_type]}\n'
        for entity in entity_list:
          response += f'{entity_type}: {entity}\n'
    response += '\n To accept the #1 tool choice, hit enter; to choose 2 or 3, enter that number.'
    response += '\n To start over, click the clear button and enter a new query.' 
    chat_idx = 1

    local_chat_history.append(response)
    chat_history.append(local_chat_history)

    return '', chat_history, None


with gr.Blocks() as forest:
  top = gr.Markdown(
      """
      # Chat with MoDrAg! The MOdular DRug design AGent!
      - Currently using the drug design tools below:
      """)
  with gr.Row():
    with gr.Accordion("Tasks available - Click to open/close..", open=False)as prot:
      gr.Markdown('''
                  - Find Uniprot IDs for a protein/gene name.
                  - report the number of bioactive molecules for a protein, organized by Chembl ID.
                  - report the SMILES and IC50 values of bioactive molecules for a particular Chembl ID.
                  - find protein sequences, report number of chains.
                  - find small molecules present in a PDB structure.
                  - find PDB IDs that match a protein.
                  - predict the IC50 value of a small molecule based on a Chembl ID.
                  - generate novel molecules based on a Chembl ID.
                  - find the name of a molecule from the SMILES string.
                  - find the SMILES string of a molecule from the name
                  - find similar or related molecules with some basic properties from a name or SMILES.
                  - calculate Lipinski properties from a name or SMILES string.
                  - find the pharmacophore-similarity between two molecules (a molecule and a reference).
                  - Find the docking score and pose coordinates for a molecules defined by a name or a SMILES string in on of the proteins below:
                  - IGF1R,JAK2,KIT,LCK,MAPK14,MAPKAPK2,MET,PTK2,PTPN1,SRC,ABL1,AKT1,AKT2,CDK2,CSF1R,EGFR,KDR,MAPK1,FGFR1,ROCK1,MAP2K1,
                  PLK1,HSD11B1,PARP1,PDE5A,PTGS2,ACHE,MAOB,CA2,GBA,HMGCR,NOS1,REN,DHFR,ESR1,ESR2,NR3C1,PGR,PPARA,PPARD,PPARG,AR,THRB,
                  ADAM17,F10,F2,BACE1,CASP3,MMP13,DPP4,ADRB1,ADRB2,DRD2,DRD3,ADORA2A,CYP2C9,CYP3A4,HSP90AA1
        ''')


  with gr.Row():
    tools = gr.Radio(choices = ["AI", "Manual"],label="AI or manual??",interactive=True, value = "AI", scale = 2)
    voice_choice = gr.Radio(choices = ['On', 'Off'],label="Audio Voice Response?", interactive=True, value='Off', scale = 2)

  chatbot = gr.Chatbot()


  msg = gr.Textbox(label="Type your messages here and hit enter.", scale = 2)
  with gr.Row():
    chat_btn = gr.Button(value = "Send", scale = 2)
    clear = gr.ClearButton([msg, chatbot], scale = 2)

  image_holder = gr.Image()

  chat_btn.click(new_chat.chat, [msg, tools], [msg, chatbot, image_holder])
  msg.submit(new_chat.chat, [msg, tools], [msg, chatbot, image_holder])
  clear.click(new_chat.reset_chat)


if __name__ == "__main__":
    forest.launch(debug=False, share=True)