import numpy as np # linear algebra
import pandas as pd
import torch
from molscribe import MolScribe
from huggingface_hub import hf_hub_download
import gradio as gr
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import io
import pubchempy as pcp

def name_node(smiles: str) -> (str):
  '''
    Queries Pubchem for the name of the molecule based on the smiles string.
      Args:
        smiles: the input smiles string
      Returns:
        names_list: the list of names of the molecules
        name_string: a string of the tool results
  '''
  print("name tool")
  print('===================================================')

  name_string = ''
  try:
      res = pcp.get_compounds(smiles, "smiles")
      name = res[0].iupac_name
      name_string += f'{smiles}: IUPAC molecule name: {name}\n'
      syn_list = pcp.get_synonyms(res[0].cid)
      for alt_name in syn_list[0]['Synonym'][:5]:
          name_string += f'{smiles}: alternative or common name: {alt_name}\n'
  except:
      name = "unknown"
      name_string += f'{smiles}: Fail\n'

  return name_string

ckpt_path = hf_hub_download('yujieq/MolScribe', 'swin_base_char_aux_1m.pth')
model = MolScribe(ckpt_path, device=torch.device('cpu'))

def make_smiles(img):
  '''
    Takes in an image file and returns the smiles string and a new image of the molecule.
      Args:
        img: the input image file
      Returns:
        name_string: a string of the tool results
        new_img: a new image of the molecule
  '''
  output = model.predict_image_file(img, return_atoms_bonds=True, return_confidence=True)
  mol = Chem.MolFromSmiles(output['smiles'])
  new_img_raw = Draw.MolsToGridImage([mol], molsPerRow=1, legends=[output['smiles']])
  filename = "chat_image.png"
  new_img_raw.save(filename)
  new_img = Image.open(filename)

  smiles = output['smiles']
  name_string = name_node(smiles)
  return name_string, new_img

def agent_make_smiles(api_flag, img):
  '''
    Takes in an image file and returns the smiles string and a new image of the molecule.
      Args:
        img: the input image file
      Returns:
        name_string: a string of the tool results
        img_list: a list of new images of the molecule
  '''
  output = model.predict_image_file(img, return_atoms_bonds=True, return_confidence=True)
  mol = Chem.MolFromSmiles(output['smiles'])
  new_img_raw = Draw.MolsToGridImage([mol], molsPerRow=1, legends=[output['smiles']])

  smiles = output['smiles']
  name_string = name_node(smiles)

  if api_flag == 'True':
      return name_string, new_img_raw
  else:
      return name_string, None


with gr.Blocks() as imgsmiles:
  top = gr.Markdown(
      """
      # Convert a molecule image to a SMILES string and name with MolScribe
      - Black on white iamges work best
      """)

  agent_flag_choice = gr.Radio(choices = ['True', 'False'],label="Are you an Agent?", interactive=True, value='False', scale = 2)
  with gr.Row():
    inputs=gr.Image(type="filepath")
    with gr.Column():
       text_out = gr.Textbox(lines=2, label="SMILES")
       img_out = gr.Image(label="New Image")

  submit_button = gr.Button("Submit")
  clear_button = gr.ClearButton([inputs, text_out, img_out], value = "Clear")
  agent_button = gr.Button("Agent use only")

  submit_button.click(make_smiles, [inputs], [text_out, img_out])
  agent_button.click(agent_make_smiles, [agent_flag_choice, inputs], [text_out, img_out])

imgsmiles.launch(mcp_server=True, share=True)