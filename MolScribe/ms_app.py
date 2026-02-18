import numpy as np # linear algebra
import pandas as pd
import torch
from molscribe import MolScribe
from huggingface_hub import hf_hub_download
import gradio as gr
from rdkit import Chem
from PIL import Image
import io

ckpt_path = hf_hub_download('yujieq/MolScribe', 'swin_base_char_aux_1m.pth')
model = MolScribe(ckpt_path, device=torch.device('cpu'))

def make_smiles(img):
  '''
  '''
  output = model.predict_image_file(img, return_atoms_bonds=True, return_confidence=True)
  mol = Chem.MolFromSmiles(output['smiles'])
  new_img_raw = Chem.Draw.MolsToGridImage([mol], molsPerRow=1, legends=[output['smiles']])
  new_img = Image.open(io.BytesIO(new_img_raw.data))
  return output['smiles'], new_img

demo = gr.Interface(
    fn = make_smiles,
    inputs=gr.Image(type="filepath"), 
    outputs=[gr.Textbox(lines=2, label="SMILES"), gr.Image(label="New Image")],
    title="MolScribe",
)
    
demo.launch(mcp=True)