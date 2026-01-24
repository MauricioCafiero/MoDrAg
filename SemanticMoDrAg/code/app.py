import torch
import os, re
import gradio as gr
import numpy as np
from intake_function import chat_manager

device = "cuda" if torch.cuda.is_available() else "cpu"

new_chat = chat_manager()
new_chat.start_model_tokenizer()
new_chat.start_support_models()

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


forest.launch(debug=True)