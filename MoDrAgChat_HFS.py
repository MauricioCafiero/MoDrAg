import openai
import os
from openai import OpenAI
import gradio as gr 
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
import base64

api_key = os.getenv("OPENAI_API_KEY")
eleven_key = os.getenv("eleven_key")

elevenlabs = ElevenLabs(api_key=eleven_key)
client = OpenAI(api_key=api_key)

cafchem_tools = [
		 {
        "type" : "mcp",
        "server_label":"cafiero-proteinagent",
        "server_url":"https://cafierom-proteinagent.hf.space/gradio_api/mcp/",
        "require_approval": "never",
        "allowed_tools": ["ProteinAgent_ProteinAgent"],
    },
    {
        "type" : "mcp",
        "server_label":"cafiero-propagent",
        "server_url":"https://cafierom-propagent.hf.space/gradio_api/mcp/",
        "require_approval": "never",
        "allowed_tools": ["PropAgent_PropAgent"],
    },
    {
        "type" : "mcp",
        "server_label":"cafiero-moleculeagent",
        "server_url":"https://cafierom-moleculeagent.hf.space/gradio_api/mcp/",
        "require_approval": "never",
        "allowed_tools": ["MoleculeAgent_MoleculeAgent"],
    },
    {
        "type" : "mcp",
        "server_label":"cafiero-dockagent",
        "server_url":"https://cafierom-dockagent.hf.space/gradio_api/mcp/",
        "require_approval": "never",
        "allowed_tools": ["DockAgent_DockAgent"],
    },
		 ]

chat_history = []
global last_id
last_id = None

def chat(prompt, tools, voice_choice):
    chat_history.append(
                    {"role": "user", "content": prompt}
    )
    global last_id

    if tools == "Yes":
        if (last_id != None):
            response = client.responses.create(
                        instructions = 'You are a drug design assistant. Use tools provided for information; do not insert \
                        your own knowledge if the information can be obtained from a tool. When calling a tool give complete \
                        sentences for the task parameter. For example, do not define the task as "fetch smiles", rather, say: \
                        "find the smiles string for the molecule".',
                        model = "o4-mini",
                        tools = cafchem_tools,
                        input=prompt,
                        previous_response_id = last_id
            )
        else:
            response = client.responses.create(
                        instructions = 'You are a drug design assistant. Use tools provided for information; do not insert \
                        your own knowledge if the information can be obtained from a tool. When calling a tool give complete \
                        sentences for the task parameter. For example, do not define the task as "fetch smiles", rather, say: \
                        "find the smiles string for the molecule".',
                        model = "o4-mini",
                        tools = cafchem_tools,
                        input=prompt
            )
    else:
        if (last_id != None):
            response = client.responses.create(
                        model = "o4-mini",
        				input=prompt,
                        previous_response_id = last_id
            )
        else:
            response = client.responses.create(
                    model = "o4-mini",
    				input=prompt
            )

    chat_history.append(
                    {"role": "assistant", "content": response.output_text}
    )
    last_id = response.id

    if voice_choice == "On":
        elita_text = response.output_text
    
        voice_settings = {
              "stability": 0.37,
              "similarity_boost": 0.90,
              "style": 0.0,
              "speed": 1.05
              }
    
        audio_stream = elevenlabs.text_to_speech.convert(
          text = elita_text,
          voice_id = 'vxO9F6g9yqYJ4RsWvMbc',
          model_id = 'eleven_multilingual_v2',
          output_format='mp3_44100_128',
          voice_settings=voice_settings
        )
    
        audio_converted = b"".join(audio_stream)
        audio = base64.b64encode(audio_converted).decode("utf-8")
        audio_player = f'<audio src="data:audio/mpeg;base64,{audio}" controls autoplay></audio>'
    else:
        audio_player = ''
    
    return "", chat_history, audio_player

def clear_history():
	global chat_history
	chat_history = []
	global last_id
	last_id = None

def voice_from_file(file_name):
  audio_file = file_name
  with open(audio_file, 'rb') as audio_bytes:
              audio = base64.b64encode(audio_bytes.read()).decode("utf-8")
  audio_player = f'<audio src="data:audio/mpeg;base64,{audio}" controls autoplay></audio>'
  return audio_player
    
def prot_workflow():
  elita_text = "Starting with a protein, try searching for Uniprot IDs, followed by Chembl IDs. \
Then you can look for bioactive molecules for each Chembl ID. You can also search for crystal structures \
in the PDB and get titles of those structures, sequences, numbers of chains, and small molecules in the structure. \
Generate novel bioactive molecules based on a protein Chembl ID using a GPT, or predict an IC50 for a molecule \
based on a protein Chembl ID using a gradient-boosting model."
  messages = [{'role': 'assistant', 'content': elita_text}]
  audio_player = voice_from_file('protein_chat.mp3')
  return audio_player, messages

def mol_workflow():
  elita_text = "Starting with a molecule, try finding it's Lipinski properties, or its pharmacophore similarity to a known active. \
Find similar molecules or, if it has substituted rings, find analogues."
  messages = [{'role': 'assistant', 'content': elita_text}]
  audio_player = voice_from_file('mol_wf.mp3')
  return audio_player, messages

def combo_workflow():
  elita_text ="Starting with a protein and a molecule, try docking the molecule in the protein. If you have a Chembl ID, predict the IC50 \
value of the molecule in the protein."
  messages = [{'role': 'assistant', 'content': elita_text}]
  audio_player = voice_from_file('combo_wf.mp3')
  return audio_player, messages

def prot_accordions():
  elita_text = 'Try queries like: find UNIPROT IDs for the protein MAOB; find PDB IDs for MAOB; how many chains \
are in the PDB structure 4A7G; find PDB IDs matching the protein MAOB; list the bioactive molecules for the CHEMBL \
ID CHEMBL2039; dock the molecule CCCC(F) in the protein DRD2; predict the IC50 value for CCCC(F) based on the CHEMBL \
ID CHEMBL2039; or generate novel molecules based on the CHEMBL ID CHEMBL2039.'
  messages = [{'role': 'assistant', 'content': elita_text}]
  audio_player = voice_from_file('protein.mp3')
  return audio_player, messages

def mol_accordions():
  elita_text = 'Try queries like: Find the name of CCCF, find the smiles for paracetamol, or find molecules similar to paracetamol.'
  messages = [{'role': 'assistant', 'content': elita_text}]
  audio_player = voice_from_file('mol.mp3')
  return audio_player, messages 

def prop_accordions():
  elita_text = 'Try queries like: Find Lipinski properties for CCCF, find pharmacophore-similarity between \
  CCCF and CCCBr, or generate analogues of c1ccc(O)cc1.'
  messages = [{'role': 'assistant', 'content': elita_text}]
  audio_player = voice_from_file('Props.mp3')
  return audio_player, messages

def dock_accordions():
  elita_text = 'Try queries like: dock CCC(F) in the protein MAOB'
  messages = [{'role': 'assistant', 'content': elita_text}]
  audio_player = voice_from_file('Dock.mp3')
  return audio_player, messages

with gr.Blocks() as forest:
  gr.Markdown(
      """
      # Chat with MoDrAg! OpenAI 04-mini can tap into Modrag through an MCP and use all of your favourite drug design tools.
      - Currently using the tools below:
      """)
  with gr.Row():
    with gr.Accordion("Protein Agent - Click to open/close.", open=False)as prot:
      gr.Markdown('''
                  - Find Uniprot IDs for a protein/gene name.
                  - report the number of bioactive molecules for a protein, organized by Chembl ID.
                  - report the SMILES and IC50 values of bioactive molecules for a particular Chembl ID.
                  - find protein sequences, report number fo chains.
                  - find small molecules present in a PDB structure.
                  - find PDB IDs that match a protein.
                  - predict the IC50 value of a small molecule based on a Chembl ID.
                  - generate novel molecules based on a Chembl ID.
      ''')
    with gr.Accordion("Molecule Agent - Click to open/close.", open=False) as mol:
      gr.Markdown('''
                  - find the name of a molecule from the SMILES string.
                  - find the SMILES string of a molecule from the name
                  - find similar or related molecules with some basic properties from a name or SMILES.
      ''')
    with gr.Accordion("Property Agent - Click to open/close.", open=False) as prop:
      gr.Markdown('''
                  - calculate Lipinski properties from a SMILES string. 
                  - find the pharmacophore-similarity between two molecules (a molecule and a reference).
                  - generate analogues of ring molecules and report their QED values.
      ''')
    with gr.Accordion("Docking Agent - Click to open/close.", open=False) as dock:
      gr.Markdown('''
                  - Find the docking score and pose coordinates for a molecules defined by a SMILES string in on of the proteins below:
                  - IGF1R,JAK2,KIT,LCK,MAPK14,MAPKAPK2,MET,PTK2,PTPN1,SRC,ABL1,AKT1,AKT2,CDK2,CSF1R,EGFR,KDR,MAPK1,FGFR1,ROCK1,MAP2K1,
                  PLK1,HSD11B1,PARP1,PDE5A,PTGS2,ACHE,MAOB,CA2,GBA,HMGCR,NOS1,REN,DHFR,ESR1,ESR2,NR3C1,PGR,PPARA,PPARD,PPARG,AR,THRB,
                  ADAM17,F10,F2,BACE1,CASP3,MMP13,DPP4,ADRB1,ADRB2,DRD2,DRD3,ADORA2A,CYP2C9,CYP3A4,HSP90AA1
        ''')
  with gr.Row():
    molecule_workflow = gr.Button(value = "Sample Molecule Workflow")
    protein_workflow = gr.Button(value = "Sample Protein Workflow")
    combined_workflow = gr.Button(value = "Sample Combined Workflow")

  with gr.Row():
    tools = gr.Radio(choices = ["Yes", "No"],label="Use CafChem tools?",interactive=True, value = "Yes", scale = 2)
    voice_choice = gr.Radio(choices = ['On', 'Off'],label="Audio Voice Response?", interactive=True, value='Off', scale = 2)
  
  chatbot = gr.Chatbot()

  with gr.Row():
    msg = gr.Textbox(label="Type your messages here and hit enter.", scale = 2)
    chat_btn = gr.Button(value = "Send", scale = 0)
  talk_ele = gr.HTML() 
  
  clear = gr.ClearButton([msg, chatbot])
  
  chat_btn.click(chat, [msg, tools, voice_choice], [msg, chatbot, talk_ele])
  msg.submit(chat, [msg, tools, voice_choice], [msg, chatbot, talk_ele])
  mol.expand(mol_accordions, outputs = [talk_ele, chatbot])
  prop.expand(prop_accordions, outputs = [talk_ele, chatbot])
  prot.expand(prot_accordions, outputs = [talk_ele, chatbot])
  dock.expand(dock_accordions, outputs = [talk_ele, chatbot])
  molecule_workflow.click(mol_workflow, outputs = [talk_ele, chatbot])
  protein_workflow.click(prot_workflow, outputs = [talk_ele, chatbot])
  combined_workflow.click(combo_workflow, outputs = [talk_ele, chatbot])
  clear.click(clear_history)

if __name__ == "__main__":
    forest.launch(debug=False, share=True)
