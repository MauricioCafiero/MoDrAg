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
                        model = "o4-mini",
                        tools = cafchem_tools,
                        input=prompt,
                        previous_response_id = last_id
            )
        else:
            response = client.responses.create(
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

def prot_accordions():
  elita_text = 'Try queries like: find UNIPROT IDs for the protein MAOB; find PDB IDs for MAOB; how many chains \
are in the PDB structure 4A7G; find PDB IDs matching the protein MAOB; list the bioactive molecules for the CHEMBL \
ID CHEMBL2039; dock the molecule CCCC(F) in the protein DRD2; predict the IC50 value for CCCC(F) based on the CHEMBL \
ID CHEMBL2039; or generate novel molecules based on the CHEMBL ID CHEMBL2039.'
  messages = [{'role': 'assistant', 'content': elita_text}]
  audio_player = voice_from_file('protein.mp3')
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
    workflow = gr.Button(value = "Sample Workflow")

  with gr.Row():
    tools = gr.Radio(choices = ["Yes", "No"],label="Use CafChem tools?",interactive=True, value = "Yes", scale = 2)
    voice_choice = gr.Radio(choices = ['On', 'Off'],label="Audio Voice Response?", interactive=True, value='Off', scale = 2)
  
  chatbot = gr.Chatbot()

  with gr.Row():
    msg = gr.Textbox(label="Type your messages here and hit enter.", scale = 2)
    chat_btn = gr.Button(value = "Send", scale = 0)
  elitas_voice = gr.HTML() 
  
  clear = gr.ClearButton([msg, chatbot])
  
  chat_btn.click(chat, [msg, tools, voice_choice], [msg, chatbot, elitas_voice])
  msg.submit(chat, [msg, tools, voice_choice], [msg, chatbot, elitas_voice])
  workflow.click(prot_workflow, outputs=[elitas_voice, chatbot])
  prot.expand(prot_accordions, outputs = [elitas_voice, chatbot])
  clear.click(clear_history)

if __name__ == "__main__":
    forest.launch(debug=False, share=True)
