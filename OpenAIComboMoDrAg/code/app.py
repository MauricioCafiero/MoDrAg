from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import gradio as gr
import spaces

from PIL import Image
from collections import Counter

from typing import Annotated, TypedDict
import time, sys, os

sys.path.append('code')
from modrag_molecule_functions import *
from modrag_property_functions import *
from modrag_protein_functions import *

openai_key = os.getenv("OPENAI_API_KEY")

tools = [name_node, smiles_node, related_node, structure_node, 
         substitution_node, lipinski_node, pharmfeature_node,
         uniprot_node, listbioactives_node, getbioactives_node,
         predict_node, gpt_node, pdb_node, find_node, docking_node,
         target_node]

model = ChatOpenAI(model_name="gpt-5.2", api_key=openai_key).bind_tools(tools)

class State(TypedDict):
  messages: Annotated[list, add_messages]

def model_node(state: State) -> State:
  res = model.invoke(state['messages'])
  return {'messages': res}

builder = StateGraph(State)
builder.add_node('model', model_node)
builder.add_node('tools', ToolNode(tools))
builder.add_edge(START, 'model')
builder.add_conditional_edges('model', tools_condition)
builder.add_edge('tools',  'model')

graph = builder.compile()
sys_message = SystemMessage(content="You are a helpful cat who says nyan and meow a lot.")
global messages
messages = [sys_message]

def start_chat():
  '''
  '''
  global chat_history, messages, reasoning
  chat_history = []
  reasoning = []
  messages.append(sys_message)

@spaces.GPU
def chat_turn(prompt: str, chat_display):
  '''
  '''  
  global messages, chat_history, reasoning
  human_message = HumanMessage(content=prompt)
  messages.append(human_message)
  local_history = [prompt]

  input = {
      'messages' : messages
  }

  for c in graph.stream(input):
    try:
      ai_mes = c['model']['messages'].content
      messages.append(AIMessage(ai_mes))
      if ai_mes != '':
        print(f'message is {ai_mes}')
        local_history.append(ai_mes)
    except:
      pass
    try:
      if os.path.exists('current_image.png'):
        if os.path.getmtime('current_image.png') > time.time() - 30:
          img = Image.open('current_image.png')
        else:
          img = None
      else:
        img = None
    except:
      img = None
    try:
      reasoning.append(c['tools']['messages'][0].content)
    except:
      pass
  
  if len(local_history) != 2:
    local_history.append('no message')

  chat_history.append({'role': 'user', 'content': local_history[0]})
  chat_history.append({'role': 'assistant', 'content': local_history[1]})
  return '', img, chat_history

def send_reasoning():
  global reasoning
  return reasoning

start_chat()

with gr.Blocks(fill_height=True) as OpenAIMoDrAg:
  gr.Markdown('''
              # MoDrAg Chatbot using ChatGPT 5.2
              - The *MOdular DRug design AGent*!
              - This chatbot can answer questions about molecules, proteins, and their interactions.
              It can also perform tasks such as predicting properties, finding similar molecules, and docking. Try it out!
              - See the tool log box at the bottom for direct tool outputs.
              ''')


  chat = gr.Chatbot()
  with gr.Row(equal_height = True):
    msg = gr.Textbox(label = 'query', scale = 8)
    sub_button = gr.Button("Submit", scale = 2)
  clear = gr.ClearButton([msg, chat])
  img_box = gr.Image()
  reasoning_box = gr.Textbox(label="Tool logs", lines = 20)
  msg.submit(chat_turn, [msg, chat], [msg, img_box, chat]).then(send_reasoning, [], [reasoning_box])
  sub_button.click(chat_turn, [msg, chat], [msg, img_box, chat])
  clear.click(start_chat, [], [])

OpenAIMoDrAg.launch(mcp_server = True)