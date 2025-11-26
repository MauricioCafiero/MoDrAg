import torch
from typing import Annotated, TypedDict, Literal
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, trim_messages, AIMessage, HumanMessage, ToolCall

from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import chain
from uuid import uuid4
import re
import matplotlib.pyplot as plt
import spaces

from rdkit import Chem
from rdkit.Chem import AllChem, QED
from rdkit.Chem import Draw
from rdkit.Chem.Draw import MolsToGridImage
from rdkit import rdBase
from rdkit.Chem import rdMolAlign
import os, re
from rdkit import RDConfig
import pubchempy as pcp
import gradio as gr
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

hf = HuggingFacePipeline.from_model_id(
    model_id= "microsoft/Phi-4-mini-instruct",
    task="text-generation",
    pipeline_kwargs = {"max_new_tokens": 500, "temperature": 0.4})

chat_model = ChatHuggingFace(llm=hf)

class State(TypedDict):
  '''
    The state of the agent.
  '''
  messages: Annotated[list, add_messages]
  query_smiles: str
  query_task: str
  query_name: str
  query_reference: str
  tool_choice: tuple
  which_tool: int
  props_string: str
  similars_img: str
  loop_again: str

def name_node(state: State) -> State:
  '''
    Queries Pubchem for the name of the molecule based on the smiles string.
      Args:
        smiles: the input smiles string
      Returns:
        name: the name of the molecule
        props_string: a string of the tool results
  '''
  print("name tool")
  print('===================================================')
  current_props_string = state["props_string"]

  try:
    smiles = state["query_smiles"]
    res = pcp.get_compounds(smiles, "smiles")
    name = res[0].iupac_name
    name_string = f'IUPAC molecule name: {name}\n'
    print(smiles, name)
    syn_list = pcp.get_synonyms(res[0].cid)
    for alt_name in syn_list[0]['Synonym'][:5]:
      name_string += f'alternative or common name: {alt_name}\n'
  except:
    name = "unknown"
    name_string = 'Look further.\n'

  state["query_name"] = name

  current_props_string += name_string
  state["props_string"] = current_props_string
  state["which_tool"] += 1
  return state

def smiles_node(state: State) -> State:
  '''
    Queries Pubchem for the smiles string of the molecule based on the name.
      Args:
        smiles: the molecule name
      Returns:
        smiles: the smiles string of the molecule
        props_string: a string of the tool results
  '''
  print("smiles tool")
  print('===================================================')
  current_props_string = state["props_string"]

  try:
    name = state["query_name"]
    res = pcp.get_compounds(name, "name")
    smiles = res[0].smiles
    smiles = smiles.replace('#','~')
    smiles_string = f'The SMILES string for the molecule is: {smiles}\n'
  except:
    smiles = "unknown"
    smiles_string = 'Look further.\n'

  state["query_smiles"] = smiles

  current_props_string += smiles_string
  state["props_string"] = current_props_string
  state["which_tool"] += 1
  return state

def related_node(state: State) -> State:
  '''
    Queries Pubchem for similar molecules based on the smiles string or name
      Args:
        smiles: the input smiles string, OR
        name: the molecule name
      Returns:
        props_string: a string of the tool results.
  '''
  print("related tool")
  print('===================================================')
  current_props_string = state["props_string"]

  print(state['query_name'])
  try:
  #while x == 2:
    if (state['query_smiles'] == None) or (state['query_smiles'] == '') or (state['query_smiles'] == 'none') or (state['query_smiles'] == ' '):
        try:
          name = state["query_name"]
          res = pcp.get_compounds(name, "name")
          smiles = res[0].smiles
          state["query_smiles"] = smiles
          print('got smiles! ', smiles)

          print('trying with smiles')
          res = pcp.get_compounds(smiles, "smiles", searchtype="similarity",listkey_count=50)
          props_string = f'The following molecules are similar to {smiles}: \n'
          print('got related molecules with smiles')
        except:
          print('Not enough information to run related tool with name')
    else:
        print('trying with smiles')
        smiles = state["query_smiles"]
        res = pcp.get_compounds(smiles, "smiles", searchtype="similarity",listkey_count=50)
        props_string = f'The following molecules are similar to {smiles}: \n'
        print('got related molecules with smiles')

    sub_smiles = []

    i = 0
    for compound in res:
      if i == 0:
        print(compound.iupac_name)
        i+=1
      sub_smiles.append(compound.smiles)
      props_string += f'Name: {compound.iupac_name}\n'
      props_string += f'SMILES: {compound.smiles}\n'
      props_string += f'Molecular Weight: {compound.molecular_weight}\n'
      props_string += f'LogP: {compound.xlogp}\n'
      props_string += '==================='

    sub_mols = [Chem.MolFromSmiles(smile) for smile in sub_smiles]
    legend = [str(compound.smiles) for compound in res]

    img = Draw.MolsToGridImage(sub_mols, legends=legend, molsPerRow=4, subImgSize=(250, 250))
    #pic = img.data

    filename = "Similars_image.png"
    # with open(filename+".png",'wb+') as outf:
    #     outf.write(pic)
    img.save(filename)
  except:
    props_string = ''
    filename = None

  current_props_string += props_string
  state["props_string"] = current_props_string
  state['similars_img'] = filename
  state["which_tool"] += 1
  return state

def first_node(state: State) -> State:
  '''
    The first node of the agent. This node receives the input and asks the LLM
    to determine which is the best tool to use to answer the QUERY TASK.
      Input: the initial prompt from the user. should contain only one of more of the following:
             smiles: the smiles string, task: the query task, path: the path to the file,
             reference: the reference smiles
             the value should be separated from the name by a ':' and each field should
             be separated from the previous one by a ','.
             All of these values are saved to the state
      Output: the tool choice
  '''
  query_smiles = None
  state["query_smiles"] = query_smiles
  query_task = None
  state["query_task"] = query_task
  query_name = None
  state["query_name"] = query_name
  query_reference = None
  state["query_reference"] = query_reference
  state['similars_img'] = None
  props_string = ""
  state["props_string"] = props_string
  state["loop_again"] = None

  raw_input = state["messages"][-1].content
  #print(raw_input)
  parts = raw_input.split(',')
  for part in parts:
    if 'query_smiles' in part:
      query_smiles = part.split(':')[1]
      if query_smiles.lower() == 'none':
        query_smiles = None
      state["query_smiles"] = query_smiles
    if 'query_task' in part:
      query_task = part.split(':')[1]
      state["query_task"] = query_task
    if 'query_name' in part:
      query_name = part.split(':')[1]
      if query_name.lower() == 'none':
        query_name = None
      state["query_name"] = query_name
    if 'query_reference' in part:
      query_reference = part.split(':')[1]
      state["query_reference"] = query_reference

  prompt = f'For the QUERY_TASK given below, determine if one or two of the tools descibed below \
can complete the task. If so, reply with only the tool names followed by "#". If two tools \
are required, reply with both tool names separated by a comma and followed by "#". \
If the tools cannot complete the task, reply with "None #".\n \
QUERY_TASK: {query_task}.\n \
The information provided by the user is:\n \
QUERY_SMILES: {query_smiles}.\n \
QUERY_NAME: {query_name}.\n \
Tools: \n \
smiles_tool: queries Pubchem for the smiles string of the molecule based on the name.\n \
name_tool: queries Pubchem for the NAME of the molecule based on the smiles string.\n \
related_tool: queries Pubchem for related or similar molecules based on the smiles string or name and returns 20 results. \
returns the names, SMILES strings, molecular weights and logP values for the related or similar molecules. \n \
'

  res = chat_model.invoke(prompt)
  print(res)

  tool_choices = str(res).replace('smilars', 'smiles').split('<|assistant|>')[1].split('#')[0].strip()
  tool_choices = tool_choices.split(',')
  print(tool_choices)

  if len(tool_choices) == 1:
    tool1 = tool_choices[0].strip()
    if tool1.lower() == 'none':
      tool_choice = (None, None)
    else:
      tool_choice = (tool1, None)
  elif len(tool_choices) == 2:
    tool1 = tool_choices[0].strip()
    tool2 = tool_choices[1].strip()
    if tool1.lower() == 'none' and tool2.lower() == 'none':
      tool_choice = (None, None)
    elif tool1.lower() == 'none' and tool2.lower() != 'none':
      tool_choice = (None, tool2)
    elif tool2.lower() == 'none' and tool1.lower() != 'none':
      tool_choice = (tool1, None)
    else:
      tool_choice = (tool1, tool2)
  else:
    tool_choice = (None, None)

  state["tool_choice"] = tool_choice
  state["which_tool"] = 0
  print(f"First Node. The chosen tools are: {tool_choice}")

  return state

def retry_node(state: State) -> State:
  '''
    If the previous loop of the agent does not get enough informartion from the 
    tools to answer the query, this node is called to retry the previous loop.
      Input: the previous loop of the agent.
      Output: the tool choice
  '''
  query_task = state["query_task"]
  query_smiles = state["query_smiles"]
  query_name = state["query_name"]

  prompt = f'You were previously given the QUERY_TASK below, and asked to determine if one \
or two of the tools descibed below could complete the task. The tool choices did not succeed. \
Please re-examine the tool choices and determine if one or two of the tools descibed below \
can complete the task. If so, reply with only the tool names followed by "#". If two tools \
are required, reply with both tool names separated by a comma and followed by "#". \
If the tools cannot complete the task, reply with "None #".\n \
The information provided by the user is:\n \
QUERY_SMILES: {query_smiles}.\n \
QUERY_NAME: {query_name}.\n \
The task is: \
QUERY_TASK: {query_task}.\n \
Tool options: \n \
smiles_tool: queries Pubchem for the smiles string of the molecule based on the name as input.\n \
name_tool: queries Pubchem for the NAME (IUPAC) of the molecule based on the smiles string as input. \
Also returns a short list of common names for the molecule. \n \
related_tool: queries Pubchem for related or similar molecules based on the smiles string or name as input and returns 20 results. \
Returns the names, SMILES strings, molecular weights and logP values for the related or similar molecules. \n \
'

  res = chat_model.invoke(prompt)

  tool_choices = str(res).replace('smilars', 'smiles').split('<|assistant|>')[1].split('#')[0].strip()
  tool_choices = tool_choices.split(',')
  if len(tool_choices) == 1:
    if tool_choices[0].strip().lower() == 'none':
      tool_choice = (None, None)
    else:
      tool_choice = (tool_choices[0].strip().lower(), None)
  elif len(tool_choices) > 1:
    if tool_choices[0].strip().lower() == 'none':
      tool_choice = (None, tool_choices[1].strip().lower())
    elif tool_choices[1].strip().lower() == 'none':
      tool_choice = (tool_choices[0].strip().lower(), None)
    else:
      tool_choice = (tool_choices[0].strip().lower(), tool_choices[1].strip().lower())
  # elif 'none' in tool_choices[0].strip().lower():
  #   tool_choice = None
  else:
    tool_choice = None

  state["tool_choice"] = tool_choice
  state["which_tool"] = 0
  print(f"The chosen tools are (Retry): {tool_choice}")

  return state

def loop_node(state: State) -> State:
  '''
    This node accepts the tool returns and decides if it needs to call another
    tool or go on to the parser node.
      Input: the tool returns.
      Output: the next node to call.
  '''
  return state

def parser_node(state: State) -> State:
  '''
    This is the third node in the agent. It receives the output from the tool,
    puts it into a prompt as CONTEXT, and asks the LLM to answer the original
    query.
      Input: the output from the tool.
      Output: the answer to the original query.
  '''
  props_string = state["props_string"]
  query_task = state["query_task"]
  tool_choice = state["tool_choice"]

  if type(tool_choice) != tuple and tool_choice == None:
    state["loop_again"] = "finish_gracefully"
    return state
  elif type(tool_choice) == tuple and (tool_choice[0] == None) and (tool_choice[1] == None):
    state["loop_again"] = "finish_gracefully"
    return state

  prompt = f'Using the CONTEXT below, answer the original query, which \
was to answer the QUERY_TASK. End your answer with a "#" \
CONTEXT: {props_string}.\n \
QUERY_TASK: {query_task}.\n '

  res = chat_model.invoke(prompt)
  trial_answer = str(res).split('<|assistant|>')[1]
  print('parser 1 ', trial_answer)
  state["messages"] = res

  check_prompt = f'Determine if the TRIAL ANSWER below answers the original \
QUERY TASK. If it does, respond with "PROCEED #" . If the TRIAL ANSWER did not \
answer the QUERY TASK, respond with "LOOP #" \n \
Only loop again if the TRIAL ANSWER did not answer the QUERY TASK. \
TRIAL ANSWER: {trial_answer}.\n \
QUERY_TASK: {query_task}.\n'

  res = chat_model.invoke(check_prompt)
  print('parser, loop again? ', res)

  if str(res).split('<|assistant|>')[1].split('#')[0].strip().lower() == "loop":
    state["loop_again"] = "loop_again"
    return state
  elif str(res).split('<|assistant|>')[1].split('#')[0].strip().lower() == "proceed":
    state["loop_again"] = None
    print('trying to break loop')
  elif "proceed" in str(res).split('<|assistant|>')[1].lower():
    state["loop_again"] = None
    print('trying to break loop')

  return state

def reflect_node(state: State) -> State:
  '''
    This is the fourth node of the agent. It recieves the LLMs previous answer and
    tries to improve it.
      Input: the LLMs last answer.
      Output: the improved answer.
  '''
  previous_answer = state["messages"][-1].content
  props_string = state["props_string"]

  prompt = f'Look at the PREVIOUS ANSWER below which you provided and the \
TOOL RESULTS. Write an improved answer based on the PREVIOUS ANSWER and the \
TOOL RESULTS by adding additional clarifying and enriching information. End \
your new answer with a "#" \
PREVIOUS ANSWER: {previous_answer}.\n \
TOOL RESULTS: {props_string}. '

  res = chat_model.invoke(prompt)

  return {"messages": res}

def graceful_exit_node(state: State) -> State:
  '''
    Called when the Agent cannot assign any tools for the task
  '''
  props_string = state["props_string"]
  prompt = f'Summarize the information in the CONTEXT, including any useful chemical information. Start your answer with: \
Here is what I found: \n \
CONTEXT: {props_string}'

  res = chat_model.invoke(prompt)

  return {"messages": res}
    

def get_chemtool(state):
  '''
  '''
  which_tool = state["which_tool"]
  tool_choice = state["tool_choice"]
  print('in get_chemtool ',tool_choice)
  if tool_choice == None:
    return None
  if which_tool == 0 or which_tool == 1:
    current_tool = tool_choice[which_tool]
    if current_tool == "smiles_tool" and ("query_name" not in state.keys()):
      current_tool = "name_tool"
      print("Switching from smiles tool to name tool")
    elif current_tool == "name_tool" and ("query_smiles" not in state.keys()):
      current_tool = "smiles_tool"
      print("Switching from name tool to smiles tool")

  elif which_tool > 1:
    current_tool = None

  return current_tool

def loop_or_not(state):
  '''
  '''
  print(f"(line 417) Loop? {state['loop_again']}")
  if state["loop_again"] == "loop_again":
    return True
  elif state["loop_again"] == "finish_gracefully":
    return 'lets_get_outta_here'
  else:
    return False

def pretty_print(answer):
  final = str(answer['messages'][-1]).split('<|assistant|>')[-1].split('#')[0].strip("n").strip('\\').strip('n').strip('\\')
  for i in range(0,len(final),100):
    print(final[i:i+100])

def print_short(answer):
  for i in range(0,len(answer),100):
    print(answer[i:i+100])

builder = StateGraph(State)
builder.add_node("first_node", first_node)
builder.add_node("retry_node", retry_node)
builder.add_node("smiles_node", smiles_node)
builder.add_node("name_node", name_node)
builder.add_node("related_node", related_node)
builder.add_node("loop_node", loop_node)
builder.add_node("parser_node", parser_node)
builder.add_node("reflect_node", reflect_node)
builder.add_node("graceful_exit_node", graceful_exit_node)

builder.add_edge(START, "first_node")
builder.add_conditional_edges("first_node", get_chemtool, {
    "smiles_tool": "smiles_node",
    "name_tool": "name_node",
    "related_tool": "related_node",
    None: "parser_node"})

builder.add_conditional_edges("retry_node", get_chemtool, {
    "smiles_tool": "smiles_node",
    "name_tool": "name_node",
    "related_tool": "related_node",
    None: "parser_node"})

builder.add_edge("smiles_node", "loop_node")
builder.add_edge("name_node", "loop_node")
builder.add_edge("related_node", "loop_node")

builder.add_conditional_edges("loop_node", get_chemtool, {
    "smiles_tool": "smiles_node",
    "name_tool": "name_node",
    "related_tool": "related_node",
    "loop_again": "first_node",
    None: "parser_node"})

builder.add_conditional_edges("parser_node", loop_or_not, {
    True: "retry_node",
    'lets_get_outta_here': "graceful_exit_node",
    False: "reflect_node"})

builder.add_edge("reflect_node", END)
builder.add_edge("graceful_exit_node", END)

graph = builder.compile()

@spaces.GPU
def MoleculeAgent(task, smiles, name):

  #if Similars_image.png exists, remove it
  if os.path.exists('Similars_image.png'):
    os.remove('Similars_image.png')

  input = {
    "messages": [
        HumanMessage(f'query_smiles: {smiles}, query_task: {task}, query_name: {name}')
    ]
  }
  #print(input)

  replies = []
  for c in graph.stream(input): #, stream_mode='updates'):
    m = re.findall(r'[a-z]+\_node', str(c))
    if len(m) != 0:
      try:
          reply = c[str(m[0])]['messages']
          if 'assistant' in str(reply):
            reply = str(reply).split("<|assistant|>")[-1].split('#')[0].strip()
            replies.append(reply)
      except:
          reply = str(c).split("<|assistant|>")[-1].split('#')[0].strip()
          replies.append(reply)
  #check if image exists
  if os.path.exists('Similars_image.png'):
    img_loc = 'Similars_image.png'
    img = Image.open(img_loc)
  #else create a dummy blank image
  else:
    img = Image.new('RGB', (250, 250), color = (255, 255, 255))

  return replies[-1], img

with gr.Blocks(fill_height=True) as forest:
  gr.Markdown('''
              # Molecule Agent - calls the PubChem API to:
              - fetch names
              - fetch SMILES
              - find related or similar molecules
              ''')

  name, smiles = None, None
  with gr.Row():
    with gr.Column():
      smiles = gr.Textbox(label="Molecule SMILES of interest (optional): ", placeholder='none')
      name = gr.Textbox(label="Molecule Name of interest (optional): ", placeholder='none')
      task = gr.Textbox(label="Task for Agent: ")
      calc_btn = gr.Button(value = "Submit to Agent")
    with gr.Column():
      props = gr.Textbox(label="Agent results: ", lines=20 )
      pic = gr.Image(label="Molecule")


      calc_btn.click(MoleculeAgent, inputs = [task, smiles, name], outputs = [props, pic])
      task.submit(MoleculeAgent, inputs = [task, smiles, name], outputs = [props, pic])

forest.launch(debug=False, mcp_server=True)