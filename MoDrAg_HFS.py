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
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
import base64

from rdkit import Chem
from rdkit.Chem import AllChem, QED
from rdkit.Chem import Draw
from rdkit.Chem.Draw import MolsToGridImage
from rdkit import rdBase
from rdkit.Chem import rdMolAlign
import os
from rdkit import RDConfig
import pubchempy as pcp
import gradio as gr
from PIL import Image
from gradio_client import Client
from anthropic import Anthropic

device = "cuda" if torch.cuda.is_available() else "cpu"

hf = HuggingFacePipeline.from_model_id(
    model_id= "microsoft/Phi-4-mini-instruct",
    task="text-generation",
    pipeline_kwargs = {"max_new_tokens": 1000, "temperature": 0.2})

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
  query_protein: str
  query_up_id: str
  query_chembl: str
  query_pdb: str
  tool_choice: tuple
  which_tool: int
  props_string: str
  similars_img: str
  loop_again: str

def molecule_node(state: State) -> State:
  '''
    Calls the Molecule Agent, which can find Names and SMILES strings of molecules
    and return a list of similar molecules with names, SMILES, molecular weights and logP values.
      Args:
        smiles: the input smiles string or name
      Returns:
        name: the name of the molecule
        smiles: the smiles string of the molecule
        related: a list of related or similar molecules with names, SMILES, molecular weights and logP values
  '''
  print("molecule tool")
  print('===================================================')
  current_props_string = state["props_string"]
  query_smiles = state["query_smiles"]
  query_name = state["query_name"]
  full_query_task = state["query_task"]
  print(f"in mol node input: {query_smiles}, {query_name}, {full_query_task}")
  print('===================================================')

  ## separate query task into the part needed for this agent ans disregard the rest
#   prompt = f'Read the FULL_QUERY below. Also read the AGENT_DESCRIPTION. If the FULL_QUERY contains \
# tasks that cannot be solved by the agent described in the AGENT_DESCRIPTION, then separate put the \
# portion of the FULL_QUERY that *can* be solved by this agent. \
# If all of the QUERY_TASK can be completed by the agent, the return the original FULL_QUERY only. If you \
# do separate a portion of the FULL_QUERY, be sure to formulate it as a complete sentence. \n \
# The only text in the response should be either: 1. the orginal FULL_QUERY *or* 2. Just the part of the \
# query that can be answered by the agent. DO NOT ADD ANY OTHER COMMENTARY TO THE OUTPUT. \
# FULL_QUERY: {full_query_task} \n \
# AGENT_DESCRIPTION: Can complete three different tasks: query Pubchem for a molecule name based on the SMILES string \
# or query Pubchem for a SMILES string based on the molecule name, or find molecules related or similar to the given \
# molecule based on the SMILES string or name. \n'

#   process_message = anth_client.messages.create(
#   model="claude-3-haiku-20240307",
#   max_tokens=200,
#   system = "You are part of a node in an AI Agent that separating tasks into parts fort each agent.",
#   messages=[
#       {"role": "user", "content": prompt},
#   ]
#   )

#   query_task = process_message.content[0].text
#   print('new_task: ',query_task)

  client = Client("cafierom/MoleculeAgent")
  try:
    new_text, img = client.predict(full_query_task, query_smiles, query_name, api_name="/MoleculeAgent")
  except:
    new_text = ''
  current_props_string += new_text
  #print(f"in mol node output: {new_text}")
  #print('===================================================')

  filename = "Similars_image.png"
  #img.save(filename)
  #print(type(filename))

  state["similars_img"] = filename
  state["props_string"] = current_props_string
  state["which_tool"] += 1

#   prompt = f'query_name is a name of a molecule. query_smiles is the SMILES string of a molecule. \
# Read the PROPS STRING below. It may contain a query_name or a query_smiles. If so, respond with \
# the following only: # query_name: molecule name # query_smiles: molecule smiles #. If the molecule name \
# is not present in the answer but the SMILES is, respond with: # query_smiles: molecule smiles #. If the molecule \
# SMILES is not present but the name is present, respond with: # query_name: molecule name #. \n \
# PROPS STRING: {new_text} \
# '
#   res = chat_model.invoke(prompt)

#   reply = str(res).split("<|assistant|>")[-1].split('#')[1:]
#   reply[-1] = reply[-1].split("\' additional_kwargs={}")[0]

#   for part in reply:
#     query = part.split(':')
#     if 'name' in part:
#       state['query_name'] = part[1].strip()
#     if 'smiles' in part:
#       state['query_smiles'] = part[1].strip()

  #print(f"in mol node output: {state['query_smiles']}, {state['query_name']}")
  #print('===================================================')

  return state

def property_node(state: State) -> State:
  '''
    Calls the property agent, which can calculate Lipinski properties of molecules, find the
    similarity between two pharmacophores, and generate analogues of molecules with their QED
    values.
      Args:
        smiles: the input smiles string or name
        reference (optional): the smiles string of a reference molecule
      Returns:
        prop_string: a string containing the properties of the molecule
  '''
  print("property tool")
  print('===================================================')
  current_props_string = state["props_string"]
  query_smiles = state["query_smiles"]
  query_reference = state["query_reference"]
  query_task = state["query_task"]
  print(f"in prop node input: {query_smiles}, {query_reference}, {query_task}")
  print('===================================================')

   ## separate query task into the part needed for this agent ans disregard the rest

  client = Client("cafierom/PropAgent")

  try:
    new_text, img = client.predict(query_task, query_smiles, query_reference, api_name="/PropAgent")
  except:
    new_text = ''
  current_props_string += new_text

  filename = "analogues_image.png"
  #img.save(filename)
  print(type(filename))

  state["similars_img"] = filename
  state["props_string"] = current_props_string
  state["which_tool"] += 1
  return state

def protein_node(state: State) -> State:
  '''
    Calls the protein agent, which can answer protein-centric questions
    regarding Uniprot, Chembl bioactivity, and PDB structural data.
  '''
  print("protein tool")
  print('===================================================')
  current_props_string = state["props_string"]
  query_task = state["query_task"]
  query_protein = state["query_protein"]
  query_up_id = state["query_up_id"]
  query_chembl = state["query_chembl"]
  query_pdb = state["query_pdb"]
  query_smiles = state["query_smiles"]
  print(f"in protein node input: task={query_task}, protein={query_protein}, up_id={query_up_id}, chembl={query_chembl}, pdb={query_pdb}, smiles={query_smiles}")
  print('===================================================')

  client = Client("cafierom/ProteinAgent")

  try:
      new_text, img = client.predict(
          query_task,
          query_protein,
          query_up_id,
          query_chembl,
          query_pdb,
          query_smiles,
          api_name="/ProteinAgent"
      )
  except:
      new_text = ''
  current_props_string += new_text

  filename = "proteinagent_image.png"
  state["similars_img"] = filename
  state["props_string"] = current_props_string
  state["which_tool"] += 1
  return state

def dock_node(state: State) -> State:
  '''
    Calls the protein agent, which can answer protein-centric questions
    regarding Uniprot, Chembl bioactivity, and PDB structural data.
  '''
  print("docking tool")
  print('===================================================')
  current_props_string = state["props_string"]
  query_task = state["query_task"]
  query_smiles = state["query_smiles"]
  query_protein = state["query_protein"]
  # add variables as needed

  print(f"in docking node input: task={query_task}, smiles = {query_smiles}, protein = {query_protein}.")
  print('===================================================')

  client = Client("cafierom/DockAgent") # fill in agent

  try:
      new_text, img = client.predict(
          query_task, #add as needed
          query_smiles,
          query_protein,
          api_name="/DockAgent"
      )
  except:
      new_text = ''
  current_props_string += new_text

  filename = "agent_image.png"
  state["similars_img"] = filename
  state["props_string"] = current_props_string
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
  user_input = state["messages"][-1].content

  query_smiles = None
  state["query_smiles"] = query_smiles
  query_task = None
  state["query_task"] = query_task
  query_name = None
  state["query_name"] = query_name
  query_reference = None
  state["query_reference"] = query_reference
  query_protein = None
  state["query_protein"] = query_protein
  query_up_id = None
  state["query_up_id"] = query_up_id
  query_chembl = None
  state["query_chembl"] = query_chembl
  query_pdb = None
  state["query_pdb"] = query_pdb
  state['similars_img'] = None
  props_string = ""
  state["props_string"] = props_string
  state["loop_again"] = None

  prompt = f'Background information: \
QUERY_TASK is the task the user is asking us to perform. It should have as much information as possible about the task. \
QUERY_SMILES is the SMILES string for a molecule that the user provided. \
QUERY_NAME is the name of a molecule that the user provided. It may be an IUPAC name or a common name, such as a drug name. \
QUERY_PROTEIN is the protein that the user provided. \
QUERY_REFERENCE is the SMILES string of a second molecule that the user provided to serve as a reference. \n \
QUERY_UP_ID is a Uniprot ID the user provided. \
QUERY_CHEMBL is a Chembl ID the user provided. \
QUERY_PDB is a PDB ID the user provided. \n \
Examine the USER INPUT below. It should contain a QUERY_TASK and either a QUERY_SMILES or QUERY_NAME. \
It may also contain a QUERY_PROTEIN and a QUERY_REFERENCE. Your task is to extract any of these that are present. \n \
Report your results in the following format: # QUERY_TASK: the task # QUERY_SMILES: the smiles string # QUERY_NAME: the name # \
QUERY_PROTEIN: the protein # QUERY_REFERENCE: the reference smiles string # QUERY_UP_ID: the uniprot id # QUERY_CHEMBL: the chembl id # QUERY_PDB: the pdb id. \
If one of the requested items is not present in the USER INPUT, use NONE as the value. \n \
The QUERY_NAME, QUERY_REFERENCE or QUERY_SMILES may appear in the QUERY_TASK as well. \n \
USER INPUT: {user_input}.\n \
'

  res1 = chat_model.invoke(prompt)


  reply = str(res1).replace('C#','C~').split("<|assistant|>")[-1].split('#')[1:]
  reply[-1] = reply[-1].split("\' additional_kwargs={}")[0]
  for chunk in reply:
    if 'QUERY_SMILES' in chunk:
      query_smiles = chunk.split(':')[1]
      if query_smiles.lower() == 'none':
        query_smiles = None
      else:
        query_smiles = query_smiles.replace('~','#').strip().strip("n").strip('\\').strip('n').strip('\\')
      state["query_smiles"] = query_smiles
    if 'QUERY_TASK' in chunk:
      query_task = chunk.split(':')[1]
      if query_task.lower() == 'none':
        query_task = None
      else:
        query_task = query_task.strip().strip("n").strip('\\').strip('n').strip('\\')
        query_task = query_task.replace('protei','protein')
      state["query_task"] = query_task
    if 'QUERY_NAME' in chunk:
      query_name = chunk.split(':')[1]
      if query_name.lower() == 'none':
        query_name = None
      else:
        query_name = query_name.strip().strip("n").strip('\\').strip('n').strip('\\')
    if 'QUERY_PROTEIN' in chunk:
      query_protein = chunk.split(':')[1]
      if query_protein.lower() == 'none':
        query_protein = None
      else:
        query_protein = query_protein.strip().strip("n").strip('\\').strip('n').strip('\\')
    if 'QUERY_REFERENCE' in chunk:
      query_reference = chunk.split(':')[1]
      if query_reference.lower() == 'none':
        query_reference = None
      else:
        query_reference = query_reference.strip().strip("n").strip('\\').strip('n').strip('\\')
    if 'QUERY_UP_ID' in chunk:
      query_up_id = chunk.split(':')[1]
      if query_up_id.lower() == 'none':
        query_up_id = None
      else:
        query_up_id = query_up_id.strip().strip("n").strip('\\').strip('n').strip('\\')
    if 'QUERY_CHEMBL' in chunk:
      query_chembl = chunk.split(':')[1]
      if query_chembl.lower() == 'none':
        query_chembl = None
      else:
        query_chembl = query_chembl.strip().strip("n").strip('\\').strip('n').strip('\\')
    if 'QUERY_PDB' in chunk:
      query_pdb = chunk.split(':')[1]
      if query_pdb.lower() == 'none':
        query_pdb = None
      else:
        query_pdb = query_pdb.strip().strip("n").strip('\\').strip('n').strip('\\')

  state["query_name"] = query_name
  state["query_task"] = query_task
  state["query_smiles"] = query_smiles
  state['query_protein'] = query_protein
  state['query_up_id'] = query_up_id
  state['query_chembl'] = query_chembl
  state['query_pdb'] = query_pdb
  state['messages'] = res1
  state["query_reference"] = query_reference


  return state

def calling_node(state: State) -> State:
  '''
  '''
  query_task = state["query_task"]
  query_smiles = state["query_smiles"]
  query_name = state["query_name"]
  query_protein = state["query_protein"]
  query_reference = state["query_reference"]
  query_up_id = state["query_up_id"]
  query_chembl = state["query_chembl"]
  query_pdb = state["query_pdb"]

  prompt = f'Examine the QUERY_TASK below as well as the other information provided (SMILES, NAME, PROTEIN, PDB, CHEMBL, UP_ID, REFERENCE) \
and determine if ONE or TWO of the AGENTS descibed below could complete the task. If the AGENTS can complete \
the task, reply as follows. If only one agent is needed: # first_agent_name; if two agents are needed: \
# first_agent_name, second_agent_name. Carefully consider of two agents are needed by the QUERY TASK. \
If the AGENTS cannot complete the task, reply with "# None ". \n \
Do not offer any additional information. \n \
MOLECULE_AGENT: Can complete three different tasks: query Pubchem for a molecule name based on the SMILES string \
or query Pubchem for a SMILES string based on the molecule name, or find molecules related or similar to the given molecule based on the SMILES string \
or name. \n \
PROPERTY_AGENT: Can calculate Lipinski properties of molecules, find the pharmacophore-similarity between two molecule (a molecule and a reference), \
and generate analogues of molecules with their QED values. \n \
PROTEIN_AGENT: Can call Uniprot to find uniprot ids for a protein, can call Chembl to find hits for a given uniprot id and report the  \
number of bioactive molecules in the hit, can call Chembl to find a list bioactive molecules for a given chembl id and their IC50 values, \
can call PDB to find the number of chains in a protein, the protein sequence and any small molecules in the protein structure. \n \
DOCK_AGENT: Can dock a molecule in a protein using AutoDock Vina and return a docking score and the coordinates/XYZ positions of conformation of \
the docked molecule. \n \
QUERY_TASK: {query_task}.\n \
QUERY_SMILES: {query_smiles}.\n \
QUERY_NAME: {query_name}.\n \
QUERY_PROTEIN: {query_protein}.\n \
QUERY_REFERENCE: {query_reference}.\n \
QUERY_UP_ID: {query_up_id}.\n \
QUERY_CHEMBL: {query_chembl}.\n \
QUERY_PDB: {query_pdb}.\n \
'

  res2 = chat_model.invoke(prompt)
  state["messages"] = res2

  reply = str(res2).split("<|assistant|>")[-1].split("\' additional_kwargs={}")[0]
  agents = reply.split(',')
  agents_list = []
  for agent in agents:
    #use regex to replace a space between two letters with an underscore
    agent = re.sub(r'([a-z]) ([A-Z])', r'\1_\2', agent)
    agent = agent.upper()
    agents_list.append(agent.strip('#').strip('*').strip(';').strip('.').strip())

  #   ['# Protein Agent']
  print('in calling node: ',agents_list)

  if len(agents_list) == 1:
    agent = agents_list[0]
    if agent.lower() == 'none':
      tool_choice = (None, None)
    else:
      tool_choice = (agent, None)
  elif len(agents_list) == 2:
    agent1 = agents_list[0]
    agent2 = agents_list[1]
    if agent1.lower() == 'none' and agent2.lower() == 'none':
      tool_choice = (None, None)
    elif agent1.lower() == 'none' and agent2.lower() != 'none':
      tool_choice = (None, agent2)
    elif agent2.lower() == 'none' and agent1.lower() != 'none':
      tool_choice = (agent1, None)
    else:
      tool_choice = (agent1, agent2)
  else:
    tool_choice = (None, None)

  state["tool_choice"] = tool_choice
  state["which_tool"] = 0
  print(f"The chosen tools are: {tool_choice}")

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

  prompt = f'Using the CONTEXT below, answer the original QUERY_TASK. Include any useful context provided \
in the CONTEXT. Remeber that any docking scores reported were calculatd with AutoDock Vina. Begin your answer with a "#" \n \
QUERY_TASK: {query_task}.\n \
CONTEXT: {props_string}.\n '

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
  # print('*'*50)
  print('parser, loop again? ', res)
  # print('*'*50)
  if str(res).split('<|assistant|>')[1].split('#')[0].strip().lower() == "loop":
    state["loop_again"] = "loop_again"
    state["messages"] = res
    return state
  elif str(res).split('<|assistant|>')[1].split('#')[0].strip().lower() == "proceed":
    state["loop_again"] = None

    return state

def gracefulexit_node(state: State) -> State:
  '''
    Called when the Agent cannot assign any tools for the task
  '''
  props_string = state["props_string"]
  prompt = f'Summarize the information in the CONTEXT, including any useful chemical information. Start your answer with: \
Here is what I found: \n \
CONTEXT: {props_string}'

  res = chat_model.invoke(prompt)

  return {"messages": res}

def get_agent(state):
  '''
  '''
  which_tool = state["which_tool"]
  tool_choice = state["tool_choice"]
  #print(tool_choice)
  if tool_choice is None or tool_choice == (None, None):
    return None
  if which_tool == 0 or which_tool == 1:
    current_tool = tool_choice[which_tool]
    if current_tool is None:
      return None
  elif which_tool > 1:
    current_tool = None

  return current_tool

def loop_or_not(state):
  '''
  '''
  print(f"(line 482) Loop? {state['loop_again']}")
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
builder.add_node("calling_node", calling_node)
builder.add_node("molecule_node", molecule_node)
builder.add_node("property_node", property_node)
builder.add_node("protein_node", protein_node)
builder.add_node("dock_node", dock_node)
builder.add_node("loop_node", loop_node)
builder.add_node("parser_node", parser_node)
builder.add_node("gracefulexit_node", gracefulexit_node)

builder.add_edge(START, "first_node")
builder.add_edge("first_node", "calling_node")
builder.add_conditional_edges("calling_node", get_agent, {
    "MOLECULE_AGENT": "molecule_node",
    "PROPERTY_AGENT": "property_node",
    "PROTEIN_AGENT": "protein_node",
    "DOCK_AGENT": "dock_node",
    None: "parser_node"})

builder.add_edge("molecule_node", "loop_node")
builder.add_edge("property_node", "loop_node")
builder.add_edge("protein_node", "loop_node")
builder.add_edge("dock_node", "loop_node")

builder.add_conditional_edges("loop_node", get_agent, {
    "MOLECULE_AGENT": "molecule_node",
    "PROPERTY_AGENT": "property_node",
    "PROTEIN_AGENT": "protein_node",
    "DOCK_AGENT": "dock_node",
    None: "parser_node"})

builder.add_conditional_edges("parser_node", loop_or_not, {
    True: "calling_node",
    'lets_get_outta_here': "gracefulexit_node",
    False: END})

builder.add_edge("gracefulexit_node", END)

graph = builder.compile()

chat_history = []
claude_key = os.getenv("anthropic_key")
anth_client = Anthropic(api_key=claude_key)

@spaces.GPU
def DDAgent(task):

  chat_history.append(
                    {"role": "user", "content": task}
  )
  #if Similars_image.png exists, remove it
  if os.path.exists('Similars_image.png'):
    os.remove('Similars_image.png')

#   prompt_for_claude = f'Read the QUERY_TASK and the CONTEXT below. The QUERY_TASK should contain one or more of the following: \
# QUERY_SMILES is the SMILES string for a molecule that the user provided. \
# QUERY_NAME is the name of a molecule that the user provided. It may be an IUPAC name or a common name, such as a drug name. \
# QUERY_PROTEIN is the protein that the user provided. \
# QUERY_REFERENCE is the SMILES string of a second molecule that the user provided to serve as a reference. \n \
# QUERY_UP_ID is a Uniprot ID the user provided. \
# QUERY_CHEMBL is a Chembl ID the user provided. \
# QUERY_PDB is a PDB ID the user provided. \n \
# Decide if any information from the CONTEXT \
# If the QUERY_TASK provided by the user requires one of QUERY properties but it is not present in the QUERY_TASK, look through the CONTEXT \
# to see if they are present there. If they are, rewrite the QUERY_TASK to include the specific information from the CONTEXT. \
# For example, if the QUERY_TASK says: "find bioactive molecules for that chembl id", and the CONTEXT contains a \
# chembl id (P9877 for example), the you can rewrite the QUERY_TASK as "find bioactive molecules for the chembl ID P9877. \
# Remeber that the rewritten task should contain the original task *and* any additional specific information extracted from the context. \
# Do not refer to the CONTEXT in your response, just insert the needed specific information when appropriate. \
# Output the rewritten task and nothing else. If there is no extra helpful information in the CONTEXT, then output the original \
# QUERY_TASK in its original form with no changes. \n \
# QUERY_TASK: {task}.\n \
# CONTEXT: {str(chat_history)}.\n '

#   process_message = anth_client.messages.create(
#   model="claude-3-haiku-20240307",
#   max_tokens=200,
#   system = "You are part of a node in an AI Agent that is looking through past conversations for context to add to new queries..",
#   messages=[
#       {"role": "user", "content": prompt_for_claude},
#   ]
#   )

#   new_task = process_message.content[0].text
#   print('new_task: ',new_task)

  input = {
    "messages": [
        HumanMessage(f'{task}')
    ]
  }
  #print(input)

  replies = []
  reply = None
  for c in graph.stream(input): #, stream_mode='updates'):
    m = re.findall(r'[a-z]+\_node', str(c))
    if len(m) != 0:
      #print(c)
      if 'messages' in str(c):
        reply = c[str(m[0])]['messages']
      else:
        reply = c[str(m[0])]
      if 'assistant' in str(reply):
        reply = str(reply).split("<|assistant|>")[-1].split('#')[1:]
        reply = ' '.join(reply).split("\' additional_kwargs={}")[0]
        reply = reply.replace('~', '#')
        print(reply)
        print('===================================================')
        replies.append(reply)

  if reply is None:
    reply = "No response generated."
    replies.append(reply)

  #check if image exists
  if os.path.exists('Similars_image.png'):
    img_loc = 'Similars_image.png'
    img = Image.open(img_loc)
  #else create a dummy blank image
  else:
    img = Image.new('RGB', (250, 250), color = (255, 255, 255))
  
  chat_history.append(
                    {"role": "assistant", "content": replies[-1]}
  )

  return "", chat_history, img

def clear_history():
  global chat_history
  chat_history = []

eleven_key = os.getenv("eleven_key")
elevenlabs = ElevenLabs(api_key=eleven_key)

def render_voice(text_in: str):
  voice_settings = {
            "stability": 0.37,
            "similarity_boost": 0.90,
            "style": 0.0,
            "speed": 0.95
        }

  audio_stream = elevenlabs.text_to_speech.convert(
        text = text_in,    
        voice_id = 'vxO9F6g9yqYJ4RsWvMbc',
        model_id = 'eleven_multilingual_v2',
        output_format='mp3_44100_128',
        voice_settings=voice_settings
    )

  audio_converted = b"".join(audio_stream)
  audio = base64.b64encode(audio_converted).decode("utf-8")
  audio_player = f'<audio src="data:audio/mpeg;base64,{audio}" controls autoplay></audio>'

  return audio_player

def mol_accordions():
  elita_text = 'Try queries like: Find the name of CCCF, find the smiles for paracetamol, or find molecules similar to paracetamol.'
  audio_player = render_voice(elita_text)
  return audio_player

def prop_accordions():
  elita_text = 'Try queries like: Find Lipinski properties for CCCF, find pharmacophore-similarity between CCCF and CCCBr, or generate analogues of c1ccc(O)cc1.'
  audio_player = render_voice(elita_text)
  return audio_player

def prot_accordions():
  elita_text = 'Try queries like: Find Uniprot IDs for MAOB, Find PDB IDs for MAOB, or How many chains are in the PDB structure 4A7G' 
  audio_player = render_voice(elita_text)
  return audio_player

def dock_accordions():
  elita_text = 'Try queries like: Dock CCCF in the protein MAOB.'
  audio_player = render_voice(elita_text)
  return audio_player

with gr.Blocks() as forest:
  top = gr.Markdown('''
              # MoDrAg - the *Mo*dular *Dr*ug Design *Ag*ent!
              - Here to perform all of your small molecule and protein based drug design tasks! Currently directing the sub-agents below. Click to see what each agent can do.
              ''')

  with gr.Row():
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
    with gr.Accordion("Protein Agent - Click to open/close.", open=False)as prot:
      gr.Markdown('''
                  - Find Uniprot IDs for a protein/gene name.
                  - report the number of bioactive molecules for a protein, organized by Chembl ID.
                  - report the SMILES and IC50 values of bioactive molecules for a particular Chembl ID.
                  - find protein sequences, report number fo chains.
                  - find small molecules present in a PDB structure,
      ''')
    with gr.Accordion("Docking Agent - Click to open/close.", open=False) as dock:
      gr.Markdown('''
                  - Find the docking score and pose coordinates for a molecules defined by a SMILES string in on of the proteins below:
                  - IGF1R,JAK2,KIT,LCK,MAPK14,MAPKAPK2,MET,PTK2,PTPN1,SRC,ABL1,AKT1,AKT2,CDK2,CSF1R,EGFR,KDR,MAPK1,FGFR1,ROCK1,MAP2K1,
                  PLK1,HSD11B1,PARP1,PDE5A,PTGS2,ACHE,MAOB,CA2,GBA,HMGCR,NOS1,REN,DHFR,ESR1,ESR2,NR3C1,PGR,PPARA,PPARD,PPARG,AR,THRB,
                  ADAM17,F10,F2,BACE1,CASP3,MMP13,DPP4,ADRB1,ADRB2,DRD2,DRD3,ADORA2A,CYP2C9,CYP3A4,HSP90AA1
        ''')

  chatbot = gr.Chatbot(type="messages", placeholder="## Hello, I'm MoDrAg! Let's design together!")

  task = gr.Textbox(label="Type your messages here and hit enter.", scale = 2)
  chat_btn = gr.Button(value = "Send")
  
  clear = gr.ClearButton([task])
  pic = gr.Image(label="Molecules (if needed)")
  talk_ele = gr.HTML()
    
  chat_btn.click(DDAgent, inputs = [task], outputs = [task, chatbot, pic])
  task.submit(DDAgent, [task], [task, chatbot, pic])
  mol.expand(mol_accordions, outputs = [talk_ele])
  prop.expand(prop_accordions, outputs = [talk_ele])
  prot.expand(prot_accordions, outputs = [talk_ele])
  dock.expand(dock_accordions, outputs = [talk_ele])
  clear.click(clear_history)
  
  @gr.render(inputs=top)
  def get_speech(args):
    elita_text = "Hi, I'm Modrag! Let's design together!"
    audio_player = render_voice(elita_text)
    talk_ele = gr.HTML(audio_player)
  

forest.launch(debug=False, mcp_server=True)
