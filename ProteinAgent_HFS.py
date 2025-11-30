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
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, QED
from rdkit.Chem import Draw
from rdkit.Chem.Draw import MolsToGridImage
from rdkit import rdBase
from rdkit.Chem import rdMolAlign
import os, re
from rdkit import RDConfig
import gradio as gr
from PIL import Image

import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from tqdm.auto import tqdm
import requests 
import spaces
from rcsbapi.search import TextQuery
import requests
import itertools

import lightgbm as lgb
from lightgbm import LGBMRegressor
import deepchem as dc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import random
from finetune_gpt import *

device = "cuda" if torch.cuda.is_available() else "cpu"

hf = HuggingFacePipeline.from_model_id(
    model_id= "microsoft/Phi-4-mini-instruct",
    task="text-generation",
    pipeline_kwargs = {"max_new_tokens": 800, "temperature": 0.4})

chat_model = ChatHuggingFace(llm=hf)

class State(TypedDict):
  '''
    The state of the agent.
  '''
  messages: Annotated[list, add_messages]
  query_smiles: str
  query_task: str
  query_protein: str
  query_up_id: str
  query_pdb: str
  query_chembl: str
  tool_choice: tuple
  which_tool: int
  props_string: str
  loop_again: str
  which_pdbs: int
  #(Literal["lipinski_tool", "substitution_tool", "pharm_feature_tool"],
  #                   Literal["lipinski_tool", "substitution_tool", "pharm_feature_tool"])


def uniprot_node(state: State) -> State:
  '''
    This tool takes in the user requested protein and searches UNIPROT for matches.
    It returns a string scontaining the protein ID, gene name, organism, and protein name.
      Args:
        query_protein: the name of the protein to search for.
      Returns:
        protein_string: a string containing the protein ID, gene name, organism, and protein name.

  '''
  print("UNIPROT tool")
  print('===================================================')

  protein_name = state["query_protein"]
  current_props_string = state["props_string"]

  try:
    url = f'https://rest.uniprot.org/uniprotkb/search?query={protein_name}&format=tsv'
    response = requests.get(url).text

    f = open(f"{protein_name}_uniprot_ids.tsv", "w")
    f.write(response)
    f.close()

    prot_df = pd.read_csv(f'{protein_name}_uniprot_ids.tsv', sep='\t')
    prot_human_df = prot_df[prot_df['Organism'] == "Homo sapiens (Human)"]
    print(f"Found {len(prot_human_df)} Human proteins out of {len(prot_df)} total proteins")

    prot_ids = prot_df['Entry'].tolist()
    prot_ids_human = prot_human_df['Entry'].tolist()

    genes = prot_df['Gene Names'].tolist()
    genes_human = prot_human_df['Gene Names'].tolist()

    organisms = prot_df['Organism'].tolist()

    names = prot_df['Protein names'].tolist()
    names_human = prot_human_df['Protein names'].tolist()

    protein_string = ''
    for id, gene, organism, name in zip(prot_ids, genes, organisms, names):
      protein_string += f'Protein ID: {id}, Gene: {gene}, Organism: {organism}, Name: {name}\n'

  except:
    protein_string = 'No proteins found'

  current_props_string += protein_string
  state["props_string"] = current_props_string
  state["which_tool"] += 1
  return state

def get_qed(smiles):
  '''
    Helper function to compute QED for a given molecule.
      Args:
        smiles: the input smiles string
      Returns:
        qed: the QED score of the molecule.
  '''
  mol = Chem.MolFromSmiles(smiles)
  qed = Chem.QED.default(mol)

  return qed

def listbioactives_node(state: State) -> State:
  '''
    Accepts a UNIPROT ID and searches for bioactive molecules
      Args:
        up_id: the UNIPROT ID of the protein to search for.
      Returns:
        props_string: the number of bioactive molecules for the given protein
  '''
  print("List bioactives tool")
  print('===================================================')

  up_id = state["query_up_id"].strip()
  current_props_string = state["props_string"]

  targets = new_client.target
  bioact = new_client.activity

  try:
    target_info = targets.get(target_components__accession=up_id).only("target_chembl_id","organism", "pref_name", "target_type")
    target_info = pd.DataFrame.from_records(target_info)
    print(target_info)
    if len(target_info) > 0:
      print(f"Found info for Uniprot ID: {up_id}")

    chembl_ids = target_info['target_chembl_id'].tolist()

    chembl_ids = list(set(chembl_ids))
    print(f"Found {len(chembl_ids)} unique ChEMBL IDs")

    len_all_bioacts = []
    bioact_string = ''
    for chembl_id in chembl_ids:
      bioact_chosen = bioact.filter(target_chembl_id=chembl_id, type="IC50", relation="=").only(
          "molecule_chembl_id",
          "type",
          "standard_units",
          "relation",
          "standard_value",
      )
      len_this_bioacts = len(bioact_chosen)
      len_all_bioacts.append(len_this_bioacts)
      this_bioact_string = f"Lenth of Bioactivities for ChEMBL ID {chembl_id}: {len_this_bioacts}"

      bioact_string += this_bioact_string + '\n'
  except:
    bioact_string = 'No bioactives found\n'

  current_props_string += bioact_string
  state["props_string"] = current_props_string
  state["which_tool"] += 1
  return state

def getbioactives_node(state: State) -> State:
  '''
    Accepts a Chembl ID and get all bioactives molecule SMILES and IC50s for that ID
      Args:
        chembl_id: the chembl ID to query
      Returns:
        props_string: the bioactive molecule SMILES and IC50s for the chembl ID
  '''
  print("Get bioactives tool")
  print('===================================================')

  chembl_id = state["query_chembl"].strip()
  current_props_string = state["props_string"]

  #check if f'{chembl_id}_bioactives.csv' exists
  if os.path.exists(f'{chembl_id}_bioactives.csv'):
    print(f'Found {chembl_id}_bioactives.csv')
    total_bioact_df = pd.read_csv(f'{chembl_id}_bioactives.csv')
    print(f"number of records: {len(total_bioact_df)}")
  else:

    compounds = new_client.molecule
    bioact = new_client.activity

    bioact_chosen = bioact.filter(target_chembl_id=chembl_id, type="IC50", relation="=").only(
        "molecule_chembl_id",
        "type",
        "standard_units",
        "relation",
        "standard_value",
    )

    chembl_ids = []
    ic50s = []
    for record in bioact_chosen:
        if record["standard_units"] == 'nM':
            chembl_ids.append(record["molecule_chembl_id"])
            ic50s.append(float(record["standard_value"]))

    bioact_dict = {'chembl_ids' : chembl_ids, 'IC50s': ic50s}
    bioact_df = pd.DataFrame.from_dict(bioact_dict)
    bioact_df.drop_duplicates(subset=["chembl_ids"], keep= "last")
    print(f"Number of records: {len(bioact_df)}")
    print(bioact_df.shape)


    compounds_provider = compounds.filter(molecule_chembl_id__in=bioact_df["chembl_ids"].to_list()).only(
        "molecule_chembl_id",
        "molecule_structures"
    )

    cids_list = []
    smiles_list = []

    for record in compounds_provider:
        cid = record['molecule_chembl_id']
        cids_list.append(cid)

        if record['molecule_structures']:
            if record['molecule_structures']['canonical_smiles']:
                smile = record['molecule_structures']['canonical_smiles']
            else:
                print("No canonical smiles")
                smile = None
        else:
            print('no structures')
            smile = None
        smiles_list.append(smile)

    new_dict = {'SMILES': smiles_list, 'chembl_ids_2': cids_list}
    new_df = pd.DataFrame.from_dict(new_dict)

    total_bioact_df = pd.merge(bioact_df, new_df, left_on='chembl_ids', right_on='chembl_ids_2')
    print(f"number of records: {len(total_bioact_df)}")

    total_bioact_df.drop_duplicates(subset=["chembl_ids"], keep= "last")
    print(f"number of records after removing duplicates: {len(total_bioact_df)}")

    total_bioact_df.dropna(axis=0, how='any', inplace=True)
    total_bioact_df.drop(["chembl_ids_2"],axis=1,inplace=True)
    print(f"number of records after dropping Null values: {len(total_bioact_df)}")

    total_bioact_df.sort_values(by=["IC50s"],inplace=True)

    if len(total_bioact_df) > 0:
      total_bioact_df.to_csv(f'{chembl_id}_bioactives.csv')

  limit = 50
  if len(total_bioact_df) > limit:
    total_bioact_df = total_bioact_df.iloc[:limit]

  bioact_string = f'Results for top bioactivity (IC50 value) for molecules in ChEMBL ID: {chembl_id}. \n'
  for smile, ic50 in zip(total_bioact_df['SMILES'], total_bioact_df['IC50s']):
    smile = smile.replace('#','~')
    bioact_string += f'Molecule SMILES: {smile}, IC50 (nM): {ic50}\n'

  mols = [Chem.MolFromSmiles(smile) for smile in total_bioact_df['SMILES'].to_list()]
  legends = [f'IC50: {ic50}' for ic50 in total_bioact_df['IC50s'].to_list()]
  img = MolsToGridImage(mols, molsPerRow=5, legends=legends, subImgSize=(200,200))
  filename = "Substitution_image.png"
    # pic = img.data
    # with open(filename,'wb+') as outf:
    #   outf.write(pic)
  img.save(filename)

  current_props_string += bioact_string
  state["props_string"] = current_props_string
  state["which_tool"] += 1
  return state

def predict_node(state: State) -> State:
  '''
    uses the current_bioactives.csv file from the get_bioactives node to fit the
    Light GBM model and predict the IC50 for the current smiles.
  '''
  print("Predict Tool")
  print('===================================================')
  current_props_string = state["props_string"]
  smiles = state["query_smiles"]
  chembl_id = state["query_chembl"].strip()
  print(f"in predict node, smiles: {smiles}")

  try:
    df = pd.read_csv(f'{chembl_id}_bioactives.csv')
    #if length of the dataframe is over 2000, take a random sample of 2000 points
    if len(df) > 2000:
      df = df.sample(n=2000, random_state=42)

    y_raw = df["IC50s"].to_list()
    smiles_list = df["SMILES"].to_list()
    ions_to_clean = ["[Na+].",".[Na+]","[Cl-].",".[Cl-]","[K+].",".[K+]"]
    Xa = []
    y = []
    for smile, value in zip(smiles_list, y_raw):
      for ion in ions_to_clean:
        smile = smile.replace(ion,"")
      y.append(np.log10(value))
      Xa.append(smile)

    mols = [Chem.MolFromSmiles(smile) for smile in Xa]
    print(f"Number of molecules: {len(mols)}")

    featurizer=dc.feat.RDKitDescriptors()
    featname="RDKitDescriptors"
    f = featurizer.featurize(mols)

    nan_indicies = np.isnan(f)
    bad_rows = []
    for i, row in enumerate(nan_indicies):
        for item in row:
            if item == True:
                if i not in bad_rows:
                    print(f"Row {i} has a NaN.")
                    bad_rows.append(i)

    print(f"Old dimensions are: {f.shape}.")

    for j,i in enumerate(bad_rows):
        k=i-j
        f = np.delete(f,k,axis=0)
        y = np.delete(y,k,axis=0)
        Xa = np.delete(Xa,k,axis=0)
        print(f"Deleting row {k} from arrays.")

    print(f"New dimensions are: {f.shape}")
    if f.shape[0] != len(y) or f.shape[0] != len(Xa):
      raise ValueError("Number of rows in X and y do not match.")

    X_train, X_test, y_train, y_test = train_test_split(f, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LGBMRegressor(metric='rmse', max_depth = 50, verbose = -1, num_leaves = 31,
                          feature_fraction = 0.8, min_data_in_leaf = 20)
    modelname = "LightGBM Regressor"
    model.fit(X_train, y_train)

    train_score = model.score(X_train,y_train)
    print(f"score for training set: {train_score:.3f}")

    valid_score = model.score(X_test, y_test)
    print(f"score for validation set: {valid_score:.3f}")

    for ion in ions_to_clean:
      smiles = smiles.replace(ion,"")
    test_mol = Chem.MolFromSmiles(smiles)
    test_feat = featurizer.featurize([test_mol])
    test_feat = scaler.transform(test_feat)
    prediction = model.predict(test_feat)
    test_ic50 = 10**(prediction[0])
    print(f"Predicted IC50: {test_ic50}")
    prop_string = f"The predicted IC50 value for the test molecule is : {test_ic50:.3f} nM. \
The Bioactive data was fitted with the LightGMB model, using RDKit descriptors. The trainin score \
was {train_score:.3f} and the testing score was {valid_score:.3f}. "
    print(prop_string)

  except:
    prop_string = ''

  current_props_string += prop_string
  state["props_string"] = current_props_string
  state["which_tool"] += 1
  return state

def gpt_node(state: State) -> State:
  '''
    Uses a Chembl dataset, previously stored in a CSV file by the get_bioactives node, to
    to finetune a GPT model to generate novel molecules for the target protein.

    Args:
      chembl_id
    returns:
      prop_string: a string of the novel, generated molecules
  '''
  print("GPT node")
  print('===================================================')
  current_props_string = state["props_string"]
  chembl_id = state["query_chembl"].strip()
  print(f"in gpt node, chembl id: {chembl_id}")

  try:
    df = pd.read_csv(f'{chembl_id}_bioactives.csv')
    prop_string, img = finetune_gpt(df, chembl_id) 
    prop_string = prop_string.replace("#","~")
    filename = "Substitution_image.png"
    # pic = img.data
    # with open(filename,'wb+') as outf:
    #   outf.write(pic)
    img.save(filename)
  except:
    prop_string = ''

  current_props_string += prop_string
  state["props_string"] = current_props_string
  state["which_tool"] += 1
  return state

def get_protein_from_pdb(pdb_id):
  url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
  r = requests.get(url)
  return r.text

def one_to_three(one_seq):
  rev_aa_hash = {
      'A': 'ALA',
      'R': 'ARG',
      'N': 'ASN',
      'D': 'ASP',
      'C': 'CYS',
      'Q': 'GLN',
      'E': 'GLU',
      'G': 'GLY',
      'H': 'HIS',
      'I': 'ILE',
      'L': 'LEU',
      'K': 'LYS',
      'M': 'MET',
      'F': 'PHE',
      'P': 'PRO',
      'S': 'SER',
      'T': 'THR',
      'W': 'TRP',
      'Y': 'TYR',
      'V': 'VAL'
  }

  try:
    three_seq = rev_aa_hash[one_seq]
  except:
    three_seq = 'X'

  return three_seq

def three_to_one(three_seq):
  aa_hash = {
      'ALA': 'A',
      'ARG': 'R',
      'ASN': 'N',
      'ASP': 'D',
      'CYS': 'C',
      'GLN': 'Q',
      'GLU': 'E',
      'GLY': 'G',
      'HIS': 'H',
      'ILE': 'I',
      'LEU': 'L',
      'LYS': 'K',
      'MET': 'M',
      'PHE': 'F',
      'PRO': 'P',
      'SER': 'S',
      'THR': 'T',
      'TRP': 'W',
      'TYR': 'Y',
      'VAL': 'V'
  }

  one_seq = []
  for residue in three_seq:
    try:
      one_seq.append(aa_hash[residue])
    except:
      one_seq.append('X')

  return one_seq

def pdb_node(state: State) -> State:
  '''
    Accepts a PDB ID and queires the protein databank for the sequence of the protein, as well as other
    information such as ligands.
      Args:
        pdb: the PDB ID to query
      Returns:
        props_string: a string of the
  '''
  test_pdb = state["query_pdb"].strip()
  current_props_string = state["props_string"]

  print(f"pdb tool using {test_pdb}")
  print('===================================================')

  pdb_str = get_protein_from_pdb(test_pdb)
  chains = {}
  other_molecules = {}

  #print(pdb_str.split('\n')[0])
  for line in pdb_str.split('\n'):
    parts = line.split()
    try:
      if parts[0] == 'SEQRES':
        if parts[2] not in chains:
          chains[parts[2]] = []
        chains[parts[2]].extend(parts[4:])
      if parts[0] == 'HETNAM':
        j = 1
        if parts[1].strip() in ['2','3','4','5','6','7','8','9']:
          j = 2
        print(parts[j])
        if parts[j] not in other_molecules:
          other_molecules[parts[j]] = []
        other_molecules[parts[j]].extend(parts[2:])
    except:
      print('Blank line')

    chains_ol = {}
    for chain in chains:
      chains_ol[chain] = three_to_one(chains[chain])

  props_string = f"Chains in PDB ID {test_pdb}: {', '.join(chains.keys())} \n"
  for chain in chains_ol:
    props_string += f"Chain {chain}: {''.join(chains_ol[chain])} \n"
    print(f"Chain {chain}: {''.join(chains_ol[chain])}")
  props_string += f"Ligands in PDB ID {test_pdb}.\n"
  for mol in other_molecules:
    props_string += f"Molecule {mol}: {''.join(other_molecules[mol])} \n"

  current_props_string += props_string
  state["props_string"] = current_props_string
  state["which_tool"] += 1
  return state

def find_node(state: State) -> State:
  '''
    Accepts a protein name and searches the protein databack for PDB IDs that match along with the entry titles.
      Args:
        protein_name: the protein to query
      Returns:
        props_string: a string of the
  '''
  test_protein = state["query_protein"].strip()
  which_pdbs = state["which_pdbs"]
  current_props_string = state["props_string"]

  print(f"find tool using {test_protein}")
  print('===================================================')

  try:
    query = TextQuery(value=test_protein)
    results = query()

    def pdb_gen():
      for rid in results:
        yield(rid)

    take10 = itertools.islice(pdb_gen(), which_pdbs, which_pdbs+10, 1)

    pdb_string = f'10 PDBs that match the protein {test_protein} are: \n'
    for pdb in take10:
      data = requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb}").json()
      title = data['struct']['title']
      pdb_string += f'PDB ID: {pdb}, with title: {title} \n'
    state["which_pdbs"] = which_pdbs+10
  except:
    pdb_string = ''


  current_props_string += pdb_string
  state["props_string"] = current_props_string
  state["which_tool"] += 1
  return state

def first_node(state: State) -> State:
  '''
    The first node of the agent. This node receives the input and asks the LLM
    to determine which is the best tool to use to answer the QUERY TASK.
      Input: the initial prompt from the user. should contain only one of more of the following:
             query_protein: the name of the protein to search for.
             query_up_id: the Uniprot ID of the protein to search for.
             query_chembl: the chembl ID to query
             query_pdb: the PDB ID to query
             query_smiles: the smiles string
             query_task: the query task
             the value should be separated from the name by a ':' and each field should
             be separated from the previous one by a ','.
             All of these values are saved to the state
      Output: the tool choice
  '''
  query_smiles = None
  state["query_smiles"] = query_smiles
  query_task = None
  state["query_task"] = query_task
  query_protein = None
  state["query_protein"] = query_protein
  query_up_id = None
  state["query_up_id"] = query_up_id
  query_pdb = None
  state["query_pdb"] = query_pdb
  query_chembl = None
  state["query_chembl"] = query_chembl
  props_string = ""
  state["props_string"] = props_string
  state["loop_again"] = None
  state['which_pdbs'] = 0

  raw_input = state["messages"][-1].content
  parts = raw_input.split(',')
  for part in parts:
    if 'smiles' in part:
      query_smiles = part.split(':')[1]
      if query_smiles.lower() == 'none':
        query_smiles = None
      state["query_smiles"] = query_smiles
    if 'task' in part:
      query_task = part.split(':')[1]
      state["query_task"] = query_task
    if 'protein' in part:
      query_protein = part.split(':')[1]
      if query_protein.lower() == 'none':
        query_protein = None
      state["query_protein"] = query_protein
    if 'up_id' in part:
      query_up_id = part.split(':')[1]
      if query_up_id.lower() == 'none':
        query_up_id = None
      state["query_up_id"] = query_up_id
    if 'pdb' in part:
      query_pdb = part.split(':')[1]
      if query_pdb.lower() == 'none':
        query_pdb = None
      state["query_pdb"] = query_pdb
    if 'chembl' in part:
      query_chembl = part.split(':')[1]
      if query_chembl.lower() == 'none':
        query_chembl = None
      state["query_chembl"] = query_chembl

  prompt = f'For the QUERY_TASK given below, determine if one or two of the tools descibed below \
can complete the task. If so, reply with only the tool names followed by "#". If two tools \
are required, reply with both tool names separated by a comma and followed by "#". \
If the tools cannot complete the task, reply with "None #".\n \
QUERY_TASK: {query_task}.\n \
Tools: \n \
uniprot_tool: this tool takes in the user requested protein and searches UNIPROT for matches. \
It returns a string containing the protein ID, gene name, organism, and protein name.\n \
list_bioactives_tool: Accepts a given UNIPROT ID and searches for bioactive molecules \n \
get_bioactives_tool: Accepts a Chembl ID and get all bioactives molecule SMILES and IC50s for that ID\n \
pdb_tool: Accepts a PDB ID and queires the protein databank for the number of chains in and sequence of the \n \
protein, as well as other information such as ligands in the structure. \
find_tool: Accepts a protein name and seaches for PDB IDs that match, returning the PDB ID and the title. \
predict_tool: Predicts the IC50 value for the molecule indicated by the SMILES string provided. \
Uses the LightGBM model. \n \
gpt_tool: Uses a machine-learning GPT model to generate novel molecules for a chembl dataset. It returns a list  \
of novel molecules generated by the GPT. \
'
  res = chat_model.invoke(prompt)

  tool_choices = str(res).split('<|assistant|>')[1].split('#')[0].strip()
  tool_choices = tool_choices.split(',')

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
  print(f"The chosen tools are: {tool_choice}")

  return state

def retry_node(state: State) -> State:
  '''
    If the previous loop of the agent does not get enough information from the
    tools to answer the query, this node is called to retry the previous loop.
      Input: the previous loop of the agent.
      Output: the tool choice
  '''
  query_task = state["query_task"]
  query_protein = state["query_protein"]
  query_up_id = state["query_up_id"]
  query_chembl = state["query_chembl"]
  query_pdb = state["query_pdb"]
  query_smiles = state["query_smiles"]

  prompt = f'You were previously given the QUERY_TASK below, and asked to determine if one \
or two of the tools described below could complete the task. The tool choices did not succeed. \
Please re-examine the tool choices and determine if one or two of the tools described below \
can complete the task. If so, reply with only the tool names followed by "#". If two tools \
are required, reply with both tool names separated by a comma and followed by "#". \
If the tools cannot complete the task, reply with "None #".\n \
The information provided by the user is:\n \
QUERY_PROTEIN: {query_protein}.\n \
QUERY_UP_ID: {query_up_id}.\n \
QUERY_CHEMBL: {query_chembl}.\n \
QUERY_PDB: {query_pdb}.\n \
QUERY_SMILES: {query_smiles}.\n \
The task is: \
QUERY_TASK: {query_task}.\n \
Tool options: \n \
uniprot_tool: this tool takes in the user requested protein and searches UNIPROT for matches. \
It returns a string containing the protein ID, gene name, organism, and protein name.\n \
list_bioactives_tool: Accepts a given UNIPROT ID and searches for bioactive molecules \n \
get_bioactives_tool: Accepts a Chembl ID and get all bioactives molecule SMILES and IC50s for that ID\n \
pdb_tool: Accepts a PDB ID and queires the protein databank for the number of chains in and sequence of the \
protein, as well as other information such as ligands in the structure. \n \
find_tool: Accepts a protein name and seaches for PDB IDs that match, returning the PDB ID and the title. \
predict_tool: Predicts the IC50 value for the molecule indicated by the SMILES string provided. \
Uses the LightGBM model. \n \
gpt_tool: Uses a machine-learning GPT model to generate novel molecules for a chembl dataset. It returns a list  \
of novel molecules generated by the GPT. \
'

  res = chat_model.invoke(prompt)

  tool_choices = str(res).split('<|assistant|>')[1].split('#')[0].strip()
  tool_choices = tool_choices.split(',')

  if len(tool_choices) == 1:
    tool1 = tool_choices[0].strip()
    if tool1.lower() == 'none':
      tool_choice = (None, None)
    else:
      tool_choice = (tool1.strip(), None)
  elif len(tool_choices) > 1:
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
was to answer the QUERY_TASK. Remember that novel molecules generated in the CONTEXT \
were made using a fine-tuned GPT. End your answer with a "#" \
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

def get_chemtool(state):
  '''
  '''
  which_tool = state["which_tool"]
  tool_choice = state["tool_choice"]

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
  print(f"(line 690) Loop? {state['loop_again']}")
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
builder.add_node("uniprot_node", uniprot_node)
builder.add_node("listbioactives_node", listbioactives_node)
builder.add_node("getbioactives_node", getbioactives_node)
builder.add_node("pdb_node", pdb_node)
builder.add_node("find_node", find_node)
builder.add_node("predict_node", predict_node)
builder.add_node("gpt_node", gpt_node)

builder.add_node("loop_node", loop_node)
builder.add_node("parser_node", parser_node)
builder.add_node("reflect_node", reflect_node)
builder.add_node("gracefulexit_node", gracefulexit_node)

builder.add_edge(START, "first_node")
builder.add_conditional_edges("first_node", get_chemtool, {
    "uniprot_tool": "uniprot_node",
    "list_bioactives_tool": "listbioactives_node",
    "get_bioactives_tool": "getbioactives_node",
    "pdb_tool": "pdb_node",
    "find_tool": "find_node",
    "predict_tool": "predict_node",
    "gpt_tool": "gpt_node",
    None: "parser_node"})

builder.add_conditional_edges("retry_node", get_chemtool, {
    "uniprot_tool": "uniprot_node",
    "list_bioactives_tool": "listbioactives_node",
    "get_bioactives_tool": "getbioactives_node",
    "pdb_tool": "pdb_node",
    "find_tool": "find_node",
    "predict_tool": "predict_node",
    "gpt_tool": "gpt_node",
    None: "parser_node"})

builder.add_edge("uniprot_node", "loop_node")
builder.add_edge("listbioactives_node", "loop_node")
builder.add_edge("getbioactives_node", "loop_node")
builder.add_edge("pdb_node", "loop_node")
builder.add_edge("find_node", "loop_node")
builder.add_edge("predict_node", "loop_node")
builder.add_edge("gpt_node", "loop_node")

builder.add_conditional_edges("loop_node", get_chemtool, {
    "uniprot_tool": "uniprot_node",
    "list_bioactives_tool": "listbioactives_node",
    "get_bioactives_tool": "getbioactives_node",
    "pdb_tool": "pdb_node",
    "find_tool": "find_node",
    "predict_tool": "predict_node",
    "gpt_tool": "gpt_node",
    None: "parser_node"})

builder.add_conditional_edges("parser_node", loop_or_not, {
    True: "retry_node",
    'lets_get_outta_here': "gracefulexit_node",
    False: "reflect_node"})

builder.add_edge("reflect_node", END)
builder.add_edge("gracefulexit_node", END)

graph = builder.compile()

@spaces.GPU
def ProteinAgent(task, protein, up_id, chembl_id, pdb_id, smiles):
  '''
  This Agent can perform several protein-related tasks. It can find UNIPROT IDs for a protein, or, given
  a UNIPROT ID it can find Chembl IDs that match. It can find numbers of and lists of bioactive molecules
  based on a Chembl ID. It can query the protein databank to find PDB IDs matching a protein name and return
  the IDs and titles. It can find a particular PDB ID and report information such as how many chains it contains, 
  the sequence, and any small molecules or ligands bound in the structure. It can predict the IC50 value of a molecule
  based on a Chembl dataset using the LightGBM model. It can generate novel molecules using a finetuned GPT based on
  a Chembl dataset.

      Args:
          task: the task to carry out
          protein: a protein name
          up_id: a UNIPROT ID
          chembl_id: a chembl ID
          pdb_id: a PDB ID
          smiles: a SMILES string for a molecule.

      Returns:
          replies[-1]: a text string containing the information requested
          img: an image if appropriate, otherwise a blank image.
  '''
  input = {
    "messages": [
        HumanMessage(f'query_task: {task}, query_protein: {protein}, query_up_id: {up_id}, query_chembl: {chembl_id}, query_pdb: {pdb_id}, query_smiles: {smiles}')
    ]
  }

  #if Substitution_image.png exists, remove it
  if os.path.exists('Substitution_image.png'):
    os.remove('Substitution_image.png')

  #print(input)
  replies = []
  for c in graph.stream(input): #, stream_mode='updates'):
    m = re.findall(r'[a-z]+\_node', str(c))
    if len(m) != 0:
      reply = c[str(m[0])]['messages']
      if 'assistant' in str(reply):
        reply = str(reply).split("<|assistant|>")[-1].split('#')[0].strip()
        reply = reply.replace("~","#")
        replies.append(reply)
  #check if image exists
  if os.path.exists('Substitution_image.png'):
    img_loc = 'Substitution_image.png'
    img = Image.open(img_loc)
  #else create a dummy blank image
  else:
    img = Image.new('RGB', (250, 250), color = (255, 255, 255))

  return replies[-1], img

with gr.Blocks(fill_height=True) as forest:
  gr.Markdown('''
              # Protein Agent
              - calls Uniprot to find uniprot ids
              - calls Chembl to find hits for a given uniprot id and reports number of bioactive molecules in the hit
              - calls Chembl to find a list bioactive molecules for a given chembl id and their IC50 values
              - calls PDB to find the number of chains in a protein, proteins sequences and small molecules in the structure
              - calls PDB to find PDB IDs that match a protein name.
              - Uses Bioactive molecules to predict IC50 values for novel molecules with a LightGBM model.
              - Uses Bioactive molecules to generate novel molecules using a fine-tuned GPT.
              ''')

  with gr.Row():
    with gr.Column():
      protein = gr.Textbox(label="Protein name of interest (optional): ", placeholder='none')
      up_id = gr.Textbox(label="Uniprot ID of interest (optional): ", placeholder='none')
      chembl_id = gr.Textbox(label="Chembl ID of interest (optional): ", placeholder='none')
      pdb_id = gr.Textbox(label="PDB ID of interest (optional): ", placeholder='none')
      smiles = gr.Textbox(label="Molecule SMILES of interest (optional): ", placeholder='none')
      task = gr.Textbox(label="Task for Agent: ")
      calc_btn = gr.Button(value = "Submit to Agent")
    with gr.Column():
      props = gr.Textbox(label="Agent results: ", lines=20 )
      pic = gr.Image(label="Molecule")


      calc_btn.click(ProteinAgent, inputs = [task, protein, up_id, chembl_id, pdb_id, smiles], outputs = [props, pic])
      task.submit(ProteinAgent, inputs = [task, protein, up_id, chembl_id, pdb_id, smiles], outputs = [props, pic])

forest.launch(debug=False, mcp_server=True)