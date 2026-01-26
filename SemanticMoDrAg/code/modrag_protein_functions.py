from rdkit import Chem
from rdkit.Chem import AllChem, QED
from rdkit.Chem import Draw
from rdkit.Chem.Draw import MolsToGridImage
from rdkit import rdBase
from rdkit.Chem import rdMolAlign
import os, re
from rdkit import RDConfig
from PIL import Image

import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from tqdm.auto import tqdm
import requests 
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
from dockstring import load_target


def uniprot_node(protein_names: list[str], human_flag: bool = False) -> (list[str], str):
  '''
    This tool takes in the user requested protein and searches UNIPROT for matches.
    It returns a string scontaining the protein ID, gene name, organism, and protein name.
      Args:
        query_protein: the name of the protein to search for.

      Returns:
        total_ids: a list of UNIPROT IDs for the given protein names.
        protein_string: a string containing the protein ID, gene name, organism, and protein name.

  '''
  print("UNIPROT tool")
  print('===================================================')

  total_ids = []
  protein_string = ''

  for protein_name in protein_names:
    try:
      url = f'https://rest.uniprot.org/uniprotkb/search?query={protein_name}&format=tsv'
      response = requests.get(url).text

      f = open(f"{protein_name}_uniprot_ids.tsv", "w")
      f.write(response)
      f.close()

      prot_df_raw = pd.read_csv(f'{protein_name}_uniprot_ids.tsv', sep='\t')
      if human_flag:
        prot_df = prot_df_raw[prot_df_raw['Organism'] == "Homo sapiens (Human)"]
        print(f"Found {len(prot_df)} Human proteins out of {len(prot_df_raw)} total proteins")
      else:
        prot_df = prot_df_raw

      prot_ids = prot_df['Entry'].tolist()
      genes = prot_df['Gene Names'].tolist()
      organisms = prot_df['Organism'].tolist()
      names = prot_df['Protein names'].tolist()

      sub_ids = []
      for id, gene, organism, name in zip(prot_ids, genes, organisms, names):
        protein_string += f'Protein {protein_name}, ID: {id}, Gene: {gene}, Organism: {organism}, Name: {name}\n'
        sub_ids.append(id)

      protein_string += '==========================================================================================\n'
      total_ids.append(sub_ids)
    except:
      protein_string += f'No proteins found for {protein_name}'
      protein_string += '==========================================================================================\n'
      total_ids.append([])

  return total_ids, protein_string, None

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

def listbioactives_node(up_ids_list: list[str]) -> (list[int], list[str], str):
  '''
    Accepts a UNIPROT ID and searches for bioactive molecules
      Args: 
        up_ids_list: the UNIPROT IDs of the proteins to search for.
      Returns:
        total_bioacts_list: a list of the number of bioactive molecules for each protein
        total_chembl_ids_list: a list of the ChEMBL IDs for each protein
        bioact_string: a string containing the results of the search.
  '''
  print("List bioactives tool")
  print('===================================================')

  total_bioacts_list = []
  total_chembl_ids_list = []
  bioact_string = ''

  for up_id in up_ids_list:

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
        bioact_string += f"For Uniprot {up_id}: length of Bioactivities for ChEMBL ID {chembl_id}: {len_this_bioacts}\n"

      bioact_string += f'================================================================================================\n'
      total_chembl_ids_list.append(chembl_ids)
      total_bioacts_list.append(len_all_bioacts)

    except:
      bioact_string += f'No bioactives found for Uniprot {up_id}\n'
      bioact_string += f'================================================================================================\n'
      total_chembl_ids_list.append([])
      total_bioacts_list.append([])

  return total_bioacts_list, bioact_string, total_chembl_ids_list

def getbioactives_node(chembl_ids_list: list[str]) -> (list[str], str):
  '''
    Accepts a Chembl ID and get all bioactives molecule SMILES and IC50s for that ID
      Args:
        chembl_id: the chembl ID to query
      Returns:
        bioactives_list: a list of the bioactive molecules for each chembl ID
        bioactives_string: a string containing the results of the search.
        bioactives_images: a list of images for each bioactive molecule.
  '''
  print("Get bioactives tool")
  print('===================================================')

  bioactives_list = []
  bioactives_images = []
  bioactives_string = ''

  for chembl_id in chembl_ids_list:
    try:
      #check if f'{chembl_id}_bioactives.csv' exists
      chembl_id = chembl_id.upper()
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

      bioact_tuple_list = []
      bioactives_string += f'Results for top bioactivity (IC50 value) for molecules in ChEMBL ID: {chembl_id}. \n'
      for smile, ic50 in zip(total_bioact_df['SMILES'], total_bioact_df['IC50s']):
        bioactives_string += f'Molecule SMILES: {smile}, IC50 (nM): {ic50}\n'
        bioact_tuple_list.append((smile, ic50))
      bioactives_string += f'=========================================================================================\n'

      mols = [Chem.MolFromSmiles(smile) for smile in total_bioact_df['SMILES'].to_list()]
      legends = [f'IC50: {ic50}' for ic50 in total_bioact_df['IC50s'].to_list()]
      img = MolsToGridImage(mols, molsPerRow=5, legends=legends, subImgSize=(200,200))
      bioactives_images.append(img)
      bioactives_list.append(bioact_tuple_list)
    except: 
      bioactives_list.append([])
      bioactives_string += f'No bioactives found for ChEMBL ID: {chembl_id}\n'
      bioactives_string += f'=========================================================================================\n'
      bioactives_images.append(None)

  return bioactives_list, bioactives_string, bioactives_images

def predict_node(smiles_list_in: list[str], chembl_id: str) -> (list[float],str):
  '''
    uses the current_bioactives.csv file from the get_bioactives node to fit the
    Light GBM model and predict the IC50 for the current smiles.
      Args:
        smiles_list: the SMILES strings of the molecules to predict
        chembl_id: the chembl ID to query
      Returns:
        preds: a list of predicted IC50 values for the input SMILES
        preds_string: a string containing the predicted IC50 values for the input SMILES
  '''
  print("Predict Tool")
  print('===================================================')

  # if f'{chembl_id}_bioactives.csv' does not exist, call the bioactives node
  if not os.path.exists(f'{chembl_id}_bioactives.csv'):
    _, _, _ = getbioactives_node([chembl_id])
  
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
  except:
    return [], 'Model training failed, unable to predict.'

  preds = []
  preds_string = ''

  for smiles in smiles_list_in:
    print(f"in predict node, smiles: {smiles}")
    try:
      for ion in ions_to_clean:
        smiles = smiles.replace(ion,"")
      test_mol = Chem.MolFromSmiles(smiles)
      test_feat = featurizer.featurize([test_mol])
      test_feat = scaler.transform(test_feat)
      prediction = model.predict(test_feat)
      test_ic50 = 10**(prediction[0])
      print(f"Predicted IC50 for {smiles}: {test_ic50}")
      preds_string += f"The predicted IC50 value for {smiles} is : {test_ic50:.3f} nM.\n"
      
      preds.append(test_ic50)
    except:
      preds.append(None)
      preds_string += f"The prediction for {smiles} failed.\n"

  preds_string += f"The Bioactive data was fitted with the LightGMB model, using RDKit descriptors. The training score \
was {train_score:.3f} and the testing score was {valid_score:.3f}. "

  return preds, preds_string, None

def gpt_node(chembl_id: str) -> (list[str], str, Image.Image):
  '''
    Uses a Chembl dataset, previously stored in a CSV file by the get_bioactives node, to
    to finetune a GPT model to generate novel molecules for the target protein.

    Args:
      chembl_id: the ChEMBL ID to query
    returns:
      smiles_list: a list of generated SMILES strings
      gpt_string: a string containing the results of the GPT finetuning and generation.
      img: an image containing the generated molecules.
  '''
  print("GPT node")
  print('===================================================')
  
  # if f'{chembl_id}_bioactives.csv' does not exist, call the bioactives node
  if not os.path.exists(f'{chembl_id}_bioactives.csv'):
    _, _, _ = getbioactives_node([chembl_id])

  try:
    df = pd.read_csv(f'{chembl_id}_bioactives.csv')
    smiles_list, gpt_string, img = finetune_gpt(df, chembl_id) 

  except:
    gpt_string = ''
    smiles_list = []
    img = None

  return smiles_list, gpt_string, [img]

def get_protein_from_pdb(pdb_id):
  '''
    Helper function to get the protein information from the PDB database.
    Args:
      pdb_id: the PDB ID of the protein
    Returns:
      r.text: the PDB information as a string
  '''
  url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
  r = requests.get(url)
  return r.text

def one_to_three(one_seq):
  '''
    Converts a one-letter amino acid sequence to a three-letter sequence.
    Args:
      one_seq: the one-letter amino acid sequence
    Returns:
      three_seq: the three-letter amino acid sequence
  '''
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
  '''
  Converts a three-letter amino acid sequence to a one-letter sequence.
  Args:
    three_seq: the three-letter amino acid sequence
  Returns:
    one_seq: the one-letter amino acid sequence
  '''
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

def pdb_node(test_pdb_list: list[str]) -> (list[str], str):
  '''
    Accepts a PDB ID and queires the protein databank for the sequence of the protein, as well as other
    information such as ligands.
      Args:
        test_pdb_list: the PDB IDs to query
      Returns:
        all_seqs: a list of the sequences for each PDB ID
        total_pdb_string: a string containing the results of the PDB query.
      (collects all ligands but does not return them currently)
  '''

  print(f"pdb toolS")
  print('===================================================')

  total_pdb_string = ''
  all_seqs = []
  all_ligands = []

  for test_pdb in test_pdb_list:
    try:
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

      sub_seqs = []
      sub_ligands = []
      total_pdb_string += f"Chains in PDB ID {test_pdb}: {', '.join(chains.keys())} \n"
      for chain in chains_ol:
        total_pdb_string += f"Chain {chain}: {''.join(chains_ol[chain])} \n"
        sub_seqs.append(''.join(chains_ol[chain]))
        print(f"Chain {chain}: {''.join(chains_ol[chain])}")
      total_pdb_string += f"Ligands in PDB ID {test_pdb}.\n"
      for mol in other_molecules:
        total_pdb_string += f"Molecule {mol}: {''.join(other_molecules[mol])} \n"
        sub_ligands.append(''.join(other_molecules[mol]))
      total_pdb_string += f'=========================================================================================\n'

      all_seqs.append(sub_seqs)
      all_ligands.append(sub_ligands)
    except:
      total_pdb_string += f'Failed to get data for PDB ID {test_pdb}\n'
      total_pdb_string += f'=========================================================================================\n'
      all_seqs.append([])
      all_ligands.append([])

  return all_seqs, total_pdb_string, None

def find_node(test_protein_list: list[str]) -> (list[str], str):
  '''
    Accepts a protein name and searches the protein databack for PDB IDs that match along with the entry titles.
      Args:
        test_protein_list: the protein names to query
      Returns:
        total_ids: a list of the PDB IDs for each protein name
        pdb_string: a string containing the results of the PDB search.
  '''

  print(f"PDB search tool")
  print('===================================================')

  total_ids = []
  pdb_string = ''
  which_pdbs = 0

  for test_protein in test_protein_list:
    try:
      query = TextQuery(value=test_protein)
      results = query()

      def pdb_gen():
        for rid in results:
          yield(rid)

      take10 = itertools.islice(pdb_gen(), which_pdbs, which_pdbs+10, 1)

      local_ids = []
      pdb_string += f'10 PDBs that match the protein {test_protein} are: \n'
      for pdb in take10:
        data = requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb}").json()
        title = data['struct']['title']
        pdb_string += f'PDB ID: {pdb}, with title: {title} \n'
        local_ids.append(pdb)
      total_ids.append(local_ids)
    except:
      pdb_string += f'Failed to get PDB IDs for protein {test_protein}\n'
      total_ids.append([])

  return total_ids, pdb_string, None

def docking_node(smiles_list: list[str], query_protein: str) -> (list[float], str):
  '''
    Docking tool: uses dockstring to dock the molecule into the protein
    Args:
      smiles_list: the SMILES strings of the molecules to dock
      protein: the protein to dock into
    Returns:
      docking_scores: a list of docking scores for each molecule
      docking_string: a string containing the results of the docking.
  '''
  print("docking tool")
  print('===================================================')
  cpuCount = os.cpu_count()
  print(f"Number of CPUs: {cpuCount}")

  print(f'query_protein: {query_protein}')

  scores_list = []
  scores_string = ''

  for query_smiles in smiles_list:
    try:
      query_smiles = query_smiles.replace('.[Na+]','').replace('.[Na+]','').replace('.[K+]','').replace('[K+].','').replace('.[Cl-]','').replace('[Cl-].','')
      target = load_target(query_protein)
      print("===============================================")
      print(f"Docking molecule with {cpuCount} cpu cores.")
      score, aux = target.dock(query_smiles, num_cpus = cpuCount)
      scores_list.append(score)
      mol = aux['ligand']
      print(f"Docking score: {score}")
      print("===============================================")
      atoms_list = ""
      template = mol
      molH = Chem.AddHs(mol)
      AllChem.ConstrainedEmbed(molH,template, useTethers=True)
      xyz_string = f"{molH.GetNumAtoms()}\n\n"
      for atom in molH.GetAtoms():
        atoms_list += atom.GetSymbol()
        pos = molH.GetConformer().GetAtomPosition(atom.GetIdx())
        xyz_string += f"{atom.GetSymbol()} {pos[0]} {pos[1]} {pos[2]}\n"
      scores_string += f"Docking score for {query_smiles}: {score} kcal/mol \n\n"
      scores_string += f"pose structure: {xyz_string}\n"
      scores_string += f"=========================================================\n"

    except:
      print(f"Molecule {query_smiles} could not be docked!")
      scores_string = ''
      scores_list.append(None)

  return scores_list, scores_string, None