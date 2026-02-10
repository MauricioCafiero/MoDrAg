from gliner import GLiNER
import re
import numpy as np
from rdkit import Chem

smiles_pattern = r'[CHONFClBrISPKacnosp0-9@+\-\[\]\(\)\/.=#$%]{5,}'
UPA_pattern = r'[OPQ][0-9][A-Z0-9]{3}[0-9]'
UPA_pattern_2 = r'[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9][A-Z]?[A-Z0-9]*[0-9]?'
PDB_pattern = r'\s[0-9][A-Z0-9]{3}'
chembl_pattern = r'[Cc][Hh][Ee][Mm][Bb][Ll][0-9]{3,}'

def start_ner():
  '''
  Starts the NER model for biomedical named entity recognition.
    Returns:
        model: The NER model.
  '''
  model_name =  "Shoriful025/biomedical_ner_roberta_base"
  model = GLiNER.from_pretrained("anthonyyazdaniml/gliner-biomed-large-v1.0-disease-chemical-gene-variant-species-cellline-ner")

  return model

def smiles_regex(query: str):
  '''
  Accepts a query string and returns the detected SMILES strings.
    Args:
        query: The input query string.
    Returns:
        matches: A list of detected SMILES strings.
  '''
  matches = re.findall(smiles_pattern, query)
  matches = [m for m in matches if any(char not in ['a','c','n','o','s','p','l','r'] for char in m)]
  matches = [m for m in matches if any(char not in ['0','1','2','3','4','5','6','7','8','9','l','P','O','Q'] for char in m)]
  matches = [m for m in matches if any(char not in ['0','1','2','3','4','5','6','7','8','9','.','-','+'] for char in m)]
  #matches = [m.strip(' ') for m in matches]
  print(f'Initial SMILES matches: {matches}')
  modified_matches = []
  for m in matches:
    try:
      mol = Chem.MolFromSmiles(m)
      if mol is not None:
        modified_matches.append(m)
    except:
      continue
  print(f'Modified SMILES matches: {modified_matches}')
  return modified_matches

def uniprot_regex(query: str):
  '''
  Accepts a query string and returns the detected Uniprot IDs.
    Args:
        query: The input query string.
    Returns:
        matches: A list of detected Uniprot IDs.
  '''
  matches = re.findall(UPA_pattern, query)
  matches = matches + re.findall(UPA_pattern_2, query)
  
  return matches

def pdb_regex(query: str):
  '''
  Accepts a query string and returns the detected PDB IDs.
    Args:
        query: The input query string.
    Returns:
        matches: A list of detected PDB IDs.
  '''
  matches = re.findall(PDB_pattern, query)
  matches = [m[1:] for m in matches]

  return matches

def chembl_regex(query: str):
  '''
  Accepts a query string and returns the detected ChEMBL IDs.
    Args:
        query: The input query string.
    Returns:
        matches: A list of detected ChEMBL IDs.
  '''
  matches = re.findall(chembl_pattern, query)

  return matches

def name_protein_ner(query: str, model):
  '''
  Accepts a query string and returns the detected protein, disease and molecule entities.
    Args:
        query: The input query string.
        model: The NER model to use.
    Returns:
        proteins: A list of detected protein names.
        molecules: A list of detected molecule names.
        diseases: A list of detected disease names.
  '''
  labels = ['Disease or phenotype', 'Chemical entity', 'Gene or gene product',
  'Sequence variant', 'Organism', 'Cell line']

  entities = model.predict_entities(query, labels, threshold=0.90)
  molecules = []
  proteins = []
  diseases = []

  for entity in entities:
    if entity['label'] == 'Gene or gene product':
      start_idx = entity['start']
      end_idx = entity['end']
      if ' ' not in query[start_idx:end_idx]:
        proteins.append(query[start_idx:end_idx])

    elif entity['label'] == 'Chemical entity':
      start_idx = entity['start']
      end_idx = entity['end']
      if ' ' not in query[start_idx:end_idx]:
        molecules.append(query[start_idx:end_idx])

    elif entity['label'] == 'Disease or phenotype':
      start_idx = entity['start']
      end_idx = entity['end']
      print('Found disease label: ', query[start_idx:end_idx])
      diseases.append(query[start_idx:end_idx].strip())

  molecules = [m.lower() for m in molecules]
  molecules = list(set(molecules))
  proteins = list(set(proteins))
  diseases = list(set(diseases))

  return proteins, molecules, diseases

def parse_input(query: str, model):
  '''
  Accepts a query string and returns the detected entities.
    Args:
        query: The input query string.
        model: The NER model to use.
    Returns:
        present: A dictionary with counts of each entity type found.
        proteins_list: A list of detected protein names.
        molecules_list: A list of detected molecule names.
        smiles_list: A list of detected SMILES strings.
        uniprot_list: A list of detected Uniprot IDs.
        pdb_list: A list of detected PDB IDs.
        chembl_list: A list of detected ChEMBL IDs.
  '''
  proteins_list, molecules_list, diseases_list = name_protein_ner(query, model)
  smiles_list = smiles_regex(query)
  uniprot_list = uniprot_regex(query)
  pdb_list = pdb_regex(query)
  chembl_list = chembl_regex(query)

  # drop duplicates in each list
  proteins_list = list(set(proteins_list))
  molecules_list = list(set(molecules_list))
  diseases_list = list(set(diseases_list))
  smiles_list = list(set(smiles_list))
  uniprot_list = list(set(uniprot_list))
  pdb_list = list(set(pdb_list))
  chembl_list = list(set(chembl_list))
  
  present = {
      'proteins': len(proteins_list),
      'molecules': len(molecules_list),
      'diseases': len(diseases_list),
      'smiles': len(smiles_list),
      'uniprot': len(uniprot_list),
      'pdb': len(pdb_list),
      'chembl': len(chembl_list)
  }

  return present, proteins_list, molecules_list, diseases_list, smiles_list, uniprot_list, pdb_list, chembl_list

def caf_parse(input_text: str, model):
  '''
  '''
  present, proteins_list, molecules_list, diseases_list, smiles_list, uniprot_list, pdb_list, chembl_list = parse_input(input_text, model)

  return {
      'proteins': proteins_list,
      'molecules': molecules_list,
      'diseases': diseases_list,
      'smiles': smiles_list,
      'uniprot': uniprot_list,
      'pdb': pdb_list,
      'chembl': chembl_list
  }