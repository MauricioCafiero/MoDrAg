import sys
from just_parse import *

from transformers import pipeline
import torch
from openai import OpenAI
from anthropic import Anthropic
import os, time


''' Queries to test ================================================================'''


def test_caf_parse(query, ner_model):
  '''
   Tests the caf_parse function and prints the results and time taken.
     Args:
         query: The input query string to test.
         ner_model: The NER model to use for parsing.
  '''
  print('Testing Caf parser...')
  start = time.time()
  response = caf_parse(query, ner_model)
  end = time.time() 
  print(f'Time taken: {end-start} seconds')
  print(response)
  print('================================================================')
  return response

def start_huggingface_model(model_name):
  '''
   Loads a Hugging Face model and returns the generator pipeline.
     Args:
         model_name: The name of the Hugging Face model to load.
     Returns:
         generator: The text generation pipeline for the loaded model.
  '''
  print('Loading Hugging Face model...')
  generator = pipeline('text-generation', model = model_name, device='cuda')
  return generator

def parse_huggingface(query, pipe):
  '''
   Tests a Hugging Face model and prints the results and time taken.
     Args:
         query: The input query string to test.
         model_name: The name of the Hugging Face model to use.
  '''
  print('Testing Hugging Face model...')
  messages = [
      [
          {
              "role": "system",
              "content": [{"type": "text", "text": '''
      ## Your job is to extract drug-design related information from the user's input. You will look for:
      - SMILES: smiles strings for molecules,
      - MOLECULE NAMES: names of molecules,
      - PROTEIN NAMES: names of proteins,
      - PDB IDs: protein databank ids for proteins,
      - DISEASE NAMES: names of diseases,
      - UNIPROT ACCESSION CODES: Uniprot accession codes for proteins,
      - CHEMBL IDS: chembl ids for datasets.

      ## You will format the output as a JSON object with the following keys:
      - smiles_list
      - names_list
      - proteins_list
      - pdb_list
      - diseases_list
      - uniprot_list
      - chembl_list

      the corresponding values will be lists containing the requested items.
      DO NOT ADD ANY OF YOUR OWN INFORMATION. LOOK ONLY AT THE USER INPUT.
    '''},]
          },
          {
              "role": "user",
              "content": [{"type": "text", "text": query},]
          },
      ],
  ]

  output = pipe(messages, max_new_tokens=200)

  return output[-1][0]['generated_text'][2]['content']

def test_huggingface(query, pipe):
  '''
   Tests a Hugging Face model and prints the results and time taken.
     Args:
         query: The input query string to test.
         model_name: The name of the Hugging Face model to use.
  '''
  start = time.time()
  response = parse_huggingface(query, pipe)
  end = time.time() 
  print(f'Time taken: {end-start} seconds')
  print(response)
  print('================================================================')
  return response


def parse_chatgpt(query: str, client):
  '''

  '''
  dd_message = client.responses.create(
                          instructions = '''
    ## Your job is to extract drug-design related information from the user's input. You will look for:
    - SMILES: smiles strings for molecules,
    - MOLECULE NAMES: names of molecules,
    - PROTEIN NAMES: names of proteins,
    - PDB IDs: protein databank ids for proteins,
    - DISEASE NAMES: names of diseases,
    - UNIPROT ACCESSION CODES: Uniprot accession codes for proteins,
    - CHEMBL IDS: chembl ids for datasets.

    ## You will format the output as a JSON object with the following keys:
    - smiles_list
    - names_list
    - proteins_list
    - pdb_list
    - diseases_list
    - uniprot_list
    - chembl_list

    the corresponding values will be lists containing the requested items.
    DO NOT ADD ANY OF YOUR OWN INFORMATION. LOOK ONLY AT THE USER INPUT.
  ''',
            model = "gpt-5.2",
            input=query,
              )

  response = dd_message.output_text
  return response

def test_chatgpt(query, client):
  '''
  '''
  start = time.time()
  response = parse_chatgpt('Dock paracetamol in DRD2', client)
  end = time.time()
  print(f'Time taken: {end-start} seconds')
  print(response)
  return response

def parse_anthropic(query: str, client):
  '''
  '''
  dd_message = client.messages.create(
  model="claude-haiku-4-5-20251001",
  max_tokens=1000,
  system = '''
    ## Your job is to extract drug-design related information from the user's input. You will look for:
    - SMILES: smiles strings for molecules,
    - MOLECULE NAMES: names of molecules,
    - PROTEIN NAMES: names of proteins,
    - PDB IDs: protein databank ids for proteins,
    - DISEASE NAMES: names of diseases,
    - UNIPROT ACCESSION CODES: Uniprot accession codes for proteins,
    - CHEMBL IDS: chembl ids for datasets.

    ## You will format the output as a JSON object with the following keys:
    - smiles_list
    - names_list
    - proteins_list
    - pdb_list
    - diseases_list
    - uniprot_list
    - chembl_list

    the corresponding values will be lists containing the requested items.
    DO NOT ADD ANY OF YOUR OWN INFORMATION. LOOK ONLY AT THE USER INPUT.
  ''',
  messages=[
      {"role": "user", "content": query},
  ]
  )

  extracted_text = dd_message.content[0].text

  return extracted_text

def test_anthropic(query, client):
  '''
  '''
  start = time.time()
  response = parse_anthropic(query, client)
  end = time.time()
  print(f'Time taken: {end-start} seconds')
  print(response)
  return response

def prep_tests():
  '''
  '''
  gemma_pipe = start_huggingface_model('google/gemma-3-1b-it')
  granite_pipe = start_huggingface_model('ibm-granite/granite-4.0-1b')
  ner_model = start_ner_model()

  return gemma_pipe, granite_pipe, ner_model

def run_tests(query: str, gemma_pipe, granite_pipe, openai_client, anthropic_client, ner_model):
  '''   Runs all the tests for the different parsing methods and prints the results.
     Args:
         query: The input query string to test.
         gemma_pipe: The Hugging Face pipeline for the Gemma model.
         granite_pipe: The Hugging Face pipeline for the Granite model.
         openai_client: The OpenAI client for ChatGPT.
         anthropic_client: The Anthropic client for Claude.
         ner_model: The NER model to use for the Caf parser.
  '''

  result = test_caf_parse(query, ner_model)
  print('================================================================')
  result = test_huggingface(query, gemma_pipe)
  print('================================================================')
  result = test_huggingface(query, granite_pipe)
  print('================================================================')
  result = test_chatgpt(query, openai_client)
  print('================================================================')
  result = test_anthropic(query, anthropic_client)