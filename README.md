# MoDrAg
A *Mo*dular *Dr*ug-design AI *Ag*ent. 
- This page describes the *semantic* version of the agent. For the LanGraph-based agent, see [this page](https://github.com/MauricioCafiero/MoDrAg/blob/main/Langgraph_modrag.md).

## Philosophy
- Let LLMs do what they do best, employ small models for specific tasks, and leave the rest for deterministic code!
- Uses an embedding model (Embedding Gemma), a named entity recognition model, and REGEX for input parsing.
- LLM is used for chat and for interpretation of tool results.
- Everything ‘open,' avoiding using paid services, i.e. not using the OpenAI or Anthropic APIs, etc. where possible.
- Uses a small LM (Gemma 1B) for use almost anywhere!

## How to add a function to the Agent
- A function can be easily added if it only requires input in the form of: SMILES, molecule names, protein names, Chembl IDs, Uniprot accession codes, or PBD IDs.
- The function should be of the form (substitute the function task for the word function):
```
def function_node(agr: type, arg2: type):
  '''
  Doc string with args and returns
  '''
  [function body]

  returns a_list, a_string, an_image_list
```
The function can take any number of arguements but must return exactly 3: a list (can be nested), a string containing the function results in text form (to be given to the LLM as context), and an optional image or list of images. If it is a single image it should still be given as a list.
- The function can be aded to either *modrag_protein_functions, modrag_property_functions,* or *modrag_molecule_functions*. Dependencies should be added to the top of the file.
- Dependencies should also be added to the *requirements.txt* file for installation in the container.
- A description should be aded to the ```full_tool_descriptions``` dictionary in app.py (intake_function.py for Colab). This is the description that will be shown to the user when the AI has selected a tool (for human-in-the-loop approval).
-  A description should be aded to the ```tool_descriptions``` dictionary in input_parsing.py. This is the same query that will be used by the embedding model to compare against user embeddings to select the tool.
- The function name and arguments list should be added to the ```define_tool_hash``` function in input_parsing.py. This hash table is used to run the selected tool.
- The function name, arguments list, and human-readable version of the arguments should be added to the ```define_tool_reqs``` function in input_parsing.py. This hash table is used to check for the required data before running a tool and for asking the user to provide any missing data. 



## Abilities:
A nicely featured drug-design pipeline with voice reposponse, including: 

**Premium Features**
- Dock a molecule in a protein using AutoDock Vina, get the score and the pose.
- Use LightGBM to create a model to predict IC50 values of novel molecules. Trains itself using a Chembl Dataset which Modrag can find for you.
- Use a GPT, finetuned on a Chembl dataset (found by MoDrAg), to generate novel ligands for a protein. <br>

##### *these features will take a bit longer than the standards features. The last two require a Chembl dataset ID, which you can find using modrag. 
  
**Standard Features**  
- Find Uniprot IDs for a protein,
- Find Chembl IDs for a given Uniprot ID,
- Find bioactive molecules for a given Chembl ID,
- Find PDB IDs for a given protein,
- Find sequences, ligand and numbers of chains for a PDB ID.
- Get SMILES strings for molecules, or
- get names for SMILES strings.
- Search Pubchem for similar molecules or generate analogues.
- Find Lipinski properties of molecules.
- Find pharmacophore overlap between two molecules.



