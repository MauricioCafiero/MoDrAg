# MoDrAg
A modular drug-design AI Agent. See the descption of [abilities](#abilities) below and scroll down for a [description of all the files](#files) in this repo.

### Find a working demo of MoDrAg [on HuggingFace Spaces!](https://huggingface.co/spaces/cafierom/DDAgent)
<img width="600" height="4200" alt="image" src="https://github.com/user-attachments/assets/019db235-cf98-4b98-8057-2252e948d139" />

## Abilities:
A nicely featured drug-design pipeline, including:
- Find Uniprot IDs for a protein,
- Find Chembl IDs for a given Uniprot ID,
- Find bioactive molecules for a given Chembl ID,
- Find PDB IDs for a given protein,
- Find sequences, ligand and numbers of chains for a PDB ID.
- Get SMILES strings for molecules, or
- get names for SMILES strings.
- Search Pubchem for similar molecules or generate analogues.
- Find Lipinki properties of molecules.
- Find pharmacophore overlap between two molecules.
- Dock a molecule in a protein, get the score and the pose.

## Files:
- ```MoDrAg_HFS.py```: complete app for the Drug design agent. Upload to HuggingFace spaces and rename app.py.
- ```MoleculeAgent_HFS.py, PropAgent_HFS.py, ProteinAgent_HFS.py, DockAgent_HFS.py```: sub-Agents for carrying out various tasks. Also ready to upload to a Hugginface space and rename as app.py.
- Various ```requirements.txt``` files: the requirements for each of the Agents. 
- ```Agent_template.py```: populate this file with one or more tools, add tool descriptions, and add the tools to the Graph and this file can be uploaded to a HuggingFace Space and serve as an Agent or MCP immediately.
- ```node_template.py```: this inlcudes templates for adding your agent as a sub-agent to MoDrAg, code for adding your agent description to the Gradio GUI, and step-by-step directions on how to add all of the bits needed to integrate the new agent into the Graph. 
