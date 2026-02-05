# LangGraph-based Agent
See the descption of the Agent's [abilities](#abilities), the [design philosphy](#philosophy), a [description of all the files](#files) in this repo, or some [sample images](#images) below.

### Find a working demo of MoDrAg [on HuggingFace Spaces!](https://huggingface.co/spaces/cafierom/DDAgent)
### Use MoDrAg's tools with OpenAI models with [Modrag Chat](https://github.com/MauricioCafiero/MoDrAg/blob/main/MoDrAg_Chat.md)
### See the MoDrAg [demo video!](https://www.linkedin.com/posts/mauricio-cafiero-5481259b_excited-to-final-post-my-entry-for-gradio-activity-7401037808611577858-U9sd?utm_medium=ios_app&rcm=ACoAABUygsgB1ocjqc_UIBRFQJEZ3QMcuNRWSas&utm_source=social_share_send&utm_campaign=copy_link)
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

**Speech-to-text provided by a generous donation from [Eleven Labs](https://elevenlabs.io).**


## Philosophy
- Everything ‘open,' avoiding using paid services, i.e. not using the OpenAI or Anthropic APIs, etc. where possible.
- Modular sub-Agents each perform a family of tasks, easily upgradeable.
- Hardwire tool and Agent calls into the Graph – meaning that you can plug in any AI/LLM; it does not have to be trained to call tools.
- Use a small LM (~4B parameters) to save GPU/CPU for sub-agents performing calculations.
  * HuggingFace’s smolLM has trouble parsing the chemistry
  * Microsoft’s Phi4-mini-instruct works well!
  * Looking to try IBM’s Granite 4 Nano!


## Files:
- ```MoDrAg_HFS.py```: complete app for the Drug design agent. Upload to HuggingFace spaces and rename app.py.
- ```MoleculeAgent_HFS.py, PropAgent_HFS.py, ProteinAgent_HFS.py, DockAgent_HFS.py```: sub-Agents for carrying out various tasks. Also ready to upload to a Hugginface space and rename as app.py.
- Various ```requirements.txt``` files: the requirements for each of the Agents. 
- ```Agent_template.py```: populate this file with one or more tools, add tool descriptions, and add the tools to the Graph and this file can be uploaded to a HuggingFace Space and serve as an Agent or MCP immediately.
- ```node_template.py```: this inlcudes templates for adding your agent as a sub-agent to MoDrAg, code for adding your agent description to the Gradio GUI, and step-by-step directions on how to add all of the bits needed to integrate the new agent into the Graph.

## Images:

### Agent GUI:
<img width="730" height="619" alt="image" src="https://github.com/user-attachments/assets/45a46be3-a6c4-4331-9e60-94363eff1e7d" />

### Chat GUI
<img width="719" height="591" alt="image" src="https://github.com/user-attachments/assets/77cabc7a-c971-4f02-8082-5dfa0056c94a" />


### Agent workflow:
<img width="825" height="393" alt="image" src="https://github.com/user-attachments/assets/044fb2ab-8e7a-4942-9e8d-9b33554387af" />

### SubAgent workflow:
<img width="839" height="388" alt="image" src="https://github.com/user-attachments/assets/a35a562d-ac5d-481c-a0bd-f8d8291cf5f1" />
