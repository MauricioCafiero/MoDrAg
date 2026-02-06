'''
1. Replace X with the name of your agent/change the spaces repo from cafierom.
2. Add variables as needed

3. in the MoDrAg agent, add an agent description in the calling node
4. add a node to the graph
5. add a conditional edge from calling_node
6. add edge to loop_node
7. add a conditional edge from loop_node
8. add a description to the GUI (see below)
'''
def X_node(state: State) -> State:
  '''
    Calls the protein agent, which can answer protein-centric questions
    regarding Uniprot, Chembl bioactivity, and PDB structural data.
  '''
  print("X tool")
  print('===================================================')
  current_props_string = state["props_string"]
  query_task = state["query_task"]
  query_smiles = state["query_smiles"]
  # add variables as needed

  print(f"in X node input: task={query_task}")
  print('===================================================')

  client = Client("cafierom/XAgent") # fill in agent

  try:
      new_text, img = client.predict(
          query_task, #add as needed
          api_name="/XAgent"
      )
  except:
      new_text = ''
  current_props_string += new_text

  filename = "agent_image.png"
  state["similars_img"] = filename
  state["props_string"] = current_props_string
  state["which_tool"] += 1
  return state

  with gr.Accordion("X Agent - Click to open/close.", open=False):
      gr.Markdown('''
                  - Find ...
      ''')