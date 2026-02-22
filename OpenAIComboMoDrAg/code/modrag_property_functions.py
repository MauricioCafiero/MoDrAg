from rdkit import Chem
from rdkit.Chem import AllChem, QED
from rdkit.Chem import Draw
from rdkit import rdBase
from rdkit.Chem import rdMolAlign
import os, re
from rdkit import RDConfig
from rdkit.Chem.Features.ShowFeats import _featColors as featColors
from rdkit.Chem.FeatMaps import FeatMaps
from PIL import Image
from langchain_core.tools import tool

fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef'))

fmParams = {}
for k in fdef.GetFeatureFamilies():
    fparams = FeatMaps.FeatMapParams()
    fmParams[k] = fparams

@tool
def substitution_node(smiles_list: list[str]) -> (list[str], str, list):
  '''
    A simple substitution routine that looks for a substituent on a phenyl ring and
    substitutes different fragments in that location. Returns a list of novel molecules and their
    QED score (1 is most drug-like, 0 is least drug-like).

      Args:
        smiles: the input smiles string
      Returns:
        new_smiles_list: a list of novel molecules and their QED scores.
        new_smiles_string: a string of the tool results
  '''
  print("substitution tool")
  print('===================================================')

  new_fragments = ["c(Cl)c", "c(F)c", "c(O)c", "c(C)c", "c(OC)c", "c([NH3+])c",
                   "c(Br)c", "c(C(F)(F)(F))c"]

  total_sub_smiles_list = []
  total_sub_smiles_string = ''
  total_sub_images = []

  for smiles in smiles_list:
    try:
        new_smiles = []
        for fragment in new_fragments:
            m = re.findall(r"c(\D\D*)c", smiles)
            if len(m) != 0:
                for group in m:
                    #print(group)
                    if fragment not in group:
                        new_smile = smiles.replace(group[1:], fragment)
                        new_smiles.append(new_smile)

        qeds = []
        for new_smile in new_smiles:
            qeds.append(get_qed(new_smile))
        original_qed = get_qed(smiles)

        total_sub_smiles_string += "Substitution or Analogue creation tool results: \n"
        total_sub_smiles_string += f"The original molecule SMILES was {smiles} with QED {original_qed}.\n"
        total_sub_smiles_string += "Novel Molecules or Analogues and QED values: \n"
        for i in range(len(new_smiles)):
            total_sub_smiles_string += f"SMILES: {new_smiles[i]}, QED: {qeds[i]:.3f}\n"
        total_sub_smiles_list.append(new_smiles)

        mols = [Chem.MolFromSmiles(smile) for smile in new_smiles]
        img = Draw.MolsToGridImage(mols,legends=new_smiles, molsPerRow=4, subImgSize=(250, 250))
        total_sub_images.append(img)
    except:        
        total_sub_smiles_list.append([])
        total_sub_smiles_string += f"SMILES: {smiles}, Fail\n"
        total_sub_images.append(None)   

  pic = img.data
  with open('current_image.png', 'wb') as f:
    f.write(pic)
  img = Image.open('current_image.png')

  return total_sub_smiles_list, total_sub_smiles_string, img

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

@tool
def lipinski_node(smiles_list: list[str]) -> (list[float], str):
  '''
    A tool to calculate QED and other lipinski properties of a molecule.
      Args:
        smiles: the input smiles string
      Returns:
        total_lipinski_list: a list of the QED and other lipinski properties of the molecules,
                      including Molecular Weight, LogP, HBA, HBD, Polar Surface Area,
                      Rotatable Bonds, Aromatic Rings and Undesireable Moieties.
        total_lipinski_string: a string of the tool results
  '''
  print("lipinski tool")
  print('===================================================')

  total_lipinski_list = []
  total_lipinski_string = ''

  for smiles in smiles_list:
    for ion in ['.[Na+]', '.[K+]', '.[Cl-]', '.[Br-]', '[Na+].', '[K+].', '[Cl-].', '[Br-].']:
        smiles = smiles.replace(ion, '')
    lipinski_list = []
    try:
        mol = Chem.MolFromSmiles(smiles)
        qed = Chem.QED.default(mol)

        p = Chem.QED.properties(mol)
        mw = p[0]
        logP = p[1]
        hba = p[2]
        hbd = p[3]
        psa = p[4]
        rb = p[5]
        ar = p[6]
        um = p[7]

        lipinski_list.append(qed)
        lipinski_list.append(mw)
        lipinski_list.append(logP)
        lipinski_list.append(hba)
        lipinski_list.append(hbd)
        lipinski_list.append(psa)
        lipinski_list.append(rb)
        lipinski_list.append(ar)
        lipinski_list.append(um)

        total_lipinski_string += f"Properties of SMILES: {smiles}: QED: {qed:.3f}\n"
        total_lipinski_string += f"Molecular Weight: {mw:.3f}, LogP: {logP:.3f}\n"
        total_lipinski_string += f"Hydrogen bond acceptors: {hba}, Hydrogen bond donors: {hbd}\n" 
        total_lipinski_string += f"Polar Surface Area: {psa:.3f}, Rotatable Bonds: {rb}\n" 
        total_lipinski_string += f"Aromatic Rings: {ar}, Undesireable moieties: {um}\n"
        total_lipinski_string += "===================================================\n"    
        total_lipinski_list.append(lipinski_list)
    except:
        total_lipinski_list.append([])
        total_lipinski_string += f"SMILES: {smiles}, Could not get properties\n"  
  return total_lipinski_list, total_lipinski_string, None

@tool
def pharmfeature_node(known_smiles: str, test_smiles: list[str]) -> (list[float], str):
  '''
    A tool to compare the pharmacophore features of a query molecule against
    a those of a reference molecule and report the pharmacophore features of both and the feature
    score of the query molecule.

      Args:
        known_smiles: the reference smiles string
        test_smiles: the query smiles string
      Returns:
        total_pharmfeature_scores: a list of the pharmacophore feature scores of the query molecules.
        total_pharmfeature_string: a string of the tool results
  '''
  print("pharmfeature tool")
  print('===================================================')

  keep = ('Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 'ZnBinder', 'Aromatic', 'LumpedHydrophobe')
  feat_hash = {'Donor': 'Hydrogen bond donors', 'Acceptor': 'Hydrogen bond acceptors',
               'NegIonizable': 'Negatively ionizable groups', 'PosIonizable': 'Positively ionizable groups',
               'ZnBinder': 'Zinc Binders', 'Aromatic': 'Aromatic rings', 'LumpedHydrophobe': 'Hydrophobic/non-polar groups' }


  smiles = [known_smiles, *test_smiles]
  mols = [Chem.MolFromSmiles(x) for x in smiles]

  mols = [Chem.AddHs(m) for m in mols]
  ps = AllChem.ETKDGv3()

  for m in mols:
      AllChem.EmbedMolecule(m,ps)

  total_pharmfeature_scores = []
  total_pharmfeature_string = ''

  #i = 1
  for i in range(1, len(mols)):
    o3d = rdMolAlign.GetO3A(mols[i],mols[0])
    o3d.Align()

    feat_vectors = []
    for m in [mols[0], mols[i]]:
        rawFeats = fdef.GetFeaturesForMol(m)
        feat_vectors.append([f for f in rawFeats if f.GetFamily() in keep])

    feat_maps = [FeatMaps.FeatMap(feats = x,weights=[1]*len(x),params=fmParams) for x in feat_vectors]
    test_score = feat_maps[0].ScoreFeats(feat_maps[1].GetFeatures())/(feat_maps[0].GetNumFeatures())

    feats_known = {}
    feats_test = {}
    for feat in feat_vectors[0]:
        if feat.GetFamily() not in feats_known.keys():
            feats_known[feat.GetFamily()]  = 1
        else:
            feats_known[feat.GetFamily()] += 1

    for feat in feat_vectors[1]:
        if feat.GetFamily() not in feats_test.keys():
            feats_test[feat.GetFamily()]  = 1
        else:
            feats_test[feat.GetFamily()] += 1

    total_pharmfeature_string += f"PharmFeature tool results for SMILES: {smiles[i]}: \n"
    total_pharmfeature_string += f"The Pharmacophore Feature Overlap Score of the test molecule \
versus the reference molecule is {test_score:.3f}. \n\n"
    total_pharmfeature_scores.append(test_score)

    for feat in feats_known.keys():
        total_pharmfeature_string += f"There are {feats_known[feat]} {feat_hash[feat]} in the reference molecule. \n"

    for feat in feats_test.keys():
        total_pharmfeature_string += f"There are {feats_test[feat]} {feat_hash[feat]} in the test molecule. \n"
    #i += 1
    total_pharmfeature_string += "===================================================\n"

  return total_pharmfeature_scores, total_pharmfeature_string, None