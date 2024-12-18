import streamlit as st
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, Draw
from transformers import EsmModel, EsmTokenizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from model import MAGNETDTI
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Custom CSS to center text
st.markdown("""
    <style>
    .centered-text {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

def load_checkpoint(checkpoint_path, model, model_name):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        best_val_accuracy = checkpoint.get('best_val_acc', None)
        if best_val_accuracy is not None:
            st.success(f"Validation Accuracy for model {model_name} : {best_val_accuracy:.4f}")
        return model
    else:
        st.warning("No checkpoint found. Initializing with default weights.")
        return model

@st.cache_resource
def load_models():
    esm_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(base_dir, 'model/pca_dc.pkl'), 'rb') as f:
        pca_dc = pickle.load(f)
    with open(os.path.join(base_dir, 'model/pca_esm.pkl'), 'rb') as f:
        pca_esm = pickle.load(f)
    with open(os.path.join(base_dir, 'model/pca_pcp.pkl'), 'rb') as f:
        pca_pcp = pickle.load(f)
    with open(os.path.join(base_dir, 'model/pca_maccs.pkl'), 'rb') as f:
        pca_maccs = pickle.load(f)
    dti_model = MAGNETDTI(nprotein=1876, ndrug=1767, nproteinfeat=256, ndrugfeat=256, nhid=16, nheads=4, alpha=0.2)
    checkpoint = torch.load(os.path.join(base_dir, 'model/magnetdti_model.pkl'), map_location=torch.device('cpu'))
    print(checkpoint.keys())
    dti_model.load_state_dict(checkpoint, strict=False)
    dti_model.eval()
    
    return esm_model, esm_tokenizer, pca_dc, pca_esm, pca_pcp, pca_maccs, dti_model

esm_model, esm_tokenizer, pca_dc, pca_esm, pca_pcp, pca_maccs, dti_model = load_models()

def calculate_pcp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    pcp_features = []
    for name, func in Descriptors._descList:
        try:
            value = func(mol)
            pcp_features.append(value)
        except:
            pcp_features.append(0)
    return np.array(pcp_features)

def calculate_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    maccs = MACCSkeys.GenMACCSKeys(mol)
    return np.array(maccs)

def calculate_dc(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    dc_vector = np.zeros(800)
    for gap in [0, 1]:
        for i in range(len(sequence) - gap - 1):
            pair = sequence[i] + sequence[i + gap + 1]
            index = amino_acids.index(pair[0]) * 20 + amino_acids.index(pair[1])
            dc_vector[gap * 400 + index] += 1
    return dc_vector / (len(sequence) - 1)

def calculate_esm(sequence):
    inputs = esm_tokenizer(sequence, return_tensors="pt")
    with torch.no_grad():
        outputs = esm_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

def process_drug(smiles):
    pcp = calculate_pcp(smiles)
    maccs = calculate_maccs(smiles)
    if pcp is None or maccs is None:
        return None
    
    pcp = pcp[:202]
    
    pcp_pca = pca_pcp.transform(pcp.reshape(1, -1))
    maccs_pca = pca_maccs.transform(maccs.reshape(1, -1))
    drug_features = np.concatenate([pcp_pca, maccs_pca], axis=1)
    return drug_features.flatten()

def process_target(sequence):
    dc = calculate_dc(sequence)
    esm = calculate_esm(sequence)
    dc_pca = pca_dc.transform(dc.reshape(1, -1))
    esm_pca = pca_esm.transform(esm.reshape(1, -1))
    target_features = np.concatenate([dc_pca, esm_pca], axis=1)
    return target_features.flatten()

def calculate_similarity_matrix(features):
    return cosine_similarity(features.reshape(1, -1))

def fuse_vectors(vector1, vector2):
    return np.concatenate([vector1, vector2])

def get_model_prediction(model, protein_features, protein_adj, drug_features, drug_adj, index, device):
    model.eval()
    with torch.no_grad():
        index_tensor = torch.tensor(index, dtype=torch.long).to(device)
        y_pred = model(protein_features, protein_adj, drug_features, drug_adj, index_tensor, device)
        probabilities = torch.sigmoid(y_pred)
    return probabilities.cpu().numpy()

st.title("Drug-Target Interaction Prediction")

drug_smiles = st.text_input("Enter drug SMILES:")
target_sequence = st.text_area("Enter target protein sequence:")

if st.button("Visualize"):
    col1, col2 = st.columns(2)
    
    with col1:
        if drug_smiles:
            mol = Chem.MolFromSmiles(drug_smiles)
            if mol:
                img = Draw.MolToImage(mol)
                st.image(img, use_container_width=True)
                st.markdown('<p class="centered-text">Drug Structure</p>', unsafe_allow_html=True)
            else:
                st.error("Invalid SMILES string. Please check and try again.")
        else:
            st.warning("Please enter a drug SMILES.")

    with col2:
        if target_sequence:
            # Analyze the sequence
            analysis = ProteinAnalysis(str(target_sequence))

            # Get amino acid percentages
            aa_percent = analysis.get_amino_acids_percent()

            # Prepare data for plotting
            amino_acids = list(aa_percent.keys())
            percentages = list(aa_percent.values())

            # Set up the plot style
            sns.set_style("whitegrid")
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create bar plot
            bars = ax.bar(amino_acids, percentages, color=sns.color_palette("husl", 20))

            # Customize the plot
            ax.set_title("Amino Acid Composition", fontsize=20, fontweight='bold')
            ax.set_xlabel("Amino Acids", fontsize=16)
            ax.set_ylabel("Percentage", fontsize=16)
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            ax.set_ylim(0, max(percentages) * 1.1)  # Set y-axis limit with some padding

            # Add percentage labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1%}',
                        ha='center', va='bottom', fontsize=12)

            # Add a light grid for better readability
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Improve layout and display the plot
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('<p class="centered-text">Target Protein Amino Acid Composition</p>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a target protein sequence.")

if st.button("Predict Interaction"):
    if drug_smiles and target_sequence:
        drug_features = process_drug(drug_smiles)
        target_features = process_target(target_sequence)
        
        if drug_features is None:
            st.error("Invalid SMILES string. Please check and try again.")
        else:
            drug_sim_matrix = calculate_similarity_matrix(drug_features)
            target_sim_matrix = calculate_similarity_matrix(target_features)
            
            fused_features = fuse_vectors(drug_features, target_features)
            fused_sim_matrix = np.logical_or(drug_sim_matrix, target_sim_matrix).astype(int)
            
            model_input = np.concatenate([fused_features, fused_sim_matrix.flatten()])
            
            print("protein_features shape:", target_features.shape)
            print("protein_adj shape:", target_sim_matrix.shape)
            print("drug_features shape:", drug_features.shape)
            print("drug_adj shape:", drug_sim_matrix.shape)
            print("model_input shape:", model_input.shape)

            progress_bar = st.progress(0)
        
            prediction = get_model_prediction(dti_model, target_features, target_sim_matrix, drug_features, drug_sim_matrix, model_input, device='cpu')

            # If you need the probability for a specific class (e.g., positive interaction)
            interaction_probability = prediction[0]
            st.success(f"Probability of interaction: {interaction_probability:.4f}")
    else:
        st.error("Please enter both drug SMILES and target protein sequence.")

st.markdown("---")
st.write("Note: This app uses pre-trained models for ESM embeddings, PCA, and DTI prediction.")

st.markdown("## Model Performance Metrics")
metrics = {
    "RMSE_test": 0.261734,
    "MAE_test": 0.159984,
    "PCC_test": 0.847308,
    "R2_test": 0.717755,
    "AUC_test": 0.969526,
    "AUPR_test": 0.964155
}

metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
metrics_df.index.name = 'Metric'

st.table(metrics_df)
