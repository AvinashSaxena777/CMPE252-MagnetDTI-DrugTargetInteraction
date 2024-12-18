# CLass: CMPE252 Section 3-Artificial Intelligence and Data Engr
* This project is a part of Artificial Intelligence and Data Engr course at San Jose State University, serving as a semester project aimed at applying theoritical knowledge to practical machine learning applications.

# MAGNET-DTI: Multi-Attention Graph NETwork for Enhanced Drug Target Interaction Prediction
Drug Target Interaction system, predicts whether a drug SMILE sequence and target protein sequence are interactable or not.


## Project Team
    1. Avinash Saxena
        Email ID: avinash.saxena@sjsu.edu
    2. Anthony Komareddy 
        Email ID: anthonysandeshreddy.kommareddy@sjsu.edu
    3. Vasav Patel
        Email ID: vasavswetangbhai.patel@sjsu.edu

## Project Stacks

  * Pytorch: Model defintion and training
  * wandb: Model Versioning and Observability
  * streamlit: User Interface
  * transformers: Huggingface library transformer models for accessing esm model
  * rdkit: For extracting and visualizing chemical property. 
  * numpy, pandas, sklearn, matplotlib: Data processing and visualization.

## How To Run Training Scripts
  * Install the required libraries
  * pip install -r requirements.txt
  * Just Run the training script notebooks, just change the path names as per your directories

## How To Run User Interface
  * Install the required libraries
  * pip install -r requirements.txt
  * streamlit run app.py

## User Interface Screenshot
<centre>
<p float="centre">

  <img src="/images/UI/1.png" width="800" />
</p>

<p float="centre">
  <img src="/images/UI/2.png" width="800" />
</p>

<p float="centre">
  <img src="/images/UI/3.png" width="800" />
</p>
</centre>


# Key Accomplishments
* We have made significant progress in our Drug Target Interaction:
* Successfully able to extract features from Drug sequence and Target Protein Sequence
* Implemented a novel Graph Attention plus MultiHead Self Attention model for drug target interaction.
* Our project's GitHub repository contains all the code and results: https://github.com/AvinashSaxena777/CMPE252-MagnetDTI-DrugTargetInteraction/


# Results
## Loss Curves
<p float="left">
<img src="/images/metrics/magnet-dti-train-loss.png" alt="Training Loss Curve" width="40%"> <img src="/images/metrics/magnet-dti-val-loss.png" alt="Test Loss Curve" width="40%">
</p>

## Accuracy Curves
<p float="left">
<img src="/images/metrics/magnet-dti-train-auc.png" alt="Training Accuracy Curve" width="40%"> <img src="/images/metrics/magnet-dti-val-auc.png" alt="Test Accuracy Curve" width="40%">
</p>





# References
Throughout our project, we relied on several key resources:
* https://huggingface.co/facebook/esm1b_t33_650M_UR50S
* https://github.com/Search-AB/MIFAM-DTI
* https://github.com/Diego999/pyGAT.git
* https://pytorch.org/docs/stable/index.html

# Challenges Encountered
* Our main challenge occurred early in the project when we attempted to integrate mutiple features from drug and target sequence, but we were able to overcome that by using libraries such rdkit.
* Another challenge we encountered was GPU memory while training, the model was not able to train on T4 GPU and it requires >15GB memory, so we used colab's L4 GPU which has 24GB GPU memory.

# Future Work
* Use esm-2, recent model released by meta team as an alternative of esm-1b and check the performance.

