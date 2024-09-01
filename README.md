# DeepPROTACs-2.0
 DeepPROTAC 2.0
Training:
1. Unzip the "DeepPROTACs 2.0 code.tgz" and go into the directory

2. Prepare the data. This script needs the ligase pocket, target pocket, ligase 
    ligand, target ligand, linker and the label. Here we have a script 
    "prepare_data.ipynb" which can be used to extract the pockets and ligands 
    from the complexes and encode them.We provide 10 example datas for testing, and you can get the whole dataset from https://bailab.siais.shanghaitech.edu.cn/services/deepprotac-db

3. Prepare the environment. Here we export our anaconda environment as the
   file "env.yaml". You can use the following commands:
       $ conda env create -f env.yaml
       $ conda activate DeepPROTAC
   to obtain the same environment. 
4. Run the training script.
        $ python main.py

Predicting:
1. Prepare the ligase ligand, ligase pocket, target ligand, target pocket and linker SMILES we need. Run the predicting script
        $ python predict.py
