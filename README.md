# DeepPROTAC 2.0

DeepPROTAC 2.0 has expanded its dataset from the original version, introduced spatial structure information into the algorithm, updated the model, and achieved better results.

## Training

1. **Unzip the Archive**:
   Unzip "DeepPROTACs 2.0 code.tgz" and navigate into the directory.

2. **Prepare the Data**:
   Use the "prepare_data.ipynb" script to extract the ligase pocket, target pocket, ligase ligand, target ligand, linker, and the label from the complexes and encode them. We provide 10 example datasets for testing. The complete dataset is available at [DeepPROTAC-DB](https://bailab.siais.shanghaitech.edu.cn/services/deepprotac-db).

3. **Prepare the Environment**:
   Export our Anaconda environment using "env.yaml". Create and activate this environment with the following commands:
```bash
$ conda env create -f env.yaml 
$ conda activate DeepPROTAC
```

4. **Run the Training Script**:
```bash
$ python main.py
```

## Predicting

**Prepare for Prediction**:
Prepare the ligase ligand, ligase pocket, target ligand, target pocket, and linker SMILES as required. Then run the prediction script:
```bash
$ python predict.py
```
