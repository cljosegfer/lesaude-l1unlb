
import h5py
import pandas as pd
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset

class PTBXL():
    def __init__(self, hdf5_path = '/home/josegfer/datasets/ptbxl/ptbxl.h5', 
                 metadata_path = '/home/josegfer/datasets/ptbxl/metadata.csv'):
        self.hdf5_file = h5py.File(hdf5_path, "r")
        self.metadata = pd.read_csv(metadata_path)

        trn_metadata, val_metadata, tst_metadata = self.split()
        self.check_dataleakage(trn_metadata, val_metadata, tst_metadata)
        
        self.trn_idx_dict = self.get_idx_dict(trn_metadata)
        self.val_idx_dict = self.get_idx_dict(val_metadata)
        self.tst_idx_dict = self.get_idx_dict(tst_metadata)

    def split(self, fold_col = 'fold'): # authors use this split setup
        trn_metadata = self.metadata.loc[(self.metadata[fold_col] != 9) * (self.metadata[fold_col] != 10)]
        val_metadata = self.metadata.loc[self.metadata[fold_col] == 9]
        tst_metadata = self.metadata.loc[self.metadata[fold_col] == 10]

        return trn_metadata, val_metadata, tst_metadata
    
    def check_dataleakage(self, trn_metadata, val_metadata, tst_metadata, exam_id_col = 'exam_id', patient_id_col = 'patient_id'):
        print('checking exam_id leakage')
        trn_ids = set(trn_metadata[exam_id_col].unique())
        val_ids = set(val_metadata[exam_id_col].unique())
        tst_ids = set(tst_metadata[exam_id_col].unique())
        assert (len(trn_ids.intersection(val_ids)) == 0), "Some IDs are present in both train and validation sets."
        assert (len(trn_ids.intersection(tst_ids)) == 0), "Some IDs are present in both train and test sets."
        assert (len(val_ids.intersection(tst_ids)) == 0), "Some IDs are present in both validation and test sets."

        print('checking patient_id leakage')
        trn_ids = set(trn_metadata[patient_id_col].unique())
        val_ids = set(val_metadata[patient_id_col].unique())
        tst_ids = set(tst_metadata[patient_id_col].unique())
        assert (len(trn_ids.intersection(val_ids)) == 0), "Some patient IDs are present in both train and validation sets."
        assert (len(trn_ids.intersection(tst_ids)) == 0), "Some patient IDs are present in both train and test sets."
        assert (len(val_ids.intersection(tst_ids)) == 0), "Some patient IDs are present in both validation and test sets."

    def get_idx_dict(self, split_metadata, exam_id_col = 'exam_id'):
        split_exams, split_h5_idx, temp = np.intersect1d(self.hdf5_file[exam_id_col], split_metadata[exam_id_col].values, return_indices = True)
        split_csv_idx = split_metadata.iloc[temp].index.values
        split_idx_dict = {exam_id_col: split_exams, 'h5_idx': split_h5_idx, 'csv_idx': split_csv_idx}

        print('checking exam_id consistency in idx dict')
        for idx, exam_id in tqdm(enumerate(split_idx_dict[exam_id_col])):
            assert self.hdf5_file[exam_id_col][split_idx_dict['h5_idx'][idx]] == exam_id
            assert self.metadata[exam_id_col][split_idx_dict['csv_idx'][idx]] == exam_id
        return split_idx_dict

class PTBXLsplit(Dataset):
    def __init__(self, database, split_idx_dict, 
                 tracing_col = 'tracings', exam_id_col = 'exam_id', output_col = ['NORM', 'MI', 'STTC', 'CD', 'HYP']):
        self.database = database
        self.split_idx_dict = split_idx_dict

        self.tracing_col = tracing_col
        self.exam_id_col = exam_id_col
        self.output_col = output_col
    
    def __len__(self):
        return len(self.split_idx_dict[self.exam_id_col])
    
    def __getitem__(self, idx):
        return {'X': self.database.hdf5_file[self.tracing_col][self.split_idx_dict['h5_idx'][idx]], 
                'y': self.database.metadata[self.output_col].loc[self.split_idx_dict['csv_idx'][idx]].values}