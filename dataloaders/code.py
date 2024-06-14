
import h5py
import pandas as pd
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset

class CODE():
    def __init__(self, hdf5_path = '/home/josegfer/code/code14/code14.h5', 
                 metadata_path = '/home/josegfer/code/code14/exams.csv', 
                 reports_path = '/home/josegfer/code/code14/BioBERTpt_text_report_crop.h5', 
                 val_size = 0.05, trn_size = 0.05):
        self.hdf5_file = h5py.File(hdf5_path, "r")
        self.metadata = pd.read_csv(metadata_path)
        self.reports = h5py.File(reports_path, "r")

        self.val_size = val_size
        self.tst_size = trn_size

        trn_metadata, val_metadata, tst_metadata = self.split()
        self.check_dataleakage(trn_metadata, val_metadata, tst_metadata)
        
        self.pretrn_idx_dict = self.get_idx_dict(trn_metadata)
        self.val_idx_dict = self.get_idx_dict(val_metadata)
        # self.trn_idx_dict = self.get_idx_dict(tst_metadata)
        self.trn_idx_dict = self.get_idx_dict(trn_metadata)
        self.tst_idx_dict = 'test'

    def split(self, patient_id_col = 'patient_id'):
        patient_ids = self.metadata[patient_id_col].unique()

        num_trn = int(len(patient_ids) * (1 - self.tst_size - self.val_size))
        num_val = int(len(patient_ids) * self.val_size)

        trn_ids = set(patient_ids[:num_trn])
        val_ids = set(patient_ids[num_trn : num_trn + num_val])
        tst_ids = set(patient_ids[num_trn + num_val :])

        trn_metadata = self.metadata.loc[self.metadata[patient_id_col].isin(trn_ids)]
        val_metadata = self.metadata.loc[self.metadata[patient_id_col].isin(val_ids)]
        tst_metadata = self.metadata.loc[self.metadata[patient_id_col].isin(tst_ids)]

        return trn_metadata, val_metadata, tst_metadata
    
    def check_dataleakage(self, trn_metadata, val_metadata, tst_metadata, exam_id_col = 'exam_id'):
        trn_ids = set(trn_metadata[exam_id_col].unique())
        val_ids = set(val_metadata[exam_id_col].unique())
        tst_ids = set(tst_metadata[exam_id_col].unique())
        assert (len(trn_ids.intersection(val_ids)) == 0), "Some IDs are present in both train and validation sets."
        assert (len(trn_ids.intersection(tst_ids)) == 0), "Some IDs are present in both train and test sets."
        assert (len(val_ids.intersection(tst_ids)) == 0), "Some IDs are present in both validation and test sets."

    def get_idx_dict(self, split_metadata, exam_id_col = 'exam_id'):
        split_exams, split_h5_idx, temp = np.intersect1d(self.hdf5_file[exam_id_col], split_metadata[exam_id_col].values, return_indices = True)
        split_csv_idx = split_metadata.iloc[temp].index.values
        split_idx_dict = {exam_id_col: split_exams, 'h5_idx': split_h5_idx, 'csv_idx': split_csv_idx}

        print('checking exam_id consistency in idx dict')
        for idx, exam_id in tqdm(enumerate(split_idx_dict[exam_id_col])):
            assert self.hdf5_file[exam_id_col][split_idx_dict['h5_idx'][idx]] == exam_id
            assert self.metadata[exam_id_col][split_idx_dict['csv_idx'][idx]] == exam_id
        return split_idx_dict

class CODEsplit(Dataset):
    def __init__(self, database, split_idx_dict, 
                 tracing_col = 'tracings', exam_id_col = 'exam_id', output_col = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"], text_col = 'embeddings'):
        if split_idx_dict == 'test':
            self.database = CODEtest()
            self.split_idx_dict = self.database.metadata # so we can get use the same __len__()
            self.text_col = tracing_col # so we can get use the same __getitem__()
            print('using test ds, H is treated as X')
        else:
            self.database = database
            self.split_idx_dict = split_idx_dict
            self.text_col = text_col

        self.tracing_col = tracing_col
        self.exam_id_col = exam_id_col
        self.output_col = output_col
    
    def __len__(self):
        return len(self.split_idx_dict[self.exam_id_col])
    
    def __getitem__(self, idx):
        return {'X': self.database.hdf5_file[self.tracing_col][self.split_idx_dict['h5_idx'][idx]], 
                'H': self.database.reports[self.text_col][self.split_idx_dict['h5_idx'][idx]], 
                'y': self.database.metadata[self.output_col].loc[self.split_idx_dict['csv_idx'][idx]].values}

class CODEtest():
    def __init__(self, hdf5_path = '/home/josegfer/code/codetest/data/ecg_tracings.hdf5', 
                 metadata_path = '/home/josegfer/code/codetest/data/annotations/gold_standard.csv'):
        self.hdf5_file = h5py.File(hdf5_path, "r")
        self.metadata = pd.read_csv(metadata_path)

        self.metadata['exam_id'] = self.metadata.index # so we can get use the same __len__()
        self.metadata['h5_idx'] = self.metadata.index # so we can get use the same __getitem__()
        self.metadata['csv_idx'] = self.metadata.index # so we can get use the same __getitem__()
        self.reports = self.hdf5_file # so we can get use the same __getitem__()