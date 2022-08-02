import logging
import sys
import os
from pyrsistent import optional
import torch
import torch.utils.data
import monai.transforms
import radio.data as radata 
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
from project.misc import nifti_helpers
import pandas as pd
import numpy as np

import utils



"""
Dataset based on PyTorch data primitive
Abstract Base Class - 'BaseVisionDataset' for vision datasets.
"""


Sample = List[Tuple[Path, Any]]

OneSample = Union[Dict[str, Tuple[Any, ...]], Tuple[Any, ...]]

""" (Anzu's Notes)
    Functions from parent class 
    
    find_classes(directory: Path) (Finds class folders in a dataset)
    make_dataset (Generates list of samples of form (path_to_sample, class) )
"""


# TODO: make something to compile subject ID, scan ID and target


# TODO: rename this class to be AmyloidDataset 
# Original class name (DepDataModule)
class AmyloidDataset(radata.BaseVisionDataset):
    
    """
    This class will be based on the folder dataset
    in radio for our specific project
    
    BaseVisionDataset - base class for making datasets that are compatible with torchvision 
    
    Must override __getitem__ and __len__ methods.
    
    To create subclass must implement __init__ , __len__ , __getitem__ 
    
    """
    
    def __init__( 
                 
        self,
        root: Path, 
        base_csv: pd.DataFrame, 
        loader: Callable[[Path], Any], 
        
        subjectCol: Optional[str] = None, 
        scanCol: Optional[str] = None, 
        target_col: Optional[list[str]] = None,
        
        transform: Optional[Callable] = None,
        
        target_transform: Optional[Callable] = None,
        
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[Path], bool]] = None,
        return_paths: bool = False,
        max_class_size: int = 10,
        max_dataset_size: int = 9223372036854775807,
        
        subject_list: list[str] = None, #array of IDs to be used (?)
        
    ) -> None:
        
        # still inside init function 
        # calling init of parent class (inherited from radio) 
        
        super().__init__(
            root,
            loader,
            transform,
            target_transform,
            extensions,
            is_valid_file,
            return_paths,
            max_class_size,
            max_dataset_size,
        )
        
        # read in subject list csv file 
        logging.info("Using subject list") 
        
        self.subject_list = subject_list
        
        self.data = pd.read_csv(base_csv)
        
        # make dataset 
        samples = self.make_dataset(
            self.root, self.subject_list, self.targets, extensions,
        )

        if len(samples) == 0:
            msg = (f"Found 0 samples in: {root}. \n Supported ",)
            raise RuntimeError(msg)

        self.subjectCol = subjectCol 
        self.scanCol = scanCol
        if self.subjectCol is None or self.scanCol is None:
            self.IDs = self.get_IDs(root=root)
        else:
            self.IDs = self.data[subjectCol, scanCol]
        self.targets = self.get_targets(target_col)

        samples = self.make_dataset(
            self.root,
            extensions,
            is_valid_file,
            self.max_class_size,
            self.max_dataset_size,
        )

    @staticmethod 
    def get_IDs(root, extra_folder: str = ""):
        folders = os.listdir(root)
        sub_names = []
        scan_names = []
        temp = []

        for root, dirs, files in os.walk(root):
            for name in dirs:
                base = os.path.basename(root)
                if base in sub_names:
                    if base in scan_names:
                        pass  # TODO: figure out what to do with the extra folders
                    else:
                        scan_names.append(name)
                        temp.append([base, name])
                else:
                    sub_names.append(name)

        combined_sub_scan = np.array(temp)
        print(combined_sub_scan[:, 0])
        return pd.DataFrame(
            {"subjectID": combined_sub_scan[:, 0], "scanID": combined_sub_scan[:, 1]}
        )

    def get_targets(self, target_col):
        if self.subjectCol is None or self.scanCol is None:
            self.subjectCol = self.data.columns[
                np.where(self.data.isin([self.IDs.iloc[0]["subjectID"]]))[1][0]
            ]
            self.scanCol = self.data.columns[
                np.where(self.data.isin([self.IDs.iloc[0]["scanID"]]))[1][0]
            ]
        target_vals = []
        for index, row in self.IDs.iterrows():
            target_vals.append(
                self.data[
                    self.data[self.subjectCol] == row["subjectID"]
                    and self.data[self.scanCol] == row["scanID"]
                ]
            )
        self.subset_data = pd.DataFrame(
            {
                "subjectID": self.IDs["subjectID"],
                "scanID": self.IDs["scanID"],
                "targetVal": target_vals,
            }
        )


    # TODO: start here!
    # HELPFUL FUNCTION TO MAKE DATASET! 
    def make_dataset(
        self,
        directory: Path,
        subject_list: list[str],
        target: pd.DataFrame,
        extensions: Optional[Tuple[str, ...]] = None,
    ) -> Sample:
        """
        Generates a list of images of a form (path_to_sample, class).
        This can be overridden to e.g. read files from a compressed zip file
        instead of from the disk.

        Parameters
        ----------
        directory : Path
            root dataset directory, corresponding to ``self.root``.
        extensions : Tuple[str]
            A list of allowed extensions.
        subject_list: list[str]
            A list of subject ids to use for the dataset. Can contain multiple scan IDs.
        target: pd.Dataframe
            Dataframe containing subject ID and scan_IDs of relavent data 

        Raises
        ------
        FileNotFoundError: In case no valid file was found for any class.

        Returns
        -------
        _: Sample
            Samples of a form (path_to_sample, targetVal).

        Notes
        -----
        Both `extensions` and `is_valid_file` cannot be None or not None at the
        same time.
        """

        instances: Sample = []
        for subjectID in subject_list:
            # get scanID
            index = np.where(target["subjectID"] == subjectID)[0]
            scanID = target.iloc[index]["scanID"]
            targetVal = target.iloc[index]["targetVal"]
            instances.append((os.path.join(directory, subjectID, scanID), targetVal))

        return instances

    # Must implement this method because it's an abstract method in the base class 
    def __getitem__(self, idx: int) -> OneSample:
        # check tensor dtype
        # return formatted as a tuple (not dict) 
        
        
        """
        Parameters
        ----------
        idx : int
            A (random) integer for data intexing.

        Returns
        -------
        _: Tuple[Any, ...]
            (sample, target) where target is class_index of the target class.
            (sample, target, path) if ``self.return_paths`` is True.
        """
        path, target = self.samples[idx]
        
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        if self.return_paths:
            return sample, target, path
        
        return sample, target

    # Must implement because it's an abstract method in the parent class 
    def __len__(self) -> int:
        """Return the total number of images""" 
        return len(self.samples)


# TODO: check this class (?) 
class AmyloidDataloader( DataLoader ):

    def __init__(self, dataset=None, **Params):
        
        """dataloader for Amyloid data
    
        This class allows user to create a dataloader to automatically wrap the 
        dataset class for experiment being performed.
        
        Args:
            batch_size (int): batch size used for training
            shuffle (bool): if you want
            num_workers (int): number of workers to multithread dataloading
            pin_memory (bool): set whether to pin memory for cuda acceleration.
                Set in the train.py script
            subjects_list (list[filepaths]): array of csv filepaths with subjects
                if you use this dataset and loader will be dictionaries Dataset parameters
                detailed above if initializing dataset within this class
            oversample (str): If specified used to eval() which algorithm to use for
                oversampling minority. Do not use with 'undersample.'
            undersample (str): If specified used to eval() which algorithm to use for
                undersampling majority 'oversample.'
        
        """
        
        self.dataset = dataset 
        self.batch_size = Params["batch_size"]
        self.shuffle = Params["shuffle"]
        self.min = None
        self.max = None 
    
    # TODO: continue making DataLoader class
    
    def getLoader(self):
        return self.loader 
    
    

if __name__ == "__main__":
    
    test_params = {
        "transformations":"None",
        "csv_file":"../data/filename.csv", #TODO: check this
        "batch_size": 1,
        "shuffle":"True",
        "features": [ 
            #TODO: add features here
        ],
        "labels": [ 
            #TODO: add labels here 
        ]
         
    }
    
    utils.set_logger(os.path.join(".","debug.log"))
    logging.info("Loading the datasets...") 
    
    test_loader = AmyloidDataLoader(**test_params)


