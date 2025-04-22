import os
import json
import torchvision
import numpy as np
import math

from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment
from semilearn.datasets.utils import split_ossl_data, reassign_target


import os.path
import pickle
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from PIL import Image




class CIFAR10():
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    meta = {
            "filename": "batches.meta",
            "key": "label_names",
            "md5": "5ff9c542aee3614f3951f8cda6e48888",
        }

    def __init__(
        self,
        root,
        dataset,service,shape,date,
        train= True,
        transform = None,
        target_transform = None,
        download = False,
        
    ) -> None:

        #super().__init__(root, transform=transform, target_transform=target_transform)
        self.root=root
        self.dataset=dataset
        self.service=service
        self.date=date
        self.shape=shape
        self.train = train  # training set or test set
        self.transform=transform
        self.target_transform=target_transform


        self.train_list = [
        ["data_"+self.dataset+'_'+'incre'+'_'+date+'_'+self.service+".pkl", ""],
# =============================================================================
#         ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
#         ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
#         ["data_batch_4", "634d18415352ddfa80567beed471001a"],
#         ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
# =============================================================================
    ]

        self.test_list = [
        ["data_"+self.dataset+'_'+'incre'+'_'+date+'_'+self.service+".pkl", ""],
        ]
        

    
        #self.download=download
        
        if download:
            print('disabled')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, self.shape[0],self.shape[1])
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


mean, std = {}, {}
mean['cifar10'] = [0.485, 0.456, 0.406]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]

std['cifar10'] = [0.229, 0.224, 0.225]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]


def get_cifar_openset(args, alg, name,date, num_labels, num_classes, data_dir='./data', pure_unlabeled=False):
    #name = name.split('_')[0]  # cifar10_openset -> cifar10
    for i in range(len(name.split('_'))):
        if(i==0):
            dataset_name=name.split('_')[i]
        elif(i<=len(name.split('_'))-3):
            dataset_name+='_'+name.split('_')[i]
    print('dataset_name',dataset_name)
    service=name.split('_')[-2]
    shape=args.img_size
    
    data_dir = os.path.join(data_dir, 'cifar10')#dataset_name+'_'+service.lower()
    print('data_dir',data_dir)
    #dset = getattr(torchvision.datasets, name.upper())
    #dset = dset(data_dir, train=True, download=False)
    dset= CIFAR10(data_dir,dataset=dataset_name,service=service,shape=shape,date=date,train=True)
    data, targets = dset.data, dset.targets

    #crop_size = args.img_size
    crop_ratio = args.crop_ratio

    seen_classes = set(range(0, 5))|set(range(6, 8))|set(range(9, 27))
    num_all_classes = 28

    #if name == 'cifar10':
        #seen_classes = set(range(0, 27))#set(range(0, 8))|set(range(9, 15))
        
    #elif name == 'cifar100':
        #num_super_classes = num_classes // 5  # args.num_super_classes
        #num_all_classes = 100
        #seen_classes = set(np.arange(num_all_classes)[super_classes < num_super_classes])
    #else:
        #raise NotImplementedError

    lb_data, lb_targets, ulb_data, ulb_targets = split_ossl_data(args, data, targets, num_labels, num_all_classes,
                                                                 seen_classes, None, True)

    if alg == 'fullysupervised':
        lb_data = data
        lb_targets = targets
    #print('lb_data',lb_data)
    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, None, False, None, False)

    if pure_unlabeled:
        seen_indices = np.where(ulb_targets < num_classes)[0]
        ulb_data = ulb_data[seen_indices]
        ulb_targets = ulb_targets[seen_indices]

    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_all_classes, None, True, None, False)

    #dset = getattr(torchvision.datasets, name.upper())
    #dset = dset(data_dir, train=False, download=False)
    dset= CIFAR10(data_dir,dataset=dataset_name,service=service,shape=shape,date=date, train=False)
    test_data, test_targets = dset.data, reassign_target(dset.targets, num_all_classes, seen_classes)
    
    seen_indices = np.where(test_targets < num_classes)[0]
    print('len_seen_indices',len(seen_indices))
    eval_dset = BasicDataset(alg, test_data[seen_indices], test_targets[seen_indices],
                             len(seen_classes), None, False, None, False)
    test_full_dset = BasicDataset(alg, test_data, test_targets, num_all_classes, None, False, None, False)
    return lb_dset, ulb_dset, eval_dset, test_full_dset
