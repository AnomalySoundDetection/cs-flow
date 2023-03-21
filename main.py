'''This is the repo which contains the original code to the WACV 2022 paper
"Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection"
by Marco Rudolph, Tom Wehrbein, Bodo Rosenhahn and Bastian Wandt.
For further information contact Marco Rudolph (rudolph@tnt.uni-hannover.de)'''

import config as c
from train import train
# from utils import load_datasets, make_dataloaders
from utils import load_datasets, make_dataloaders
from dataset import AudioDataset, get_machine_id_list, select_dirs

for m in c.machine_type:
    root_path = c.dataset_path + "/" + c.class_name
    id_list = get_machine_id_list(target_dir=root_path, dir_type="train")
    train_data = select_dirs(machine=m)
    test_data = select_dirs(machine=m, dir_type="test")
    print(len(id_list))
    for _id in id_list:
        # train_set, test_set = load_datasets(c.dataset_path, c.class_name)
        train_set = AudioDataset(data=train_data, _id=_id, root=root_path, sample_rate=c.sample_rate, train=True)
        test_set = AudioDataset(data=test_data, _id=_id, root=root_path, sample_rate=c.sample_rate, train=False)
        # exit(1)
        # train_loader, test_loader = make_dataloaders(train_set, test_set)
        # train(train_loader, test_loader)
        train(train_set, test_set)
    break
