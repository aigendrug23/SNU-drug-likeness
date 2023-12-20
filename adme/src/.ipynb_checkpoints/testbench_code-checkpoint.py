from torch import nn
from .ginet_finetune import GINet_Feat_MTL
from .dataset_mtl import MolTestDatasetWrapper
from .tdc_data import get_scaler
from .tdc_constant import TDC

# find files in folder and returns path
def find_pt_files(folder_path_list):
    import glob

    pt_files = []
    for folder_path in folder_path_list:
        pt_files = pt_files + glob.glob(f"{folder_path}/*.pt")
    return pt_files


def extract_tdcList(path):
    li = []
    for tdc in TDC.allList:
        if str(tdc) in path:
            li.append(tdc)
    scaled = "sc" in path
    return li, scaled


# Prepare for all kinds of test
class TestbenchHelper:
    def __init__(self, device, pred_layer_depth):
        self.models = [None] * 5
        for i in range(1, 4 + 1):
            self.models[i] = GINet_Feat_MTL(
                pool="mean",
                drop_ratio=0,  # does not matter in eval mode
                pred_layer_depth=pred_layer_depth,
                num_tasks=i,
                pred_act="relu",
            ).to(device)

        self.testloaders = {}
        for tdc in TDC.allList:
            _, _, testloader = MolTestDatasetWrapper(
                [tdc], scaled=False
            ).get_data_loaders()
            self.testloaders[tdc] = testloader

        self.criterions = {}
        for tdc in TDC.allList:
            if tdc.isRegression():
                self.criterions[tdc] = nn.MSELoss()
            else:
                self.criterions[tdc] = nn.BCEWithLogitsLoss()

        self.scalers = {}
        for tdc in TDC.allList:
            if tdc.isRegression():
                self.scalers[tdc] = get_scaler(tdc)

    def get_model(self, tdcList):
        return self.models[len(tdcList)]

    def get_testloader(self, tdc):
        return self.testloaders[tdc]

    def get_criterion(self, tdc):
        return self.criterions[tdc]

    def get_scaler(self, tdc):
        return self.scalers[tdc]
