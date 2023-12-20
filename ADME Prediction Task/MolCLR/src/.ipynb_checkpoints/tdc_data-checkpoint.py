import pandas as pd
from .tdc_constant import TDC

# data_name for saved file & tdc.single_pred.ADME API
data_names = {
    TDC.Solubility: "Solubility_AqSolDB",
    TDC.BBB: "BBB_Martins",
    TDC.CYP3A4: "CYP3A4_Veith",
    TDC.Clearance: "Clearance_Hepatocyte_AZ",
}


# Get Predefined train, validation, test data set
def get_data_df(tdc, yLabel="Y", datasetType="train"):
    data_path = _datapath(tdc, datasetType)
    try:
        df = pd.read_csv(data_path, index_col=0)
    except FileNotFoundError:
        _save_data_csv(tdc)
        df = pd.read_csv(data_path, index_col=0)
    df = df.drop(columns="Drug_ID")  # Some data Drug matches but Drug_ID does not match
    df = df.rename(columns={"Y": yLabel})
    return df


# Usage: get_mtl_data_df([TDC.BBB, TDC.Clearance], datasetType: 'train' | 'valid' | 'test', scaled: Boolean)
def get_mtl_data_df(tdc_list, datasetType="train", scaled=False):
    data_path = _mtl_datapath(tdc_list, datasetType, scaled)
    try:
        df = pd.read_csv(data_path, index_col=0)
    except FileNotFoundError:
        _save_mtl_data_csv(tdc_list, scaled)
        df = pd.read_csv(data_path, index_col=0)

    return df


# Get Standard Scaler for Clearance, Solubility
def get_scaler(tdc):
    if tdc.isRegression():
        from sklearn.preprocessing import StandardScaler

        df = get_data_df(tdc, yLabel="Y", datasetType="train")
        values = df["Y"].values.reshape(-1, 1)
        # values.shape = (849, 1)
        scaler = StandardScaler().fit(values)
        return scaler
    else:
        raise Exception("tdc should be one of TDC.Clearance or TDC.Solubility")

# Print some info for mtl dataframe
def mtl_data_df_info(df):
    df.info()
    print()
    y_columns = [col for col in df.columns if col != "Drug"]

    together_rows = df[df[y_columns].notnull().sum(axis=1) >= 2]
    print(f"Record with more than two data: {len(together_rows)}")
    lonely_rows = df[df[y_columns].notnull().sum(axis=1) == 1]
    print(f"Record with only one data: {len(lonely_rows)}")

    
####### INTERNAL GENERATION CODE #######
def _save_data_csv(tdc):
    from tdc.single_pred import ADME

    raise Exception("Not used. Please use existing data.")

    data_name = data_names[tdc]
    data = ADME(name=data_name)
    splits = data.get_split()
    for split in ["train", "valid", "test"]:
        data_path = _datapath(tdc, datasetType)
        splits[split].to_csv(data_path)


def _make_mtl_data_df(tdc_list, datasetType="train", scaled=False):
    # Initialize a boolean flag to check if it's the first iteration
    first_iteration = True

    # Iterate through tdcs
    for tdc in TDC.get_ordered_list(tdc_list):
        yLabel = str(tdc)

        df = get_data_df(tdc, yLabel=str(tdc), datasetType=datasetType)
        if scaled and tdc.isRegression():
            scaler = get_scaler(tdc)
            df[yLabel] = scaler.transform(df[yLabel].values.reshape(-1, 1))

        if first_iteration:
            df_acc = df
            first_iteration = False
        else:
            # Merge DataFrames based on "Drug"
            # how: outer -> include all keys from both DataFrames, even if some values are missing
            df_acc = pd.merge(df_acc, df, on=["Drug"], how="outer")

    return df_acc


def _save_mtl_data_csv(tdc_list, scaled=False):
    for datasetType in ["train", "valid", "test"]:
        df = _make_mtl_data_df(tdc_list, datasetType=datasetType, scaled=scaled)
        datapath = _mtl_datapath(tdc_list, datasetType, scaled)
        df.to_csv(datapath)
        print(f"New file generated in {datapath}")


def _datapath(tdc, datasetType):
    return f"/data/project/aigenintern/2023-2/TDC/{data_names[tdc]}_{datasetType}.csv"


def _mtl_datapath(tdc_list, datasetType, scaled):
    return f'/data/project/aigenintern/2023-2/TDC/MTL/{TDC.get_ordered_list(tdc_list)}_{datasetType}{"_sc" if scaled else ""}.csv'
