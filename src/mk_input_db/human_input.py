import logging, beautifullogger
import sys
import pandas as pd, numpy as np
from toolbox import matlab_loader, df_loader, Manager
from typing import List, Any
from mk_input_db.common import DataPath, common_metadata_cols, all_input_cols
import pathlib, toolbox
from tqdm import tqdm
import scipy

logger = logging.getLogger(__name__)


def read_human_stn_matlab_database(file, key, computation_m: toolbox.Manager):
    mat = matlab_loader.load(str(file))
    df = pd.DataFrame(mat[key])
    for col in df.columns:
        df[col] = df.apply(lambda row: np.reshape(row[col], -1)[0] if row[col].size == 1 else None if row[col].size == 0 else row[col], axis=1)
        if df[col].isnull().all():
            df.drop(columns=[col], inplace=True)
    df.columns = df.iloc[0]
    df = df.iloc[1:, :]
    df.columns = ["StructDateH"] + list(df.columns[1:])
    meta = df['StructDateH'].astype(str).str.isdigit()
    df.loc[meta, 'StructDateH'] = np.nan
    df['StructDateH'].ffill(inplace=True)
    df= df.loc[meta, :].reset_index(drop=True)

    df["Species"] = "Human"
    df["Condition"] = "Park"
    df["Structure"] = "STN_"+ df['StructDateH'].str.slice(0,4)
    df["Date"] = df['StructDateH'].str.slice(5,15)
    df["Hemisphere"] = df['StructDateH'].str.slice(16)
    df["Electrode"] = df["channel"]
    df["Depth"] = df["file"].str.extract('(\d+)').astype(str)
    df["Subject"] = np.nan
    df["file_path"] = df["Structure"] + "/" + df['StructDateH'].str.slice(5) + "/"+ df["file"].str.replace("map", "mat")
    df["Start"] = 0
    def get_duration(fp):
        mat = scipy.io.loadmat("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/Human_STN_Correct_All/"+fp, variable_names=['CElectrode1_TimeBegin', 'CElectrode1_TimeEnd'])
        dur= np.squeeze(mat['CElectrode1_TimeEnd']) - np.squeeze(mat['CElectrode1_TimeBegin'])
        return dur
    # tqdm.pandas(desc="Computing durations")
    df["End"] = df["file_path"].apply(lambda fp: computation_m.declare_computable_ressource(get_duration, {"fp":fp}, toolbox.float_loader, "human_input_durations", True))
    return df

def add_raw_signals(df, computation_m: toolbox.Manager):
    raw_df = df.copy()
    raw_df["file_keys"] = df.apply(lambda row: ["CElectrode{}".format(row["Electrode"])], axis=1) 
    raw_df["signal_path"] = raw_df.apply(lambda row: DataPath(row["file_path"], row["file_keys"]), axis=1)
    raw_df["signal_fs_path"] = raw_df.apply(lambda row: DataPath(row["file_path"], ["CElectrode{}_KHz".format(row["Electrode"])]), axis=1)
    raw_df["signal_type"] = "mua"
    import scipy
    def declare_raw_sig(dp):
        mat =  scipy.io.loadmat("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/Human_STN_Correct_All/"+dp.file_path, variable_names=dp.keys[0])
        raw = np.squeeze(mat[dp.keys[0]])
        logger.info("Got ressource {}. Shape is {}".format(dp, raw.shape))
        return raw
    
    def declare_raw_fs(dp):
        mat =  scipy.io.loadmat("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/Human_STN_Correct_All/"+dp.file_path, variable_names=dp.keys[0])
        fs = np.squeeze(mat[dp.keys[0]])
        # logger.info("Got ressource {}. Value is {}".format(dp, fs))
        return fs
    
    raw_df["signal"] = raw_df.apply(lambda row: computation_m.declare_computable_ressource(declare_raw_sig, {"dp": row["signal_path"]}, toolbox.np_loader, "human_input_mua", True), axis=1)
    raw_df["signal_fs"] = raw_df.apply(lambda row: computation_m.declare_computable_ressource(declare_raw_fs, {"dp": row["signal_fs_path"]}, toolbox.float_loader, "human_input_fs", True), axis=1)
    raw_df["neuron_num"] = np.nan
    return raw_df[all_input_cols]



def add_neuron_signals(df, computation_m):
    for i in range(4):
        df["neuron_data{}".format(i)] = list(zip(df["Unit {} Rate".format(i+1)], df["Unit {} Isolaton".format(i+1)]))
        df.drop(columns = ["Unit {} Rate".format(i+1), "Unit {} Isolaton".format(i+1)], inplace=True)

    neuron_df = pd.wide_to_long(df, stubnames="neuron_data", i = ["Species", "Condition", "Structure", "Date", "Hemisphere", "Electrode", "Depth", "Subject", "file"], j="neuron_num").reset_index()

    neuron_df["Rate"] = neuron_df["neuron_data"].str[0]
    neuron_df["Isolation"] = neuron_df["neuron_data"].str[1]
    neuron_df.drop(columns = ["neuron_data"], inplace=True)
    neuron_df = neuron_df.loc[~neuron_df["Rate"].isna()].reset_index()
    neuron_df = neuron_df.loc[neuron_df["Isolation"] > 0.6].reset_index()

    neuron_df["spike_file_path"] = neuron_df["Structure"] + "/" + neuron_df['StructDateH'].str.slice(5) + "/sorting results_"+ neuron_df['StructDateH'].str.slice(5) + "-" + neuron_df["file"].str[0:-4] + "-channel" + neuron_df["Electrode"].astype(str) + "-1.mat"
    neuron_df["signal_path"] = neuron_df.apply(lambda row: row["spike_file_path"], axis=1)
    neuron_df["signal_fs"] = 1
    neuron_df["signal_type"] = "spike_times"

    def declare_spike_sig(signal_path, neuron):
        mat = matlab_loader.load("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/Human_STN_Correct_All/"+ signal_path, variable_names="sortingResults")
        spike_df = np.squeeze(mat["sortingResults"])[()]
        spike_arr = np.squeeze(spike_df[6])
        res = pd.DataFrame(spike_arr, columns=["neuron_num", "spike_time"] + ["pca_{}".format(i) for i in range(spike_arr.shape[1]-2)])
        spike = res.loc[res["neuron_num"]==neuron+1, "spike_time"].to_numpy()
        logger.info("Got ressource {}, neuron = {}. Shape is {}".format(signal_path, neuron, spike.shape))
        return spike
    
    neuron_df["signal"] = neuron_df.apply(lambda row: computation_m.declare_computable_ressource(declare_spike_sig, {"signal_path": row["signal_path"], "neuron": row["neuron_num"]}, toolbox.np_loader, "human_input_spike", True), axis=1)
    return neuron_df[all_input_cols]

def add_human_session_ids(df: pd.DataFrame):
    df = df.copy()
    session_id = {"Date", "Hemisphere", "Depth", "Species", "Condition"}
    sensor_id = session_id | {"Electrode",  "Structure"}
    signal_id = sensor_id | {"signal_type",  "neuron_num"}

    logger.info("nb_entries: {}, n sessions: {}, nsensors: {}, n_signals: {}".format(
        len(df.index), 
        df.groupby(list(session_id), dropna=False).ngroups, 
        df.groupby(list(sensor_id), dropna=False).ngroups, 
        df.groupby(list(signal_id), dropna=False).ngroups
    ))

    if df.duplicated(subset=signal_id).any():
        grp = df.copy()
        grp['count'] = df.groupby(list(signal_id), dropna=False)[list(signal_id)[0]].transform('count')
        pbs = grp[grp["count"] > 1].copy()
        logger.warning("Duplication. Examples:\n{}.\nContinuing while ignoring them".format(pbs.to_string()))
        df = df.drop_duplicates(subset=signal_id, keep=False, ignore_index=True)

    df["Session"] = "HS_#"+ df.groupby(by=list(session_id)).ngroup().astype(str)
    return df


def read_stn_inputs(file, key, computation_m):
    df = read_human_stn_matlab_database(file, key)
    raw_df = add_raw_signals(df, computation_m)
    neuron_df = add_neuron_signals(df, computation_m)
    all_df = pd.concat([raw_df, neuron_df], ignore_index=True, axis=0)
    return all_df[all_input_cols]


# def read_human_stn_metadata(file, key):
#     mat = matlab_loader.load(str(file))
#     df = pd.DataFrame(mat[key])
#     for col in df.columns:
#         df[col] = df.apply(lambda row: np.reshape(row[col], -1)[0] if row[col].size == 1 else None if row[col].size == 0 else row[col], axis=1)
#         if df[col].isnull().all():
#             df.drop(columns=[col], inplace=True)
#     df.columns = df.iloc[0]
#     df = df.iloc[1:, :]
#     df.columns = ["StructDateH"] + list(df.columns[1:])
#     meta = df['StructDateH'].astype(str).str.isdigit()
#     df.loc[meta, 'StructDateH'] = np.nan
#     df['StructDateH'].ffill(inplace=True)
#     df= df.loc[meta, :].reset_index(drop=True)

#     df["Species"] = "Human"
#     df["Condition"] = "Park"
#     df["Structure"] = "STN_"+ df['StructDateH'].str.slice(0,4)
#     df["Date"] = df['StructDateH'].str.slice(5,15)
#     df["Hemisphere"] = df['StructDateH'].str.slice(16)
#     df["Electrode"] = df["channel"]
#     df["Depth"] = df["file"].str.extract('(\d+)').astype(str)
#     df["Subject"] = np.nan


#     raw_df = df[["Species", "Condition", "Structure", "Date", "Hemisphere", "Electrode", "Depth", "Subject"]].copy()
#     raw_df["file_path"] = df["Structure"] + "/" + df['StructDateH'].str.slice(5) + "/"+ df["file"].str.replace("map", "mat")
#     raw_df["file_keys"] = df.apply(lambda row: ["CElectrode{}".format(row["Electrode"])], axis=1) 

#     raw_df["signal"] = raw_df.apply(lambda row: DataPath(row["file_path"], row["file_keys"]), axis=1)
#     raw_df["signal_fs"] = raw_df.apply(lambda row: DataPath(row["file_path"], ["CElectrode{}_KHz".format(row["Electrode"])]), axis=1)
#     raw_df["signal_type"] = "mua"

#     for i in range(4):
#         df["neuron_data{}".format(i)] = list(zip(df["Unit {} Rate".format(i+1)], df["Unit {} Isolaton".format(i+1)]))
#         df.drop(columns = ["Unit {} Rate".format(i+1), "Unit {} Isolaton".format(i+1)], inplace=True)

#     neuron_df = pd.wide_to_long(df, stubnames="neuron_data", i = ["Species", "Condition", "Structure", "Date", "Hemisphere", "Electrode", "Depth", "Subject", "file"], j="neuron_num").reset_index()

#     neuron_df["Rate"] = neuron_df["neuron_data"].str[0]
#     neuron_df["Isolation"] = neuron_df["neuron_data"].str[1]
#     neuron_df.drop(columns = ["neuron_data"], inplace=True)
#     neuron_df = neuron_df.loc[~neuron_df["Rate"].isna()].reset_index()
#     neuron_df = neuron_df.loc[neuron_df["Isolation"] > 0.6].reset_index()

#     neuron_df["spike_file_path"] = neuron_df["Structure"] + "/" + neuron_df['StructDateH'].str.slice(5) + "/sorting results_"+ neuron_df['StructDateH'].str.slice(5) + "-" + neuron_df["file"].str[0:-4] + "-channel" + neuron_df["Electrode"].astype(str) + "-1.mat"
#     neuron_df["signal"] = neuron_df.apply(lambda row: DataPath(row["spike_file_path"], ["sortingResults", 6]), axis=1)
#     neuron_df["signal_fs"] = 1
#     neuron_df["signal_type"] = "spike_times"
    
#     raw_df.drop(columns=["file_path", "file_keys"], inplace=True)
#     neuron_df=neuron_df[["Species", "Condition", "Structure", "Date", "Hemisphere", "Electrode", "Depth", "Subject", "signal", "signal_fs", "signal_type", "neuron_num"]].copy()
#     return pd.concat([raw_df, neuron_df], ignore_index=True)



# def get_file_ressource(signal: DataPath, neuron_num):
#     dp, neuron = signal, neuron_num
#     try:
#         if np.isnan(neuron):
#             mat =  matlab_loader.load("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/Human_STN_Correct_All/"+dp.file_path)
#             raw = np.squeeze(mat[dp.keys[0]])
#             logger.info("Got ressource {}, neuron = {}. Shape is {}".format(dp, neuron, raw.shape))
#             return raw
#         else:
#             mat = matlab_loader.load("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/Human_STN_Correct_All/"+ dp.file_path)
#             spike_df = np.squeeze(mat["sortingResults"])[()]
#             spike_arr = np.squeeze(spike_df[6])
#             res = pd.DataFrame(spike_arr, columns=["neuron_num", "spike_time"] + ["pca_{}".format(i) for i in range(spike_arr.shape[1]-2)])
#             spike = res.loc[res["neuron_num"]==neuron+1, "spike_time"].to_numpy()
#             logger.info("Got ressource {}, neuron = {}. Shape is {}".format(dp, neuron, spike.shape))
#             return spike
#     except KeyboardInterrupt as e:
#         raise e
#     except BaseException as e:
#         logger.error("Error while getting ressource {}, neuron = {}. Error is:\n{}".format(dp, neuron, e))
#         return np.nan