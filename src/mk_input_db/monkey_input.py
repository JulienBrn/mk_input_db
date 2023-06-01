import logging, beautifullogger
import sys
import pandas as pd, numpy as np
from toolbox import matlab_loader, df_loader, Manager
from typing import List, Any
from mk_input_db.common import DataPath, common_metadata_cols, all_input_cols
import pathlib, toolbox
from tqdm import tqdm

logger = logging.getLogger(__name__)

import re

recup_wild = re.compile('{.+?}')

def extract_wildcards(strlist, pattern):
    print("pattern: ", pattern)
    def msub(match):
        val = match.group()
        print(val)
        return '(?P<{}>.*)'.format(val[1:-1])
    regex_str = recup_wild.sub(msub, pattern)
    regex = re.compile(regex_str)
    res=[]
    for s in strlist:
        d = regex.match(s)
        if not d is None:
          d=d.groupdict()
          d["full_path"] = s
          # print("dict: ", d)
          res.append(d)
    df = pd.DataFrame(
        res, 
      )
    return df

def read_monkey_database(file):
    df = pd.read_csv(str(file), sep=",")
    df["Structure"]=df["Structure"].str.slice(0,3)
    df["file_path"] = df["Condition"].astype(str) + "/" + df["Subject"].astype(str) + "/" + df["Structure"].astype(str) + "/" + df["Date"].astype(str) + "/unit" + df["unit"].astype(str) + ".mat"
    df["Species"] = "Monkey"
    df_depth = df.sort_values(by=["Date","Species","Condition", "Subject", "Electrode",  "Structure", "Start", "End"])
    # print(df_depth)
    df_depth["diff"] = (df_depth.shift(1, fill_value=np.inf)["End"] <= df_depth["Start"]).astype(int)
    # print( df_depth.shift(1, fill_value=np.inf)["Start"])
    df_depth["Depth_num"] = df_depth.groupby(by=["Date","Species", "Condition", "Subject", "Electrode",  "Structure"])["diff"].cumsum()
    df_depth["Depth"] = "#"+ df_depth["Depth_num"].astype(str)
    # print(df_depth)
    # input()
    return df_depth.drop(columns=["diff", "Depth_num"])

def add_raw_signals(df: pd.DataFrame, base_folder, computation_m): 
    if df.duplicated(subset=["Date","Species", "Condition", "Subject", "Electrode", "Depth", "Structure", "unit"]).any():
        logger.warning("Duplication problem before counting raw signals")
    raw_df = df.groupby(["Date","Species", "Condition", "Subject", "Electrode", "Depth", "Structure"]).aggregate(lambda x: tuple(x)).reset_index()
    # print(raw_df.to_string())
    # input()
    raw_df["signal_fs"] = 25000
    raw_df["signal_type"] = "raw"
    raw_df["signal_path"] = raw_df.apply(lambda row: [(s, e, DataPath(f, ["RAW"])) for s,e,f in zip(row["Start"], row["End"], row["file_path"])], axis=1)
    raw_df["Start"] = raw_df["Start"].apply(min)
    raw_df["End"] = raw_df["End"].apply(max)
    import scipy

    def declare_raw(dpl):
        new_dpl=[]
        for s,e,dp in dpl:
            mat =  scipy.io.loadmat(str(base_folder)+"/"+dp.file_path, variable_names=dp.keys[0])
            raw = np.squeeze(mat[dp.keys[0]])
            # logger.info("Got ressource {}. Shape is {}".format(dp, raw.shape))
            new_dpl.append((25000*s, 25000*s+raw.size, raw))
        start = min([s for s, e,d in new_dpl])
        end = max([e for s, e,d in new_dpl])
        res = np.empty(shape=(len(new_dpl), end-start))
        # print(start, end, res.shape)
        res[:] = np.nan
        for i, (s,e,r) in enumerate(new_dpl):
            # dp = dp[0][2] #CHANGE HERE
            # mat =  scipy.io.loadmat(str(base_folder)+"/"+dp.file_path, variable_names=dp.keys[0])
            
            # raw = np.squeeze(mat[dp.keys[0]])
            # logger.info("Got ressource {}. Shape is {}".format(dp, raw.shape))
            # print(s, e, e*25000-s*25000)
            # mlen = min((e-s)*25000, raw.size)
            try:
                res[i,s-start:e-start] = r
            except BaseException as err:
                raise BaseException("Error in declare raw for i = {}. \nInitial Error is \n{}\n. \nExpected sizes in slice affectation are {} and {}. dpl is\n{}\nnew_dpl is\n{}".format(i, err, e-s, raw.size, dpl, new_dpl))

        res_agg_max = np.nanmax(res,axis=0)
        res_agg_min = np.nanmin(res, axis=0)
        if np.count_nonzero(np.nan_to_num(res_agg_max - res_agg_min, nan=0)):
            print("ex computation")
            ex = np.nonzero(np.nan_to_num(res_agg_max - res_agg_min, nan=0))[0][0]
            print("ex is", ex)
            raise BaseException(
                "Error in declare raw: Differences found." 
                +"dpl is\n{}\nnew_dpl is\n{}.\nExample of differences starting at index {}. First elements of array are:\n\n{}".format(dpl, new_dpl, ex, pd.DataFrame(np.transpose(res[:, ex:ex+1000])).to_string()))
        elif np.isnan(res_agg_max).any():
            logger.error("nans")
        elif len(dpl) > 1:
            logger.info("Check was successful")
        # print(res.shape, res_agg_max.shape)
        return res_agg_max
    
    raw_df["signal"] = raw_df.apply(lambda row: computation_m.declare_computable_ressource(declare_raw, {"dpl": row["signal_path"]}, toolbox.np_loader, "monkey_input_raw", True), axis=1)
    return raw_df
    

def add_neuron_signals(df, base_folder, computation_m): 
    neuron_df = df.copy()
    neuron_df["neuron_num"] = df["unit"]
    neuron_df["signal_fs"] = 40000
    neuron_df["signal_type"] = "spike_times"
    neuron_df["signal_path"] = neuron_df["file_path"].apply(lambda fp: DataPath(fp, ["SUA"]))
    import scipy

    def declare_spikes(dp):
        mat =  scipy.io.loadmat(str(base_folder)+"/"+dp.file_path, variable_names=dp.keys[0])
        spikes = np.squeeze(mat[dp.keys[0]])
        logger.info("Got ressource {}. Shape is {}".format(dp, spikes.shape))
        return spikes
    
    neuron_df["signal"] = neuron_df.apply(lambda row: computation_m.declare_computable_ressource(declare_spikes, {"dp": row["signal_path"]}, toolbox.np_loader, "monkey_input_spikes", True), axis=1)
    return neuron_df

def read_monkey_inputs(file, computation_m):
    df = read_monkey_database(file)
    raw_df = add_raw_signals(df, str(pathlib.Path(file).parent), computation_m)
    neuron_df = add_neuron_signals(df, str(pathlib.Path(file).parent), computation_m)
    columns_df = pd.DataFrame(columns=all_input_cols)
    all_df = pd.concat([columns_df, raw_df, neuron_df], ignore_index=True, axis=0)
    return all_df[all_input_cols]


def add_monkey_session_ids(df: pd.DataFrame):
    df = df.copy()
    session_id = {"Date", "Subject", "Condition"}
    sensor_id = session_id | {"Electrode",  "Structure", "Depth"}
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
        for r in pbs.loc[pbs["signal_type"]=="spike_times", "signal"]:
            r.get_result()
        logger.warning("Duplication. Examples:\n{}.\nContinuing while ignoring them".format(pbs.to_string()))
        tmp = pbs[pbs["signal_type"]=="raw"].head(2)
        start1 = tmp["Start"].iat[0] * tmp["signal_fs"].iat[0]
        start2 = tmp["Start"].iat[1] * tmp["signal_fs"].iat[1]
        end1 = tmp["End"].iat[0] * tmp["signal_fs"].iat[0]
        end2 = tmp["End"].iat[1] * tmp["signal_fs"].iat[1]
        s1 = tmp["signal"].iat[0].get_result()
        s2 = tmp["signal"].iat[1].get_result()
        mstart = max(start1, start2)
        mend = min(end1, end2)
        mstart1 = mstart - start1
        mstart2 = mstart - start2
        mend1 = mend-start1
        mend2 = mend-start2
        v1 = s1[mstart1:mend1]
        v2 = s2[mstart2:mend2]
        diff = v1-v2
        logger.info("Nb samples in common:{}, nb notequal:{}".format(diff.size, np.count_nonzero(diff)))
        # np.savetxt("s1.txt", s1[mstart1:mstart1+1000])
        # np.savetxt("s2.txt", s2[mstart2:mstart2+1000])
        # np.savetxt("s2.txt", s2)
        df = df.drop_duplicates(subset=signal_id, keep=False, ignore_index=True)

    df["Session"] = "MS_#"+ df.groupby(by=list(session_id)).ngroup().astype(str)
    return df