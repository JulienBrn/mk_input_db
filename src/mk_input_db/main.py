import logging, beautifullogger
import sys
import pandas as pd, numpy as np
from toolbox import matlab_loader, df_loader, Manager
from typing import List, Any
from mk_input_db.human_input import read_human_stn_matlab_database, add_raw_signals, add_neuron_signals, add_human_session_ids, read_stn_inputs
from mk_input_db.monkey_input import read_monkey_inputs, add_monkey_session_ids
import pathlib, toolbox
from tqdm import tqdm

logger = logging.getLogger(__name__)

def setup_nice_logging():
    beautifullogger.setup(logmode="w")
    logging.getLogger("toolbox.ressource_manager").setLevel(logging.WARNING)
    logging.getLogger("toolbox.signal_analysis_toolbox").setLevel(logging.WARNING)

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        logger.info("Keyboard interupt")
        sys.exit()
        return
    else:
        sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = handle_exception

computation_m = Manager("./cache/computation")


def run():
    from tqdm import tqdm
    setup_nice_logging()
    logger.info("Running start")
    dfs = []
    for f, key in zip(
        ["/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/Human_STN_Correct/all_Sorting_data_DLOR_Part{}.mat".format(i) for i in range(1, 4)],
        ["allSortingResultsDatabase_DLOR_"+str(i) for i in range(1, 4)]):
        df = read_human_stn_matlab_database(f, key, computation_m)
        # print(df)
        dfs.append(df)
    for f, key in zip(
        ["/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/Human_STN_Correct/all_Sorting_data_VMNR_Part{}.mat".format(i) for i in range(1, 4)],
        ["allSortingResultsDatabase_VMNR_"+str(i) for i in range(1, 4)]):
        df = read_human_stn_matlab_database(f, key, computation_m)
        # print(df)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    print(df)
    raw_df  = add_raw_signals(df, computation_m)
    print(raw_df)
    neuron_df  = add_neuron_signals(df, computation_m)
    print(neuron_df)
    raw_df = raw_df.sample(frac=1).head(100).reset_index(drop=True)
    neuron_df = neuron_df.sample(frac=1).head(100).reset_index(drop=True)
    all_df = pd.concat([raw_df, neuron_df], ignore_index=True, axis=0)
    all_df = add_human_session_ids(all_df)
    print(all_df)

    monkey_df = read_monkey_inputs("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/MarcAnalysis/Inputs/MonkeyData4Review/BothMonkData_withTime.csv", computation_m)
    monkey_df = add_monkey_session_ids(monkey_df)
    monkey_df["num_source"] = monkey_df["signal_path"].apply(lambda x: len(x) if hasattr(x, "__len__") else 1)
    monkey_df = monkey_df.sort_values(by="num_source", ascending=False)
    print(monkey_df)
    tqdm.pandas(desc="Computing monkey signals")
    for sig in tqdm(monkey_df["signal"]):
        sig.get_result()
    # monkey_df["signal"].progress_apply(lambda r: r.get_result())
    
    # tqdm.pandas(desc="Computing file ressources fs")
    # from toolbox import Profile
    # with Profile() as p:
    #     raw_df["signal_fs"].progress_apply(lambda r: r.get_result())
    # r = p.get_results()
    # df_loader.save("profiling.tsv", r)
    # tqdm.pandas(desc="Computing spike file ressources")
    # neuron_df["signal"].progress_apply(lambda r: r.get_result())
    # tqdm.pandas(desc="Computing mua file ressources")
    # raw_df["signal"].progress_apply(lambda r: r.get_result())
    
    raise BaseException("stop")
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
        grp['count'] = df.groupby(list(signal_id))[list(signal_id)[0]].transform('count')
        logger.warning("Duplication. Examples:\n{}.\nContinuing while ignoring them".format(grp[grp["count"] > 1]))
        df = df.drop_duplicates(subset=signal_id, keep=False, ignore_index=True)

    
    df_loader.save("test.tsv", df)

    import pathlib, toolbox
    from tqdm import tqdm
    def get_file_ressource(signal: DataPath, neuron_num):
        dp, neuron = signal, neuron_num
        try:
            if np.isnan(neuron):
                mat =  matlab_loader.load("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/Human_STN_Correct_All/"+dp.file_path)
                raw = np.squeeze(mat[dp.keys[0]])
                logger.info("Got ressource {}, neuron = {}. Shape is {}".format(dp, neuron, raw.shape))
                return raw
            else:
                mat = matlab_loader.load("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/Human_STN_Correct_All/"+ dp.file_path)
                spike_df = np.squeeze(mat["sortingResults"])[()]
                spike_arr = np.squeeze(spike_df[6])
                res = pd.DataFrame(spike_arr, columns=["neuron_num", "spike_time"] + ["pca_{}".format(i) for i in range(spike_arr.shape[1]-2)])
                spike = res.loc[res["neuron_num"]==neuron+1, "spike_time"].to_numpy()
                logger.info("Got ressource {}, neuron = {}. Shape is {}".format(dp, neuron, spike.shape))
                return spike
        except KeyboardInterrupt as e:
            raise e
        except BaseException as e:
            logger.error("Error while getting ressource {}, neuron = {}. Error is:\n{}".format(dp, neuron, e))
            return np.nan
        

    tqdm.pandas(desc="Declaring file ressources")

    df = toolbox.mk_block(df, ["signal", "neuron_num"], get_file_ressource, (toolbox.np_loader, "inputs", True), computation_m)
    print(df)
    df = df.sample(frac=1).reset_index(drop=True)
    tqdm.pandas(desc="Computing file ressources")
    df["inputs"].progress_apply(lambda r: r.get_result())
    print(df)

    
    # input()
    # def myconv(x):
    #     try:
    #         return 0 if np.isnan(x.astype(float, casting="unsafe")) else int(x)
    #     except:
    #         return 0
    # df["n_units"] = df["number of units"].apply(myconv)
    # for i in range(4):
    #     df["neuron_data{}".format(i)] = list(zip(df["Unit {} Rate".format(i+1)], df["Unit {} Isolaton".format(i+1)]))
    #     df.drop(columns = ["Unit {} Rate".format(i+1), "Unit {} Isolaton".format(i+1)], inplace=True)
    # print(df)
    # input()
    # # df.melt(id_vars=list(set(df.columns) - set([])) )
    # neuron_df = pd.wide_to_long(df, stubnames="neuron_data", i = list(raw_id), j="neuron_num").reset_index()
    # neuron_df["Rate"] = neuron_df["neuron_data"].str[0]
    # neuron_df["Isolation"] = neuron_df["neuron_data"].str[1]
    # neuron_df.drop(columns = ["neuron_data"], inplace=True)
    # neuron_df = neuron_df.loc[~neuron_df["Rate"].isna()].reset_index()
    # neuron_df = neuron_df.loc[neuron_df["Isolation"] > 0.6].reset_index()
    # tmp_df = df.loc[~df["n_units"].isna()]
    # neuron_df = df.loc[df.index.repeat(df["n_units"])]
    # neuron_df["unit_id"] =  neuron_df.groupby(level=0).cumcount() +1
    # neuron_df["Rate"] = neuron_df[]
    
    # logger.info("\n{}".format(neuron_df))
    # input()
    
    # for f, key, fs_key, spike, nb in zip(df["raw_file_path"], df["raw_data_key"], df["raw_fs_key"], df["spike_file_path"], df["number of units"]):
    #     mat = matlab_loader.load("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/Human_STN_Correct_All/"+ f)
    #     try:
    #         raw = np.squeeze(mat[key])
    #     except:
    #         raw = None
    #     try:
    #         fs = np.squeeze(mat[fs_key])[()]
    #     except:
    #         fs = np.nan
    #     print(raw, fs)
    #     try:
    #         spike_mat = matlab_loader.load("/run/user/1000/gvfs/smb-share:server=filer2-imn,share=t4/Julien/Human_STN_Correct_All/"+ spike)
    #         spike_df = np.squeeze(spike_mat["sortingResults"])[()]
    #         spike_arr = np.squeeze(spike_df[6])
    #         res = pd.DataFrame(spike_arr, columns=["neuron_num", "spike_time"] + ["pca_{}".format(i) for i in range(spike_arr.shape[1]-2)])

    #         print("nb_neurons = {}".format(nb))
    #         # for entry in spike_df:
    #         #     entry = np.squeeze(entry)
                
    #         #     if len(entry.shape) == 2:
    #         #         print(pd.DataFrame(entry))
    #         #     else:
    #         #         print("sp df", entry.shape)
    #         # print()
    #         print(res)
    #         input()
    #     except BaseException as e:
    #         print("no spike file for {}. Error is {}".format(spike, e))
        # input()
    

    logger.info("Running end")