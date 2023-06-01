import logging, beautifullogger
import sys
import pandas as pd, numpy as np
from toolbox import matlab_loader, df_loader, Manager
from typing import List, Any

logger = logging.getLogger(__name__)


class DataPath:
    file_path: str
    keys: List[Any]

    def __init__(self, file_path, keys=[]):
        self.file_path = file_path
        self.keys = keys

    def __str__(self):
        if len(self.keys) > 0:
            return "File({})[{}]".format(self.file_path, ", ".join([str(key) for key in self.keys]))
        else:
            return "File({})".format(self.file_path)
        
    def __repr__(self):
        return self.__str__()
    
class TimeSeries:
    def start(self) -> float: pass
    def end(self) -> float: pass
    def duration(self) -> float: pass
    def get_table(self) -> pd.DataFrame: pass


class RegularTimeSeries(TimeSeries):
    fs: float
    signal: np.array

    def __init__(self, fs, signal):
        self.fs = fs
        if np.squeeze(self.signal).ndim != 1:
            raise BaseException("Error when creating RegurlarTimeSeries. Wrong number of dimensions.")
        self.signal = np.squeeze(self.signal)
    
    def duration(self) -> float:
        return float(self.signal.size) / self.fs
    
    def get_table(self):
        d= pd.DataFrame()
        d["value"] = self.signal
        d["times"] = np.arange(0, self.duration(), 1.0/self.fs)
        return d
    
class EventTimeSeries(TimeSeries):
    times: np.array
    values: np.array
    duration: float

    def duration(self) -> float:
        return self.duration
    
    def get_table(self):
        d= pd.DataFrame()
        d["value"] = self.signal
        d["times"] = np.arange(0, self.duration(), 1.0/self.fs)
        return d


common_metadata_cols = ["Species", "Condition", "Structure", "Date", "Hemisphere", "Electrode", "Depth", "Subject", "signal_path"]
all_input_cols = ["signal_type",  "signal_fs", "signal_path", "signal", "Start", "End", "Species", "Condition", "Structure", "Date", "Hemisphere", "Electrode", "neuron_num", "Depth", "Subject"]