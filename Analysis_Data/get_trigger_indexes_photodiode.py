import os
from tracemalloc import stop
from cv2 import threshold
import numpy as np
import mne
import scipy
from scipy import signal
from mne.utils import _TempDir
import matplotlib.pyplot as plt
import pd_parser
from pd_parser.parse_pd import _read_raw, _to_tsv
from savitzky_golay import savitzky_golay
from scipy.signal import argrelextrema

def normalized(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def get_trigger_indexes_photodiode(data_raw_file='PAT_3066_EEG_708977_Anon_CategoryLocalizer.edf',data_path = '/Volumes/GoogleDrive/My Drive/EEG_DATA_PIERRE/PAT_3066',trigger_channel='Xe1',t_start = 0,t_end=-1,threshold_val=0.8):
    if data_raw_file == 'PAT_3066_EEG_708977_Anon_CategoryLocalizer.edf':
        print('*** USING EXAMPLE FILE ***')
    out_dir = _TempDir()
    print(f'*   Data location.... : {out_dir}\n\n')

    data_raw_file = os.path.join(data_path, data_raw_file)
    raw = mne.io.read_raw_edf(data_raw_file)
    picks = mne.pick_channels_regexp(raw.info.ch_names, regexp=trigger_channel)
    if t_end == -1: t_end = raw.__len__()
    raw.get_data(picks=picks,start=t_start,stop=t_end)
    raw2 = raw[-1][0]
    raw2 = np.array(raw2[0])
    del raw

    temp3 = np.diff(savitzky_golay(raw2, 11, 3)) # window size 51, polynomial order 3
    temp3 = normalized(temp3)

    index_final_onset,_ = scipy.signal.find_peaks(temp3, height=threshold_val*temp3.max())
    index_final_offset,_ = scipy.signal.find_peaks(-1*temp3, height=threshold_val-1)

    print('*   length of onset indexes')
    print(len(index_final_onset))
    # print('*   sample:' + [str(x +'; ')for x in index_final_onset[0:4]]])
    print('*   length of offset indexes')#: '+str(len(index_final_offset))])
    print(len(index_final_offset))
    return index_final_onset,index_final_offset
    # for local maxima

    
