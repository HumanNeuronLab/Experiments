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
import pyqtgraph as pg
import pyqtgraph as plotWidget
from scipy.signal import butter, lfilter, freqz


# y = butter_lowpass_filter(data, cutoff, fs, order)
def butter_lowpass_filter(data,cutoff,fs,order):
    b,a = butter(order, cutoff, fs=fs, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y


def normalized(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def get_trigger_indexes_photodiode(data_raw_file='PAT_3066_EEG_708977_Anon_CategoryLocalizer.edf',data_path = '/Volumes/GoogleDrive/My Drive/EEG_DATA_PIERRE/PAT_3066',trigger_channel='Xe1',t_start = 0,t_end=-1,threshold_val=0.8,do_plot=True,t_shift=35):
    if data_raw_file == 'PAT_3066_EEG_708977_Anon_CategoryLocalizer.edf':
        print('*   *** USING EXAMPLE FILE ***')
    out_dir = _TempDir()
    print(f'*   Data location.... : {out_dir}\n\n')

    # load trigger data
    data_raw_file = os.path.join(data_path, data_raw_file)
    raw = mne.io.read_raw_edf(data_raw_file)
    picks = mne.pick_channels_regexp(raw.info.ch_names, regexp=trigger_channel)
    if t_end == -1: t_end = raw.__len__()
    raw = raw.get_data(picks=picks,start=t_start,stop=t_end)
    raw = normalized(raw[0])
    print(len(raw),np.shape(raw))
    time_s = np.linspace(0,len(raw),num=len(raw))

    # compute the differential
    temp3 = butter_lowpass_filter(raw, 10, 2048, order=2)
    temp3 = np.diff(savitzky_golay(temp3, 31, 5)) # window size 51, polynomial order 3
    # temp3 = np.diff(savitzky_golay(raw, 71, 3)) # window size 51, polynomial order 3
    temp3 = normalized(temp3)

    # find the peaks
    index_final_offset,_ = scipy.signal.find_peaks(temp3, height=threshold_val*temp3.max())
    index_final_onset,_ = scipy.signal.find_peaks(-1*temp3, height=threshold_val-1)

    # Information Printed
    print('*   length of onset indexes: ',len(index_final_onset))
    print('*   length of offset indexes: ',len(index_final_offset))#: '+str(len(index_final_offset))])
    print('*   length of Time & Trigger: ',len(time_s),' ',len(raw))#: '+str(len(index_final_offset))])
    # print()

    # If paramenter is true, show plotted trigger points
    if do_plot == True:
        plotWidget = pg.plot(title="offset = 0, onset = x")
        plotWidget.plot(time_s,raw)
        plotWidget.plot(index_final_offset-t_shift,raw[index_final_offset-t_shift],pen=None, symbol='o')
        plotWidget.plot(index_final_onset-t_shift,raw[index_final_onset-t_shift],pen=None, symbol='x')
        plotWidget.show()

    # return the onset and offset index values
    return index_final_onset+t_start-t_shift,index_final_offset+t_start-t_shift,raw,time_s
    # for local maxima
