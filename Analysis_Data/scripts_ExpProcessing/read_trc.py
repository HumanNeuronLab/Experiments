
"""Load sleep files.

This file contain functions to load :
- European Data Format (*.edf)
- Micromed (*.trc)
- BrainVision (*.vhdr)
- ELAN (*.eeg)
- Hypnogram (*.hyp)
"""
import os
import io
from warnings import warn
import logging
import datetime

import numpy as np
from scipy.stats import iqr

# from visbrain.io.dependencies import is_mne_installed
# from visbrain.io.dialog import dialog_load
# from visbrain.io.mneio import mne_switch
# from visbrain.io.rw_hypno import (read_hypno, oversample_hypno)
# from visbrain.io.rw_utils import get_file_ext
# from visbrain.io.write_data import write_csv
# from visbrain.io import merge_annotations

# from visbrain.utils.others import get_dsf
# from visbrain.utils.mesh import vispy_array
# from visbrain.utils.sleep.hypnoprocessing import sleepstats

# from visbrain.config import PROFILER

# logger = logging.getLogger('visbrain')

# __all__ = ['ReadSleepData', 'get_sleep_stats']


def read_trc(path):
    """Read data from a Micromed (trc) file (version 4).

    Poor man's version of micromedio.py from Neo package
    (https://pythonhosted.org/neo/)

    Parameters
    ----------
    path : str
        Filename(with full path) to .trc file
    downsample : int
        Down-sampling frequency.

    Returns
    -------
    sf : float
        The sampling frequency.
    downsample : float
        The downsampling frequency
    data : array_like
        The data organised as well(n_channels, n_points)
    chan : list
        The list of channel's names.
    n : int
        Number of samples before down-sampling.
    start_time : array_like
        Starting time of the recording (hh:mm:ss)
    annotations : array_like
        Array of annotations.
    """
    import struct

    def read_f(f, fmt):
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

    with io.open(path, 'rb') as f:
        # Read header
        f.seek(175, 0)
        header_version, = read_f(f, 'b')
        assert header_version == 4
        print(f)
        f.seek(138, 0)
        data_start_offset, n_chan, anotat, sf, nbytes = read_f(f, 'IHHHH')
        print(n_chan)
        f.seek(128, 0)
        day, month, year, hour, minute, sec = read_f(f, 'bbbbbb')
        start_time = datetime.time(hour, minute, sec)

        # Raw data
        f.seek(data_start_offset, 0)
        m_raw = np.frombuffer(f.read(), dtype='u' + str(nbytes))
        m_raw = m_raw.reshape((int(m_raw.size / n_chan), n_chan)).transpose()

        # Read label / gain
        gain = []
        chan = []
        logical_ground = []
        data = np.empty(shape=m_raw.shape, dtype=np.float32)

        f.seek(176, 0)
        zone_names = ['ORDER', 'LABCOD']
        zones = {}
        for zname in zone_names:
            zname2, pos, length = read_f(f, '8sII')
            zones[zname] = zname2, pos, length

        zname2, pos, length = zones['ORDER']
        f.seek(pos, 0)
        code = np.fromfile(f, dtype='u2', count=n_chan)

        for c in range(n_chan):
            zname2, pos, length = zones['LABCOD']
            f.seek(pos + code[c] * 128 + 2, 0)

            chan = np.append(chan, f.read(6).decode('utf-8').strip())
            logical_min, logical_max, logic_ground_chan, physical_min, \
                physical_max = read_f(f, 'iiiii')

            logical_ground = np.append(logical_ground, logic_ground_chan)

            gain = np.append(gain, float(physical_max - physical_min) /
                             float(logical_max - logical_min + 1))

    # Multiply by gain
    m_raw = m_raw - logical_ground[:, np.newaxis]
    data = m_raw * gain[:, np.newaxis].astype(np.float32)

    # Get original signal length :
    n = data.shape[1]

    # Get down-sample factor :
    sf = float(sf)
    chan = list(chan)
    print(chan)
    # dsf, downsample = get_dsf(downsample, sf)

    return sf, data, chan, n, start_time




data_path = '/Volumes/GoogleDrive/My Drive/EEG_DATA_PIERRE/PAT_3066'

data_raw_file_trc = os.path.join(data_path, 'PAT_3066_EEG_695997_Anon_FLM.TRC')

sf, data, chan, n, start_time = read_trc(data_raw_file_trc)
print(np.shape(data))