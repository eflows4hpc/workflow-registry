# -*- coding: utf8 -*-
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import sys
import numpy as np
from glob import glob
from obspy.clients.filesystem.sds import Client
from obspy.core import read, Stream, UTCDateTime


def read_traces_SDS(config, basepath):
    kwargs = {}
    if config.data_format:
        kwargs["format"] = config.data_format
        print("Data format: %s" % config.data_format)

    tmpst = Stream()

    # ---Here we expect that start_time and end_time are always provided
    start_t = UTCDateTime(config.start_time)
    end_t = UTCDateTime(config.end_time)
    time_lag = config.time_lag
    t_overlap = config.t_overlap
    # ---
    # --- Making list of time-windows
    t_bb = np.arange(config.start_t, config.end_t, time_lag - t_overlap)
    t_end = t_bb + time_lag
    # --- Adjusting time-window to the length of requested data:
    # to avoid the case when requested time-windows exceed the data length
    time_length = abs(end_t - start_t)
    t_bb = t_end[t_end <= time_length] - time_lag
    t_end = t_bb + time_lag
    # -----------------------------------------------------------

    # config.dataarchive_type == "SDS":
    print("SDS Data Archive type")
    client = Client(basepath)
    tmpst = client.get_waveforms(config.data_network, "*", "*", "*", start_t, end_t)

    # --- Selecting stations:
    # Get the intersection between the list of available stations
    # and the list of requested stations:
    tmpst_select = Stream()
    for ch in config.channel:
        tmpst_select += tmpst.select(channel=ch)
    tmpst_stations = [tr.stats.station for tr in tmpst_select]
    stations = sorted(set(tmpst_stations) & set(config.stations))

    # Retain only requested channel and stations:
    st = Stream(tr for tr in tmpst_select if tr.stats.station in stations)
    if not st:
        print("Could not read any trace!")
        sys.exit(1)
    st.sort()

    # --- Removing stations with short record
    for ssta in stations:
        tmpst = st.select(station=ssta)
        # --- Checking start-end of the trace
        lendat_sec = 0
        for trr in tmpst:
            tr_start = abs(trr.stats.starttime - start_t)
            tr_end = abs(trr.stats.endtime - start_t)
            lendat_sec += tr_end - tr_start
            # print(trr, lendat_sec, end_t-start_t)
        if lendat_sec < (end_t - start_t) / 3:
            print(ssta, "short data")
            # Remove stations with too short data
            for tr in tmpst:
                st.remove(tr)
    if len(st) < 1:
        print("No data to process, try shorter time-length")
        sys.exit(1)

    # --- Updating station list ---
    tmpst_stations = [tr.stats.station for tr in st]
    stations = sorted(set(tmpst_stations) & set(config.stations))

    # --- Adjusting time-windows to data starts/ends
    rec_starts = [st.select(station=ssta)[0].stats.starttime for ssta in stations]
    rec_end = [st.select(station=ssta)[-1].stats.endtime for ssta in stations]
    max_tr_start = max(rec_starts)
    min_tr_end = min(rec_end)
    # --- start adjustment:
    if (t_bb[0] - (max_tr_start - start_t)) < 0:
        t_bb = t_bb[t_bb >= (max_tr_start - start_t)]
        t_end = t_bb + time_lag
    # --- end adjustment:
    if (min_tr_end - start_t - t_end[-1]) < 0:
        t_bb = t_end[t_end <= (min_tr_end - start_t)] - time_lag
        t_end = t_bb + time_lag
    # print(t_bb[0], t_bb[-1], max_tr_start, start_t)
    # --- Check for gaps and do time-window adjustment:
    gaps_sta = {}
    stagaps = []
    for ssta in stations:
        gaps_sta[ssta] = []
        tmpst = st.select(station=ssta)
        # --- getting list with gap info including start and end times of the gaps
        gaps_tmp = tmpst.get_gaps()
        #
        # -- list of gap starts & ends in seconds from start_t
        sta_gapbeg = [gg[4] - start_t for gg in gaps_tmp]
        sta_gapend = [gg[5] - start_t for gg in gaps_tmp]
        for idg in range(len(gaps_tmp)):
            gstart = sta_gapbeg[idg]
            gend = sta_gapend[idg]
            idx1 = (np.abs(t_bb - gstart)).argmin()
            idx2 = (np.abs(t_bb - gend)).argmin()
            # dte = min(t_bb-gend)
            # print(gstart, idx1, gend, idx2)
            gaps_sta[ssta].append([idx1, idx2])
        # ---
        if len(gaps_sta[ssta]) > 0 and ssta not in stagaps:
            stagaps.append(ssta)
    # Removing some of the stations with gaps if the number of sta < 10%
    persta = np.round(100.0 / len(stations) * len(stagaps), 0)
    # print('------', persta)
    if persta < 10.0:
        for sta in stagaps:
            for tr in st.select(station=sta):
                st.remove(tr)
        tbb_adj = t_bb
        tend_adj = np.array(tbb_adj) + time_lag
    else:
        st.merge(method=1, fill_value="interpolate")
        idwin_not = []
        for kk in gaps_sta.keys():
            #
            for gg in gaps_sta[kk]:
                ids = [i for i in range(gg[0], gg[1] + 1) if i not in idwin_not]
                for ii in ids:
                    idwin_not.append(ii)
        if len(idwin_not) > 0:
            #
            tbb_adj = [t_bb[i] for i in range(len(t_bb)) if i not in idwin_not]
            tend_adj = np.array(tbb_adj) + time_lag
        else:
            # print('No gaps in data')
            tbb_adj = t_bb
            tend_adj = np.array(tbb_adj) + time_lag
    # -------------------------------------------------------------------------

    # Check sampling rate -----------------------------------------------------
    config.delta = None
    for tr in st:
        tr.detrend(type="constant")
        tr.taper(type="hann", max_percentage=0.005, side="left")
        sampling_rate = tr.stats.sampling_rate
        # Resample data, if requested
        if config.sampl_rate_data:
            if sampling_rate >= config.sampl_rate_data:
                dec_ct = int(sampling_rate / config.sampl_rate_data)
                tr.decimate(dec_ct, strict_length=False, no_filter=True)
            else:
                raise ValueError(
                    "Sampling frequency for trace %s is lower than %s"
                    % (tr.id, config.sampl_rate_data)
                )
        delta = tr.stats.delta
        if config.delta is None:
            config.delta = delta
        else:
            if delta != config.delta:
                raise ValueError(
                    "Trace %s has different delta: %s (expected: %s)"
                    % (tr.id, delta, config.delta)
                )
    # Recompute sampling rate after resampling
    config.sampl_rate_data = st[0].stats.sampling_rate

    # ---Will not actually need it with Gap check  --- Check if true!!!
    # Check for common starttime and endtime of the traces
    # st_starttime = max([tr.stats.starttime for tr in st])
    # st_endtime = min([tr.stats.endtime for tr in st])
    # if config.start_time:
    #     st.trim(max(st_starttime, UTCDateTime(config.start_time)),
    #             min(st_endtime, UTCDateTime(config.end_time)))
    # else:
    #     st.trim(st_starttime, st_endtime)

    # --- cut the data to the selected length dt------------------------------
    if config.cut_data:
        st.trim(
            st[0].stats.starttime + config.cut_start,
            st[0].stats.starttime + config.cut_start + config.cut_delta,
        )
    else:
        config.cut_start = 0.0

    # config.starttime = st[0].stats.starttime
    config.starttime = start_t

    # attach station list and trace ids to config file
    config.stations = [tr.stats.station for tr in st]
    config.trids = [tr.id for tr in st]

    print("Number of traces in stream = ", len(st))
    # print('nr adjusted windows=', len(tbb_adj), 'nr initial windows=', len(t_bb))
    return st, tbb_adj
