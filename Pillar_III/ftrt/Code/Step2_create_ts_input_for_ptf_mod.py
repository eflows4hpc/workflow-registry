#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np
import xarray as xr
import h5py

def parse_line():
    """
    """
    Description = "Nc - THySEA file handler"
    examples = "Example:\n" + sys.argv[0] + " --ts_path path_to_scenario_folders --depth_file depth.dat"
  
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=Description, epilog=examples)
  
    parser.add_argument("--ts_path", default = "",
                                help = "Input folder containing nc files. Default: None")
    parser.add_argument("--depth_file", default = None,
                                      help = "Input POIs depth file. Default: None")
    parser.add_argument("--out_path", default = "",
                                help = "Output folder for the .nc file. Default: working folder")
    args = parser.parse_args()
  
    return args


def read_depth(depthfile):
    """
    """

    # if depthfile is None:
    #     sys.exit("Depth file is missing")
    # else:
    #     depthfile = depthfile
        
    if(os.path.isfile(depthfile) == False):
        sys.exit("File {0} not Found".format(depthfile))

    poi_dep = []
    with open(depthfile) as f:
        for line in f:
            tmp = float(line.split()[0])
            poi_dep.append(tmp)
            
    return np.array(poi_dep)


def save_ts_out_ptf_tot(scenarios, pois, outdir, seis_type, errfile):
    """
    """

    n_scenarios = len(scenarios)
    n_pois = len(pois)

    ts_min = np.zeros((n_scenarios, n_pois))
    ts_max = np.zeros((n_scenarios, n_pois))
    ts_max_gl = np.zeros((n_scenarios, n_pois))
    ts_min_off = np.zeros((n_scenarios, n_pois))
    ts_max_off = np.zeros((n_scenarios, n_pois))
    ts_max_off_gl = np.zeros((n_scenarios, n_pois))
    ts_p2t = np.zeros((n_scenarios, n_pois))
    ts_p2t_gl = np.zeros((n_scenarios, n_pois))
    fail = np.zeros((n_scenarios))

    errfile = os.path.join(outdir, "Step2_" + seis_type + "_failed.log")
    ferr = open(errfile,'w')

    for isc, scenario in enumerate(scenarios):

        #tsfile = os.path.join(path, scenario, "out_ts_ptf.nc")
        tsfile = scenario
        print(tsfile)
        if(os.path.isfile(tsfile) == False):
            ferr.write(tsfile + ' not found' + '\n')
            ts_max[isc,:] = -9999
            ts_min[isc,:] = -9999
            ts_p2t[isc,:] = -9999
            ts_max_off[isc,:] = -9999
            ts_min_off[isc,:] = -9999
            ts_max_gl[isc,:] = -9999
            ts_max_off_gl[isc,:] = -9999
            ts_p2t_gl[isc,:] = -9999
        else:
            #nc = h5py.File(tsfile,'r')
            nc = xr.open_dataset(tsfile)
            ts_p2t_tmp = np.array(nc["ts_p2t_gl"].values)
            max_val = np.max(ts_p2t_tmp)
            if max_val>100:
               fail[isc]=1
            else:
               fail[isc]=0
            ts_max[isc,:] = nc["ts_max"].values
            ts_min[isc,:] = nc["ts_min"].values
            ts_p2t[isc,:] = nc["ts_p2t"].values
            ts_max_off[isc,:] = nc["ts_max_off"].values
            ts_min_off[isc,:] = nc["ts_min_off"].values
            ts_max_gl[isc,:] = nc["ts_max_gl"].values
            ts_max_off_gl[isc,:] = nc["ts_max_off_gl"].values
            ts_p2t_gl[isc,:] = nc["ts_p2t_gl"].values
            #print(ts_max[isc,:].shape, nc["ts_max"].values.shape)
            #ts_max[isc,:] = np.array(nc["ts_max"])
            #ts_min[isc,:] = np.array(nc["ts_min"])
            #ts_p2t[isc,:] = np.array(nc["ts_p2t"])
            #ts_max_off[isc,:] = np.array(nc["ts_max_off"])
            #ts_min_off[isc,:] = np.array(nc["ts_min_off"])
            #ts_max_gl[isc,:] = np.array(nc["ts_max_gl"])
            #ts_max_off_gl[isc,:] = np.array(nc["ts_max_off_gl"])
            #ts_p2t_gl[isc,:] cenario np.array(nc["ts_p2t_gl"])


    pois = range(n_pois)
    scenarios = range(n_scenarios)
    outfile = os.path.join(outdir, "Step2_" + seis_type + "_hmax.nc")

    ds = xr.Dataset(
            data_vars={"ts_max": (["scenarios", "pois"], ts_max),
                       "ts_min": (["scenarios", "pois"], ts_min),
                       "ts_max_gl": (["scenarios", "pois"], ts_max_gl),
                       "ts_max_off": (["scenarios", "pois"], ts_max_off),
                       "ts_min_off": (["scenarios", "pois"], ts_min_off),
                       "ts_max_off_gl": (["scenarios", "pois"], ts_max_off_gl),
                       "ts_p2t": (["scenarios", "pois"], ts_p2t),
                       "ts_p2t_gl": (["scenarios", "pois"], ts_p2t_gl)},
            coords={"pois": pois, "scenarios": scenarios},
            attrs={"description": outfile})


    encode = {"zlib": True, "complevel": 9, "dtype": "float32", 
              "_FillValue": False}
    encoding = {var: encode for var in ds.data_vars}
    print("Writting file " + outfile, flush=True) 
    ds.to_netcdf(outfile,'w',engine="scipy")
    #ds.to_netcdf(outfile, format="NETCDF4", encoding=encoding)

    ferr.close()

    return fail

######################################################################

def step2_create_ptf_input(ts_path, out_path, depth_file, log_file):

    #args = parse_line()
    scenarios_bs = ts_path
    print("Scenarios:" + str(scenarios_bs))
    #ts_path = ts_path
    depth_file = depth_file
    out_path = out_path
    print("OutPath:" + out_path)
    #if(ts_path is not None):
    #    ts_path = ts_path
    #    bs_path = os.path.join(ts_path, "Step2_BS")
    #    ps_path = os.path.join(ts_path, "Step2_PS")
    #else:
    #    sys.exit("ts file path is missing")
    
    if(depth_file is not None):
        poi_depth = read_depth(depth_file)
    else:
        sys.exit("depth file path is missing")
    
    if(out_path is not None):
        outdir = out_path
    
    if os.path.isdir(out_path):
        #scenarios_bs = [d for d in sorted(os.listdir(bs_path)) if "BS" in d]
        fail=save_ts_out_ptf_tot(scenarios_bs, poi_depth, outdir,"BS",log_file)
    else:
        print("WARN: Outpath not a directory", flush=True)
    #if os.path.isdir(ps_path):
    #    scenarios_ps = [d for d in sorted(os.listdir(ps_path)) if "PS" in d]
    #    save_ts_out_ptf_tot(ps_path, scenarios_ps, poi_depth, outdir, "PS",log_file)

    return fail


