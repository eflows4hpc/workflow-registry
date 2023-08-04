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
    examples = "Example:\n" + sys.argv[0] + " --nc sim_ts.nc --depth_file depth.dat --ptf_file outfile.nc"
  
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=Description, epilog=examples)
  
    parser.add_argument("--nc", default = None,
                                help = "Input nc infile. Default: None")
    parser.add_argument("--depth_file", default = None,
                                      help = "POIs' depth file name (input)")
    parser.add_argument("--ptf_file", default = None, #action="store_true",
                                      help = "Output file name (.nc) for the ptf procedure")
    # parser.add_argument("--ts_file", default = None,
    #                                  help = "Output time_series file. Default = None --> same name input file but with txt extention")
    # parser.add_argument("--time_file", default = "time.dat",
    #                                   help = "Output time file. Default: time.dat")
    # parser.add_argument("--grid_txt", default = None, 
    #                                   help = "Files with points. txt file format. sequence of points: None")
    # parser.add_argument("--save_out", default = "No", 
    #                                   help = "Save generated files. Yes/[No]")
    # parser.add_argument("--postproc", default = "Yes", 
    #                                  help = "Save post-processed files. [Yes]/No")
    parser.add_argument("--sub_offset", default = "Yes",
                                        help = "Subtract initial offset. [Yes]/No")
    parser.add_argument("--get_maxmin", default = "Yes", 
                                        help = "Extract the maximum value for each POI [Yes]/No")
    parser.add_argument("--gl", default = "Yes",
                                help = "Apply green law for amplification. [Yes]/No")
  
  
    args = parser.parse_args()
  
    return args


def load_nc(ncfile):
    """
    """
    if(os.path.isfile(ncfile) == False):
        sys.exit("File {0} not Found".format(ncfile))

    nc = h5py.File(ncfile,'r')
    time = np.array(nc["time"])
    eta = np.array(nc["eta"])
    return np.nan_to_num(eta), np.nan_to_num(time)


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


def remove_offset(ts):
    """
    """
    ts0 = ts[np.newaxis,0,:]  # adding a new axis to ts0 for broadcasting 
    ts_off = ts - ts0
    return ts_off



def get_maxmin(ts):
    """
    """
    return np.amax(ts, axis=0), np.amin(ts, axis=0)



def greens_law(hmax, depth):
    """
    """
    amplification = np.zeros((len(hmax)))
    ind = np.where(depth < 1.0)[0]
    depth[ind] = 1.0
    amplification = hmax*depth**(1.0/4.0)

    return amplification
    

############################################################

def step2_extract_ts(out_ts,depth_file,ptf_file,simwdir):

    #args = parse_line()
    nc = out_ts
    depth_file = depth_file
    get_maxmin_act = "Yes"
    sub_offset_act = "Yes"
    gl_act = "Yes"
    
    # load netcdf with times series
    if(nc is not None):
        ts, time = load_nc(nc)
    
    # Load depth
    if(depth_file is not None):
        poi_depth = read_depth(depth_file)
        pois = range(len(poi_depth))
    
    
    # POST-PROCESSING
    # extract hmax
    if(get_maxmin_act == "Yes"):
        ts_max, ts_min = get_maxmin(ts)
        #ts_max, ts_min, its_max, its_min = get_peak2through(ts)
        ts_p2t = 0.5*(ts_max - ts_min)
    
    # remove offset
    if(sub_offset_act == "Yes"):
        ts_off = remove_offset(ts)
        ts_max_off, ts_min_off = get_maxmin(ts_off)
    
    # apply greens law
    if(gl_act == "Yes"):
        ts_max_gl = greens_law(ts_max, poi_depth)
        #ts_min_gl = greens_law(ts_min)
        ts_max_off_gl = greens_law(ts_max_off, poi_depth)
        #ts_min_off_gl = greens_law(ts_min_off)
        ts_p2t_gl = greens_law(ts_p2t, poi_depth)
   
    #print(ts_max,ts_min,ts_max_gl,ts_min_off,ts_max_off_gl,ts_p2t,ts_p2t_gl) 
    # save output file for ptf with max min quantities (.nc)
    if(ptf_file is not None):
        ds = xr.Dataset(
            data_vars={"ts_max": (["pois"], ts_max),
                       "ts_min": (["pois"], ts_min),
                       "ts_max_gl": (["pois"], ts_max_gl),
                       "ts_max_off": (["pois"], ts_max_off),
                       "ts_min_off": (["pois"], ts_min_off),
                       "ts_max_off_gl": (["pois"], ts_max_off_gl),
                       "ts_p2t": (["pois"], ts_p2t),
                       "ts_p2t_gl": (["pois"], ts_p2t_gl)},
            coords={"pois": pois},
            attrs={"description": ptf_file})

        encode = {"zlib": True, "complevel": 9, "dtype": "float32",
              "_FillValue": False}
        encoding = {var: encode for var in ds.data_vars}
        ds.to_netcdf(ptf_file,engine="scipy")#,encoding=encoding)
        
        #ds = xr.Dataset(
        #        data_vars=dict(
        #                 ts_max=(["pois"], ts_max),
        #                 ts_min=(["pois"], ts_min),
        #                      ),
        #                 #  "ts_max_gl": (["pois"], ts_max_gl),
        #                 #  "ts_max_off": (["pois"], ts_max_off),
        #                 #  "ts_min_off": (["pois"], ts_min_off),
        #                 #  "ts_max_off_gl": (["pois"], ts_max_off_gl),
        #                 #  "ts_p2t": (["pois"], ts_p2t),
        #                 #  "ts_p2t_gl": (["pois"], ts_p2t_gl)},
        #        coords=dict(
        #               pois=(["pois"],pois),
        #                    ),
        #        attrs=dict(description=outfile)
        #)
        #encode = {"zlib": True, "complevel": 9, "dtype": "float32", 
        #          "_FillValue": False}
        #encoding = {var: encode for var in ds.data_vars}
        #ds.to_netcdf(outfile,'w',engine="scipy",encoding=encoding)
   
