import os
import sys
import ast
import numpy as np
import h5py
import netCDF4
import math as math
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.basemap import Basemap

from ptf_save import define_save_path
from ptf_save import define_file_names



def make_figures(**kwargs):


    Config                = kwargs.get('cfg', None)
    args                  = kwargs.get('args', None)
    ee                    = kwargs.get('event_parameters', None)
    ptf                   = kwargs.get('ptf', None)
    file_hc               = kwargs.get('file_hc', None)
    fig_path              = kwargs.get('fig_path', None)
    fail                  = kwargs.get('fail', None)

    save_dict = define_save_path(cfg = Config,args = args, event = ee)
    save_dict = define_file_names(cfg = Config,args = args, event = ee, dictionary=save_dict)
    
    file_tim = Config.get('Figures','file_tim')
    file_sel = Config.get('Figures','file_user')
    file_pois = Config.get('Figures','file_pois')
    file_bathy = Config.get('Figures','file_bathy')

    ptf_hc = make_hazard_curves(cfg                = Config,
                                args               = args,
                                event_parameters   = ee,
                                file_hc            = file_hc,
                                file_pois          = file_pois,
                                file_sel           = file_sel,
                                file_tim           = file_tim,
                                file_bathy         = file_bathy,
                                fig_path           = fig_path,
                                fail               = fail)


    ptf_map = make_hazard_maps(cfg                = Config,
                               args               = args,
                               event_parameters   = ee,
                               file_hc            = file_hc,
                               file_pois          = file_pois,
                               file_tim           = file_tim,
                               file_bathy         = file_bathy,
                               fig_path           = fig_path,
                               fail               = fail)


    out=1

    return out


def make_hazard_curves(**kwargs):

    Config                = kwargs.get('cfg', None)
    args                  = kwargs.get('args', None)
    ee                    = kwargs.get('event_parameters', None)
    file_hc               = kwargs.get('file_hc', None)
    file_pois             = kwargs.get('file_pois', None)
    file_sel              = kwargs.get('file_sel', None)
    file_tim              = kwargs.get('file_tim', None)
    file_bathy            = kwargs.get('file_bathy', None)
    fig_path              = kwargs.get('fig_path', None)
    fail                  = kwargs.get('fail', None)

    percentiles=np.array((Config.get('Figures','percentiles')))

    lat=[]
    lon=[]
    f = open(file_pois, 'r')
    for line in f:
        col = line.split()
        lon.append(float(col[2]))
        lat.append(float(col[3]))
    f.close()
    lon=np.array(lon)
    lat=np.array(lat)
    tglen=len(lon)
    
    latu=[]
    lonu=[]
    f = open(file_sel, 'r')
    next(f)
    for line in f:
        col = line.split()
        lonu.append(float(col[0]))
        latu.append(float(col[1]))
    f.close()
    lonu=np.array(lonu)
    latu=np.array(latu)
    
    TIM=[]
    f = open(file_tim, 'r')
    for line in f:
        TIM.append(float(line))
    f.close
    TIM=np.array(TIM)
    
    latp=np.zeros(len(lonu))
    lonp=np.zeros(len(lonu))
    ##### Identification of the closest POIs to the users coordinates #####
    id_min=np.zeros(len(lonu))
    for upoi in range(len(lonu)):
        lon_dif = lon-lonu[upoi]
        lat_dif = lat-latu[upoi]
        diff = np.sqrt((lon_dif)**2+(lat_dif)**2)*111.0
        id_min[upoi] = np.argmin(diff)
        idm = int(np.argmin(diff))
        latp[upoi]=lat[idm]
        lonp[upoi]=lon[idm]
    
    with h5py.File(file_hc, "r") as f:
        HC_P = np.array(f['hazard_curves_bs_at_pois'])
        HC_M = np.array(f['hazard_curves_bs_at_pois_mean'])
        IM_O = np.array(f['Intensity_measure_all_bs'])
    
    #####  Map of the user POIs positions #####
    
    n=np.arange(len(lonu))
    fig=plt.figure(figsize=(20,15))
    plt.subplots_adjust(left=0.05,right=0.95,top=0.90,bottom=0.05,wspace=0.15,hspace=0.05)
    #m = Basemap(projection='spstere', llcrnrlat=y1,urcrnrlat=y2,llcrnrlon=x1,urcrnrlon=x2,boundinglat=-63,lon_0=180, resolution='l',round=True)
    m = Basemap(llcrnrlon=-10,llcrnrlat=29.0,urcrnrlon=40,urcrnrlat=46.0,resolution='l')
    m.drawmeridians(np.arange(-10,40,5),labels=[False,True,True,False])
    m.drawparallels(np.arange(29,46,5),labels=[True,False,False,True])
    m.bluemarble(alpha=0.42)
    m.drawcoastlines(color='#555566', linewidth=1)
    m.drawmapboundary()
    cm1 = plt.cm.get_cmap('RdYlBu_r')
    plt.scatter(lonu,latu,c='k',s=150,zorder=3,label='User positions')
    plt.scatter(lonp,latp,c='r',s=150,zorder=3,label='Closest POIs')
    for i, txt in enumerate(n):
        plt.annotate(txt, (lonp[i], latp[i]),color='w', weight='bold',fontsize=20)#, xytext = (0,0.1))
    plt.legend()
    plt.savefig(fig_path+'map_pois_sel.png',bbox_inches='tight')

    #####  Plot of the hazard curves #####
   
    count=0 
    for k in list(id_min):
        k=int(k)
        HC = HC_P[k,:]
        nonNull_id = np.where(HC>0)
        nonNull = HC[nonNull_id[0]]
        nonNullIm = TIM[nonNull_id[0]]
        if not nonNull.any():
            print("list empty")
            toll = 1.e-10
        else:
            xx = np.append(1, nonNull)
            yy = np.append(0, nonNullIm)
            uniq = np.where(np.diff(xx, n=1, axis=-1)!=0)
            xx_fin = np.flip(xx[uniq[0]])
            yy_fin = np.flip(yy[uniq[0]])
    
        fig = plt.figure(figsize=(8,6))
        plt.plot(yy_fin,xx_fin)
        plt.fill_between(yy_fin,xx_fin,0)
        plt.axhline(y=0.95,color='k', linestyle='dotted')
        plt.axhline(y=0.05,color='k', linestyle='dotted')
        plt.axhline(y=0.15,color='k', linestyle='dashed')
        plt.axhline(y=0.85,color='k', linestyle='dashed')
        plt.axhline(y=0.5,color='red', linestyle='dashed')
        plt.axvline(x=HC_M[k],color='red', linestyle='solid')
        plt.xlabel('Tsunami intensity (m)')
        plt.ylabel('Probability of exceedance (%)')
        plt.xscale('log')
        plt.ylim([0,1])
        plt.xlim([0.001,100])
        plt.savefig(fig_path+'%d_scenarios_histo_hazard_p%d.png'%(len(fail),count),bbox_inches='tight')
        count=count+1

    out = 1

    return out


def make_hazard_maps(**kwargs):

    Config                = kwargs.get('cfg', None)
    args                  = kwargs.get('args', None)
    ee                    = kwargs.get('event_parameters', None)
    file_hc               = kwargs.get('file_hc', None)
    file_pois             = kwargs.get('file_pois', None)
    file_tim              = kwargs.get('file_tim', None)
    fig_path              = kwargs.get('fig_path', None)
    file_bathy            = kwargs.get('file_bathy', None)
    fail                  = kwargs.get('fail', None)

    percentiles=np.fromstring(Config.get('Figures','percentiles'), dtype=float, sep=',')

    lat=[]
    lon=[]
    f = open(file_pois, 'r')
    for line in f:
        col = line.split()
        lon.append(float(col[2]))
        lat.append(float(col[3]))
    f.close()
    lon=np.array(lon)
    lat=np.array(lat)
    tglen=len(lon)
    
    TIM=[]
    f = open(file_tim, 'r')
    for line in f:
        TIM.append(float(line))
    f.close
    TIM=np.array(TIM)
    
    with h5py.File(file_hc, "r") as f:
        HC_P = np.array(f['hazard_curves_bs_at_pois'])
        HC_M = np.array(f['hazard_curves_bs_at_pois_mean'])
        IM_O = np.array(f['Intensity_measure_all_bs'])
    
    #####  Calculation of the mean, median and percentiles PTF values #####
    
    ##### Plot of the hazard maps #####
    dataset = netCDF4.Dataset(file_bathy)
    mlon = dataset.variables['x']
    mlat = dataset.variables['y']
    z = dataset.variables['z']
    vmin=0.01   #np.amin(val)
    vmax=100    #np.amax(val)
    
    #####  Map of the mean PTF values #####
    
    val_mean_prec=HC_M
    fig=plt.figure(figsize=(20,15))
    plt.subplots_adjust(left=0.05,right=0.95,top=0.90,bottom=0.05,wspace=0.15,hspace=0.05)
    #m = Basemap(projection='spstere', llcrnrlat=y1,urcrnrlat=y2,llcrnrlon=x1,urcrnrlon=x2,boundinglat=-63,lon_0=180, resolution='l',round=True)
    m = Basemap(llcrnrlon=-10,llcrnrlat=29.0,urcrnrlon=40,urcrnrlat=46.0,resolution='l')
    m.drawmeridians(np.arange(-10,40,5),labels=[False,True,True,False])
    m.drawparallels(np.arange(29,46,5),labels=[True,False,False,True])
    m.bluemarble(alpha=0.42)
    m.drawcoastlines(color='#555566', linewidth=1)
    m.drawmapboundary()
    val=val_mean_prec
    cm1 = plt.cm.get_cmap('RdYlBu_r')
    sc1=plt.scatter(lon,lat,c=val,s=50,cmap=cm1,zorder=3,norm=matplotlib.colors.LogNorm(vmin,vmax))
    cbar=plt.colorbar(sc1,fraction=0.015, pad=0.02)
    cbar.set_label('Mean of the PTF [m]', rotation=270)
    plt.savefig(fig_path+'%d_scenarios_map_mean.png'%(len(fail)),bbox_inches='tight')
    
    #####  Maps of the percentiles PTF values #####
    
    for iper in percentiles:
        per=float(iper)
        percent=1.0-per/100.0
        val_perc=np.zeros(tglen)
    
        for k in range(len(lon)):
    
            HC = HC_P[k,:]
            nonNull_id = np.where(HC>0)
            nonNull = HC[nonNull_id[0]]
            nonNullIm = TIM[nonNull_id[0]]
    
            if not nonNull.any():
                print("list empty")
                toll = 1.e-10
                val_prec[k] = toll
            else:
                xx = np.append(1, nonNull)
                yy = np.append(0, nonNullIm)
                uniq = np.where(np.diff(xx, n=1, axis=-1)!=0)
                xx_fin = np.flip(xx[uniq[0]])
                yy_fin = np.flip(yy[uniq[0]])
                val_perc[k]=np.interp(percent, xx_fin, yy_fin)
    
    
        fig=plt.figure(figsize=(20,15))
        plt.subplots_adjust(left=0.05,right=0.95,top=0.90,bottom=0.05,wspace=0.15,hspace=0.05)
        #m = Basemap(projection='spstere', llcrnrlat=y1,urcrnrlat=y2,llcrnrlon=x1,urcrnrlon=x2,boundinglat=-63,lon_0=180, resolution='l',round=True)
        m = Basemap(llcrnrlon=-10,llcrnrlat=29.0,urcrnrlon=40,urcrnrlat=46.0,resolution='l')
        m.drawmeridians(np.arange(-10,40,5),labels=[False,True,True,False])
        m.drawparallels(np.arange(29,46,5),labels=[True,False,False,True])
        m.bluemarble(alpha=0.42)
        m.drawcoastlines(color='#555566', linewidth=1)
        m.drawmapboundary()
        cm1 = plt.cm.get_cmap('RdYlBu_r')
        sc1=plt.scatter(lon,lat,c=val_perc,s=50,cmap=cm1,zorder=3,norm=matplotlib.colors.LogNorm(vmin,vmax))
        cbar=plt.colorbar(sc1,fraction=0.015, pad=0.02)
        cbar.set_label(str(per)+'th percentile of the PTF [m]', rotation=270)
        plt.savefig(fig_path+'%d_scenarios_map_p%d.png'%(len(fail),int(per)),bbox_inches='tight')
    
    out = 1    

    return out


