#!/usr/bin/evn python
# -*- coding:utf-8 -*-
'''@Author: QinXY
   @Date  : 2016-12-20
'''
import PlotOption
import os
import time
import tarfile
import multiprocessing
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from scipy.ndimage.interpolation import map_coordinates


inputDir = "Z:\\world_nwp\\ENVIRONMENT\\CUACE\\"
outputDir = "D:\\Tomcat\\webapps\\EnsoGuangxi\\sys\\mai_plts\\fcst\\country\\"
shpDir = "D:\\FigPlot\\SRCPt\gx\\shp\\"
dataDir = "D:\\FigPlot\\Indata\\MICAPS\\"

# inputDir = "E:\\work\\guangxi\\data\\"
# outputDir = "E:\\work\\guangxi\\data\\"
# shpDir = "D:\\FigPlot\SRCPt\gx\\shp\\"
# dataDir = "E:\\work\\guangxi\\data\\"


def GetTimeInfo():
    # Read last time in txt file, and calculate the datetime we need this time.
    try:
        with open(outputDir + 'tt.txt', 'r+') as f:
            last_time = time.strptime(f.readlines()[0], '%Y%m%d%H')
            tar_time = (datetime(*last_time[:4]) + timedelta(hours=4)).strftime('%Y%m%d%H')
            latest_time = (datetime(*last_time[:4]) + timedelta(hours=12)).strftime('%Y%m%d%H')
    except:
        pass
    return tar_time, latest_time


def WriteNewTime(latest_time):
    try:
        with open(outputDir + 'tt.txt', 'w+') as f:
            f.write(latest_time)
    except:
        pass
            
            
def TarFile(str_tar, latest_time):
    files = os.listdir(inputDir)
    for file in files:
        if str_tar in file:
            t = tarfile.open(inputDir + file)
            t.extractall(path=dataDir)
            t.close()
            print u'解压完成'
            return True


def Routing(fname, lt_time):
    ele_air = ['AQI_4', 'PM25_4', 'PM10_4', 'CO_4', 'SO2_4', 'NO2_4', 'O3_4',
               'pslv_4', 'pwat_4', 'VIS_4', 't2m_4', 'u_v_2', 'u10m_v10m_2', 't_4']
    layer = ['700', '850', '925', '1000']
    if not os.path.exists(os.path.join(outputDir, lt_time)):
        os.mkdir(os.path.join(outputDir, lt_time))
    pp = os.path.join(dataDir, fname)
    files = os.listdir(pp)
    pool = multiprocessing.Pool(processes=4)
    for file in files:
        if file in ele_air:
            fn_CN = os.path.join(outputDir, lt_time, 'CN')
            fn_GX = os.path.join(outputDir, lt_time, 'GX')
            fn_C = os.path.join(fn_CN, file)
            fn_G = os.path.join(fn_GX, file)
            if not os.path.exists(fn_CN):
                os.mkdir(fn_CN)
                os.mkdir(fn_GX)
                os.mkdir(fn_C)
                os.mkdir(fn_G)
            #Running(layer, pp, file)
            pool.apply_async(Running, (layer, pp, file, ))
    pool.close()
    pool.join()


def Running(layer, pp, file):
    if "_4" in file:
        elename = file.replace("_4", '')
    elif "_2"in file:
        elename = file.replace("_2", '')
    print elename

    file_ele = os.listdir(os.path.join(pp, file))
    for f in file_ele:
        if f in layer:
            print f
            fn_p = os.listdir(os.path.join(pp, file, f))
            for fp in fn_p:
                print fp
                fn_f = os.path.join(pp, file, f, fp)
                fname_out = file + '\\' + f
                if file == 'u_v_2':
                    # return fp, fn_f, elename, fname_out, f
                    data = ReadWindData(fp, fn_f)
                    MakeWindPlot(data, elename, fname_out, f, fp)
                else:
                    # return fp, fn_f, elename, pp, f, file, fname_out
                    data = ReadData(fp, fn_f)
                    MakePlot(data, elename, fname_out, f, fp)
        else:
            print f
            fn_f = os.path.join(pp, file, f)
            fname_out = file
            if file == 'u10m_v10m_2':
                # return f, fn_f, elename, fname_out, '1000'
                data = ReadWindData(f, fn_f)
                MakeWindPlot(data, elename, fname_out, '1000', f)
            else:
                # return f, fn_f, elename, pp, '1000', file, fname_out
                data = ReadData(f, fn_f)
                MakePlot(data, elename, fname_out, '1000', f)


def ReadData(*inf):
    rls_x, rls_y = 2, 2
    f, fn_f = inf[0], inf[1]
    year, month, day, hour, prep = f[0:2], f[2:4], f[4:6], f[6:8], f[10:12]
    with open(fn_f, 'r') as info:
        fl = info.readlines()[1].split(" ")
        a = filter(lambda x: x != '', fl)
        stlon, edlon = float(a[8]), float(a[9])
        stlat, edlat = float(a[10]), float(a[11])
        xnum, ynum = int(a[12]), int(a[13])
    xx = np.linspace(stlon, edlon, xnum)
    yy = np.linspace(stlat, edlat, ynum)
    X, Y = np.meshgrid(xx, yy)
    xnew = np.linspace(stlon, 137, xnum*rls_x)
    ynew = np.linspace(stlat, 55, ynum*rls_y)
    XN, YN = np.meshgrid(xnew, ynew)
    df = pd.read_table(fn_f, skiprows=2, header=None, delim_whitespace=True)
    df[df == 9999.00] = np.nan
    data = np.reshape(df.as_matrix(columns=None), (ynum, xnum))
    new_x = np.linspace(0, xnum-1, xnum*rls_x)
    new_y = np.linspace(0, ynum-1, ynum*rls_y)
    coords = np.array(np.meshgrid(new_x, new_y))
    B = map_coordinates(data.T, coords, mode='nearest', prefilter=False)
    return B, XN, YN, year, month, day, hour, prep


def ReadWindData(f, fn_f):
    year, month, day, hour, prep = f[0:2], f[2:4], f[4:6], f[6:8], f[10:12]
    df = pd.read_table(fn_f, skiprows=2, header=None, delim_whitespace=True,
                       names=['lon', 'lat', 'dirc', 'wind'],
                       usecols=[1, 2, 8, 9])
    df[df == 9999] = np.nan
    # B = df.dropna()
    B = df
    return B, year, month, day, hour, prep


def MakePlot(data, *info):
    cmap = PlotOption.CMAP.get(info[0])
    rgb = PlotOption.RGB.get(info[0])
    unit = PlotOption.UNIT.get(info[0])
    levels = PlotOption.LEVELS.get(info[0])
    PlotOption.Plot_GX(cmap, rgb, levels, unit, data, info[0], info[2], info[1], info[3])
    PlotOption.Plot_CN(cmap, rgb, levels, unit, data, info[0], info[2], info[1], info[3])


def MakeWindPlot(data, *info):
    PlotOption.PlotWind_GX(data, info[0], info[2], info[1], info[3])
    PlotOption.PlotWind_CN(data, info[0], info[2], info[1], info[3])
