#!/usr/bin/env python
# -*- coding:utf-8 -*-
''' @Author: QinXY
    @Date  : 2016-12-20
'''
import math
import os

import matplotlib.colors as mpc
import matplotlib.patches as mpatches
import matplotlib.pylab as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from mpl_toolkits.basemap import Basemap
from scipy.ndimage.filters import maximum_filter, minimum_filter

import shapefile
import southchinasea

shpDir = "D:\\FigPlot\\SRCPt\\gx\\shp\\"
# shpDir = "E:\\work\\shp\\guangxi\\"
# outputDir = "E:\\work\\guangxi\\data\\"
outputDir = "E:\\work\\guangxi\\data\\"
city = [u'南宁', u'玉林', u'防城港', u'钦州', u'北海', u'贵港', u'贺州',
        u'百色', u'来宾', u'崇左', u'柳州', u'河池', u'桂林', u'梧州']
cityId = [59431, 59453, 59635, 59632, 59644, 59249, 59065,
          59211, 59242, 59425, 59046, 59023, 57957, 59265]
LON = [108.22, 110.12, 108.35, 108.62, 109.13, 109.62, 111.50,
       106.60, 109.23, 107.35, 109.40, 108.03, 110.30, 111.30]
LAT = [22.63, 22.67, 21.62, 21.95, 21.45, 23.12, 24.42,
       23.90, 23.75, 22.40, 24.35, 24.70, 25.32, 23.48]

AQI_RGB = np.array([[0, 228, 0], [255, 255, 0], [255, 126, 0], [255, 0, 0], [153, 0, 76],
                    [126, 0, 35], [126, 0, 35]])
pwat_RGB = np.array([[255, 255, 255], [165, 243, 141], [153, 210, 202], [155, 188, 232],
                     [107, 157, 225], [59, 126, 219], [17, 44, 144], [70, 25, 129]])
t_RGB = np.array([[116, 163, 226], [155, 188, 232], [152, 214, 196], [234, 219, 112],
                  [250, 204, 79], [242, 155, 0], [239, 117, 17], [231, 75, 26],
                  [217, 51, 18], [181, 1, 9], [111, 0, 21]])
vis_RGB = np.array([[158, 98, 38], [200, 17, 169], [238, 44, 44], [239, 117, 17],
                    [238, 173, 14], [255, 255, 0], [124, 252, 0], [153, 205, 208],
                    [135, 175, 229], [255, 255, 255]])
pslv_RGB = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

# Set color cmap
CMAP = {'t': mpc.ListedColormap(t_RGB / 255.0),
        'O3': mpc.ListedColormap(AQI_RGB / 255.0),
        'CO': mpc.ListedColormap(AQI_RGB / 255.0),
        'NO2': mpc.ListedColormap(AQI_RGB / 255.0),
        'SO2': mpc.ListedColormap(AQI_RGB / 255.0),
        'AQI': mpc.ListedColormap(AQI_RGB / 255.0),
        't2m': mpc.ListedColormap(t_RGB / 255.0),
        'VIS': mpc.ListedColormap(vis_RGB / 255.0),
        'PM10': mpc.ListedColormap(AQI_RGB / 255.0),
        'PM25': mpc.ListedColormap(AQI_RGB / 255.0),
        'pwat': mpc.ListedColormap(pwat_RGB / 255.0),
        'pslv': mpc.ListedColormap(pslv_RGB / 255.0)
        }

# Set RGB color
RGB = {'t': t_RGB,
       'O3': AQI_RGB,
       'CO': AQI_RGB,
       'NO2': AQI_RGB,
       'SO2': AQI_RGB,
       'AQI': AQI_RGB,
       't2m': t_RGB,
       'VIS': vis_RGB,
       'PM10': AQI_RGB,
       'PM25': AQI_RGB,
       'pwat': pwat_RGB,
       'pslv': pslv_RGB
       }

# Set unit
UNIT = {'t': r'$^\circ$C',
        'O3': r'$ug/m^3$',
        'CO': r'$mg/m^3$',
        'AQI': '',
        'VIS': r'$km$',
        'NO2': r'$ug/m^3$',
        'SO2': r'$ug/m^3$',
        't2m': r'$^\circ$C',
        'PM10': r'$ug/m^3$',
        'PM25': r'$ug/m^3$',
        'pwat': r'$mm$'}

# Set levels
LEVELS = {'PM25': [0 - 0.01, 35, 75, 115, 150, 250, 350, 9999],
          'PM10': [0 - 0.01, 50, 150, 250, 350, 420, 500, 9999],
          'SO2': [0 - 0.01, 150, 500, 650, 800, 1600, 2100, 99999],
          'NO2': [0 - 0.01, 100, 200, 700, 1200, 2340, 3090, 99999],
          'O3': [0 - 0.01, 160, 200, 300, 400, 800, 1000, 99999],
          'CO': [0 - 0.01, 5, 10, 35, 60, 90, 120, 99999],
          'AQI': [0 - 0.01, 50, 100, 150, 200, 300, 400, 9999],
          'pwat': [0 - 0.01, 0.1, 10, 25, 50, 100, 200, 400, 9999],
          't': [-999, -10, -5, 0, 10, 15, 20, 25, 30, 35, 40, 99999],
          't2m': [-999, -15, -10, -5, 0, 10, 15, 20, 25, 30, 35, 40],
          'VIS': [0 - 0.01, 0.2, 0.5, 1, 2, 3, 5, 10, 20, 30, 9999],
          'pslv': [0, 10, 9999]
          }


def Plot_CN(cmap, rgb, levels, unit, data, ename, lay, fdname, flname):
    M_CN = Basemap(projection='cyl', resolution='c', llcrnrlat=15,
                   urcrnrlat=55, llcrnrlon=72, urcrnrlon=137)
    fig = plt.figure(figsize=(3.2, 2.0), dpi=300, facecolor='w')
    ax = fig.add_subplot(111, frame_on=False, axisbg='w')
    plt.subplots_adjust(left=0, bottom=0., right=1, top=1, wspace=0.01, hspace=0.01)
    M_CN.readshapefile(shpDir + 'bou2_4p', name='whatever', drawbounds=True,
                       linewidth=0.1, color='black')
    NORM = mpc.BoundaryNorm(levels, cmap.N)
    clevs = np.arange(900, 1100., 2.5)
    if ename == 'CO':
        cs = plt.contourf(data[1], data[2], data[0] / 1000, cmap=cmap, levels=levels, norm=NORM)
    elif ename == 'pslv':
        cs = plt.contour(data[1], data[2], data[0], clevs, colors='k', linewidths=0.3)
        x, y = M_CN(data[1], data[2])
        local_min, local_max = extrema(data, mode='wrap', window=50)
        xlows = x[local_min]
        ylows = y[local_min]
        xhighs = x[local_max]
        yhighs = y[local_max]
        lowvals = data[local_min]
        highvals = data[local_max]
        xyplotted = []
        # don't plot if there is already a L or H within dmin meters.
        yoffset = 0.022 * (M_CN.ymax - M_CN.ymin)
        dmin = yoffset
        for x, y, p in zip(xlows, ylows, lowvals):
            if x < M_CN.xmax and x > M_CN.xmin and y < M_CN.ymax and y > M_CN.ymin:
                dist = [np.sqrt((x - x0)**2 + (y - y0)**2) for x0, y0 in xyplotted]
                if not dist or min(dist) > dmin:
                    plt.text(x, y, 'L', fontsize=6, fontweight='bold',
                             ha='center', va='center', color='r')
                    plt.text(x, y - yoffset, repr(int(p)), fontsize=3,
                             ha='center', va='top', color='r',
                             bbox=dict(boxstyle="square", ec='None', fc=(1, 1, 1, 0.5)))
                    xyplotted.append((x, y))
        xyplotted = []
        for x, y, p in zip(xhighs, yhighs, highvals):
            if x < M_CN.xmax and x > M_CN.xmin and y < M_CN.ymax and y > M_CN.ymin:
                dist = [np.sqrt((x - x0)**2 + (y - y0)**2) for x0, y0 in xyplotted]
                if not dist or min(dist) > dmin:
                    plt.text(x, y, 'H', fontsize=6, fontweight='bold',
                             ha='center', va='center', color='b')
                    plt.text(x, y - yoffset, repr(int(p)), fontsize=3,
                             ha='center', va='top', color='b',
                             bbox=dict(boxstyle="square", ec='None', fc=(1, 1, 1, 0.5)))
                    xyplotted.append((x, y))

    elif ename == 't' or ename == 't2m':
        cs = plt.contourf(data[1], data[2], data[0] - 273.15, cmap=cmap, levels=levels, norm=NORM)
    else:
        cs = ax.contourf(data[1], data[2], data[0], cmap=cmap, levels=levels, norm=NORM)

    if ename == 'NO2':
        plt.text(92, 51, '全国$NO_2$未来%s小时预报' % data[7], fontsize=8, alpha=1)
    elif ename == 'SO2':
        plt.text(92, 51, '全国$SO_2$未来%s小时预报' % data[7], fontsize=8, alpha=1)
    elif ename == 'O3':
        plt.text(92, 51, '全国$O_3$未来%s小时预报' % data[7], fontsize=8, alpha=1)
    elif ename == 'pwat':
        plt.text(92, 51, '全国地面累计降水量未来%s小时预报' % data[7], fontsize=8, alpha=1)
    elif ename == 'pslv':
        plt.text(92, 51, '全国海平面气压未来%s小时预报' % data[7], fontsize=8, alpha=1)
    elif ename == 't':
        plt.text(92, 51, '全国%shPa温度未来%s小时预报' % (lay, data[7]), fontsize=8, alpha=1)
    elif ename == 't2m':
        plt.text(92, 51, '全国温度未来%s小时预报' % data[7], fontsize=8, alpha=1)
    elif ename == 'VIS':
        plt.text(92, 51, '全国能见度未来%s小时预报' % data[7], fontsize=8, alpha=1)
    elif ename == 'pslv':
        plt.text(92, 51, '全国海平面气压未来%s小时预报' % data[7], fontsize=8, alpha=1)
    else:
        plt.text(92, 51, '全国$%s$未来%s小时预报' % (ename, data[7]), fontsize=8, alpha=1)
    plt.text(96, 48.5, '起报时间：$20%s$年$%s$月$%s$日$%s$时' % (data[3], data[4], data[5], data[6]),
             fontsize=4, alpha=1)
    SouthChinaSea(ax)
    if ename == 'pslv':
        pass
    else:
        Maskout(cs, ax, shpDir + 'country1', 'China', 3)
        Legend_CN(levels, rgb, ax, unit)
    # plt.show()
    date_fcs = "20%s%s%s%s" % (data[3], data[4], data[5], data[6])
    if not os.path.exists(os.path.join(outputDir, date_fcs, 'CN', '%s') % fdname):
        os.mkdirs(os.path.join(outputDir, date_fcs, 'CN', '%s') % fdname)
    pth = os.path.join(outputDir, date_fcs, 'CN', '%s', '%s') % (fdname, flname)
    plt.savefig(pth + '.png', dpi=300)
    plt.close(fig)


def Plot_GX(cmap, rgb, levels, unit, data, ename, lay, fdname, flname):
    M_GX = Basemap(projection='cyl', resolution='c', llcrnrlat=20.6,
                   urcrnrlat=26.6, llcrnrlon=104, urcrnrlon=112.3)
    fig = plt.figure(figsize=(3.2, 2.0), dpi=300, facecolor='w')
    ax = fig.add_subplot(111, frame_on=False, axisbg='w')
    plt.subplots_adjust(left=0, bottom=0., right=1, top=1, wspace=0.01, hspace=0.01)
    M_GX.readshapefile(shpDir + 'guangxi', name='whatever', drawbounds=True,
                       linewidth=0.1, color='black')
    NORM = mpc.BoundaryNorm(levels, cmap.N)
    clevs = np.arange(900, 1100., 2.5)
    if ename == 'CO':
        cs = plt.contourf(data[1], data[2], data[0] / 1000, cmap=cmap, levels=levels, norm=NORM)
    elif ename == 'pslv':
        cs = plt.contour(data[1], data[2], data[0], clevs, colors='k', linewidths=0.4)
    elif ename == 't' or ename == 't2m':
        cs = plt.contourf(data[1], data[2], data[0] - 273.15, cmap=cmap, levels=levels, norm=NORM)
    else:
        cs = ax.contourf(data[1], data[2], data[0], cmap=cmap, levels=levels, norm=NORM)

    plt.scatter(LON, LAT, c='black', marker='o', s=7, linewidths=0.1, alpha=1)
    for i in range(len(city)):
        x, y, name = [LON[i], LAT[i], city[i]]
        plt.text(x - 0.15, y - 0.235, name, alpha=1,
                 fontdict={'color': 'black', 'weight': 'medium', 'size': 4})

    if ename == 'NO2':
        plt.text(104.3, 26.3, '广西$NO_2$未来%s小时预报' % data[7], fontsize=8, alpha=1)
    elif ename == 'SO2':
        plt.text(104.3, 26.3, '广西$SO_2$未来%s小时预报' % data[7], fontsize=8, alpha=1)
    elif ename == 'O3':
        plt.text(104.3, 26.3, '广西$O_3$未来%s小时预报' % data[7], fontsize=8, alpha=1)
    elif ename == 'pwat':
        plt.text(104.3, 26.3, '广西地面累计降水量未来%s小时预报' % data[7], fontsize=8, alpha=1)
    elif ename == 'pslv':
        plt.text(104.3, 26.3, '广西海平面气压未来%s小时预报' % data[7], fontsize=8, alpha=1)
    elif ename == 't':
        plt.text(104.3, 26.3, '广西%shPa温度未来%s小时预报' % (lay, data[7]), fontsize=8, alpha=1)
    elif ename == 't2m':
        plt.text(104.3, 26.3, '广西温度未来%s小时预报' % data[7], fontsize=8, alpha=1)
    elif ename == 'VIS':
        plt.text(104.3, 26.3, '广西能见度未来%s小时预报' % data[7], fontsize=8, alpha=1)
    elif ename == 'pslv':
        plt.text(104.3, 26.3, '广西海平面气压未来%s小时预报' % data[7], fontsize=8, alpha=1)
    else:
        plt.text(104.3, 26.3, '广西$%s$未来%s小时预报' % (ename, data[7]), fontsize=8, alpha=1)
    plt.text(105.6, 26, '起报时间：$20%s$年$%s$月$%s$日$%s$时' % (data[3], data[4], data[5], data[6]),
             fontsize=4, alpha=1)
    if ename == 'pslv':
        pass
    else:
        Maskout(cs, ax, shpDir + 'bou2_4p', [450000], 4)
        Legend_GX(levels, rgb, ax, unit)
    # plt.show()
    date_fcs = "20%s%s%s%s" % (data[3], data[4], data[5], data[6])
    if not os.path.exists(os.path.join(outputDir, date_fcs, 'GX', '%s') % fdname):
        os.mkdirs(os.path.join(outputDir, date_fcs, 'GX', '%s') % fdname)
    pth = os.path.join(outputDir, date_fcs, 'GX', '%s', '%s') % (fdname, flname)
    plt.savefig(pth + '.png', dpi=300)
    plt.close(fig)


def PlotWind_CN(data, ename, lay, fdname, flname):
    M_CN = Basemap(projection='cyl', resolution='c', llcrnrlat=15,
                   urcrnrlat=55, llcrnrlon=72, urcrnrlon=137)
    lon, lat = data[0]['lon'], data[0]['lat']
    spd = data[0]['wind']
    dirc = data[0]['dirc']
    lonn = np.reshape(lon, (260, 430))
    latt = np.reshape(lat, (260, 430))
    spdd = np.reshape(spd, (260, 430))
    dirr = np.reshape(dirc, (260, 430))
    lonx, latx = lonn[::10, ::10], latt[::10, ::10]
    spdx, dirx = spdd[::10, ::10], dirr[::10, ::10]
    lonx = np.reshape(lonx, (43 * 26, 1))
    latx = np.reshape(latx, (43 * 26, 1))
    spdx = np.reshape(spdx, (43 * 26, 1))
    dirx = np.reshape(dirx, (43 * 26, 1))
    PI = math.pi
    llon = lonx
    llat = latx
    spdd = spdx
    dirr = dirx
    tmp = (270.0 - dirr) * PI / 180.0
    u, v = [], []
    for i in range(len(tmp)):
        ux = spdd[i] * math.cos(tmp[i])
        vx = spdd[i] * math.sin(tmp[i])
        u.append(ux)
        v.append(vx)
    fig = plt.figure(figsize=(3.2, 2.0), dpi=300, facecolor='w')

    fig.add_subplot(111, frame_on=False, axisbg='w')
    plt.subplots_adjust(left=0, bottom=0., right=1, top=1, wspace=0.01, hspace=0.01)
    M_CN.readshapefile(shpDir + 'bou2_4p', name='whatever', drawbounds=True, linewidth=0.1, color='black')
    Q = M_CN.quiver(llon, llat, u, v, scale=10, units='x', pivot='tip', width=0.05, headlength=2)
    date_fcs = "20%s%s%s%s" % (data[1], data[2], data[3], data[4])
    if ename == 'u_v':
        plt.text(92, 51, u'全国%shPa风场未来%s小时预报' % (lay, data[5]), fontsize=8, alpha=1)
        if not os.path.exists(os.path.join(outputDir, date_fcs, 'CN', 'u_v_2')):
            os.mkdir(os.path.join(outputDir, date_fcs, 'CN', 'u_v_2'))
    elif ename == 'u10m_v10m':
        plt.text(92, 51, u'全国10米高风场未来%s小时预报' % data[5], fontsize=8, alpha=1)
    plt.text(96, 48.5, u'起报时间：$20%s$年$%s$月$%s$日$%s$时' % (data[1], data[2], data[3], data[4]), fontsize=4, alpha=1)
    plt.quiverkey(Q, 0.02, 0.08, 20, '        20 m/s', labelpos='S',
                  fontproperties={'size': 4}, color='black', labelcolor='black')
    # plt.show()
    if not os.path.exists(os.path.join(outputDir, date_fcs, 'CN', '%s') % fdname):
        os.mkdir(os.path.join(outputDir, date_fcs, 'CN', '%s') % fdname)
    pth = os.path.join(outputDir, date_fcs, 'CN', '%s', '%s') % (fdname, flname)
    plt.savefig(pth + '.png', dpi=300)
    plt.close(fig)


def PlotWind_GX(data, ename, lay, fdname, flname):
    M_GX = Basemap(projection='cyl', resolution='c', llcrnrlat=20.6,
                   urcrnrlat=26.6, llcrnrlon=104, urcrnrlon=112.3)
    lon, lat = data[0]['lon'], data[0]['lat']
    spd = data[0]['wind']
    dir = data[0]['dirc']
    PI = math.pi
    llon = lon.as_matrix(columns=None)
    llat = lat.as_matrix(columns=None)
    spdd = spd.as_matrix(columns=None)
    dirr = dir.as_matrix(columns=None)
    tmp = (270.0 - dirr) * PI / 180.0
    u, v = [], []
    for i in range(len(tmp)):
        ux = spdd[i] * math.cos(tmp[i])
        vx = spdd[i] * math.sin(tmp[i])
        u.append(ux)
        v.append(vx)

    fig = plt.figure(figsize=(3.2, 2.0), dpi=300, facecolor='w')

    fig.add_subplot(111, frame_on=False, axisbg='w')
    plt.subplots_adjust(left=0, bottom=0., right=1, top=1, wspace=0.01, hspace=0.01)
    M_GX.readshapefile(shpDir + 'guangxi', name='whatever', drawbounds=True, linewidth=0.1, color='black')
    M_GX.quiver(llon, llat, u, v, scale=17, units='x', pivot='tip', width=0.012, headlength=5)
    date_fcs = "20%s%s%s%s" % (data[1], data[2], data[3], data[4])
    if ename == 'u_v':
        plt.text(104.3, 26.3, u'广西%shPa风场未来%s小时预报' % (lay, data[5]), fontsize=8, alpha=1)
        if not os.path.exists(os.path.join(outputDir, date_fcs, 'CN', 'u_v_2')):
            os.mkdir(os.path.join(outputDir, date_fcs, 'CN', 'u_v_2'))
    elif ename == 'u10m_v10m':
        plt.text(104.3, 26.3, u'广西10米高风场未来%s小时预报' % data[5], fontsize=8, alpha=1)
    plt.text(105.6, 26, u'起报时间：$20%s$年$%s$月$%s$日$%s$时' % (data[1], data[2], data[3], data[4]), fontsize=4, alpha=1)
    # plt.show()
    date_fcs = "20%s%s%s%s" % (data[1], data[2], data[3], data[4])
    if not os.path.exists(os.path.join(outputDir, date_fcs, 'GX', '%s') % fdname):
        os.mkdirs(os.path.join(outputDir, date_fcs, 'GX', '%s') % fdname)
    pth = os.path.join(outputDir, date_fcs, 'GX', '%s', '%s') % (fdname, flname)
    plt.savefig(pth + '.png', dpi=300)
    plt.close(fig)


# Add South China Sea Map
def SouthChinaSea(ax):
    for i in southchinasea.southchinasea:
        x, y = [[k[j] for k in i] for j in (0, 1)]
        ax.plot(x, y, 'k-', linewidth=0.2)
        bx = plt.axes([0.83, 0.06, .1, .2], axisbg='w', frameon=True)
        bx.plot(x, y, 'k-', linewidth=0.2)
        plt.xticks([])
        plt.yticks([])


# Do mask
def Maskout(originfig, ax, shpfile, region, m):
    sf = shapefile.Reader(shpfile)
    vertices = []
    codes = []
    for shape_rec in sf.shapeRecords():
        if shape_rec.record[m] in region:
            pts = shape_rec.shape.points
            prt = list(shape_rec.shape.parts) + [len(pts)]
            for i in range(len(prt) - 1):
                for j in range(prt[i], prt[i + 1]):
                    vertices.append((pts[j][0], pts[j][1]))
                codes += [Path.MOVETO]
                codes += [Path.LINETO] * (prt[i + 1] - prt[i] - 2)
                codes += [Path.CLOSEPOLY]
            clip = Path(vertices, codes)
            clip = PathPatch(clip, transform=ax.transData)
    for contour in originfig.collections:
        contour.set_clip_path(clip)
    return clip


# Add legend
def Legend_CN(levels, rgb, ax, unit):
    for i in range(len(levels) - 1):
        patch = mpatches.FancyBboxPatch((73, 16 + 1.6 * i), 1.2, 0.7, boxstyle='round, pad=0.24',
                                        fc=rgb[i] / 255.0, ec='black', lw=0.03)
        if i == 0:
            ax.text(75, 16 + 1.6 * i, r'$< %s$  %s' % (levels[i + 1], unit), fontsize=3)
        elif i == len(levels) - 2:
            ax.text(75, 16 + 1.6 * i, r'$> %s$  %s' % (levels[i], unit), fontsize=3)
        else:
            ax.text(75, 16 + 1.6 * i, r'$%s - %s$  %s' % (levels[i], levels[i + 1], unit), fontsize=3)
        ax.add_patch(patch)


def Legend_GX(levels, rgb, ax, unit):
    for i in range(len(levels) - 1):
        patch = mpatches.FancyBboxPatch((104.32, 20.90 + 0.25 * i), 0.14, 0.07, boxstyle='round, pad=0.04',
                                        fc=rgb[i] / 255.0, ec='black', lw=0.03)
        if i == 0:
            ax.text(104.67, 20.90 + 0.25 * i, r'$< %s$  %s' % (levels[i + 1], unit), fontsize=3)
        elif i == len(levels) - 2:
            ax.text(104.67, 20.90 + 0.25 * i, r'$> %s$  %s' % (levels[i], unit), fontsize=3)
        else:
            ax.text(104.67, 20.90 + 0.25 * i, r'$%s - %s$  %s' % (levels[i], levels[i + 1], unit), fontsize=3)
        ax.add_patch(patch)


def extrema(mat, mode='wrap', window=10):
    """find the indices of local extrema (min and max)
    in the input array."""
    mn = minimum_filter(mat, size=window, mode=mode)
    mx = maximum_filter(mat, size=window, mode=mode)
    # (mat == mx) true if pixel is equal to the local max
    # (mat == mn) true if pixel is equal to the local in
    # Return the indices of the maxima, minima
    return np.nonzero(mat == mn), np.nonzero(mat == mx)
