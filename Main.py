#!/usr/bin/evn python
# -*- coding:utf-8 -*-

import DataProcessing
import sys
from pylab import mpl
from apscheduler.schedulers.blocking import BlockingScheduler


reload(sys)
sys.setdefaultencoding('utf8')

mpl.rcParams['font.sans-serif'] = ['STSong']  # Set default font
mpl.rcParams['axes.unicode_minus'] = False  # Solving the problem with display symbolic.
mpl.rcParams['axes.linewidth'] = 0.3


def AddJob():
    # Read last time in txt file, and calculate the datetime we need this time.
    TimeInfo = DataProcessing.GetTimeInfo()
    # Unpack files.
    str_tar = "ENVIRONMENT-%s" % (TimeInfo[0])
    print str_tar
    folder_name = "NWP_HAZE2MICAPS%s" % (TimeInfo[0])
    # unpack files
    tar = DataProcessing.TarFile(str_tar, TimeInfo[1])
    if tar:
        DataProcessing.Routing(folder_name, TimeInfo[1])
        DataProcessing.WriteNewTime(TimeInfo[1])


if __name__ == "__main__":
    sched = BlockingScheduler()
    sched.add_job(AddJob, 'interval', seconds=3600)
    sched.start()
    AddJob()
