#import lvm_read
import os
import cv2
import matplotlib
matplotlib.use('tkagg')     # Matplotlib has issues with cv2
import numpy as np
import pandas as pd

from scipy.signal import spectrogram
#from typing import Optional


def extract_lvm(pth, file, get_sr = False, sep="\t", skiprows=22, dropcol="Comment"):
    """
    Extracts the time and voltage data from
    lvm formatted files.
    - Input ---
    pth (str) : relative path to folder containing file
    file (str) : filename of lvm file
    - Returns --
    t (ndarray) : timestamp for samples
    x (ndarray) : measured values
    #TODO:
    #sample_rate (int) : sample rate of data extracted
    """
 
    fullfile = os.path.join(pth, file)

    header = []     # Can be parsed later
    with open(fullfile, "r") as infile:
        for i in range(skiprows):
            #line = infile.readline().split("\t")
            header.append(infile.readline().split())

    df = pd.read_csv(fullfile, sep = sep, skiprows = skiprows)
    df.drop(columns=dropcol, inplace=True)
    df.rename(columns = {"X_Value":"time",
                         "BNC_microphone_DAQ/ai0":"mic1",
                         "BNC_microphone_DAQ/ai1":"mic2",
                         "BNC_microphone_DAQ/ai2":"mic3"},
              inplace=True)

    return df
    
    '''
    base = lvm_read.read(os.path.join(pth, file))[0]
    arr = base['data']
    t = arr[:, 0]
    x = [arr[:, j] for j in range(1, arr.shape[1])]

    if get_sr:
        sample_rate = base['Samples'][0]

        return t, np.array(x), sample_rate

    return t, np.array(x)
    '''


def transform(data, sample_rate = None):
    """
    Transforms time series data into a spectrogram
    """
    if sample_rate:
        freq, time, vals = spectrogram(data, fs = sample_rate)
    else:
        freq, time, vals = spectrogram(data)

    return freq, time, vals


def slugstartinds(data, cutoff, timedelay):
    """
    Finds the index where a slug begins for a given 1D data series
    """
    inds = np.where(data > cutoff)[0]

    sluginds = [inds[0]]

    for i in range(len(inds)-1):
        if np.abs(inds[i] - inds[i+1]) > timedelay:
            sluginds.append(inds[i+1])

    return np.array(sluginds)

def splitslugs(data, cutoff, timedelay, centerratio = 2/3, getsplitpoints = False):
    """
    Split data into slug segments
    """
    #print("In splitslugs")
    sluginds = slugstartinds(data, cutoff, timedelay)
    #print("Sluginds: ", sluginds)
    #print("Len: ", len(sluginds))

    splitpoints = []        # Index which indicate split
    coupledsp = []
    for i in range(len(sluginds) - 1):
        point = int( sluginds[i] +  np.ceil((sluginds[i+1] - sluginds[i]) * centerratio) )
        splitpoints.append(point)

    segments = [data[:splitpoints[0]]]
    coupledsp.append((0, splitpoints[0]-1))

    for i in range(len(splitpoints)-1):
        segment = data[splitpoints[i]:splitpoints[i+1]]
        #print(type(segment))
        #exit(1)
        segments.append(segment)
        coupledsp.append((splitpoints[i], splitpoints[i+1]-1))

    segments.append(data[splitpoints[-1]:])
    coupledsp.append((splitpoints[-1], len(data)-1))

    if getsplitpoints:
        return segments, coupledsp
    else:
        return segments

def splitnoise(noisedata, indlist):
    """
    Split noise into segments of similar length to slug data
    """
    segments = []
    splitpoints = []
    #extrasplitpoints = []

    noiselen = len(noisedata)
    for i in range(len(indlist)):
        start, end = indlist[i]
        if end < noiselen:
            segments.append(noisedata[start:end+1])
            splitpoints.append((start, end))
        else:
            #print(f"Index overstepped, {end} > {noiselen}")
            newstart = end % noiselen
            delta = end - start
            newend = newstart + delta

            segments.append(noisedata[newstart:newend+1])
            splitpoints.append((newstart, newend))
            #print(f"({start}, {end}) -> ({newstart}, {newend})")


        '''
        try:
        except IndexError as e:
            print("Index fuckup")
            """
            j = np.random.randint(0, indlist[-2])
            end = j + int(indlist[i+1]-indlist[i])
            seg = noisedata[j : j + end]
            segments.append(seg)
            extrasplitpoints.append(tuple(j, end))
            break
            """
            ii = indlist[i+1]%len(noisedata)
            del_ind = indlist[i+1] - indlist[i]
            segments.append(noisedata[ii:ii+del_ind])
            splitpoints.append((ii, ii+del_ind - 1))
        '''
    return segments, splitpoints

def verticalsnapshots(pth, mp4file, start_s, stop_s, measureinds):
    start_ms = int(start_s*1000)
    stop_ms = int(stop_s*1000)

    filename = os.path.join(pth, mp4file)
    cap = cv2.VideoCapture(filename)

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    s = stop_s - start_s

    #print(f"fps: {fps}, w: {fps*s}")
    if not isinstance(measureinds, list):
        measureinds = list(measureinds)
    
    arr = np.zeros((len(measureinds), h, int(np.ceil(fps*s)+2)))

    # Set start close to start sec, potential massive speedup
    cap.set(cv2.CAP_PROP_POS_MSEC, int(start_ms - 100))

    ret = True
    while ret and cap.get(cv2.CAP_PROP_POS_MSEC) < start_ms:
        #print(f"Time [ms]: {cap.get(cv2.CAP_PROP_POS_MSEC)}")
        ret, img = cap.read()

    i = 0
    while ret and cap.get(cv2.CAP_PROP_POS_MSEC) <= stop_ms:
        #print(f"Time [ms]: {cap.get(cv2.CAP_PROP_POS_MSEC)}")
        ret, BGRimg = cap.read()
        grayimg = cv2.cvtColor(BGRimg, cv2.COLOR_BGR2GRAY)
        for nr in range(len(measureinds)):
            arr[nr, :, i] = grayimg[:, measureinds[nr]]
        #print(f"{grayimg[:, 1700].shape}")
        #print(f"gray.shape: {grayimg.shape}\tgray.type: {type(grayimg)}")
        i += 1

    cap.release()
    return arr
