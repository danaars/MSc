#import lvm_read
import os
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
    sample_rate (int) : sample rate of data extracted
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

    sluginds = []

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


if __name__ == "__main__":
    import numpy as np
    import glob
    import matplotlib.pyplot as plt
    import re
    import torch
    import sys
    
    if sys.argv[1] == "save":
        pth = "0811"
        files = glob.glob(os.path.join(pth, "micData_*.lvm"))
        #files = glob.glob(pth, "micData_*.lvm")
        #files = os.listdir("0811")
        files.sort()

        outdir = "0811_spectro_scaled"
        halfmin = int(30*8000)
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        shapes = []
        for file in files:
            pth, file = os.path.split(file)
            t, data = extract_lvm(pth, file)

            tens = []
            for i in range(len(data)):
                print(len(data[i][:halfmin]))
                freq, time, vals = transform(data[i][:halfmin], sample_rate=8000)
                tens.append(vals)

            tens = np.asarray(tens)
            tens = torch.from_numpy(tens)
            shapes.append(tens.shape)

            nr = re.search(r"(\d+)", file).group(1)
            filename = f"spec{nr}.pt"       # .pt for pytorch tensor extention

            #np.savetxt(os.path.join(outdir, filename), tens)
            torch.save(tens, os.path.join(outdir, filename))

    else:
        pth = "0811_spectro_scaled"
        for i in range(28, 46):
            filename = f"spec{i}.pt"

            out = torch.load(os.path.join(pth, filename))
            print(out.shape)


        '''
        pth = "../18.10/Labdata/"
        file = "micData_25.lvm"
        file2 = "micData_26.lvm"

        t, x, sr = extract_lvm(pth, file, True)
        t2, x2, sr = extract_lvm(pth, file2, True)

        plt.plot(t, x)
        plt.title("Slug 'lyd'")
        plt.xlabel("Tid")
        plt.ylabel("Spenning (Volt)")
        plt.savefig("18_10_slug_lyd.png")
        plt.show()


        plt.plot(t2, x2)
        plt.title("Ingen slug 'lyd'")
        plt.xlabel("Tid")
        plt.ylabel("Spenning (Volt)")
        plt.savefig("18_10_noslug_lyd.png")
        plt.show()

        n = int(0.2 * len(x))
        section = x[:n]

        f, t, spec = transform(section)

        plt.pcolormesh(t, f, spec)
        plt.title("Spectrogram (første 20% av 5 min)")
        plt.xlabel("Tid")
        plt.ylabel("Frekvens (?)")
        plt.savefig("18_10_slug_spec.png")
        plt.show()

        section = x2[:n]

        f2, t2, spec2 = transform(section)

        plt.pcolormesh(t2, f2, spec2)
        plt.title("Spectrogram (første 20% av 5 min)")
        plt.xlabel("Tid")
        plt.ylabel("Frekvens (?)")
        plt.savefig("18_10_noslug_spec.png")
        plt.show()

        slug_freq_vals = np.fft.fft(x)
        slug_freqs = np.fft.fftfreq(len(x))

        plt.scatter(slug_freqs, slug_freq_vals.real, alpha = 0.5, marker = '.', label = "Real")
        plt.scatter(slug_freqs, slug_freq_vals.imag, alpha = 0.5, marker = '.', label = "Imag")
        plt.title("FFT av slug lyd")
        plt.legend()
        plt.savefig("18_10_slug_FFT.png")
        plt.show()

        slug_freq_vals2 = np.fft.fft(x2)
        slug_freqs2 = np.fft.fftfreq(len(x2))

        plt.scatter(slug_freqs2, slug_freq_vals2.real, alpha = 0.5, marker = '.', label = "Real")
        plt.scatter(slug_freqs2, slug_freq_vals2.imag, alpha = 0.5, marker = '.', label = "Imag")
        plt.title("FFT av uten slug lyd")
        plt.savefig("18_10_noslug_FFT.png")
        plt.legend()
        plt.show()
        '''
