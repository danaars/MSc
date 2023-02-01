import os
import re
import sys
import numpy as np
import pandas as pd

from utils import *

def save(segments, splitpoints, meta_list_dict, savefolder, slug, runnr, channel, sample_rate):
    for i, seg in enumerate(segments):
        if slug:
            filename = f"s{i+1}_r{runnr}_ch{channel}.txt"
            slug_label = 1
        else:
            filename = f"n{i+1}_r{runnr}_ch{channel}.txt"
            slug_label = 0

        slug_velocity = None
        slug_length = None
        
        #print(f"slug: {filename}, startind: {splitpoints[i]}, endind: {splitpoints[i+1]}") 
        freq, time, vals = transform(seg, sample_rate)

        '''
        try:
            height, width = vals.shape
        except ValueError as e:
            print(f"Iter: {i}")
            print(f"Seg:{seg}\tlen: {len(seg)}")
            print("Vals:")
            print(vals)
            raise e
        '''
        height, width = vals.shape
        
        #width = 
        #print(f"Height: {height}, Width: {width}")
        startind, endind = splitpoints[i]
        #print(f"{startind}\t{endind}")
        #endind = splitpoints[i+1]-1
        starttime = startind / sample_rate
        endtime = endind / sample_rate
        time = len(seg) / sample_rate
        multichannel = False

        segdict = {
                "filename": filename,
                "slug_label": slug_label,
                "slug_velocity": slug_velocity,
                "slug_length": slug_length,
                "width": width,
                "height": height,
                "timelen": time,
                "sample_rate": sample_rate,
                "startindex": startind,
                "endindex": endind,
                "starttime": starttime,
                "endtime": endtime,
                "source": runnr,
                "channel": channel,
                "multichannel": False
            }

        meta_list_dict.append(segdict)
        saveloc = os.path.join(savefolder, filename)
        np.savetxt(saveloc, vals)
        #if i > 3:
            #break

keys = [
        "filename",
        "slug_label",
        "slug_velocity",
        "slug_length",
        "width",        # Time len
        "height",       # Freq len
        "timelen",
        "sample_rate",
        "startindex",
        "endindex",
        "starttime",
        "endtime",
        "source",
        "channel",
        "multichannel"
        ]

#df = pd.DataFrame()
abspath = "/run/media/daniel/ATLEJ-STT/data"

outdir = "test"
if not os.path.exists(os.path.join(abspath, outdir)):
    os.makedirs(os.path.join(abspath, outdir))

meta_list_dict = []
all_splitpoints = []
savefolder = os.path.join(abspath, outdir)

#nn = 60*8000

# Split slug files
for slugfile in ["micData_49.lvm", "micData_51.lvm", "micData_53.lvm"]:
    #for slugfile in ["micData_49.lvm"]:
    print(f"Reading file: {slugfile}")
    df = extract_lvm(abspath, slugfile, get_sr = False)
    sample_rate = 8000

    for channel in ["mic1", "mic2", "mic3"]:
        #for channel in ["mic1"]:
        print(f"Treating {channel}") 
        #channel = 0
        #data = df[channel].to_numpy()[:nn]
        data = df[channel].to_numpy()
        cutoff = 3*np.std(data)
        timedelay = 3*sample_rate

        segments, splitpoints = splitslugs(data, cutoff, timedelay, centerratio=2/3, getsplitpoints=True)
        #print(splitpoints)
        #print(type(splitpoints[0]))

        #splitpoints = [0] + splitpoints + [len(data)]
        all_splitpoints.append(splitpoints)
        runnr = int(re.search("\d+", slugfile)[0])

        save(segments, splitpoints, meta_list_dict, savefolder, slug=True, runnr=runnr, channel=int(channel[-1]), sample_rate=sample_rate)


# Split noise files
i = 0
for noisefile in ["micData_54.lvm", "micData_55.lvm", "micData_58.lvm"]:
    #for noisefile in ["micData_54.lvm"]:
    print(f"Reading file: {noisefile}")
    df = extract_lvm(abspath, noisefile, get_sr = False)
    sample_rate = 8000

    for channel in ["mic1", "mic2", "mic3"]:
        #for channel in ["mic1"]:
        print(f"Treating {channel}") 
        #channel = 0
        #data = df[channel].to_numpy()[:nn]
        data = df[channel].to_numpy()
        cutoff = 3*np.std(data)
        timedelay = 3*sample_rate

        indlist = all_splitpoints[i]
        segments, splitpoints = splitnoise(data, indlist)

        runnr = int(re.search("\d+", noisefile)[0])

        #print(segments)
        #print(indlist)
        #print(len(indlist))
        #print(len(data))
        save(segments, splitpoints, meta_list_dict, savefolder, slug=False, runnr=runnr, channel=int(channel[-1]), sample_rate=sample_rate)

        i = (i+1)%(len(all_splitpoints))

df = pd.DataFrame(meta_list_dict)
csvname = "meta.csv"

if sys.argv[1] == "save":
    if not os.path.exists(os.path.join(savefolder, csvname)):
        df.to_csv(os.path.join(savefolder, csvname), mode = "a", index=False)
    else:
        df.to_csv(os.path.join(savefolder, csvname), mode = "a", index=False, header=False)
else:
    print(df)

print("Exiting successfully")
