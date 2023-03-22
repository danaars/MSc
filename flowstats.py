import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from utils import *

nr = 49
threshold = 220     # 0 - 255
datapath = "/run/media/daniel/ATLEJ-STT/data"
mp4path = "/run/media/daniel/ATLEJ-STT/mp4"
for nr in [49, 51, 53, 59, 60]:
    fil = f"micData_{nr}.lvm"
    mp4 = f"run{nr}.mp4"

    df = extract_lvm(datapath, fil)
    data = df["mic2"].to_numpy()

    segments, coupledidx = splitslugs(data, 
                                      3*np.std(data), # Threshold for interest
                                      5*8000,
                                      getsplitpoints=True)
    t = df["time"].to_numpy()

    kl = 5
    runavg = np.ones(kl)/kl
    #if not os.path.exists(f"{nr}stats"):
        #os.makedirs(f"{nr}stats")

    slices = [i for i in range(60, 1850, 50)]
    # Remove potential zip measurements
    for col in slices:
        if 30 <= col <= 45:
            slices.remove(col)
        if 1045 <= col <= 1060:
            slices.remove(col)

    slugstats = []
    measure = {"max":np.max,
               "min":np.min,
               "mean":np.mean,
               "var":np.var
            }

    print(f"Number of slugs: {len(coupledidx)}")
    for slugnr, pair in enumerate(coupledidx):
        print(f"Treating slug {slugnr+1}/{len(coupledidx)}")
        startind, stopind = pair
        start_t = startind/8000
        stop_t = stopind/8000
        print(f"mp4 time: {int(start_t//60)}:{start_t%60:2.0f} - {int(stop_t//60)}:{stop_t%60:2.0f}")

        cap = verticalsnapshots(mp4path, mp4, start_t, stop_t, slices)
        box = boxintens(mp4path, mp4, start_t, stop_t)
        box[np.where(box < threshold)]=0
        meanbox = box.mean(axis=(1, 2))
        #frame = np.arange(1, len(meanbox)+1)
        #plt.plot(frame, meanbox)
        #plt.show()
        #continue

        normcap = []
        for arr in cap:
            #print(arr.shape)
            tmp = arr - arr.mean(axis=1, keepdims=True)
            tmp[np.where(tmp < threshold)] = 0
            normcap.append(tmp)

        normintcap = []     # norm intensity
        for arr in normcap:
            normintcap.append(arr.mean(axis=0))

        fullcap = np.array(normintcap)  # All intensity plots for one slug, x slices
        #Ftrans = np.fft.fft(fullcap).mean(axis=0)
        #print(Ftrans)
        #Ffreq = np.fft.fftfreq(Ftrans.size)
        #print(Ffreq)
        #print(Ftrans.shape)
        #print(Ffreq.shape)
        #inds = np.where(Ffreq > 0)

        #plt.title(f"Slug: {slugnr+1}")
        #plt.plot(Ffreq[inds], np.abs(Ftrans)[inds])
        #plt.show()
        #continue
        #exit(1)
        '''
        fig, ax = plt.subplots(2, 2, figsize=(12,8))
        fig.suptitle(f"Slug nr. {slugnr+1}")
        ax[0, 0].imshow(normcap[0])
        ax[0, 1].imshow(normcap[-1])
        x = np.arange(len(fullcap[0]))
        ax[1, 0].plot(x, fullcap[0])
        ax[1, 0].axhline(np.std(fullcap[0]), ls='--', c='k', alpha=0.4)
        ax[1, 0].axhline(2*np.std(fullcap[0]), ls='--', c='k', alpha=0.4)
        x = np.arange(len(fullcap[-1]))
        ax[1, 1].plot(x, fullcap[-1])
        ax[1, 1].axhline(np.std(fullcap[1]), ls='--', c='k', alpha=0.4)
        ax[1, 1].axhline(2*np.std(fullcap[1]), ls='--', c='k', alpha=0.4)

        plt.show()
        '''
        # Get max, maxind, and variance from each of the slices
        Max = np.max(fullcap, axis=1)
        Maxind = np.argmax(fullcap, axis=1)
        Var = np.var(fullcap, axis=1)

        slicevals = {"max":Max, "maxind":Maxind, "var":Var}

        d = {
                "run": nr,
                "slugnr":slugnr+1,
                "startind": startind,
                "endind": stopind,
                "starttime": start_t,
                "endtime": stop_t,
                "max_boxintens": np.max(meanbox)
             }

        for vals in slicevals.keys():
            for stat in measure.keys():
                d[f"{stat}_{vals}"] = measure[stat](slicevals[vals])

        slugstats.append(d)
        # For storing into csv file
        """
        for i, arr in enumerate(fullcap):
            #print(f"[{i+1}/{len(slices)}]")
            #lowinds = np.where(arr < -np.std(arr))[0]
            #highinds = np.where(arr > np.std(arr))[0]
            #conv = np.convolve(arr, runavg, mode="same")
            #convlowinds = np.where(conv < -np.std(conv))[0]
            #convhighinds = np.where(conv > np.std(conv))[0]
            '''
            d = {
                "run":              nr,
                "slugnr":           slugnr+1,
                "startind":         startind,
                "endind":           stopind,
                "length":           int(stopind - startind),
                "starttime":        start_t,
                "endtime":          stop_t,
                "timelen":          stop_t - start_t,
                "measurepixel":     slices[i],
                "avg":              np.mean(arr),
                "var":              np.var(arr),
                "std":              np.std(arr),
                "min":              np.min(arr),
                "max":              np.max(arr),
                "minidx":           np.argmin(arr),
                "maxidx":           np.argmax(arr),
                "lowinds":          len(lowinds),
                "highinds":         len(highinds),
                "minidxtime":       np.argmin(arr)/8000,
                "maxidxtime":       np.argmax(arr)/8000,
                "convavg":          np.mean(conv),
                "convvar":          np.var(conv),
                "convstd":          np.std(conv),
                "convmin":          np.min(conv),
                "convmax":          np.max(conv),
                "convminidx":       np.argmin(conv),
                "convmaxidx":       np.argmax(conv),
                "convminidxtime":   np.argmin(conv)/8000,
                "convmaxidxtime":   np.argmax(conv)/8000,
                "convlowinds":      len(convlowinds),
                "convhighinds":     len(convlowinds)
                } 

            slugstats.append(d)
            '''
        """

    # unindent in order to use
    df = pd.DataFrame(slugstats)
    if not os.path.exists("boxint.csv"):
        df.to_csv(f"boxint.csv", mode='a', index=False, header=True)
    else:
        df.to_csv(f"boxint.csv", mode='a', index=False, header=False)
#continue
print("CSV file written")
exit(1)
"""
    allimgs = np.array(normcap)
    #exit(1) # Remove for visualization, crashes tentativly
    first_intensity = fullcap[0]
    final_intensity = fullcap[-1]

    index = np.arange(len(first_intensity))
    
    fig, ax = plt.subplots(2, 2)
    
    a = first_intensity[:-5]
    b = final_intensity[:-5]
    x = index[:-5]

    #ax[0, 1].imshow(first)
    #ax[1, 1].imshow(final)

    ax[0,0].plot(x, a, label = "vid_idx: 60", c="C0")#, alpha = 0.2)
    ax[0,0].plot(x, b, label = "vid_idx: 1850", c="C1")#, alpha = 0.2)
    #ax[0].axhline(3*np.std(a), c="C0", ls="--", alpha=0.3)
    #ax[0].axhline(3*np.std(b), c="C1", ls="--", alpha=0.3)
    ax[0,0].legend()

    arunavg = np.convolve(a, runavg, mode="same")
    brunavg = np.convolve(b, runavg, mode="same")

    afullpipe = np.where(arunavg < -np.std(arunavg))[0]
    bfullpipe = np.where(brunavg < -np.std(brunavg))[0]
    #print(f"afullpipe: {afullpipe}\t bool: {bool(afullpipe)}")
    #print(f"bfullpipe: {bfullpipe}\t bool: {bool(bfullpipe)}")
    at = 1  # Aerated tightness threshold for number of standard deviations threshold
    if afullpipe.size > 0:
        #print("Full slug")
        apeak = arunavg[:afullpipe[0]].argmax()
        ax[1,0].scatter(x[apeak], arunavg[apeak], c="green", marker="x")
        ax[1,0].scatter(x[afullpipe], arunavg[afullpipe], c="red", alpha=0.4)
        ax[0,1].axvline(afullpipe[-1]+1, c='r', ls='--', alpha=0.4)
        ax[0,1].axvline(apeak, c='k', ls='--', alpha=0.4)
        ax[0,0].scatter(x[apeak], a[apeak], c="green", marker="x", zorder=5)
        alastind = afullpipe[-1]+1
        ax[0,0].scatter(x[alastind], a[alastind], c="r", zorder=5)
        asluglen = alastind - apeak
        ax[0,1].set_title(f"Slug length (time) : {asluglen}")
    else:
        #print("Aerated slug")
        apeak = arunavg.argmax()
        ax[1,0].scatter(x[apeak], arunavg[apeak], c="green", marker="x")
        bef_ap = arunavg[:apeak]
        aft_ap = arunavg[apeak:]
        aslug_start = np.where(bef_ap < at*np.std(arunavg))[0][-1]
        aslug_end = np.where(aft_ap < at*np.std(arunavg))[0][0]+apeak
        ax[1,0].scatter(x[aslug_start], arunavg[aslug_start], c='k')
        ax[1,0].scatter(x[aslug_end], arunavg[aslug_end], c='r')
        ax[0,1].axvline(aslug_start, c='k', ls='--', alpha=0.4)
        ax[0,1].axvline(aslug_end, c='r', ls='--', alpha=0.4)
        
        

    if bfullpipe.size > 0:
        #print("Full slug")
        bpeak = brunavg[:bfullpipe[0]].argmax()
        ax[1,0].scatter(x[bpeak], brunavg[bpeak], c="green", marker="x")
        ax[1,0].scatter(x[bfullpipe], brunavg[bfullpipe], c="red", alpha=0.4)
        ax[1,1].axvline(bfullpipe[-1]+1, c='r', ls='--', alpha=0.4)
        ax[1,1].axvline(bpeak, c='k', ls='--', alpha=0.4)
        ax[0,0].scatter(x[bpeak], b[bpeak], c="green", marker="x")
        blastind = bfullpipe[-1]+1
        ax[0,0].scatter(x[blastind], b[blastind], c="r", zorder=5)
        bsluglen = blastind - bpeak
        ax[1,1].set_title(f"Slug length (time) : {bsluglen}")

        xdiff = bpeak - apeak
        sekdiff = xdiff*1/30
        pixeldiff = slices[-1]-slices[0]
        slugvel = pixeldiff/sekdiff
        fig.suptitle(f"Slug velocity: {slugvel} pixels/s")

    else:
        #print("Aerated slug")
        bpeak = brunavg.argmax()
        ax[1,0].scatter(x[bpeak], brunavg[bpeak], c="green", marker="x")
        bef_bp = brunavg[:bpeak]
        aft_bp = brunavg[bpeak:]
        bslug_start = np.where(bef_bp < at*np.std(brunavg))[0][-1]
        bslug_end = np.where(aft_bp < at*np.std(brunavg))[0][0]+bpeak
        ax[1,0].scatter(x[bslug_start], brunavg[bslug_start], c='k')
        ax[1,0].scatter(x[bslug_end], brunavg[bslug_end], c='r')
        ax[1,1].axvline(bslug_start, c='k', ls='--', alpha=0.4)
        ax[1,1].axvline(bslug_end, c='r', ls='--', alpha=0.4)

    ax[1,0].plot(x, arunavg, label = "vid_idx: 60", c="C0")
    ax[1,0].plot(x, brunavg, label = "vid_idx: 1850", c="C1")
    ax[1,0].legend()
    ax[1,0].set_title("Average")
    #ax[1,0].axhline(np.std(arunavg), ls="--", c="C0", alpha=0.3)
    #ax[1,0].axhline(-np.std(arunavg), ls="--", c="C0", alpha=0.3)
    #ax[1,0].axhline(np.std(brunavg), ls="--", c="C1", alpha=0.3)
    #ax[1,0].axhline(-np.std(brunavg), ls="--", c="C1", alpha=0.3)
    ax[1,0].fill_between(x, 0.5*np.std(arunavg), at*-np.std(arunavg), color="C0", alpha =0.2)
    ax[1,0].fill_between(x, 0.5*np.std(brunavg), at*-np.std(brunavg), color="C1", alpha =0.2)

    ax[0,1].imshow(first)
    ax[1,1].imshow(final)

    '''
    awsf = np.convolve(a, wsf_k, mode="same")
    bwsf = np.convolve(b, wsf_k, mode="same")
    ax[1, 0].plot(x, awsf, label="vid_idx: 60", c="C0")
    ax[1, 0].plot(x, bwsf, label="vid_idx: 1850", c="C1")
    ax[1, 0].legend()
    ax[1, 0].set_title("Water slug front")

    awst = np.convolve(a, wst_k, mode="same")
    bwst = np.convolve(b, wst_k, mode="same")
    ax[1, 1].plot(x, awst, label="vid_idx: 60", c="C0")
    ax[1, 1].plot(x, bwst, label="vid_idx: 1850", c="C1")
    ax[1, 1].legend()
    ax[1, 1].set_title("Water slug tail")

    aaes = np.convolve(a, aes_k, mode="same")
    baes = np.convolve(b, aes_k, mode="same")
    ax[1, 2].plot(x, aaes, label="vid_idx: 60", c="C0")
    ax[1, 2].plot(x, baes, label="vid_idx: 1850", c="C1")
    ax[1, 2].legend()
    ax[1, 2].set_title("Aerated slug")
    '''
    '''
    cutoff = 1
    sek_diff = 0.6
    apeak, _aval = find_peaks(aa, height = cutoff*np.std(aa), distance = 30*sek_diff)
    bpeak, _bval = find_peaks(bb, height = cutoff*np.std(bb), distance = 30*sek_diff)
    ax[0,0].scatter(x[apeak], a[apeak], c="r", zorder=5, marker='x')
    ax[0,0].scatter(x[bpeak], b[bpeak], c="k", zorder=5, marker='x')
    ax[1,0].scatter(x[apeak], aa[apeak], c="r", zorder=5, marker='x')
    ax[1,0].scatter(x[bpeak], bb[bpeak], c="k", zorder=5, marker='x')
    ax[1,0].axhline(cutoff*np.std(aa), c="C0", ls="--", alpha=0.3)
    ax[1,0].axhline(cutoff*np.std(bb), c="C1", ls="--", alpha=0.3)

    for i, ap in enumerate(apeak):
        if i>1:
            print("multiple peaks first")
            break
        ax[0,1].axvline(ap, c="k", ls="--", zorder=5, alpha=0.25)

    for i, bp in enumerate(bpeak):
        if i>1:
            print("multiple peaks final")
            break
        ax[1,1].axvline(bp, c="k", ls="--", zorder=5, alpha=0.25)
    '''
    '''
    daa = np.gradient(aa)
    dbb = np.gradient(bb)
    ax[1].plot(x, daa, c="C0")
    ax[1].plot(x, dbb, c="C1")
    ax[1].axhline(3*np.std(daa), c="C0", ls="--", alpha=0.3)
    ax[1].axhline(3*np.std(dbb), c="C1", ls="--", alpha=0.3)
    '''

    ax[0, 0].set_title("Vertical intensity from mp4")
    ax[0, 0].set_xlabel("Frame")
    ax[1, 0].set_title("1D convolution of vertical intensity")
    ax[1, 0].set_xlabel("Frame")
    ax[0, 1].set_title("Vertical snapshot from pixelcolumn 60")
    ax[0, 1].set_xlabel("Frame")
    ax[1, 1].set_title("Vertical snapshot from pixelcolumn 1850")
    ax[1, 1].set_xlabel("Frame")

    fig.set_size_inches(12, 8)
    fig.tight_layout()

    plt.show()
    #plt.savefig(f"{nr}stats/slug{slugnr}.png")
    #plt.close()
    #exit(1)
"""

