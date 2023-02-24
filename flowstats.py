import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from utils import *

nr = 60
datapath = "/run/media/daniel/ATLEJ-STT/data"
fil = f"micData_{nr}.lvm"
mp4path = "/run/media/daniel/ATLEJ-STT/mp4"
mp4 = f"run{nr}.mp4"

df = extract_lvm(datapath, fil)

data = df["mic2"].to_numpy()

segments, coupledidx = splitslugs(data, 3*np.std(data), 5*8000,
                                  getsplitpoints=True)
#print(len(segments))
#print(len(coupledidx))

t = df["time"].to_numpy()
'''
plt.plot(t, data, zorder=1)
for start, slutt in coupledidx:
    plt.scatter(t[start], 0, c='green')
    plt.scatter(t[slutt], 0, c='red')
plt.show()
'''

kl = 5
runavg = np.ones(kl)/kl
#wsf = np.array([0, 0, 0, 1, 2, 3, 3, 3, 1.5, -2, -2, -2])           # Water slug front
#wst = np.array([-2, -2, -2, 1.5, 3, 3, 3, 2, 1, 0, 0, 0])           # Water slug tail
#aes = np.array([0, 0, 0, 3, 5, 5, 5, 3, 2, 0, 0, 0])                # Aerated slug

wsf = np.array([0, 0, 0, 0, 0, 0, 0, 10])
wst = np.array([10, 0, 0, 0, 0, 0, 0, 0])
aes = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])

wsf_k = (wsf - wsf.mean())/len(wsf)
wst_k = (wst - wst.mean())/len(wst)
aes_k = (aes - aes.mean())/len(aes)

if not os.path.exists(f"{nr}stats"):
    os.makedirs(f"{nr}stats")

print(f"Number of slugs: {len(coupledidx)}")
for slugnr, pair in enumerate(coupledidx):
    startind, stopind = pair
    start_t = startind/8000
    stop_t = stopind/8000

    #slices = [i for i in range(60, 1850, 5)]
    slices = [60, 1850]
    cap = verticalsnapshots(mp4path, mp4, start_t, stop_t, slices)

    normcap = []
    for arr in cap:
        #print(arr.shape)
        normcap.append(arr - arr.mean(axis=1, keepdims=True))

    first = normcap[0]
    final = normcap[-1]
    normintcap = []
    for arr in normcap:
        normintcap.append(arr.mean(axis=0))

    fullcap = np.array(normintcap)

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

    afullpipe = np.where(arunavg < -np.std(arunavg))
    bfullpipe = np.where(brunavg < -np.std(brunavg))
    #print(f"afullpipe: {afullpipe}\t bool: {bool(afullpipe)}")
    #print(f"bfullpipe: {bfullpipe}\t bool: {bool(bfullpipe)}")
    if afullpipe:
        #print("Full slug")
        apeak = arunavg[:afullpipe[0][0]].argmax()
        ax[1,0].scatter(x[apeak], arunavg[apeak], c="green", marker="x")
        ax[1,0].scatter(x[afullpipe], arunavg[afullpipe], c="red", alpha=0.4)
        ax[0,1].axvline(afullpipe[0][-1]+1, c='r', ls='--', alpha=0.4)
        ax[0,0].scatter(x[apeak], a[apeak], c="green", marker="x", zorder=5)
        alastind = afullpipe[0][-1]+1
        ax[0,0].scatter(x[alastind], a[alastind], c="r", zorder=5)
        asluglen = alastind - apeak
        ax[0,1].set_title(f"Slug length (time) : {asluglen}")
    else:
        #print("Aerated slug")
        apeak = arunavg.argmax()
        ax[1,0].scatter(x[apeak], arunavg[apeak], c="green", marker="x")

    if bfullpipe:
        #print("Full slug")
        bpeak = brunavg[:bfullpipe[0][0]].argmax()
        ax[1,0].scatter(x[bpeak], brunavg[bpeak], c="green", marker="x")
        ax[1,0].scatter(x[bfullpipe], brunavg[bfullpipe], c="red", alpha=0.4)
        ax[1,1].axvline(bfullpipe[0][-1]+1, c='r', ls='--', alpha=0.4)
        ax[0,0].scatter(x[bpeak], b[bpeak], c="green", marker="x")
        blastind = bfullpipe[0][-1]+1
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

    ax[1,0].plot(x, arunavg, label = "vid_idx: 60", c="C0")
    ax[1,0].plot(x, brunavg, label = "vid_idx: 1850", c="C1")
    ax[1,0].legend()
    ax[1,0].set_title("Average")
    ax[1,0].axhline(-np.std(arunavg), ls="--", c="C0", alpha=0.3)
    #ax[1].axhline(-np.std(arunavg), ls="--", c="k", alpha=0.3)
    ax[1,0].axhline(-np.std(brunavg), ls="--", c="C1", alpha=0.3)
    #ax[1].axhline(-np.std(brunavg), ls="--", c="k", alpha=0.3)
    #ax[1].fill_between(x, np.std(arunavg), -np.std(arunavg), color="C0", alpha =0.2)
    #ax[1].fill_between(x, np.std(brunavg), -np.std(brunavg), color="C1", alpha =0.2)
    ax[0,1].imshow(first)
    ax[0,1].axvline(apeak, c='k', ls='--', alpha=0.4)

    ax[1,1].imshow(final)
    ax[1,1].axvline(bpeak, c='k', ls='--', alpha=0.4)


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
    daa = np.gradient(aa)
    dbb = np.gradient(bb)
    ax[1].plot(x, daa, c="C0")
    ax[1].plot(x, dbb, c="C1")
    ax[1].axhline(3*np.std(daa), c="C0", ls="--", alpha=0.3)
    ax[1].axhline(3*np.std(dbb), c="C1", ls="--", alpha=0.3)
    '''
    """
    ax[0, 0].set_title("Vertical intensity from mp4")
    ax[0, 0].set_xlabel("Frame")
    ax[1, 0].set_title("1D convolution of vertical intensity")
    ax[1, 0].set_xlabel("Frame")
    ax[0, 1].set_title("Vertical snapshot from pixelcolumn 60")
    ax[0, 1].set_xlabel("Frame")
    ax[1, 1].set_title("Vertical snapshot from pixelcolumn 1850")
    ax[1, 1].set_xlabel("Frame")
    """

    fig.set_size_inches(12, 8)

    plt.savefig(f"{nr}stats/slug{slugnr}.png")
    plt.close()
    #exit(1)
