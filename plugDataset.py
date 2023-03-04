from utils import *
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.linear_model import RANSACRegressor

datapath = "/run/media/daniel/ATLEJ-STT/data"
mp4path = "/run/media/daniel/ATLEJ-STT/mp4"

def consec(arr, step):
    return np.split(arr, np.where(np.diff(arr) > step)[0] + 1)

allslugs = []

for nr in [59, 60]:
    fil = f"micData_{nr}.lvm"
    mp4 = f"run{nr}.mp4"

    df = extract_lvm(datapath, fil)

    m1 = df["mic1"].to_numpy()
    m2 = df["mic2"].to_numpy()
    m3 = df["mic3"].to_numpy()
    time = df["time"].to_numpy()

    IntersertSegments, InterestIdx = splitslugs(m2,
                                                3*np.std(m2), 
                                                5*8000, 
                                                centerratio = 0.5,
                                                getsplitpoints=True)
    #slices = [60, 1850]
    slices = [i for i in range(60, 1850, 50)]
    #zipinter = [(30,45), (1045,1060)]       
    # Remove potential zip measurements
    for col in slices:
        if 30 <= col <= 45:
            slices.remove(col)
        if 1045 <= col <= 1060:
            slices.remove(col)

    cw = 11      # Convolution Width
    for slugnr, pair in enumerate(InterestIdx):
        print(f"Run {nr}\tSlug: {slugnr+1}/{len(InterestIdx)}")
        startind, endind = pair

        start_t = startind/8000
        end_t = endind/8000

        cap = verticalsnapshots(mp4path, mp4, start_t, end_t, slices)

        # Treatment
        scaled = []
        peakinds = []
        timelens = []

        for i, img in enumerate(cap):
            img = img - img.mean(axis=1, keepdims=True)

            intensity = img.mean(axis=0)[:-5]
            intensity = np.convolve(intensity, np.ones(cw)/cw, mode='same')
            x = np.arange(len(intensity))

            cut = np.std(intensity)
            low = np.where(intensity < -cut)[0]
            if len(low) == 0:
                print(f"No low indicies for slug: {slugnr}, slice: {slices[i]}")
                continue
        
            highcriteria = 0.7*intensity.max()
            highinds = np.where(intensity > highcriteria)[0]
            #print(f"Highinds: {len(highinds)}")
            peaks = consec(highinds, step=5)
            #print(f"Peaks: {len(peaks)}\t{peaks}")
            first_peak_inds = peaks[0]# if len(peaks)>1 else peaks
            #print(f"First peak inds: {first_peak_inds}")
            peakind = first_peak_inds[intensity[first_peak_inds].argmax()]
            #peakind = intensity[:low[0]].argmax()
            #peakind = intensity.argmax()
            slugendind = low[-1]
            
            peakinds.append((slices[i], peakind))
            timelens.append((slices[i], slugendind-peakind))
        
        x, y = zip(*peakinds)   #x=pixel, y=frame
        x = np.array(x)
        y = np.array(y)

        x2, t = zip(*timelens)
        x2 = np.array(x2)
        t = np.array(t)

        reg = RANSACRegressor(random_state = 0).fit(x.reshape(-1, 1), y.reshape(-1, 1))

        slope = reg.estimator_.coef_[0][0]
        intercept = reg.estimator_.intercept_[0]

        regline = slope*x + intercept
        v = 1/slope * 0.000457 * 30     #pixels/frame * pixels_to_meter * (frame_to_sec)^-1
        s = np.median(t)/30
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))
        ax[0].plot(x, regline, c='r', ls='--',
                   label = f"RANSAC Regression\n{1/slope:0.3f} Pixels/Frame\n{slope:0.3f} Frames/Pixel")
        ax[0].scatter(x, y)
        ax[0].set_title(f"Peakindexes\nVelocity estimate: {v:0.4f} m/s")
        ax[0].set_ylabel("Frame [s]")
        ax[0].set_xlabel("Pixel [m]")
        ax[0].legend()

        ax[1].scatter(x2, t)
        ax[1].axhline(np.median(t), c='r', ls='--',
                      label=f"Median: {np.median(t):0.2f} frames\n{s:0.2f} sec")
        ax[1].axhline(np.mean(t), c='k', ls='--',
                      label=f"Mean: {np.mean(t):0.2f} frames\n{np.mean(t)/30:0.2f} sec")
        ax[1].legend()
        ax[1].set_ylabel("Time Length [s]")
        ax[1].set_xlabel("Pixel [m]")
        ax[1].set_title("Time Length estimates")

        fig.suptitle(f"Slug {slugnr+1}\n{len(slices)} pixelcolumns measured\n Estimated slug length: {v*s} m")
        fig.tight_layout()
        plt.savefig(f"{nr}stats/regression/slug{slugnr+1}.png")
        plt.close()
        #plt.show()
        # Write file
        #data_segment = np.array([m1[startind:endind+1],
                                 #m2[startind:endind+1],
                                 #m2[startind:endind+1]])
        filename = f"s{slugnr+1}_r{nr}.txt"
        saveloc = "/run/media/daniel/ATLEJ-STT/plugslug"
        #out = os.path.join(saveloc, filename)
        #np.savetxt(out, data_segment)
        # create dict
        d = {
                "run":          nr,
                "filename":     filename,
                "slugnr":       slugnr+1,
                "label":        1,
                "velocity":     v,
                "length":       v*s,
                "startind":     startind,
                "endind":       endind,
                "starttime":    start_t,
                "endtime":      end_t
                }
        allslugs.append(d)

df = pd.DataFrame(allslugs)
df.to_csv("/run/media/daniel/ATLEJ-STT/plugslug/plugmeta.csv", index=False)
