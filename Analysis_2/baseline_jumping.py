'''
This script loops through RMS noise files (as they are saved in the data folder in the file system), checks each of 
the four SLK configurations in each chennel for jumps from the baseline noise level, and grades each channel based on
the prevalance of noise jumps. 
They are graded on a scale of A-D with A being a waveform uncompromised by baseline jumps and a D being fully compromised. 
Author: C. A. Chinatti
'''


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, peak_prominences
import csv
import struct
import platform
from matplotlib.backends.backend_pdf import PdfPages

system_info = platform.system()  # Detect OS for timestamp handling

def deframe(words):
    """Decode a single WIB frame into a dictionary of CD_data and metadata."""
    data_begin = 3
    tick_word_length = 14
    frame_data = words[data_begin:]
    num_ticks = len(frame_data) // tick_word_length

    frame_dict = {
        "TMTS": words[0],
        "FEMB_CD0TS": words[1] & 0x7fff,
        "FEMB_CD1TS": (words[1] >> 16) & 0x7fff,
        "CRC_error": (words[1] >> 33) & 0x3,
        "Link_valid": (words[1] >> 35) & 0x3,
        "LOL": (words[1] >> 37) & 0x1,
        "WS": (words[1] >> 38) & 0x1,
        "FS": (words[1] >> 39) & 0x3,
        "Pulser": (words[1] >> 41) & 0x1,
        "Cal": (words[1] >> 42) & 0x1,
        "Ready": (words[1] >> 43) & 0x1,
        "Context_code": (words[1] >> 44) & 0xff,
        "Version": (words[1] >> 52) & 0xf,
        "Chn_ID": (words[1] >> 56) & 0xff,
        "CD_data": [[0 for _ in range(64)] for _ in range(num_ticks)]
    }

    for tick in range(num_ticks):
        tick_data = frame_data[tick * tick_word_length:(tick + 1) * tick_word_length]
        for ch in range(64):
            low_bit = ch * 14
            low_word = low_bit // 64
            high_bit = (ch + 1) * 14 - 1
            high_word = high_bit // 64

            if low_word == high_word:
                frame_dict["CD_data"][tick][ch] = (tick_data[low_word] >> (low_bit % 64)) & 0x3fff
            else:
                high_off = high_word * 64 - low_bit
                val = (tick_data[low_word] >> (low_bit % 64)) & (0x3fff >> (14 - high_off))
                val |= (tick_data[high_word] << high_off) & ((0x3fff << high_off) & 0x3fff)
                frame_dict["CD_data"][tick][ch] = val
    return frame_dict



def wib_dec(data, fembs=range(4), spy_num= 1, fastchk = False, cd0cd1sync=True): #data from one WIB  
    spy_num_all = len(data)
    if spy_num_all < spy_num:
        spy_num = spy_num_all
    wibdata = []
    if fastchk == True:
        spy_num=1

    for sn in range(spy_num):
        tmts = [[],[],[],[],[],[],[],[]]
        cd_tmts = [[],[],[],[],[],[],[],[]]
        femb00 = []
        femb01 = []
        femb10 = []
        femb11 = []
        femb20 = []
        femb21 = []
        femb30 = []
        femb31 = []

        raw  = data[sn]
        bufs = raw[0]
        buf_end_addr = raw[1]
        spy_rec_ticks = raw[2]
        trig_cmd      = raw[3]
        #print (len(bufs),buf_end_addr,  spy_rec_ticks ) 
        if trig_cmd == 0:
            trigmode="SW"
        else:
            trigmode="HW"
        dec_data = wib_spy_dec_syn(bufs, trigmode, buf_end_addr, spy_rec_ticks, fembs, fastchk)
        if fastchk:
            for fembno in fembs:
                if (dec_data[fembno*2] != False) and (dec_data[fembno*2+1] != False) and (dec_data[fembno*2] == dec_data[fembno*2+1]) :
                    print ("FEMB%d_SYNCED"%fembno, hex(dec_data[fembno*2]), hex(dec_data[fembno*2+1]) )
                #CD0 and CD1 of the same FEMB has different time stamp
                    #return False
                else:
                    print ("Not SYNCED", hex(dec_data[fembno*2]), hex(dec_data[fembno*2+1]) )
                    print ("Data of FEMB{} is not synchoronized...".format(fembno))
                    return False
            return True
            
        if 0 in fembs:
            flen = min(len(dec_data[0]), len(dec_data[1]))
            for i in range(flen):
                chdata_64ticks0 = [dec_data[0][i]["CD_data"][tick] for tick in range(64)]        
                chdata_64ticks1 = [dec_data[1][i]["CD_data"][tick] for tick in range(64)]        
                femb00 = femb00 + chdata_64ticks0        
                cd_tmts[0].append(dec_data[0][i]["FEMB_CD0TS"])
                tmts[0].append(dec_data[0][i]["TMTS"])
                femb01 = femb01 + chdata_64ticks1        
                cd_tmts[1].append(dec_data[1][i]["FEMB_CD1TS"])
                tmts[1].append(dec_data[1][i]["TMTS"])

        if 1 in fembs:        
            flen = min(len(dec_data[2]), len(dec_data[3]))
            for i in range(flen):
                chdata_64ticks0 = [dec_data[2][i]["CD_data"][tick] for tick in range(64)]        
                chdata_64ticks1 = [dec_data[3][i]["CD_data"][tick] for tick in range(64)]        
                femb10 = femb10 + chdata_64ticks0        
                cd_tmts[2].append(dec_data[2][i]["FEMB_CD0TS"])
                tmts[2].append(dec_data[2][i]["TMTS"])
                femb11 = femb11 + chdata_64ticks1        
                cd_tmts[3].append(dec_data[3][i]["FEMB_CD1TS"])
                tmts[3].append(dec_data[3][i]["TMTS"])

        if 2 in fembs:       
            flen = min(len(dec_data[4]), len(dec_data[5]))
            for i in range(flen):
                chdata_64ticks0 = [dec_data[4][i]["CD_data"][tick] for tick in range(64)]        
                chdata_64ticks1 = [dec_data[5][i]["CD_data"][tick] for tick in range(64)]        
                femb20 = femb20 + chdata_64ticks0        
                cd_tmts[4].append(dec_data[4][i]["FEMB_CD0TS"])
                tmts[4].append(dec_data[4][i]["TMTS"])
                femb21 = femb21 + chdata_64ticks1        
                cd_tmts[5].append(dec_data[5][i]["FEMB_CD1TS"])
                tmts[5].append(dec_data[5][i]["TMTS"])

        if 3 in fembs:
            flen = min(len(dec_data[6]), len(dec_data[7]))
            for i in range(flen):
                chdata_64ticks0 = [dec_data[6][i]["CD_data"][tick] for tick in range(64)]        
                chdata_64ticks1 = [dec_data[7][i]["CD_data"][tick] for tick in range(64)]        
                femb30 = femb30 + chdata_64ticks0        
                cd_tmts[6].append(dec_data[6][i]["FEMB_CD0TS"])
                tmts[6].append(dec_data[6][i]["TMTS"])
                femb31 = femb31 + chdata_64ticks1        
                cd_tmts[7].append(dec_data[7][i]["FEMB_CD1TS"])
                tmts[7].append(dec_data[7][i]["TMTS"])

        if cd0cd1sync:
            t0s = [-1, -1, -1, -1, -1, -1, -1, -1]
            if 0 in fembs:
                t0s[0]=tmts[0][0]
                t0s[1]=tmts[1][0]
            if 1 in fembs:
                t0s[2]=tmts[2][0]
                t0s[3]=tmts[3][0]
            if 2 in fembs:
                t0s[4]=tmts[4][0]
                t0s[5]=tmts[5][0]
            if 3 in fembs:
                t0s[6]=tmts[6][0]
                t0s[7]=tmts[7][0]

            t0max = np.max(t0s)
            for i in range(8):
                if t0s[i] != -1:
                    t0s[i] = (t0max - t0s[i])//32 
        else:
            t0s = [0, 0, 0, 0, 0, 0, 0, 0]
            t0max = 0

        if 0 in fembs:
            femb00 = list(zip(*femb00))
            for i in range(len(femb00)):
                femb00[i]=femb00[i][t0s[0]:]
            femb01 = list(zip(*femb01))
            for i in range(len(femb01)):
                femb01[i]=femb01[i][t0s[1]:]
            femb0 = femb00 + femb01
        else:
            femb0 = None
        if 1 in fembs:
            femb10 = list(zip(*femb10))
            for i in range(len(femb10)):
                femb10[i]=femb10[i][t0s[2]:]
            femb11 = list(zip(*femb11))
            for i in range(len(femb11)):
                femb11[i]=femb11[i][t0s[3]:]
            femb1 = femb10 + femb11
        else:
            femb1 = None
        if 2 in fembs:
            femb20 = list(zip(*femb20))
            for i in range(len(femb20)):
                femb20[i]=femb20[i][t0s[4]:]
            femb21 = list(zip(*femb21))
            for i in range(len(femb21)):
                femb21[i]=femb21[i][t0s[5]:]
            femb2 = femb20 + femb21
        else:
            femb2 = None
        if 3 in fembs:
            femb30 = list(zip(*femb30))
            for i in range(len(femb30)):
                femb30[i]=femb30[i][t0s[6]:]
            femb31 = list(zip(*femb31))
            for i in range(len(femb31)):
                femb31[i]=femb31[i][t0s[7]:]
            femb3 = femb30 + femb31
        else:
            femb3 = None

        wibdata.append([femb0, femb1, femb2, femb3, t0max, tmts, cd_tmts]) #temp for graphing 
    return wibdata


def wib_spy_dec_syn(bufs, trigmode="SW", buf_end_addr=0x0, trigger_rec_ticks=0x3f000, fembs=range(4), fastchk=False): #synchronize samples in 8 spy buffers
    #^change default trigger_rec_ticks?
    frames = [[],[],[],[],[],[],[],[]] #frame buffers
    
    for femb in fembs:
        buf0 = bufs[femb*2]
        buf1 = bufs[femb*2+1]
        
        frames[femb*2] = spymemory_decode(buf=buf0, trigmode=trigmode, buf_end_addr=buf_end_addr, trigger_rec_ticks=trigger_rec_ticks, fastchk=fastchk)
        frames[femb*2+1] = spymemory_decode(buf=buf1,trigmode=trigmode,buf_end_addr=buf_end_addr, trigger_rec_ticks=trigger_rec_ticks, fastchk=fastchk)
    return frames

def spymemory_decode(buf, trigmode="SW", buf_end_addr = 0x0, trigger_rec_ticks=0x7fff, fastchk=False): #change default trigger_rec_ticks?
    PKT_LEN = 899 #in words
    pkgn =  trigger_rec_ticks//PKT_LEN

    if trigmode == "SW" :
        try_num = 2
    else:
        try_num = 1
    for tryi in range(try_num):
        f_heads = []
        #tryi : find the position of frame with minimum timestamp, re-order the buffer
        #tryi==2: decode it only when trigmode is software trigger
        if tryi == 0:
            if trigmode == "SW" :
                deoding_start_addr = 0x0
            else:
                spy_addr_word = buf_end_addr
                if spy_addr_word <= trigger_rec_ticks:
                    deoding_start_addr = spy_addr_word + 0x8000 - trigger_rec_ticks #? to be updated
                else:
                    deoding_start_addr = spy_addr_word  - trigger_rec_ticks
                buf2=buf+buf
                buf = buf2[deoding_start_addr*8: deoding_start_addr*8 + trigger_rec_ticks*8]

        buflen=len(buf)
        num_words = int (buflen// 8) 
        words = list(struct.unpack_from("<%dQ"%(num_words),buf)) #unpack [num_words] big-endian 64-bit unsigned integers
        i = 0
        xlen = num_words-PKT_LEN
        while i < xlen:       
            if   (words[i+PKT_LEN] - words[i]==0x800) and (words[i+1]&0x7fff == (words[i+1]>>16)&0x7fff) and  (words[i+2]==0):
                tmts = words[i]
                f_heads.append([i,tmts])
                i = i + PKT_LEN
            else:
                i = i + 1   

#        with open("./tmp_data/hexdata.txt", "w") as fp:
#            for x in range(0, len(buf), 8):
#                fp.write ("%02x%02x%02x%02x%02x%02x%02x%02x\n"%(buf[x+7], buf[x+6], buf[x+5], buf[x+4], buf[x+3], buf[x+2], buf[x+1], buf[x+0]))
#        exit()

        if tryi == 0:
            if (len(f_heads) < pkgn-2) and (len(f_heads) < 30):
                print ("Invalid data length...")
                return False
            w_sofs, tmsts = zip(*f_heads)
            tmst0 = tmsts[0]
            w_sof0 = w_sofs[0]
            f_heads = sorted(f_heads, key=lambda ts: ts[1]) 
            w_min = f_heads[0][0]
            tmstmin=f_heads[0][1]
            w_max = f_heads[-1][0]
            if (tmst0-tmstmin)//0x20 < ((buflen//8//899)*64 + 64): #ring buffer was rolled back (data is larger than the length of ring buffer)
                buf = buf[w_min*8:] + buf[:w_min*8]
#                with open("./tmp_data/hexdata2.txt", "w+") as fp:
#                    for x in range(0, len(buf), 8):
#                        fp.write ("%02x%02x%02x%02x%02x%02x%02x%02x\n"%(buf[x+7], buf[x+6], buf[x+5], buf[x+4], buf[x+3], buf[x+2], buf[x+1], buf[x+0]))
#                exit()
            else:
                buf = buf[w_sof0*8:w_max*8+PKT_LEN ]

    if fastchk:
        if (len(f_heads) > pkgn-2):
            return f_heads[0][1] 
        else:
            return False

    w_sofs, tmsts = zip(*f_heads)
    ordered_frames = []
    for i in range( len(w_sofs)):
        frame_dict = deframe(words = words[w_sofs[i]:w_sofs[i]+PKT_LEN])
        ordered_frames.append(frame_dict)
   
    return ordered_frames


def decodeRawData2(fembs, rawdata, needTimeStamps=False, period=500):
    """
    Decode WIB raw data, concatenate spy buffers by pulser timestamps,
    print number of points from each spy buffer per channel,
    and return both concatenated data and list of SPY buffer lengths.
    """
    tmpwibdata = wib_dec(rawdata, fembs, spy_num=20, cd0cd1sync=False)

    dat_tmts_l = []
    dat_tmts_h = []

    for i, wibdata in enumerate(tmpwibdata):
        try:
            if wibdata is None or len(wibdata) < 6 or wibdata[5] is None:
                continue

            tms_l = wibdata[5][fembs[0]*2] if len(wibdata[5])>fembs[0]*2 else []
            tms_h = wibdata[5][fembs[0]*2+1] if len(wibdata[5])>fembs[0]*2+1 else []

            if len(tms_l) == 0 or len(tms_h) == 0:
                continue

            dat_tmts_l.append(tms_l[0])
            dat_tmts_h.append(tms_h[0])
        except Exception:
            continue

    if len(dat_tmts_l) == 0 or len(dat_tmts_h) == 0:
        return [], []

    dat_tmtsl_oft = (np.array(dat_tmts_l)//32)%period
    dat_tmtsh_oft = (np.array(dat_tmts_h)//32)%period

    spy_buffer_lengths = []

    all_data = []
    for achn in range(128):
        conchndata = np.array([], dtype=np.uint32)
        # print(f"\nChannel {achn}:")
        for i in range(len(tmpwibdata)):
            wibdata = tmpwibdata[i]
            if wibdata is None:
                continue

            datd = [wibdata[0], wibdata[1], wibdata[2], wibdata[3]][fembs[0]]
            if datd is None or len(datd) <= achn:
                continue

            oft = dat_tmtsl_oft[i] if achn < 64 else dat_tmtsh_oft[i]

            chndata = np.array(datd[achn],dtype=np.uint32)
            lench = len(chndata)
            tmp = int(period-oft)

            num_points=((lench-tmp)//period)*period
            # print(f"  Buffer {i}: {num_points} points (offset {oft})")
            
            if achn == 0:
                spy_buffer_lengths.append(num_points)

            conchndata=np.concatenate((conchndata, chndata[tmp : tmp + num_points]))

        # print(f"  Total points after concatenation: {len(conchndata)}")
        all_data.append(conchndata)

    data=[]
    iichn=0
    for nchip in range(8):
        onechipData=[]
        for ichn in range(16):
            if iichn<len(all_data):
                onechipData.append(list(all_data[iichn]))
            else:
                onechipData.append([])
            iichn += 1
        data.append(onechipData)

    return data,spy_buffer_lengths


root_folder = "/data/rts/RTS_DAT_LArASIC_QC"
n_sockets = 8
n_channels = 16
window = 50
max_jump_gap = 1500
fs = 1e9/512           # ADC sampling rate
hp_cutoff = 3e3          # High-pass cutoff frequency (Hz)
filter_order = 4         # Butterworth order

b, a = butter(N=filter_order, Wn=hp_cutoff / (fs / 2), btype='highpass')

configs_to_check = {
    "first":  'RMS_SLK_SDD0_SDF0_SLK00_SLK11_SNC0_ST01_ST11_SG00_SG10', #5nA
    "second": 'RMS_SLK_SDD0_SDF0_SLK01_SLK11_SNC0_ST01_ST11_SG00_SG10', #1nA
    "third":  'RMS_SLK_SDD0_SDF0_SLK00_SLK10_SNC0_ST01_ST11_SG00_SG10', #500pA
    "fourth": 'RMS_SLK_SDD0_SDF0_SLK01_SLK11_SNC0_ST01_ST11_SG00_SG10'  #100pA
}

def is_bimodal(buffer, bins=30, prominence=0.005, min_peak_separation=0.1, valley_fraction=0.5):
    counts, bin_edges = np.histogram(buffer, bins=bins)
    bin_centers = 0.5*(bin_edges[:-1]+bin_edges[1:])
    peaks, _ = find_peaks(counts, prominence=prominence * np.max(counts))
    if len(peaks) < 2:
        return False

    peak_heights = counts[peaks]
    prominences = peak_prominences(counts, peaks)[0]
    primary_idx = np.argmax(peak_heights)
    primary_peak = peaks[primary_idx]
    primary_value = bin_centers[primary_peak]
    amplitude_range = np.max(buffer)-np.min(buffer)
    min_sep = amplitude_range*min_peak_separation

    for i, pk in enumerate(peaks):
        if i == primary_idx:
            continue
        separation=abs(bin_centers[pk]-primary_value)
        prom_ok=prominences[i] >= prominence*np.max(counts)
        left_idx, right_idx=sorted([primary_peak, pk])
        valley_height=np.min(counts[left_idx:right_idx+1])
        valley_ok=valley_height <= counts[pk]*valley_fraction
        if separation >= min_sep and prom_ok and valley_ok:
            return True
    return False


# --- Main loop ---
for top_folder in sorted(os.listdir(root_folder)):
    top_path=os.path.join(root_folder, top_folder)
    if not os.path.isdir(top_path):
        continue

    print(f"\n Processing top folder: {top_folder}")

    grading_results={}
    grade_counts={"A": 0, "B": 0, "C": 0, "D": 0}
    config_jump_counts={cfg: 0 for cfg in configs_to_check}
    total_waveforms=0

    for time_folder in sorted(os.listdir(top_path)):
        time_path=os.path.join(top_path, time_folder)
        if not os.path.isdir(time_path) or not time_folder.startswith("Time_"):
            continue

        for ln_folder in sorted(os.listdir(time_path)):
            ln_path=os.path.join(time_path, ln_folder)
            if not os.path.isdir(ln_path) or not ln_folder.startswith("LN_"):
                continue

            print(f"\n Processing folder: {ln_path}")

            for filename in sorted(os.listdir(ln_path)):
                if not filename.endswith(".bin") or "RMS" not in filename:
                    continue

                file_path=os.path.join(ln_path, filename)
                with open(file_path, "rb") as f:
                    rms_data=pickle.load(f)

                for socket_index in range(n_sockets):
                    for channel_index in range(n_channels):
                        jumps_in_configs={cfg: False for cfg in configs_to_check}

                        for cfg_name, cfg_key in configs_to_check.items():
                            if cfg_key not in rms_data:
                                continue

                            fembs=rms_data[cfg_key][0]
                            raw_data=rms_data[cfg_key][1]
                            data_dec, spy_buffer_lengths=decodeRawData2(fembs=fembs, rawdata=raw_data)
                            raw_signal = np.array(data_dec[socket_index][channel_index])
                            if len(raw_signal)==0:
                                continue

                            signal_hp=filtfilt(b, a, raw_signal)

                            start=0
                            all_buffer_jumps=[]
                            for length in spy_buffer_lengths:
                                end=start+length
                                if end>len(signal_hp):
                                    end=len(signal_hp)
                                buffer_segment = signal_hp[start:end]
                                if len(buffer_segment)<2*window:
                                    start=end
                                    continue

                                mini, maxi = np.min(buffer_segment), np.max(buffer_segment)
                                spread = maxi-mini
                                jump_threshold=spread*0.4

                                candidates=[]
                                for j in range(window, len(buffer_segment) - window):
                                    mean_left=np.mean(buffer_segment[j - window:j])
                                    mean_right=np.mean(buffer_segment[j:j + window])
                                    diff=mean_right-mean_left
                                    if abs(diff) >= jump_threshold:
                                        candidates.append((j, np.sign(diff)))

                                paired_jumps=[]
                                i=0
                                while i<len(candidates)-1:
                                    idx1,dir1 = candidates[i]
                                    idx2,dir2 = candidates[i+1]
                                    if dir2==-dir1 and (idx2-idx1) <= max_jump_gap:
                                        paired_jumps.append((idx1, idx2))
                                        i+=2
                                    else:
                                        i+=1

                                if paired_jumps and is_bimodal(buffer_segment):
                                    global_jumps = [start+idx for pair in paired_jumps for idx in pair]
                                    all_buffer_jumps.extend(global_jumps)

                                start = end

                            if len(all_buffer_jumps)>2:
                                jumps_in_configs[cfg_name]=True
                                config_jump_counts[cfg_name]+=1

                        grade="A"
                        if jumps_in_configs["first"]:
                            grade="D"
                        elif jumps_in_configs["second"]:
                            grade="C"
                        elif jumps_in_configs["third"] or jumps_in_configs["fourth"]:
                            grade="B"

                        grading_results[(ln_path, socket_index, channel_index)] = grade
                        grade_counts[grade] += 1
                        total_waveforms += 1

                        print(f"{filename} | Sock {socket_index}, Ch {channel_index} → Grade: {grade}")

                        ## OPTIONAL: plot any waveforms that are graded B, C, or D
                        if grade != "A":
                          fig, axes = plt.subplots(4, 2, figsize=(14, 12))
                          fig.suptitle(f"{filename} | Sock {socket_index}, Ch {channel_index} | Grade {grade}", fontsize=14)
                      
                          for row_idx, (cfg_name, cfg_key) in enumerate(configs_to_check.items()):
                              if cfg_name not in raw_signals:
                                  axes[row_idx, 0].set_title(f"{cfg_name} (missing)")
                                  axes[row_idx, 1].axis("off")
                                  continue
                      
                              raw_sig = raw_signals[cfg_name]
                              hp_sig = hp_signals[cfg_name]
                              spy_buffer_lengths = spy_buffer_lengths_per_cfg[cfg_name]
                              colors = plt.cm.viridis(np.linspace(0, 1, len(spy_buffer_lengths)))
                      
                              # --- Raw (left column) ---
                              start = 0
                              for i, length in enumerate(spy_buffer_lengths):
                                  end = start + length
                                  axes[row_idx, 0].plot(range(start, end), raw_sig[start:end], color=colors[i], linewidth=0.8)
                                  start = end
                              axes[row_idx, 0].set_title(f"{cfg_name} (Raw)")
                              axes[row_idx, 0].grid(alpha=0.3)
                      
                              # --- High-pass (right column) ---
                              start = 0
                              for i, length in enumerate(spy_buffer_lengths):
                                  end = start + length
                                  axes[row_idx, 1].plot(range(start, end), hp_sig[start:end], color=colors[i], linewidth=0.8)
                                  start = end
                              axes[row_idx, 1].set_title(f"{cfg_name} (High-pass)")
                              axes[row_idx, 1].grid(alpha=0.3)
                      
                          for ax in axes.flatten():
                              ax.set_xlabel("Sample index")
                              ax.set_ylabel("ADC counts")
                      
                          plt.tight_layout(rect=[0, 0, 1, 0.95])
                          plt.show()
    

    csv_path = f"/home/cchinatti/grading_{top_folder}.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ln_folder", "socket", "channel", "grade"])
        for (ln_folder,socket,channel), grade in grading_results.items():
            writer.writerow([ln_folder,socket,channel, grade])

    print(f"\n=== SUMMARY for {top_folder} ===")
    print(f"Total waveforms analyzed: {total_waveforms}")
    for g, c in grade_counts.items():
        print(f"  Grade {g}: {c}")
    print("\nPaired jumps by configuration:")
    for cfg, count in config_jump_counts.items():
        print(f"  {cfg}: {count} waveforms with paired baseline jumps")

    print(f"\n Results saved to: {csv_path}")
