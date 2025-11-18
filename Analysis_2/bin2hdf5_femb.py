'''
Reads out Pickles binary files from FEMN QC testing and converts them to HDF5 files
by: C. Chinatti - September 2025
Works for the following binary files: 
   QC_PWR_Cycle_t2.bin
   QC_femb_adc_sync_pat_t15.bin
   QC_femb_rms_t5.bin
   QC_femb_leakage_cur_t3.bin
   QC_femb_test_pattern_pll_t16.bin
   logs_env.bin
   QC_Cali01_t6.bin
   QC_Cali02_t7.bin
   QC_Cali03_t8.bin
   QC_Cali04_t9.bin
   QC_Cali05_t13.bin
   QC_Cali06_t14.bin
   femb_chk_pulse_t4.bin

Input:
    - Binary file from the FEMB QC.
Output:
    - HDF5 format version of the binary file.

'''



import os, sys
import h5py, pickle
import numpy as np
import re

def read_bin(filename, path_to_file):
    # Loads and reads a Pickle binary file
    with open('/'.join([path_to_file, filename]), 'rb') as f:
        data = pickle.load(f)
        return data
    
def write_hdf5(f, data, group_name='/'):
    # Writes the data to the HDF5 file
    if group_name!='/':
        grp = f.create_group(group_name)
    else:
        grp = f
    for key,val in data.items():
        if isinstance(val, dict):
            if 'attrs' in key:
                for k, v in val.items():
                    grp.attrs[k] = v
            else:
                write_hdf5(f, val, group_name=f'{group_name}/{key}')
        else:
            grp.create_dataset(key,data=val)

def get_allKeys(data):
    # Extracts the keys that are specific to the item being tested during the QC.
    SpecificKeys_inBin = []
    GeneralKeys_inBin = []
    for key in data.keys():
        if isinstance(data[key], tuple) | isinstance(data[key], list):
        # if (key not in GeneralKeys_inBin) & (key != LogKey) & (key != 'QCstatus'):
            SpecificKeys_inBin.append(key)
        else:
            GeneralKeys_inBin.append(key)
    return SpecificKeys_inBin, GeneralKeys_inBin

def specKeyData2Dict(data, specKeys_list, PLL):
    # Converts structured experimental data from a list of specified keys into a nested dictionary format suitable for saving to an HDF5 file
    # Parameters:
    #     data (dict): Dictionary where each key corresponds to a speckey and holds a list of 4 elements:
    #         [0] rawdata (list)
    #         [1] power consumption data (dict)
    #         [2] FEMB info (list of tuples)
    #         [3] configuration data (list)
    #     specKeys_list (list): List of speckey strings to process from `data`

    # Returns:
    #     dict: A nested dictionary where each key corresponds to a speckey and contains processed data for rawdata, power consumption, FEMBs, and configuration

    all_data_dict = dict()
    for i, speckey in enumerate(specKeys_list):
        speckeyData = data[speckey]
        rawdata = speckeyData[0]
        
        all_spybuff = dict()
        N_spybuff = len(rawdata)
        for ispy_buff in range(N_spybuff):
            rawdata_dict = rawdata2numpy_dict(rawdata=rawdata, spy_buff=ispy_buff)
            all_spybuff[f'trigger{ispy_buff}'] = rawdata_dict

        #if PLL in title
        if PLL == False:
            config = speckeyData[2]
        else:
            config = speckeyData[1]

        if isinstance(config, list):
                config_dict1 = config2dict(config_data=config, num=0)
                config_dict2 = config2dict(config_data=config, num=1)
        

        if len(speckeyData) > 3:
            testinfo = speckeyData[3]
            testinfo_dict = testinfo

        N_triggers = len(list(all_spybuff.keys()))
        all_spybuff['attrs'] = {'N_trigger': N_triggers}
         
        pwrcons = speckeyData[1]
        if isinstance(pwrcons, dict):

            new_pwrcons_dict = {}
            grouped_vals = {}

            for key, val in pwrcons.items():
                if key.endswith('_V') or key.endswith('_I'):
                    base = key.rsplit('_', 1)[0]
                    if base not in grouped_vals:
                        grouped_vals[base] = {}
                    if key.endswith('_V'):
                        grouped_vals[base]['V'] = val
                    else:
                        grouped_vals[base]['I'] = val
                else:
                    continue

            for base, vi in grouped_vals.items():
                v = vi.get('V', None)
                i = vi.get('I', None)
                if v is not None and i is not None:
                    p = v * i
                    val_np = np.array((v, i, p), dtype=np.dtype([('V', np.float32), ('I', np.float32), ('P', np.float32)]))
                    new_pwrcons_dict[base] = val_np
            new_pwrcons_dict['attrs'] = {
                'Info': 'Power consumption for each ASIC for each power rail',
                'unit_V': 'V',
                'unit_I': 'mA',
                'unit_P': 'mW'
            }
            speckeyData_dict = {
                'pwrcons': new_pwrcons_dict,
                'rawdata': all_spybuff
            }
            speckeyData_dict['attrs'] = {
                'fembs': 'FEMBs used (tuple)',
                'pwrcons': 'Power consumption in the format np.array([V, I, P])',
                'rawdata': 'Spy buffer'
            }

        
        
        else:
            # speckeyData_dict = {'fembs': fembs_np, 'rawdata': all_spybuff}
            try: 
                speckeyData_dict = {'rawdata': all_spybuff,
                                'test_info': testinfo
                                }
            except:
                speckeyData_dict = {'rawdata': all_spybuff,
                                }

            speckeyData_dict['attrs'] = {'fembs': 'FEMBs used (tuple)',
                                         'rawdata': 'Spy buffer'}


        if isinstance(config, (list, dict)) and config_dict1 is not None:
            speckeyData_dict['config1'] = config_dict1
        if isinstance(config, (list, dict)) and config_dict2 is not None:
            speckeyData_dict['config2'] = config_dict2

        try:
            if isinstance(testinfo, (list, dict)) and testinfo_dict is not None:
                speckeyData_dict['test_info'] = testinfo_dict
        except:
            pass

        all_data_dict[speckey] = speckeyData_dict
    return all_data_dict

def config2dict(config_data, num):
    # Turns config data into a dictionary to be saved in hdf5 
    femb_id = config_data[num][0]
    adcs_paras = config_data[num][1]
    regs_int8 = config_data[num][2]
    adac_pls_en = config_data[num][3]
    # cd_sel = config_data[0][4]


    out_dict = {'femb_id': femb_id,
                'adc_paras': adcs_paras,
                'regs_int8': regs_int8,
                'adac_pls_en': adac_pls_en,
                # 'cd_sel': cd_sel
                }
    return out_dict

def rawdata2numpy_dict(rawdata, spy_buff=0):
    spy_buff_data = rawdata[spy_buff]

    # Converts raw spy buffer data for a given trigger (`spy_buff`) into a structured dictionary format

    # spy_buff_data is a tuple with 4 elements:
    #   a. 1st element: a list of buffers, e.g. [bytearray, None, None, None, None, None, None]
    #      - These correspond to raw binary data buffers for FEMBs, where some entries may be None.
    #   b. 2nd element: buf_end_addrs (integer, e.g. 0)
    #   c. 3rd element: spy_rec_ticks (integer, e.g. 32767)
    #   d. 4th element: trig_cmd (integer, e.g. 0)
    
    # Conversion steps:
    #   1. Convert the buffers in the 1st element to NumPy uint8 arrays (skip None entries)
    #   2. Keep the scalar metadata values as is (buf_end_addrs, spy_rec_ticks, trig_cmd)
    
    # The output dictionary will contain converted FEMB buffer data and the scalar metadata

    out_spy_buff_data = {'femb_data': {}, 'buf_end_addrs': 0, 'spy_rec_ticks': 0, 'trig_cmd': 0}
    params = {1: 'buf_end_addrs', 2: 'spy_rec_ticks', 3: 'trig_cmd'}
    for i_tmp, tmpdata in enumerate(spy_buff_data):
        if type(tmpdata)==list:
            data = tmpdata
            # out_data = []
            out_data = {}
            ifemb = 0
            for i, d in enumerate(data):
                if i%2 ==0:
                    if d==None:
                        # out_data[f'femb{ifemb}'] = {f'buff{i%2}': np.nan}
                        pass
                    else:     
                        # out_data[f'femb{ifemb}'] = {f'buff{i%2}': np.frombuffer(d, dtype=np.uint8)}
                        out_data = {f'buff{i%2}': np.frombuffer(d, dtype=np.uint8)}
                else:
                    if d==None:
                        # out_data[f'femb{ifemb}'][f'buff{i%2}'] = np.nan
                        pass
                    else:
                        # out_data[f'femb{ifemb}'][f'buff{i%2}'] = np.frombuffer(d, dtype=np.uint8)
                        out_data[f'buff{i%2}'] = np.frombuffer(d, dtype=np.uint8)
                    ifemb += 1

            # out_spy_buff_data['femb_data'] = out_data
            out_spy_buff_data = out_data
        else:
            out_spy_buff_data[params[i_tmp]] = tmpdata
        # out_spy_buff_data['attrs'] = {'N_fembs': 'Number of FEMBs used'}
    return out_spy_buff_data

def PWRON(pwron_data):
    out_pwron = dict()
    for key, val in pwron_data.items():
        pwrdtype = np.dtype([('V', np.float32), ('I', np.float32), ('P', np.float32)])
        out_pwron[key] = np.array(tuple(val), dtype=pwrdtype)
    out_pwron['attrs'] = {'Info': 'The power consumption is in the format (V, I, P)'}
    return out_pwron

def binWithoutRAW2dict(data, FileName='QC_MON.bin'):
    # Converts the data from the binary files without raw data to dictionary format
    # Input:
    #     data from the binary file, output of the read_bin function
    # Output:
    #     dictionary format of the input data
    speckey_list, GeneralKeys_inBin = get_allKeys(data=data)
    out_data = dict()
    if 'MON' in FileName:
        custom_dtype = np.dtype([(f'FE{ichip}', np.float32) for ichip in range(8)])
        for speckey in speckey_list:
            if speckey in ['VBGR', 'MON_Temper', 'MON_VBGR']:
                # custom_dtype = np.dtype([(f'FE{ichip}', np.float32) for ichip in range(8)])
                tmp_dict = {
                    'datas' : np.array(tuple(data[speckey][0]), dtype=custom_dtype), # ADC bit
                    'data_v': np.array(tuple(data[speckey][1]), dtype=custom_dtype) # in the unit of AD_LSB : datas*AD_LSB
                }
                out_data[speckey] = tmp_dict
            elif speckey in ['MON_200BL', 'MON_900BL']:
                # custom_dtype = np.dtype([(f'FE{ichip}', np.float32) for ichip in range(8)])
                tmp_dict = {f'CHN_{data[speckey][ichn][0]}' : np.array(tuple(data[speckey][ichn][1]), dtype=custom_dtype) for ichn in range(16)}
                out_data[speckey] = tmp_dict
            else:
                tmp_dict = {f'DAC_{data[speckey][idac][0]}' : np.array(tuple(data[speckey][idac][1]), dtype=custom_dtype) for idac in range(len(data[speckey]))}
                out_data[speckey] = tmp_dict
        for key in GeneralKeys_inBin:
            out_data[key] = data[key]
    else:
        keys_for_report = [key for key in GeneralKeys_inBin if ('QC' not in key) & ('WIB' not in key) & ('PC' not in key) & ('tms' not in key)]
        for key in keys_for_report:
                out_data[key] = data[key]
    return out_data

def bin2dict(data, PLL): 
    # Calls necessary functions to convert the data from the binary files to dictionary format
    # for binary files except QC.log and QC_MON.bin
    speckey_list, GeneralKeys_inBin = get_allKeys(data=data)
    out_data = dict()
    for key in GeneralKeys_inBin:
        if 'PWRON' in key:
            out_data[key] = PWRON(pwron_data=data[key])
        elif 'status' in key:
            pass
        else:
            out_data[key] = data[key]
    data = specKeyData2Dict(data=data, specKeys_list=speckey_list, PLL=PLL)
    for key, val in data.items():
        out_data[key] = val
    return out_data
    
def process_bin_files(bin_files, output_path):
    #Processes several Pickle files at once

    os.makedirs(output_path, exist_ok=True)

    for bin_file in bin_files:
        PLL = False
        binFileName = os.path.basename(bin_file)
        hdf5_name = binFileName.split('.')[0] + '.hdf5'
        print(f"Processing: {hdf5_name}")
        if 'pll' in binFileName:
            PLL = True

        output_file = os.path.join(output_path, hdf5_name)
        root_path = os.path.dirname(bin_file)

        try:
            data = read_bin(filename=binFileName, path_to_file=root_path)
        except Exception as e:
            # print(f"Failed to read {bin_file}: {e}")
            continue

        with h5py.File(output_file, 'w') as f:
            try:
                data_dict = bin2dict(data=data, PLL=PLL)
                print("len Dict:", len(data_dict))
                write_hdf5(f=f, data=data_dict)
            except Exception:
                # fallback if bin2dict fails
                print("fallback")
                data_dict = binWithoutRAW2dict(data=data, FileName=binFileName)
                write_hdf5(f=f, data=data_dict)

        print(f"Processed: {hdf5_name}")


## Example Useage
#  process_bin_files(bin_files, output_path)
