
import os
import sys
import re
import base64
import pathlib
import tqdm
import json
import yaml
import tempfile
import subprocess
import codecs
from oct2py import octave
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from osram_data_generator.src.osram import read_osram_csv

leddar_host_path = ''

INT16_MAX = np.iinfo(np.int16).max

def uint16_to_int16(series):
    return INT16_MAX - np.invert(np.asarray(series, dtype=np.uint16)).astype(np.int16)

def uint16_to_int14(series):
    arr = (INT16_MAX - np.invert(np.asarray(series, dtype=np.uint16)).astype(np.int16))
    return (arr / 4)

def uint14_to_int14(series):
    arr = (INT16_MAX - np.invert(np.asarray(series, dtype=np.uint16) * 4).astype(np.int16))
    return (arr / 4)

def int14_to_int14(series):
    arr = (np.asarray(series, dtype=np.uint16) * 4).astype(np.int16)
    return (arr / 4).astype(np.int16)

conversion = {
    'uint16_to_int16': uint16_to_int16,
    'uint16_to_int14': uint16_to_int14,
    'uint14_to_int14': uint14_to_int14,
    'int14_to_int14':  int14_to_int14 }

def read_test_platform_json(filepath, aggressor_position=0, type_conversion=None, baseline_prefix_length=64, noise_prefix_length=64):

    with open(filepath) as f:
        data = json.load(f)

    convert = conversion.get(type_conversion, lambda x: x)
    output_df_groups = []
    trace_id = 0
    acquisition_data = data['acquisition_data']

    segments = { 'PD0':  0, 'PD4':  1, 'PD8':  2, 'PD12':  3, 'PD16':  4, 'PD20':  5, 'PD24':  6, 'PD28':  7, 'PD32':  8, 'PD36':  9, 'PD40': 10, 'PD44': 11, 'PD48': 12, 'PD52': 13, 'PD56': 14, 'PD60': 15,
                 'PD1': 16, 'PD5': 17, 'PD9': 18, 'PD13': 19, 'PD17': 20, 'PD21': 21, 'PD25': 22, 'PD29': 23, 'PD33': 24, 'PD37': 25, 'PD41': 26, 'PD45': 27, 'PD49': 28, 'PD53': 29, 'PD57': 30, 'PD61': 31}

    for frame_id in acquisition_data.keys():
        frame_data = data['acquisition_data'][frame_id]

        for opttile_id in frame_data.keys():
            optical_tile_data = frame_data[opttile_id]

            for acqtile_id in optical_tile_data.keys():
                acquisition_tile_data = optical_tile_data[acqtile_id]

                for trigger_id in acquisition_tile_data.keys():
                    for pd_id in acquisition_tile_data[trigger_id].keys():
                        for scan_id in range(len(acquisition_tile_data[trigger_id][pd_id])):
                            trace_data = np.array(acquisition_tile_data[trigger_id][pd_id][scan_id])
                            trace     = convert(trace_data)
                            baseline  = np.mean(trace[:baseline_prefix_length])
                            noise     = np.std(trace[:noise_prefix_length], ddof=1)
                            df = pd.DataFrame({
                                "id":                 f"PD{segments[pd_id]}SCAN{scan_id}",
                                "aggressor_position": aggressor_position,
                                "segment":            segments[pd_id],
                                "scan":               scan_id,
                                "trace_idx":          trace_id,
                                "sample_idx":         np.arange(len(trace)),
                                "trace_argmax":       np.argmax(trace),
                                "trace":              trace,
                                'snr_raw':            10 * np.log10(((np.amax(trace) - baseline) / noise) ** 2),
                                "baseline":           baseline,
                                "noise":              noise })
                            trace_id += 1
                            output_df_groups.append(df)
    df_0 = pd.concat(output_df_groups)
    df_0['victim_offset'] = df_0['segment'] - df_0['aggressor_position']

    num_traces = len(df_0['trace_idx'].unique())
    trace_length = df_0.groupby(['trace_idx']).count()['id'].iloc[0]
    real_trace_idx = np.repeat(np.arange(num_traces), trace_length)
    df_0['trace_idx'] = real_trace_idx

    return df_0

coordinate_re = re.compile(r"PD(?P<horizontal>\d+)SCAN(?P<vertical>\d+)")

def read_osram_file(filename, type_conversion, aggressor_position, baseline_prefix_length, noise_prefix_length):
    osram_df = read_osram_csv(filename)
    convert = conversion.get(type_conversion, lambda x: x)
    trace_df = []
    trace_id = 0
    for idx, trace_data in osram_df.iterrows():
        packet_id = coordinate_re.match(trace_data["id"])
        trace     = convert(trace_data["trace"])
        #get epoch timestamp
        timestamp = round(datetime.datetime.utcnow().timestamp() * 1000)
        baseline  = np.mean(trace[:baseline_prefix_length])
        noise     = np.std(trace[:noise_prefix_length], ddof=1)
        df = pd.DataFrame({
            "id":                 trace_data["id"],
            "aggressor_position": aggressor_position,
            "segment":            int(packet_id.group("horizontal")) if packet_id.group("horizontal") is not None else np.nan,
            "scan":               int(packet_id.group("vertical")) if packet_id.group("vertical") is not None else np.nan,
            "trace_idx":          trace_id,
            "timestamp":          timestamp,
            "sample_idx":         np.arange(len(trace)),
            "trace_argmax":       np.argmax(trace),
            "trace":              trace,
            "baseline":           baseline,
            "noise":              noise })
        trace_id += 1
        trace_df.append(df[df["segment"] < 32])
    df_0 = pd.concat(trace_df)
    df_0['victim_offset'] = df_0['segment'] - df_0['aggressor_position']

    num_traces = len(df_0['trace_idx'].unique())
    trace_length = df_0.groupby(['trace_idx']).count()['id'].iloc[0]
    real_trace_idx = np.repeat(np.arange(num_traces), trace_length)
    df_0['trace_idx'] = real_trace_idx
    return df_0

def read_ltl_file(filepath, aggressor_position, adc_freq, temp_folder=None, baseline_prefix_length=8, noise_prefix_length=8):
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    if temp_folder is None:
        temp_folder = temp_folder or pathlib.Path(tempfile.TemporaryDirectory().name)
        temp_folder.mkdir()
    csv_output_path = temp_folder / f"{name}.csv"
    #raw_trace_process = [ "LeddarHost.exe", "-p", "LtPlug_EngTools", "-f", "ExportToCSV", "-file", filepath, "-dm", "8", "-out", csv_output_path]
    raw_trace_process = "{} -p LtPlug_EngTools -f ExportToCSV -file \"{}\" -dm 8 -out \"{}\" -v".format(pathlib.Path(leddar_host_path), pathlib.Path(filepath), pathlib.Path(csv_output_path))
    subprocess.Popen(raw_trace_process, stdout=subprocess.PIPE, stderr=subprocess.PIPE ).communicate()

    try:
        df = pd.read_csv(temp_folder / f"{name}_RawTraces.csv", sep=';', skiprows=2)
        (temp_folder / f"{name}_RawTraces.csv").unlink()
        df = df[df["Scanning index"] == 0]
        trace_sample_columns = [x for x in df.columns if x.startswith("Sample")]
        trace_id = 0
        trace_df_parts = []
        for idx, trace_data in df.iterrows():
            trace = trace_data[trace_sample_columns].to_numpy()
            timestamp = np.arange(len(trace)) * adc_freq
            baseline = np.mean(trace[:baseline_prefix_length])
            noise = np.std(trace[:noise_prefix_length], ddof=1)
            trace_df = pd.DataFrame({
                "id":                 trace_id,
                "aggressor_position": aggressor_position,
                "segment":            trace_data["Imaging index"],
                "scan":               trace_data["Scanning index"],
                "trace_idx":          trace_id,
                "timestamp":          timestamp,
                "sample_idx":         np.arange(len(trace)),
                "trace_argmax":       np.argmax(trace),
                "trace":              trace,
                "baseline":           baseline,
                "noise":              noise })
            trace_id += 1
            trace_df_parts.append(trace_df)
        df_0 = pd.concat(trace_df_parts)
        df_0 = df_0[df_0["scan"] == 0]
        df_0['victim_offset'] = df_0['segment'] - df_0['aggressor_position']
        num_traces = len(df_0['trace_idx'].unique())
        trace_length = df_0.groupby(['trace_idx']).count()['id'].iloc[0]
        real_trace_idx = np.repeat(np.arange(num_traces), trace_length)
        df_0['trace_idx'] = real_trace_idx
        return df_0
    except Exception as e:
        return None

def run_data_preprocessing(type_, dataset, output_file_path, storage_folder_path):
    data = yaml.load(open(dataset).read())
    processed_files = set([])

    if os.path.exists(output_file_path):
        with open(output_file_path, "r") as in_:
            previous_data = yaml.load(in_)
            if previous_data['data_files'] is not None:
                processed_files = set([x['file_name'] for x in previous_data['data_files']])
    else:
        with open(output_file_path, "w") as out:
            out.write("# OUTPUT FROM STAGE 1 PROCESSING\n\n")
            out.write("data_files:\n")

    temp_folder = pathlib.Path(tempfile.TemporaryDirectory().name)
    temp_folder.mkdir()

    for data_file in tqdm.tqdm(data):
        file_path = data_file['path']
        file_name = os.path.basename(file_path)
        root, ext = os.path.splitext(file_name)
        out_file = os.path.join(storage_folder_path, f"{root}.csv")
        if out_file in processed_files:
            try:
                pd.read_csv(out_file)
                continue
            except:
                pass

        if type_ == "mtl_bench":
            df = read_test_platform_json(data_file['path'], aggressor_position=data_file['aggressor_position'], type_conversion=None, baseline_prefix_length=64, noise_prefix_length=64)
        elif type_ == "osram":
            df = read_osram_file(data_file['path'], aggressor_position=data_file['aggressor_position'], type_conversion=data_file.get('type_conversion', None), baseline_prefix_length=64, noise_prefix_length=64)
        elif type_ == "cyclops":
            df = read_ltl_file(data_file['path'], aggressor_position=data_file['aggressor_position'], adc_freq=1.0 / 160.0e6, temp_folder=temp_folder)
        if df is not None:
            df.to_csv(out_file)
            with open(output_file_path, "a") as out:
                out.write(f"  - file_name: {str(out_file)}\n")
                out.write(f"    metadata: {{'aggressor_position': {data_file['aggressor_position']} }}\n")
                out.write(f"\n")

def write_input(df, fpga_tmp_in_folder):
    data_header_values = [
        "%time",
        "i_lca3_fpga_sys.i_dsp_fpga1.dsp0_expanded_sig",
        "i_lca3_fpga_sys.i_dsp_fpga1.dsp0_expanded_val",
        "i_lca3_fpga_sys.i_dsp_fpga1.dsp1_blinded_blafter",
        "i_lca3_fpga_sys.i_dsp_fpga1.dsp1_blinded_blbefore",
        "i_lca3_fpga_sys.i_dsp_fpga1.dsp1_blinded_noise",
        "i_lca3_fpga_sys.i_dsp_fpga1.dsp1_blinded_sig",
        "i_lca3_fpga_sys.i_dsp_fpga1.dsp1_blinded_val",
        "i_lca3_fpga_sys.i_dsp_fpga1.dsp2_mf_noise",
        "i_lca3_fpga_sys.i_dsp_fpga1.dsp2_mf_sig",
        "i_lca3_fpga_sys.i_dsp_fpga1.dsp2_mf_val",
        "i_lca3_fpga_sys.i_dsp_fpga1.dsp3_th_detind",
        "i_lca3_fpga_sys.i_dsp_fpga1.dsp3_th_sig",
        "i_lca3_fpga_sys.i_dsp_fpga1.dsp3_th_threshold",
        "i_lca3_fpga_sys.i_dsp_fpga1.dsp3_th_val",
        "i_lca3_fpga_sys.i_dsp_fpga1.dsp4_det_bin",
        "i_lca3_fpga_sys.i_dsp_fpga1.dsp4_det_mag",
        "i_lca3_fpga_sys.i_dsp_fpga1.dsp4_det_val",
        "i_lca3_fpga_sys.i_dsp_fpga1.dsp4_sati_bldiff",
        "i_lca3_fpga_sys.i_dsp_fpga1.dsp5_sat_bin",
        "i_lca3_fpga_sys.i_dsp_fpga1.dsp5_sat_mag",
        "i_lca3_fpga_sys.i_dsp_fpga1.dsp5_sat_val" ]
    data_header_str = ' '.join(data_header_values)

    with open(fpga_tmp_in_folder / f"input_0.dat", 'w') as out:
        out.write(f"{data_header_str}\n")
        traces = []
        padding = ' '.join(['0' for i in data_header_values[3:]])
        out.write(f"0 0 0 {padding}\n")
        for _, trace in df.groupby(['trace_idx']):
            time =  trace.sort_values(['sample_idx'])['timestamp'].to_numpy()
            sig =trace.sort_values(['sample_idx'])['trace'].to_numpy()
            entries = '\n'.join([f"{time} {sig} 1 {padding}" for (time, sig) in zip(time, sig)])
            traces.append(entries)
        out.write(f"\n0 0 0 {padding}\n".join(traces))
        out.write("\n")
    return fpga_tmp_in_folder / f"input_0.dat"

def create_fpga_output_dataframe(path):
    result_blinder    = pd.read_csv( path / f"result_blinder.csv", sep=';')
    result_cfar       = pd.read_csv( path / f"result_cfar.csv", sep=';')
    result_fir_filter = pd.read_csv( path / f"result_fir.csv", sep=';')

    return pd.DataFrame({
        'trace_idx':      result_blinder['trace'],
        'blinder_data':   result_blinder['data_o'],
        'cfar_data':      result_cfar['data_o'],
        'cfar_threshold': result_cfar['threshold'],
        'filtered_trace': result_fir_filter['data_o'] })

def create_fpga_pulse_dataframe(path):
    result_detect_max = pd.read_csv( path / f"result_detect_max.csv", sep=';')
    return result_detect_max.assign( trace_idx=result_detect_max['trace']).drop(['trace', 'last'], axis=1)

def clean_working_folder(path):
    p_0 = path / f"result_blinder.csv"
    p_1 = path / f"result_cfar.csv"
    p_2 = path / f"result_fir_filter.csv"
    p_3 = path / f"input_0.csv"

    for p in [p_0, p_1, p_2, p_3]:
        p.exists() and p.unlink()

def fpga_process(in_file, cwd, working_tmp_path, FPGA_SIMULATOR_EXEC_PATH, FPGA_SIMULATOR_CONFIG_PATH):
    clean_working_folder(working_tmp_path)
    os.chdir(str(working_tmp_path))

    df = None
    try:
        octave.eval(f"cd {str(working_tmp_path)}")
        octave.addpath(FPGA_SIMULATOR_EXEC_PATH.replace("\\", "/"))
        octave.fpga_model(str(in_file), FPGA_SIMULATOR_CONFIG_PATH.replace("\\", "/"))
        df = create_fpga_output_dataframe(working_tmp_path)
        pulses = create_fpga_pulse_dataframe(working_tmp_path)
        os.chdir(cwd)
        return df, pulses
    except Exception as e:
        print (e)
        os.chdir(cwd)
        return df, None

def run_fpga_preprocessing(input_file, output_file_path, storage_folder, reference_model_executable_path, reference_model_config_path):
    work_filename = input_file
    data = yaml.load(open(work_filename).read())
    files = data['data_files']
    processed_files = set([])

    cwd = os.getcwd()
    working_tmp_root = tempfile.TemporaryDirectory()
    working_tmp_path = pathlib.Path(working_tmp_root.name)

    fpga_tmp_in_folder = working_tmp_path / 'data'
    if not fpga_tmp_in_folder.exists():
        fpga_tmp_in_folder.mkdir()

    # in_folder = pathlib.Path(os.path.abspath(cl_arguments.in_folder))
    working_folder = pathlib.Path(storage_folder)
    if not working_folder.exists():
        working_folder.mkdir()

    if os.path.exists(output_file_path):
        with open(output_file_path, "r") as in_:
            previous_data = yaml.load(in_)
            if previous_data['data_files'] is not None:
                processed_files = set([x['file_name'] for x in previous_data['data_files']])
    else:
        with open(output_file_path, "w") as out:
            out.write("# OUTPUT FROM STAGE 1 PROCESSING\n\n")
            out.write("data_files:\n")

    for file_idx, file_data in tqdm.tqdm(list(enumerate(files))):
        awg_in_file  = pathlib.Path(file_data['file_name'])

        root, ext = os.path.splitext(awg_in_file.name)
        fpga_out_filename = f"{root}_post_fpga.csv"
        fpga_detection_filename = f"{root}_detections.csv"

        if fpga_out_filename in processed_files:
            continue

        fpga_output_file = (working_folder / fpga_out_filename)
        detections_output_file = (working_folder / fpga_detection_filename)

        if not fpga_output_file.parent.exists():
            fpga_output_file.parent.mkdir()

        if not detections_output_file.parent.exists():
            detections_output_file.parent.mkdir()

        awg_df = pd.read_csv(awg_in_file)
        in_file = write_input(awg_df, fpga_tmp_in_folder)

        fpga_df, detection_df = fpga_process(in_file, cwd, working_tmp_path, reference_model_executable_path, reference_model_config_path)
        data_df = awg_df.assign(
            blinder_data   = fpga_df['blinder_data'].to_numpy(),
            cfar_data      = fpga_df['cfar_data'].to_numpy(),
            cfar_threshold = fpga_df['cfar_threshold'].to_numpy(),
            filtered_trace = fpga_df['filtered_trace'].to_numpy())
        data_df['trace_idx']      += len(data_df['trace_idx'].unique()) * file_idx
        detection_df['trace_idx'] += len(data_df['trace_idx'].unique()) * file_idx

        data_df.to_csv(fpga_output_file)
        detection_df.to_csv(detections_output_file)

        with open(output_file_path, "a") as out:
            out.write(f"  - file_name: {str(fpga_output_file)}\n")
            out.write(f"    detections_file_name: {str(detections_output_file)}\n")
            out.write(f"    metadata: {{'aggressor_position': {file_data['metadata']['aggressor_position']} }}\n")
            out.write(f"\n")


def correct_saturated_amplitude(amplitude, base_level, base_level_diff, BL_DIFF_MAG_ARR, CORR_MAG_ARR):
    SCALING_FACTOR = 1
    # BL_DIFF_MAG_ARR = [(9.695 * SCALING_FACTOR),(16.06 * SCALING_FACTOR),(22.72 * SCALING_FACTOR),(35.33 * SCALING_FACTOR),(64.93 * SCALING_FACTOR),(90.06 * SCALING_FACTOR),(91.64 * SCALING_FACTOR),(195.8 * SCALING_FACTOR),(209.4 * SCALING_FACTOR),(226.1 * SCALING_FACTOR),(1054.0 * SCALING_FACTOR),(1909.0 * SCALING_FACTOR),(2974.0 * SCALING_FACTOR),(4821.0 * SCALING_FACTOR),(5789.0 * SCALING_FACTOR),(6012.0 * SCALING_FACTOR)]
    # CORR_MAG_ARR = [(4066.0 * SCALING_FACTOR),(5350.0 * SCALING_FACTOR),(6794.0 * SCALING_FACTOR),(7864.0 * SCALING_FACTOR),(13620.0 * SCALING_FACTOR),(22350.0 * SCALING_FACTOR),(35130.0 * SCALING_FACTOR),(610500.0 * SCALING_FACTOR),(1181000.0 * SCALING_FACTOR),(2198000.0 * SCALING_FACTOR),(6770000.0 * SCALING_FACTOR),(9601000.0 * SCALING_FACTOR),(13090000.0 * SCALING_FACTOR),(16620000.0 * SCALING_FACTOR),(19240000.0 * SCALING_FACTOR),(22090000.0 * SCALING_FACTOR)]
    if (base_level_diff <= BL_DIFF_MAG_ARR[0]):
        return CORR_MAG_ARR[0] + base_level

    if (base_level_diff >= BL_DIFF_MAG_ARR[-1]):
        return CORR_MAG_ARR[-1] + base_level

    idx = (np.abs(BL_DIFF_MAG_ARR - base_level_diff)).argmin()
    if (base_level_diff >= BL_DIFF_MAG_ARR[idx]):
        idx0 = idx
        idx1 = idx + 1
    else:
        idx0 = idx - 1
        idx1 = idx

    xlut = CORR_MAG_ARR[idx0] + (CORR_MAG_ARR[idx1] - CORR_MAG_ARR[idx0]) / (BL_DIFF_MAG_ARR[idx1] - BL_DIFF_MAG_ARR[idx0]) * (base_level_diff - BL_DIFF_MAG_ARR[idx0])
    return xlut + base_level

def get_corrected_amplitude(amplitude, raw_trc, bl_diff_mag, corr_mag):
    # Is this pulse saturated (max of int14) ?
    if np.max(raw_trc) >= 8050:
        # Get the sample position of the first saturation
        idx = np.where(raw_trc >= 8050)
        first_clip = idx[0][0]
        last_clip  = idx[0][len(idx[0])-1]
        try:
            base_level_before_saturation = int(np.average(raw_trc[first_clip-11:first_clip-3]))
            base_level_after_saturation = int(np.average(raw_trc[last_clip+30:last_clip+38]))
            bl_diff = np.abs(base_level_after_saturation - base_level_before_saturation)
            # print(raw_trc, first_clip, last_clip, base_level_before_saturation, base_level_after_saturation, bl_diff)
            return correct_saturated_amplitude(amplitude, base_level_before_saturation, bl_diff, bl_diff_mag, corr_mag)
        except Exception as e:
            return -1
    else :
        return amplitude


def run_trace_metadata(input_file, output_file, template_span_pre, template_span_post, bl_diff_mag, corr_mag):
    work_filename = input_file
    data = yaml.load(open(work_filename).read())
    files = data['data_files']

    rows_0 = []
    distance_resolution = 1.0
    T_x = [f"T_{i}" for i in range(template_span_pre + template_span_post + 1)]

    for file_idx, file_data in tqdm.tqdm(list(enumerate(files))):

        trace_df     = pd.read_csv(file_data['file_name'])
        detection_df = pd.read_csv(file_data['detections_file_name'])
        agg          = file_data['metadata']['aggressor_position']

        aggressor_amplitude_df = trace_df[trace_df["segment"] == agg].merge(detection_df, on=['trace_idx'], how='left')
        unnamed = [x for x in aggressor_amplitude_df.columns if x.startswith("Unnamed")]
        aggressor_amplitude_df = aggressor_amplitude_df.drop([*unnamed, "id"], axis=1).groupby(['trace_idx', "sample_idx"]).first().reset_index().dropna()
        trace_ids = aggressor_amplitude_df[aggressor_amplitude_df["segment"] == agg]['trace_idx'].unique()
        aggressor_amplitude_df = aggressor_amplitude_df.set_index(['trace_idx', "sample_idx"])

        for trace_id in trace_ids:
            data = aggressor_amplitude_df.loc[trace_id]
            raw_trace = data['trace'].to_numpy()
            fpga_amplitude = data["det_mag"].iloc[0]
            fpga_corrected_amplitude = get_corrected_amplitude(fpga_amplitude, raw_trace, bl_diff_mag, corr_mag)
            aggressor_amplitude_df.loc[(trace_id,), "det_mag"] = fpga_corrected_amplitude

        aggressor_df = aggressor_amplitude_df.reset_index()                    \
            .groupby('scan')                                                   \
            .agg( trace_idx = ("trace_idx", "first"), det_mag = ("det_mag", "first") )

        trace_df['snr_raw'] = 0
        trace_df = trace_df                                                                                                          \
            .merge(aggressor_df, on=['scan'], how='left')                                                                            \
            .drop([ "id", "snr_raw", "noise", "trace_argmax", "blinder_data", "cfar_data", "cfar_threshold"], axis=1) \
            .rename(columns={"det_mag": "aggressor_amplitude"})

        unnamed = [x for x in trace_df.columns if x.startswith("Unnamed")]
        trace_df = trace_df.drop(unnamed, axis=1).dropna()

        for agg_amplitude, trace_data in list(trace_df.groupby(['aggressor_amplitude', 'victim_offset'])):

            raw_trace    = trace_data['trace'].to_numpy()
            trace        = trace_data['filtered_trace'].to_numpy().astype(np.float64)
            trace_argmax = np.argmax(trace)

            if ((trace_argmax + template_span_post) >= len(trace)-1) or ((trace_argmax - template_span_pre) < 0):
                continue

            trace_baseline   = np.mean(trace[:8])
            trace_noise      = np.std(raw_trace[:8])
            trace_normalized = trace[(trace_argmax - template_span_pre):(trace_argmax + template_span_post + 1)]

            trace_columns = { T_x[i]: v for i, v in enumerate(trace_normalized) }
            row = {
                "aggressor_position":  trace_data['aggressor_position'].iloc[0],
                "victim_offset":       trace_data['victim_offset'].iloc[0],
                "aggressor_amplitude": trace_data['aggressor_amplitude'].iloc[0],
                "baseline":            trace_baseline,
                "raw_trace_std":       trace_noise,
                **trace_columns }
            rows_0.append(row)
    df = pd.DataFrame(rows_0)
    df.to_csv(output_file)

def filter_templates(input_file, output_file, noise_threshold, template_lut, victim_offsets):
    df = pd.read_csv(input_file).dropna()
    df = df[(df["victim_offset"].isin(victim_offsets))]
    df = df[df['raw_trace_std'] < noise_threshold]

    df_out = []

    for aggressor_position in tqdm.tqdm(template_lut):
        template_idx = template_lut[aggressor_position]
        template_data = df[df['aggressor_position'] == template_idx].copy()
        template_data["aggressor_position"] = aggressor_position
        template_data["victim_position"] = template_data["aggressor_position"] + template_data["victim_offset"]
        template_data = template_data[template_data["victim_position"] >= 0]
        template_data = template_data[template_data["victim_position"] < 32]
        df_out.append(template_data)

    df_out = pd.concat(df_out).sort_values(["aggressor_position", "victim_offset", "aggressor_amplitude"]).drop(["Unnamed: 0", "victim_position"], axis=1)
    df_out.to_csv(output_file)


def MATH_ParabolicInterpCVe( aTrc, aDistResol ):
    lDenom = 0
    lNum   = 0
    lNum2  = 0
    interpolated_distance = aDistResol
    lDenom = (aTrc[1] - aTrc[0]) + (aTrc[1] - aTrc[2])
    lNum = aTrc[1] - aTrc[0]
    if (lDenom > 0):
        lNum2 = lNum * aDistResol
        interpolated_distance += (lNum2 / lDenom)
        interpolated_distance -= aDistResol / 2

    if abs(interpolated_distance) > 100:
        print(f"\n\n\n{interpolated_distance}, {lNum}, {lDenom}, {aTrc}")
    return interpolated_distance

def OffTpl(NormTpl):
    offset     = abs(NormTpl[0] - NormTpl[-1])/2
    MimMax     = np.array([NormTpl[0], NormTpl[-1]])
    pos        = np.argmin(np.abs(MimMax))
    if(NormTpl[0] >= 0 and NormTpl[-1] >= 0):
        offset = 0 - (offset + MimMax[pos])

    if(NormTpl[0] <= 0 and NormTpl[-1] >= 0):
        offset = 0 - (offset + MimMax[pos])

    if(NormTpl[0] >= 0 and NormTpl[-1] <= 0):
        offset = 0 - (offset + MimMax[pos])

    if(NormTpl[0] <= 0 and NormTpl[-1] <= 0):
        offset = (offset + np.abs(MimMax[pos]))

    return offset


def compute_templates(input_file, output_file):
    df = pd.read_csv(input_file)

    df = df.dropna()

    bin_count = 20
    T_x = [f"T_{x}" for x in range(23)]

    out_df_rows = []

    distance_resolution = 1.0

    for offset, data in tqdm.tqdm(df.groupby(['aggressor_position', 'victim_offset'])):
        df = data.drop(['aggressor_position', 'victim_offset'], axis=1).sort_values(['aggressor_amplitude'])
        rows = df.shape[0]
        rows_per_bin    = rows // bin_count
        rows_to_discard = rows %  bin_count
        discard = [*[False for i in range(rows_to_discard)], *[True for i in range(rows_per_bin * bin_count)]]
        df['discard'] = discard
        df = df[(df['discard'])].drop(['discard'], axis=1)
        df['bin'] = np.repeat(np.arange(bin_count), rows_per_bin)
        tpl = df.groupby(['bin']).mean()
        tpl['aggressor_position'] = offset[0]
        tpl['victim_offset'] = offset[1]

        for idx in range(len(tpl.index)):
            trace_normalized     = tpl.iloc[idx][T_x].to_numpy()
            trace_normalized    += OffTpl(trace_normalized)
            trace_max            = np.amax(trace_normalized)
            trace_min            = np.amin(trace_normalized)
            trace_normalized     = trace_normalized / (trace_max - trace_min)
            normalized_trace_max = np.amax(trace_normalized)
            normalized_trace_min = np.amin(trace_normalized)
            trace_columns = { T_x[i]: v for i, v in enumerate(trace_normalized) }

            maxi = template_span_pre
            template_distance  = MATH_ParabolicInterpCVe(trace_normalized[(maxi-1):(maxi+2)], distance_resolution)
            template_distance += (template_span_pre - 1) * distance_resolution

            row = {
                "aggressor_position":         int(tpl.iloc[idx]['aggressor_position']),
                "victim_offset":              int(tpl.iloc[idx]['victim_offset']),
                "aggressor_amplitude":        int(tpl.iloc[idx]['aggressor_amplitude']),
                'bin':                        tpl.index[idx],
                'template_offset':            template_distance,
                "normalized_trace_min":       normalized_trace_min,
                "normalized_trace_max":       normalized_trace_max,
                "normalized_trace_amplitude": (normalized_trace_max - normalized_trace_min),
                **trace_columns
            }
            out_df_rows.append(row)

    out_df = pd.DataFrame(out_df_rows)
    out_df = out_df \
            .reset_index()             \
            .set_index(['aggressor_position', 'victim_offset', 'bin'])

    out_df.to_csv(output_file)
    print(out_df)

def create_skeleton(aggressor_position=None, victim_offsets=None, bin_count=1):
    AGG_POS    = pd.DataFrame({"idx": 1, "aggressor_position": np.array(aggressor_position)})
    VIC_OFFSET = pd.DataFrame({"idx": 1, "victim_offset":   np.array(victim_offsets)})
    BIN        = pd.DataFrame({"idx": 1, "bin": np.arange(bin_count)})
    skeleton   = AGG_POS.merge(VIC_OFFSET.merge(BIN, on=["idx"]), on=["idx"]).drop(["idx"], axis=1)
    return skeleton.sort_values(by=["aggressor_position", "victim_offset", "bin"])

def scale(value, scaling_factor, type_):
    return type_(value*scaling_factor)

def generate_json(filePath, template_input_path, definitions_header_path, template_length, bin_count, distance_scale_bits, amplitude_scale_bits, victim_offsets, aggressor_positions):
    distance_scale       = 2 ** distance_scale_bits
    amplitude_scale      = 2 ** amplitude_scale_bits
    tile_data            = {}
    aggressor_data       = []



    for LedPower in range(0, len(template_input_path)):
        print("Generate json Ledpower {}...".format(LedPower))
        data = pd.read_csv(template_input_path[LedPower])
        sk = create_skeleton(aggressor_position=aggressor_positions, victim_offsets=victim_offsets, bin_count=bin_count)
        data = sk.merge(data, on=["aggressor_position", "victim_offset", "bin"], how="left").fillna(0.0)
        T_x = [f"T_{x}" for x in range(template_length)]

        #Add clear template structure command
        extalkTplClearCommand = {}
        extalkTplClearCommand["extalkTplClear"] = LedPower
        aggressor_data.append(extalkTplClearCommand)

        ChannelList = {}
        ChannelList["extalkTplLedPower"]                 = LedPower
        ChannelList["extalkTplAggressorChList"]          = [x for x in aggressor_positions]
        ChannelList["extalkTplVictimChList"]             = [x for x in victim_offsets]
        aggressor_data.append(ChannelList)

        for aggressor_position, grp_0 in tqdm.tqdm(data.groupby(["aggressor_position"])):
            VictimPos = 0
            for victim_offset, grp in grp_0.groupby(['victim_offset']):
                #template_struct_data = []
                aggAmpPos = 0
                for template in grp.sort_values(['aggressor_amplitude']).iterrows():

                    template_values = template[1]
                    template_samples = [scale(x, amplitude_scale, int) for x in template_values[T_x].tolist()]

                    real_template_length = template_length
                    if np.linalg.norm(template_values[T_x].tolist()) == 0:
                        real_template_length = 0

                    victim_offset_data = {}
                    victim_offset_data["extTLP"]        = LedPower
                    victim_offset_data["extTAP"]        = aggressor_position
                    victim_offset_data["extTVP"]        = VictimPos
                    victim_offset_data["extTAAP"]       = aggAmpPos
                    #extalkTplVictimOffset
                    if victim_offset != 0:
                        victim_offset_data["extTVO"]        = victim_offset
                    #extalkTplAggressorAmplitudeLowerBound
                    AggressorAmplitudeLowerBound = scale(template_values['aggressor_amplitude'], 1, int)
                    if AggressorAmplitudeLowerBound != 0:
                        victim_offset_data["extTAALB"]      = AggressorAmplitudeLowerBound
                    #extalkTplPeakOffset
                    PeakOffset = scale(template_values['template_offset'], distance_scale, int)
                    if PeakOffset != 0:
                        victim_offset_data["extTPO"]        = PeakOffset
                    if real_template_length != 0:
                        victim_offset_data["extTL"]         = real_template_length
                        victim_offset_data["extTD"]         = template_samples
                    aggressor_data.append(victim_offset_data)
                    aggAmpPos += 1
                VictimPos += 1

        #Add copy to all tiles command
        extalkTplCopyTilesCommand = {}
        extalkTplCopyTilesCommand["extalkTplCopy"] = {
            "samIdx": LedPower,
            "src" : 0
            #"dest" : [1] #Comment to copy to all tiles
        }
        aggressor_data.append(extalkTplCopyTilesCommand)

    json.dump(aggressor_data, codecs.open(filePath, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=False, indent=None)


    with open(definitions_header_path, 'w') as out:
        out.write("/******** Electronic Xtalk templates ********/\n")
        out.write("\n")
        out.write("// This file is automatically generated and should not be edited by hand.\n")
        out.write("// Editing this file by hand may cause unexpected behaviour. Furthermore, \n")
        out.write("// all manual modifications will be overwritten when the file is regenerated.\n")
        out.write("\n")
        out.write("\n")
        out.write("#include <stdint.h>\n")
        out.write("\n")
        out.write(f"#define EXTALK_TEMPLATE_LOOKUP_MAX_BINARY_SEARCH_ITERATION_COUNT 8\n")
        out.write(f"#define EXTALK_TEMPLATE_LOOKUP_MAX_TEMPLATE_LENGTH           {template_length}\n")
        out.write(f"#define EXTALK_TEMPLATE_LOOKUP_AGGRESSOR_AMPLITUDE_BIN_COUNT {bin_count}\n")
        out.write(f"#define EXTALK_TEMPLATE_LOOKUP_AGGRESSOR_POSITION_COUNT      {len(aggressor_positions)}\n")
        out.write(f"#define EXTALK_TEMPLATE_LOOKUP_VICTIM_OFFSET_COUNT           {len(victim_offsets)}\n")
        out.write(f"#define EXTALK_TEMPLATE_LOOKUP_DISTANCE_SCALE                ((int32_t)(1 << {distance_scale_bits}))\n")
        out.write(f"#define EXTALK_TEMPLATE_LOOKUP_AMPLITUDE_SCALE               ((int32_t)(1 << {amplitude_scale_bits}))\n")
        out.write("\n")

    print("Json Tpl Done")

def generate_json_2(filePath, template_input_path, definitions_header_path, template_length, bin_count, distance_scale_bits, amplitude_scale_bits, victim_offsets, aggressor_positions):
    distance_scale       = 2 ** distance_scale_bits
    amplitude_scale      = 2 ** amplitude_scale_bits
    tile_data            = {}
    aggressor_data       = []

    for LedPower in range(0, len(template_input_path)):
        print("Generate json Ledpower {}...".format(LedPower))
        data = pd.read_csv(template_input_path[LedPower])
        sk = create_skeleton(aggressor_position=aggressor_positions, victim_offsets=victim_offsets, bin_count=bin_count)
        data = sk.merge(data, on=["aggressor_position", "victim_offset", "bin"], how="left").fillna(0.0)
        T_x = [f"T_{x}" for x in range(template_length)]

        aggressor_data.append({"extalkTplClear": 0})


        ChannelList = {}
        ChannelList["extalkTplLedPower"]                 = 0
        ChannelList["extalkTplAggressorChList"]          = [x for x in aggressor_positions]
        ChannelList["extalkTplVictimChList"]             = [x for x in victim_offsets]
        aggressor_data.append(ChannelList)
        for aggressor_position, grp_0 in tqdm.tqdm(data.groupby(["aggressor_position"])):
            VictimPos = 0
            for victim_offset, grp in grp_0.groupby(['victim_offset']):
                #template_struct_data = []
                aggAmpPos = 0
                for template in grp.sort_values(['aggressor_amplitude']).iterrows():

                    template_values = template[1]
                    template_samples = [scale(x, amplitude_scale, int) for x in template_values[T_x].tolist()]

                    real_template_length = template_length
                    if np.linalg.norm(template_values[T_x].tolist()) == 0:
                        real_template_length = 0

                    victim_offset_data = {}
                    victim_offset_data["extalkTplLedPower"]                      = LedPower
                    victim_offset_data["extalkTplAggressorPosition"]             = aggressor_position
                    victim_offset_data["extalkTplVictimPosition"]                = VictimPos
                    victim_offset_data["extalkTplAggressorAmpPosition"]          = aggAmpPos
                    victim_offset_data["extalkTplVictimOffset"]                  = victim_offset
                    victim_offset_data["extalkTplAggressorAmplitudeLowerBound"]  = scale(template_values['aggressor_amplitude'], 1, int)
                    victim_offset_data["extalkTplPeakOffset"]                    = scale(template_values['template_offset'], distance_scale, int)
                    victim_offset_data["extalkTplLength"]                        = real_template_length
                    victim_offset_data["extalkTplData"]                          = template_samples
                    aggressor_data.append(victim_offset_data)
                    aggAmpPos += 1
                VictimPos += 1
    tile_data["allTile"] = aggressor_data
    json.dump(tile_data, codecs.open(filePath, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=False, indent=None)


def generate_headers(template_input_path, definitions_header_path, data_header_path, template_length, bin_count, distance_scale_bits, amplitude_scale_bits, victim_offsets, aggressor_positions):
    distance_scale       = 2 ** distance_scale_bits
    amplitude_scale      = 2 ** amplitude_scale_bits

    data = pd.read_csv(template_input_path)
    sk = create_skeleton(aggressor_position=aggressor_positions, victim_offsets=victim_offsets, bin_count=bin_count)
    data = sk.merge(data, on=["aggressor_position", "victim_offset", "bin"], how="left").fillna(0.0)
    T_x = [f"T_{x}" for x in range(template_length)]

    aggressor_data = []
    for aggressor_position, grp_0 in tqdm.tqdm(data.groupby(["aggressor_position"])):
        victim_offset_data = []
        for victim_offset, grp in grp_0.groupby(['victim_offset']):
            template_struct_data = []
            for template in grp.sort_values(['aggressor_amplitude']).iterrows():
                template_values = template[1]
                template_samples = [f'{scale(x, amplitude_scale, int)}' for x in template_values[T_x].tolist()]

                real_template_length = template_length
                if np.linalg.norm(template_values[T_x].tolist()) == 0:
                    real_template_length = 0

                template_c_array = f"{{ {', '.join(template_samples)} }}"
                template_data_string = f"{{ {scale(template_values['aggressor_amplitude'], 1, int)}, {scale(template_values['template_offset'], distance_scale, int)}, {real_template_length}, {template_c_array}}}"
                template_struct_data.append(template_data_string)
            victim_template_data = f"{{{int(aggressor_position)}, {int(victim_offset)}, {{{', '.join(template_struct_data)}}}}}"
            victim_offset_data.append(victim_template_data)
        aggressor_data.append(victim_offset_data)

    with open(definitions_header_path, 'w') as out:
        out.write("/******** Electronic Xtalk templates ********/\n")
        out.write("\n")
        out.write("// This file is automatically generated and should not be edited by hand.\n")
        out.write("// Editing this file by hand may cause unexpected behaviour. Furthermore, \n")
        out.write("// all manual modifications will be overwritten when the file is regenerated.\n")
        out.write("\n")
        out.write("\n")
        out.write("#include <stdint.h>\n")
        out.write("\n")
        out.write(f"#define EXTALK_TEMPLATE_LOOKUP_MAX_BINARY_SEARCH_ITERATION_COUNT 8\n")
        out.write(f"#define EXTALK_TEMPLATE_LOOKUP_MAX_TEMPLATE_LENGTH           {template_length}\n")
        out.write(f"#define EXTALK_TEMPLATE_LOOKUP_AGGRESSOR_AMPLITUDE_BIN_COUNT {bin_count}\n")
        out.write(f"#define EXTALK_TEMPLATE_LOOKUP_AGGRESSOR_POSITION_COUNT      {len(aggressor_positions)}\n")
        out.write(f"#define EXTALK_TEMPLATE_LOOKUP_VICTIM_OFFSET_COUNT           {len(victim_offsets)}\n")
        out.write(f"#define EXTALK_TEMPLATE_LOOKUP_DISTANCE_SCALE                ((int32_t)(1 << {distance_scale_bits}))\n")
        out.write(f"#define EXTALK_TEMPLATE_LOOKUP_AMPLITUDE_SCALE               ((int32_t)(1 << {amplitude_scale_bits}))\n")
        out.write("\n")

    with open(data_header_path, 'w') as out:
        out.write("/******** Electronic Xtalk templates ********/\n")
        out.write("\n")
        out.write("// This file is automatically generated and should not be edited by hand.\n")
        out.write("// Editing this file by hand may cause unexpected behaviour. Furthermore, \n")
        out.write("// all manual modifications will be overwritten when the file is regenerated.\n")
        out.write("\n")
        out.write("\n")
        out.write("#include <stdint.h>\n")
        out.write("\n")
        out.write(f"#define EXTALK_AGGRESSOR_CHANNEL_LIST {{ {', '.join([str(x) for x in aggressor_positions])} }}\n")
        out.write(f"#define EXTALK_VICTIM_OFFSET_LIST {{ {', '.join([str(x) for x in victim_offsets])} }}\n")
        out.write("\n")
        out.write("#define EXTALK_AGGRESSOR_TEMPLATE_DATA { \\")
        out.write("\n")

        indent = "    "
        for agg_idx, aggressor_data_row in enumerate(aggressor_data):
            out.write(f"{indent}{{ \\\n")
            for vic_idx, victim_row in enumerate(aggressor_data_row):
                out.write(f"{indent*2}{victim_row}")
                out.write(",\\\n" if vic_idx < len(aggressor_data_row) - 1 else " \\\n")
            out.write(f"{indent}}}")
            out.write(",\\\n" if agg_idx < len(aggressor_data) - 1 else "\\\n" )
        out.write("}\n")


def exit_if_one_is_none(lst):
    if None in lst:
        sys.exit(1)

if __name__ == '__main__':
    data_folder = pathlib.Path("./")
    #data_file = data_folder / "template_data_cyclops.yaml"
    data_file = data_folder / "template_data_osram.yaml"
    data = yaml.load(open(data_file).read())
    processes = data['process']

    for process in processes:
        process_name = list(process.keys())[0]
        process_data = process[process_name]

        if process_name == "preprocessing":
            type_               = process_data.get( 'type', None )
            leddar_host_path    = ("\"{}\"").format(process_data.get( 'LeddarHostPath', None ))
            skip                = process_data.get( 'skip', False )
            dataset             = process_data.get( 'dataset', None )
            storage_folder_path = process_data.get( 'storage_folder', None )
            output_file_path    = process_data.get( 'output_file', None )
            skip_if_output_file_exists = process_data.get( 'skip_if_output_file_exists', False )

            print("Preprocessing...")
            exit_if_one_is_none([dataset, storage_folder_path, output_file_path])
            if skip:
                continue

            if not os.path.exists(storage_folder_path):
                os.makedirs(storage_folder_path)

            if not os.path.exists(os.path.dirname(output_file_path)):
                os.makedirs(os.path.dirname(output_file_path))

            run_data_preprocessing(type_, dataset, output_file_path, storage_folder_path)
        elif process_name == "fpga-processing":
            skip = process_data.get( 'skip', False )
            reference_model_executable_path = process_data.get( 'reference_model_executable_path', None )
            reference_model_config_path = process_data.get( 'reference_model_config_path', None )
            input_file = process_data.get( 'input_file', None )
            storage_folder = process_data.get( 'storage_folder', None )
            output_file_path = process_data.get( 'output_file', False )
            skip_if_output_file_exists = process_data.get( 'skip_if_output_file_exists', False )

            print("FPGA Preprocessing...")
            exit_if_one_is_none([reference_model_executable_path, reference_model_config_path, input_file, storage_folder, output_file_path])
            if skip:
                continue

            if not os.path.exists(storage_folder_path):
                os.makedirs(storage_folder_path)

            if not os.path.exists(os.path.dirname(output_file_path)):
                os.makedirs(os.path.dirname(output_file_path))

            if not os.path.exists(input_file):
                sys.exit(1)

            run_fpga_preprocessing(input_file, output_file_path, storage_folder, reference_model_executable_path, reference_model_config_path)

        elif process_name == "compute-trace-metadata":
            skip = process_data.get( 'skip', False )
            saturation_calibration_file = process_data.get("saturation_calibration", None)

            if saturation_calibration_file is not None:
                saturation_calibration = json.loads(open(saturation_calibration_file).read())
                bl_diff_mag = saturation_calibration["allTile"]["sat2Calibs"]["bl_diff_mag"]
                corr_mag = saturation_calibration["allTile"]["sat2Calibs"]["corr_mag"]
            else:
                bl_diff_mag = process_data.get("bl_diff_mag", None)
                corr_mag = process_data.get("corr_mag", None)

            input_file = process_data.get( 'input_file', None )
            output_file = process_data.get( 'output_file', None )
            template_span_pre = process_data.get( 'template_span_pre', 1 )
            template_span_post = process_data.get( 'template_span_post', 1 )

            print("Trace processing 1...")
            exit_if_one_is_none([input_file, output_file])
            if skip:
                continue

            if not os.path.exists(input_file):
                sys.exit(1)

            if not os.path.exists(os.path.dirname(output_file_path)):
                os.makedirs(os.path.dirname(output_file_path))

            run_trace_metadata(input_file, output_file, template_span_pre, template_span_post, bl_diff_mag, corr_mag)

        elif process_name == "filter-templates":
            skip = process_data.get( 'skip', False )
            template_lut = process_data.get( 'template_lut', { i:i for i in range(32) } )
            victim_offsets = process_data.get( 'victim_offsets', [] )
            template_noise_threshold = process_data.get( 'template_noise_threshold', 1000 )
            input_file = process_data.get( 'input_file', None )
            output_file = process_data.get( 'output_file', None )

            print("Trace processing 2...")
            exit_if_one_is_none([input_file, output_file])
            if skip:
                continue

            if not os.path.exists(input_file):
                sys.exit(1)

            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))

            filter_templates(input_file, output_file, template_noise_threshold, template_lut, victim_offsets)

        elif process_name == "compute-templates":
            skip = process_data.get( 'skip', False )
            input_file = process_data.get( 'input_file', None )
            output_file = process_data.get( 'output_file', None )

            print("Trace processing 2...")
            exit_if_one_is_none([input_file, output_file])
            if skip:
                continue

            if not os.path.exists(input_file):
                sys.exit(1)

            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))

            compute_templates(input_file, output_file)

        elif process_name == "generate-headers":
            skip = process_data.get( 'skip', False )
            genJson = process_data.get( 'json', True )
            template_length = process_data.get( 'template_length', None )
            bin_count = process_data.get( 'bin_count', None )
            distance_scale_bits = process_data.get( 'distance_scale_bits', None )
            amplitude_scale_bits = process_data.get( 'amplitude_scale_bits', None )
            victim_offsets = process_data.get( 'victim_offsets', None )
            aggressor_positions = process_data.get( 'aggressor_positions', None )
            template_input_path = process_data.get( 'template_input_path', None )
            definitions_header_path = process_data.get( 'definitions_header_path', None )
            data_header_path = process_data.get( 'data_header_path', None )
            definitions_json_path = process_data.get( 'data_json_path', None )


            print("generate-headers...")
            exit_if_one_is_none([template_input_path])
            if skip:
                continue

            if not os.path.exists(input_file):
                sys.exit(1)

            if not os.path.exists(os.path.dirname(data_header_path)):
                os.makedirs(os.path.dirname(data_header_path))

            if not os.path.exists(os.path.dirname(definitions_header_path)):
                os.makedirs(os.path.dirname(definitions_header_path))

            if genJson:
                generate_json_2(definitions_json_path, template_input_path, definitions_header_path, template_length, bin_count, distance_scale_bits, amplitude_scale_bits, victim_offsets, aggressor_positions)
            else:
                generate_headers(template_input_path, definitions_header_path, data_header_path, template_length, bin_count, distance_scale_bits, amplitude_scale_bits, victim_offsets, aggressor_positions)

        else:
            raise Exception()
