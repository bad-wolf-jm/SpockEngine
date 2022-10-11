import pandas as pd
import numpy as np

PATH = "C:\\GitLab\\LTSimulationEngine\\Tests\\Data\\FPGAConfiguration\\BlinderTest\\Source\\UC blinder noise, MF off_20200324-165622_tv.dat"
PATH = "C:\\GitLab\\LTSimulationEngine\\Tests\\Data\\FPGAConfiguration\\BlinderTest\\Source\\UC blinder overlap, t2_dist=-3_20200324-165701_tv.dat"
PATH = "C:\\GitLab\\LTSimulationEngine\\Tests\\Data\\FPGAConfiguration\\BlinderTest\\Source\\UC blinder overlap, t2_dist=-12.19_20200324-165740_tv.dat"
# PATH = "C:\\GitLab\\LTSimulationEngine\\Tests\\Data\\FPGAConfiguration\\BlinderTest\\Source\\UC blinder overlap, t2_dist=-70_20200324-165816_tv.dat"

df = pd.read_csv(PATH, delimiter=' ')

keep_columns = ['%time',
             'i_lca3_fpga_sys.i_dsp_fpga1.dsp0_expanded_sig',
             'i_lca3_fpga_sys.i_dsp_fpga1.dsp0_expanded_val',
             'i_lca3_fpga_sys.i_dsp_fpga1.dsp1_blinded_lvdiff',
             'i_lca3_fpga_sys.i_dsp_fpga1.dsp1_blinded_noise',
             'i_lca3_fpga_sys.i_dsp_fpga1.dsp1_blinded_satlength',
             'i_lca3_fpga_sys.i_dsp_fpga1.dsp1_blinded_sig',
             'i_lca3_fpga_sys.i_dsp_fpga1.dsp1_blinded_val']
df = df[keep_columns]

col_names = {'%time': 'Timestamp',
             'i_lca3_fpga_sys.i_dsp_fpga1.dsp0_expanded_sig': 'Input',
             'i_lca3_fpga_sys.i_dsp_fpga1.dsp0_expanded_val': 'Valid',
             'i_lca3_fpga_sys.i_dsp_fpga1.dsp1_blinded_lvdiff': 'dsp1_blinded_lvdiff',
             'i_lca3_fpga_sys.i_dsp_fpga1.dsp1_blinded_noise': 'dsp1_blinded_noise',
             'i_lca3_fpga_sys.i_dsp_fpga1.dsp1_blinded_satlength': 'SaturatedLength',
             'i_lca3_fpga_sys.i_dsp_fpga1.dsp1_blinded_sig': 'BlindedSignal',
             'i_lca3_fpga_sys.i_dsp_fpga1.dsp1_blinded_val': 'BlindedValue'}
df = df.rename(columns=col_names)

print(df[df["Valid"] != 0])
