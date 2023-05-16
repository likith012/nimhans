CHANNEL_MAPPING = {'C4:M1': 'eeg',
                'C3:M2': 'eeg',
                'O2:M1': 'eeg',
                'O1:M2': 'eeg',
                # 'F4:A1': 'eeg',
                # 'F3:A2': 'eeg',
                # 'F4:M1': 'eeg',
                # 'F3:M2': 'eeg',
                'C4:A1': 'eeg',
                'C3:A2': 'eeg',
                'O2:A1': 'eeg',
                'O1:A2': 'eeg',
                'ECG 2': 'ecg',
                'E1:M2': 'eog',
                'E2:M2': 'eog',
                'EOG1:A2': 'eog',
                'EOG2:A2': 'eog',
                'EMG': 'emg',
                # 'EMG2': 'emg',
                'Chin 1': 'emg',
                # 'Chin 2': 'emg',
                'SPO2': 'misc',
                'PLMl': 'misc',
                'PLMr': 'misc',
                'Snore': 'misc',
                'Pressure Snore': 'misc',
                'Pulse': 'misc',
                'Pleth': 'misc',
                'Pos.': 'misc',
                'Move.': 'misc',
                'Light': 'misc',
                'Sum Effort': 'resp',
                'Sum RIPs': 'resp',
                'Abdomen': 'resp',
                'RIP Abdomen': 'resp',
                'Pressure Flow': 'misc',
                'Flow Th': 'temperature',      
                'Thorax': 'misc',
                'RIP Thorax': 'misc',
                'Battery': 'misc',
    }
LABEL_MAPPING = {  
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "R": 4,
}

# @dataclass
# class channel_mapping:
#     eeg: List = ['C4:M1', # EEG with M1, M2 as reference electrodes
#                 'C3:M2',
#                 'O2:M1',
#                 'O1:M2']
#     ecg: List = ['ECG 2'] # ECG
#     eog: List =  ['E1:M2', # EOG with M2 as reference electrodes
#                   'E2:M2']
#     emg: List = ['EMG'] # EMG
#     spo2: List = ['SPO2'] # Pulse oximeter, measures oxygen saturation levels
#     limb: List = ['PLMr'] # Periodic limb movements
#     snore: List = ['Snore', # Snore signals
#                    'Pressure Snore'] # Percentage of snore signals
#     bpm: List = ['Pulse'] # Beats per minute
#     pg: List = ['Pleth'] # Plethosmography
#     position: List = ['Pos.'] # Sleep position of the person (4 types)
#     light: List = ['Light'] # Light levels, 0 during sleep
#     resp: List = ['Sum Effort', # Respiratory signals
#                   'Abdomen']
#     nasal: List = ['Pressure Flow'] # Nasal pressure flow
#     temp: List = ['Flow Th'] # Flow thremister: Thermistors utilizing change in temperature of exhaled air to assess the airflow 
#                              # have been traditionally used to detect apneas and hypopneas during PSG recording
#     thorax: List = ['Thorax']