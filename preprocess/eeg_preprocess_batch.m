%% STGE-Former EEG Preprocessing Batch Script
% EEGLAB 2023 - MATLAB 2023
% This script performs batch preprocessing of EEG data for STGE-Former

% Configuration - Update these paths for your environment
data_path = 'YOUR_DATA_PATH_HERE';      % Path to raw .mat EEG files
output_path = 'YOUR_OUTPUT_PATH_HERE';  % Path for processed .set files
chanloc_file = 'mychan';                 % Electrode location file (relative to script directory)

EEG.etc.eeglabvers = '2023.1';

% Import data
cd(data_path);
rawfile = dir('*.mat');
rawname = {rawfile.name};
for i = 1:length(rawname)
    rawdata = rawname{i};
    setname = [rawdata(1:8), '_', 'process.set'];
    EEG = pop_importdata('dataformat', 'matlab', 'nbchan', 128, 'data', rawdata, 'srate', 256, 'pnts', 0, 'xmin', 0);
    % Set dataset name
    EEG.setname = setname;

    % Load electrode locations
    EEG = pop_chanedit(EEG, 'load', {chanloc_file, 'filetype', 'loc'});
    % Bandpass filter (0.1 - 40 Hz)
    EEG = pop_eegfiltnew(EEG, 'locutoff', 0.1, 'hicutoff', 40, 'plotfreqz', 1);
    % Run ICA decomposition
    EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'interrupt', 'on', 'pca', 18);
    % ICLabel component classification
    EEG = pop_iclabel(EEG, 'default');
    pop_selectcomps(EEG, [1:18]);
    EEG = pop_iclabel(EEG, 'default');
    % Remove artifact components
    EEG = pop_icflag(EEG, [NaN NaN; 0.9 1; 0.9 1; NaN NaN; NaN NaN; NaN NaN; NaN NaN]);
    EEG = pop_subcomp(EEG, [], 0);
    EEG.setname = setname;
    EEG = eeg_checkset(EEG);
    % Apply average reference
    EEG = pop_reref(EEG, []);
    % Save processed dataset
    EEG = pop_saveset(EEG, 'filename', setname, 'filepath', output_path);
end
