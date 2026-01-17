%% STGE-Former EEG Preprocessing Batch Script
% EEGLAB 2023 - MATLAB 2023
% This script performs batch preprocessing of EEG data for STGE-Former

EEG.etc.eeglabvers = '2023.1'; % this tracks which version of EEGLAB is being used, you may ignore it
% 导入数据
data_path='E:\EEG_128channels_resting_lanzhou_2015\EEG_128channels_resting_lanzhou_2015';
cd (data_path);
rawfile=dir('*.mat');
rawname={rawfile.name};
for i=1:length(rawname)
    rawdata=rawname{i};
    setname=[rawdata(1:8),'_','process.set'];
    EEG = pop_importdata('dataformat','matlab','nbchan',128,'data',rawdata,'srate',256,'pnts',0,'xmin',0);
    % 数据集命名
    EEG.setname=setname;

    %电极定位
    EEG=pop_chanedit(EEG, 'load',{'E:\EEG_128channels_resting_lanzhou_2015\EEG_128channels_resting_lanzhou_2015\mychan','filetype','loc'});
    %选择通道
   % EEG = pop_select( EEG, 'channel',{'E9','E11','E22','E24','E33','E36','E45','E52','E58','E62','E70','E83','E92','E96','E104','E108','E122','E124','Cz'});
    %EEG = eeg_checkset( EEG );
    % 带通滤波
    EEG = pop_eegfiltnew(EEG, 'locutoff',0.1,'hicutoff',40,'plotfreqz',1);
    %pop_eegplot( EEG, 1, 1, 1);
    % 做ICA
    EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'interrupt','on','pca',18);
    %pop_eegplot( EEG, 1, 0, 1);
    EEG = pop_iclabel(EEG, 'default');
    pop_selectcomps(EEG, [1:18] );
    EEG = pop_iclabel(EEG, 'default');
    % 去除伪迹
    EEG = pop_icflag(EEG, [NaN NaN;0.9 1;0.9 1;NaN NaN;NaN NaN;NaN NaN;NaN NaN]);
    EEG = pop_subcomp( EEG, [], 0);
    EEG.setname=setname;
    EEG = eeg_checkset( EEG );
    EEG = pop_reref( EEG, []);
    EEG = pop_saveset( EEG, 'filename',setname,'filepath','E:\pre-process-dataset');
end
