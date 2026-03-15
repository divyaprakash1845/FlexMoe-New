clc;
clear all;
close all;

%% 1. Setup Paths (CHANGE THESE TWO LINES IF NEEDED)
% -> Path to EEGLAB
eegpath = 'C:\Users\dppcs\Downloads\eeglab_current\eeglab2026.0.0'; 
addpath(eegpath);
eeglab;

% -> Path to the main Neuroflow folder
rootDir = 'C:\Users\dppcs\OneDrive\Desktop\DPP_Neuroflow'; 

%% 2. Find all EEG folders and the LARGEST EDF files
fprintf('🔍 Scanning %s for subject folders...\n', rootDir);
eegFolders = dir(fullfile(rootDir, '**', '*_EEG'));
edfPaths = {};
layPaths = {};

for k = 1:length(eegFolders)
    folderPath = fullfile(eegFolders(k).folder, eegFolders(k).name);
    edfFiles = dir(fullfile(folderPath, '*.edf'));
    
    if isempty(edfFiles)
        continue;
    end
    
    % --- THE AUTOPILOT LOGIC: IF MULTIPLE FILES, PICK THE LARGEST ONE ---
    if length(edfFiles) > 1
        fprintf('⚠️ Found %d EDFs in %s. Selecting the largest one to avoid false starts...\n', length(edfFiles), eegFolders(k).name);
        [~, maxIdx] = max([edfFiles.bytes]); % Find the index of the biggest file
        edfFiles = edfFiles(maxIdx);         % Keep ONLY the biggest file
    end
    
    currentEdfPath = fullfile(edfFiles(1).folder, edfFiles(1).name);
    edfPaths{end+1} = currentEdfPath; %#ok<SAGROW>
    
    % Find the matching .lay file
    [~, baseName, ~] = fileparts(currentEdfPath);
    layFile = dir(fullfile(folderPath, [baseName, '.lay']));
    if ~isempty(layFile)
        layPaths{end+1} = fullfile(layFile(1).folder, layFile(1).name); %#ok<SAGROW>
    else
        layPaths{end+1} = ''; 
    end
end

fprintf('✅ Found %d valid (largest) EDF files to process.\n', length(edfPaths));

%% 3. Process Each File (The Loop)
for fileIdx = 1:length(edfPaths)
    currentEdf = edfPaths{fileIdx};
    currentLay = layPaths{fileIdx};
    [filepath, name, ~] = fileparts(currentEdf);
    
    fprintf('\n=========================================================\n');
    fprintf('⚙️ Processing File %d of %d: %s\n', fileIdx, length(edfPaths), name);
    
    % --- Read .lay file for info ---
    info = struct();
    info.sample_rate = 250; % Default fallback
    if ~isempty(currentLay) && isfile(currentLay)
        fid = fopen(currentLay, 'r');
        lines = textscan(fid, '%s', 'Delimiter', '\n');
        fclose(fid);
        lines = lines{1};
        for i = 1:length(lines)
            line = strtrim(lines{i});
            if contains(line, 'samplerate=') || contains(line, 'samplerate =')
                parts = strsplit(line, '=');
                info.sample_rate = str2double(strtrim(parts{2}));
            end
        end
    end
    
    % --- Load EDF Data ---
    try
        fprintf('   -> Loading EDF data...\n');
        data = edfread(currentEdf);
        
        % --- THE 8 CHANNELS FOR MAESTRO ---
        select_channels = {'FZ', 'CZ', 'C3', 'C4', 'O1', 'O2', 'F3_BLUE_', 'F4_RED_'}; 
        standard_names = {'Fz', 'Cz', 'C3', 'C4', 'O1', 'O2', 'F3', 'F4'};
        
        data_subset = data(:, select_channels);
        
        % Unpack the table into an array
        first_col = data_subset{:, 1}; 
        if iscell(first_col)
            total_samples = sum(cellfun(@numel, first_col));
        else
            total_samples = numel(first_col);
        end
        
        data_new = zeros(length(select_channels), total_samples);
        for i = 1:length(select_channels)
            col_data = data_subset{:, i};
            if iscell(col_data)
                data_new(i, :) = vertcat(col_data{:})';
            else
                data_new(i, :) = col_data(:)';
            end
        end
        
        % --- Import to EEGLAB ---
        fprintf('   -> Importing to EEGLAB...\n');
        EEG = pop_importdata('dataformat', 'array', 'data', data_new, 'srate', info.sample_rate);
        
        % Rename to standard names EXACTLY ONCE, then lookup coordinates
        for i = 1:length(standard_names)
            EEG.chanlocs(i).labels = standard_names{i}; 
        end
        EEG = pop_chanedit(EEG, 'lookup','standard-10-5-cap385.elp');
        
        % --- Filtering ---
        fprintf('   -> Filtering noise (0.5 - 35 Hz)...\n');
        EEG_filt = pop_eegfiltnew(EEG, 'locutoff',  0.5, 'hicutoff',  35, 'filtorder', 3300, 'plotfreqz', 0);
        
        % --- Resampling ---
        fprintf('   -> Resampling to 500 Hz for MAESTRO...\n');
        EEG_filt = pop_resample(EEG_filt, 500);
        
        % --- Automatic Artifact Removal (ICA) ---
        fprintf('   -> Running ICA (Artifact Removal). This will take a moment...\n');
        EEG_filt = pop_runica(EEG_filt, 'icatype', 'runica', 'interrupt','off');
        EEG_filt = iclabel(EEG_filt);
        
        threshold_signal = 0.1; 
        cls_score = EEG_filt.etc.ic_classification.ICLabel.classifications;
        bad_comp = [];
        for cmp=1:size(EEG_filt.icachansind,2)
            if cls_score(cmp,1) < threshold_signal
                bad_comp = [bad_comp, cmp];
            end
        end
        fprintf('   -> Removing %d noisy components...\n', length(bad_comp));
        EEG_ica = pop_subcomp(EEG_filt, bad_comp, 0);
        
        % --- Re-reference ---
        EEG_reref = pop_reref(EEG_ica, []);
        
        % --- SAVE THE CLEAN DATA ---
        saveFileName = sprintf('%s_cleaned_EEG_500Hz.csv', name);
        savePath = fullfile(filepath, saveFileName);
        
        % Transpose so it saves as [Time x 8 Channels]
        writematrix(EEG_reref.data', savePath);
        fprintf('   ✅ Saved pristine 500Hz EEG to: %s\n', savePath);
        
    catch ME
        fprintf('   ❌ Error processing %s: %s\n', currentEdf, ME.message);
    end
end
fprintf('\n🎉 All multi-subject EEG preprocessing complete!\n');
