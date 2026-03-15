import os
import glob
import torch
import numpy as np
import pandas as pd
import scipy.signal

def process_subject(subject_folder):
    print(f"\n⚙️ Processing folder: {subject_folder}")
    
    # 1. Look for the clean EEG CSV and the Empatica BVP CSV
    clean_eeg_files = glob.glob(os.path.join(subject_folder, "**", "*cleaned_EEG_500Hz.csv"), recursive=True)
    bvp_files = glob.glob(os.path.join(subject_folder, "**", "BVP.csv"), recursive=True)
    
    if not clean_eeg_files:
        print("   ⚠️ Missing cleaned EEG CSV. Skipping...")
        return
    if not bvp_files:
        print("   ⚠️ Missing BVP.csv file. Skipping...")
        return

    # AUTO-PILOT: Pick the largest CSV if there are multiple (avoids false starts)
    if len(clean_eeg_files) > 1:
        print(f"   ⚠️ Found {len(clean_eeg_files)} clean EEG files. Selecting the largest one...")
        eeg_path = max(clean_eeg_files, key=os.path.getsize)
    else:
        eeg_path = clean_eeg_files[0]
        
    bvp_path = bvp_files[0]
    
    print(f"   -> Found Cleaned EEG: {os.path.basename(eeg_path)}")
    print(f"   -> Found BVP: {os.path.basename(bvp_path)}")
    
    # 2. Load the Cleaned EEG (Shape should be [Time, 8_Channels] from MATLAB)
    print("   -> Loading Cleaned EEG CSV...")
    eeg_df = pd.read_csv(eeg_path, header=None)
    eeg_data = eeg_df.values.T # Transpose to [8_Channels, Time]
    
    # 3. Load and extract BVP (1 channel)
    print("   -> Loading BVP CSV...")
    bvp_df = pd.read_csv(bvp_path)
    bvp_raw = bvp_df.iloc[:, 0].values 
    
    # Resample BVP to perfectly match the cleaned EEG length
    target_length = eeg_data.shape[1]
    bvp_resampled = scipy.signal.resample(bvp_raw, target_length)
    bvp_data = bvp_resampled.reshape(1, -1) # Shape: [1, Time]
    
    # 4. Fuse them together
    print("   -> Fusing into 9-channel tensor...")
    fused_numpy = np.vstack((eeg_data, bvp_data)) 
    fused_tensor = torch.tensor(fused_numpy, dtype=torch.float32).transpose(0, 1) # Final shape: [Time, 9]
    
    # 5. Save the final .pt file in the main subject folder
    save_path = os.path.join(subject_folder, "fused_tensor.pt")
    torch.save(fused_tensor, save_path)
    print(f"   ✅ Saved: {save_path} | Shape: {fused_tensor.shape}")

if __name__ == "__main__":
    print("🚀 Starting Preprocessing Pipeline...")
    
    # Looks one directory up for the raw_data folder
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'raw_data'))
    
    # We use os.path.join so it works on both Windows and Linux/Colab
    folders = glob.glob(os.path.join(raw_dir, "*/")) 
    
    if len(folders) == 0:
        print(f"⚠️ No subject folders found inside '{raw_dir}'!")
    else:
        for folder in folders:
            process_subject(folder)
            
    print("\n🎉 All subjects fused into PyTorch tensors!")
