import os
import numpy as np

raw_num_proj = 1800
new_proj = 720

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
raw_folder = f'CT_26-01-21_{raw_num_proj}'

new_folder = f'CT_26-01-21_{new_proj}'
new_folder = os.path.join(directory, new_folder, 'phantom_scan', 'Data')
os.makedirs(new_folder, exist_ok=True)

raw_data = np.load(os.path.join(directory, raw_folder, 'phantom_scan', 'Data', 'data.npy'))

num_slices_to_sum = int(raw_num_proj/new_proj)

raw_shape = np.shape(raw_data)
new_shape = (int(raw_shape[0]/num_slices_to_sum), num_slices_to_sum, *(raw_shape[1:]))

num_to_add = int(raw_shape[0] / num_slices_to_sum) * num_slices_to_sum

new_data = np.reshape(raw_data[0:num_to_add], new_shape)
new_data = np.sum(new_data, axis=1)

np.save(os.path.join(new_folder, 'data.npy'), new_data)
