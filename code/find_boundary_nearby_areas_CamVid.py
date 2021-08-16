import SimpleITK as sitk
import numpy as np
from PIL import Image
import glob

common_file_path = '/YOUR_PATH_TO_CamVid/SegNet-Tutorial/CamVid/testannot/'
common_save_path = '/YOUR_PATH_TO_CamVid/SegNet-Tutorial/CamVid/testboundary/'

val_img_list = glob.glob('/YOUR_PATH_TO_CamVid/SegNet-Tutorial/CamVid/testannot/*.png')
val_img_list.sort()

for ind in range(len(val_img_list)):
    print('{} out of {}'.format(ind, len(val_img_list)))
    file_name = val_img_list[ind]
    if file_name[-17] == 'e':
        save_file_name_b = common_save_path + file_name[-18:-4] + '_boundary.png'
    else:
        save_file_name_b = common_save_path + file_name[-17:-4] + '_boundary.png'
    label_map = Image.open(file_name)
    label_map_array = np.array(label_map)

    label_map_x_1 = np.roll(label_map_array, -1, axis=0)
    label_map_x_2 = np.roll(label_map_array, 1, axis=0)
    label_map_y_1 = np.roll(label_map_array, -1, axis=1)
    label_map_y_2 = np.roll(label_map_array, 1, axis=1)
    label_map_x_3 = np.roll(label_map_array, -2, axis=0)
    label_map_x_4 = np.roll(label_map_array, 2, axis=0)
    label_map_y_3 = np.roll(label_map_array, -2, axis=1)
    label_map_y_4 = np.roll(label_map_array, 2, axis=1)

    boundary_areas = (label_map_array != label_map_x_1) | (label_map_array != label_map_x_2) \
                     | (label_map_array != label_map_y_1) | (label_map_array != label_map_y_2) \
                     | (label_map_array != label_map_x_3) | (label_map_array != label_map_x_4) \
                     | (label_map_array != label_map_y_3) | (label_map_array != label_map_y_4)
    boundary_areas[:,:3] = 0
    boundary_areas[:,-3:] = 0
    boundary_areas[:3,:] = 0
    boundary_areas[-3:,:] = 0
    rb = Image.fromarray(boundary_areas.astype('uint8')).resize(label_map.size)
    rb.save(save_file_name_b)
