from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.41189489566336, 0.4251328133025, 0.4326707089857], 
        std=[0.27413549931506, 0.28506257482912, 0.28284674400252]
        ),
])

class TIRAMISU_CALIBRATION(Dataset):
    def __init__(self, logits_list, exp_name):
        self.logits_list = logits_list
        self.exp_name = exp_name

    def __len__(self):
        return len(self.logits_list)

    def __getitem__(self, id):
        id_logits = torch.load(self.logits_list[id])
        ## modify the following path if needed
        if self.exp_name == 'val':
            logit_id = self.logits_list[id][-21:-9]
        elif self.exp_name == 'test':
            logit_id = self.logits_list[id][-23:-9]
        else:
            raise ValueError('wrong experiment name!')

        label_item_file = '/YOUR_PATH_TO_CamVid/SegNet-Tutorial/CamVid/'+self.exp_name+'annot/'+logit_id+'.png'
        image_item_file = '/YOUR_PATH_TO_CamVid/SegNet-Tutorial/CamVid/'+self.exp_name+'/'+logit_id+'.png'
        boundary_item_file = '/YOUR_PATH_TO_CamVid/SegNet-Tutorial/CamVid/'+self.exp_name+'boundary/'+logit_id+'_boundary.png'
        id_boundary = Image.open(boundary_item_file)
        id_boundary_array = np.array(id_boundary)
        id_label = Image.open(label_item_file)
        id_label_array = np.array(id_label)
        id_image = Image.open(image_item_file).convert("RGB")
        id_image_tensor = preprocess(id_image)
        pred_item_file = '/YOUR_PATH_TO_CamVid/results/'+self.exp_name+'/'+logit_id+'_pred.png'
        id_pred = Image.open(pred_item_file)
        id_pred_array = np.array(id_pred)
        sample = {
        'image': id_image_tensor, 
        'logits': id_logits.squeeze(), 
        'label': torch.from_numpy(id_label_array), 
        'preds': torch.from_numpy(id_pred_array), 
        'boundary': torch.from_numpy(id_boundary_array)
        }

        return sample['image'], sample['logits'], sample['label'], sample['preds'], sample['boundary']