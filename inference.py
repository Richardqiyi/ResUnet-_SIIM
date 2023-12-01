import warnings

warnings.simplefilter("ignore", (UserWarning, FutureWarning))
from utils.hparams import HParam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import dataloader
from utils import metrics
from core.res_unet_plus import ResUnetPlusPlus
import torch
import argparse
import os
import numpy as np
from PIL import Image
import glob
from skimage import io


criterion = metrics.BCEDiceLoss()

def validation(model, criterion):
    valid_acc = metrics.MetricTracker()
    valid_loss = metrics.MetricTracker()

    # switch to evaluate mode
    model.eval()

    input_files = glob.glob('/code/ResUnet/SIIM_test/input_1/*.png')
    for input_file in input_files:
        # Read input and label images
        image = io.imread(input_file)
        label_file = input_file.replace('input_1', 'output_1')
        label = io.imread(label_file)

        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            # Add other necessary transformations here
        ])
        inputs = transforms.functional.to_tensor(image).unsqueeze(0).cuda()
        labels = torch.from_numpy(label).unsqueeze(0).float().div(255).cuda()

        filename = os.path.basename(input_file)

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        valid_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
        valid_loss.update(loss.data.item(), outputs.size(0))

        # Process outputs
        output_np = outputs.cpu().detach().squeeze().numpy()
        output_np_uint8 = (output_np * 255).astype(np.uint8)

        output_resized = Image.fromarray(output_np_uint8).resize((1024, 1024), Image.NEAREST)
        output_binarized = np.where(np.array(output_resized) > 128, 255, 0)

        # Save result
        save_path = os.path.join("/code/ResUnet/ResUnet_cosdecay_infer", filename)
        Image.fromarray(output_binarized.astype(np.uint8)).save(save_path)

    print("Validation Loss: {:.4f} Acc: {:.4f}".format(valid_loss.avg, valid_acc.avg))
    return {"valid_loss": valid_loss.avg, "valid_acc": valid_acc.avg}



model = ResUnetPlusPlus(3).cuda()

resume = '/code/ResUnet/checkpoints/cosdecay/cosdecay_checkpoint_94000.pt'


if os.path.isfile(resume):
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(resume)

    start_epoch = checkpoint["epoch"]

    best_loss = checkpoint["best_loss"]
    model.load_state_dict(checkpoint["state_dict"])
    print(
        "=> loaded checkpoint '{}' (epoch {})".format(
            resume, checkpoint["epoch"]
        )
    )

validation(model, criterion)
    
