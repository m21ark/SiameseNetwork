import os
import torch
import shutil
import glob
import datetime

from src.siamese import Siamese
from src.dataset import get_dataloaders
from src.train import Trainer

def save_code_state(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cwd = os.getcwd()
    
    for py_file in glob.glob("*.py"):
        shutil.copy(py_file, save_path)
        
    for root, dirs, files in os.walk(cwd + '/src'):
        for file in files:
            if file.endswith(".py"):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, cwd) 
                dest_path = os.path.join(save_path, rel_path)

                os.makedirs(os.path.dirname(dest_path), exist_ok=True) 
                shutil.copy2(src_path, dest_path)
                
    # zip the code folder
    shutil.make_archive(save_path + "/../siamese", 'zip', save_path)
    shutil.rmtree(save_path)

current_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# if dir already exists, delete it and its contents
if os.path.exists("../siamese/saved_models/"):
    shutil.rmtree("../siamese/saved_models/")
save_code_state("../siamese/saved_models/" + current_date_time)

device = "mps" if torch.backends.mps.is_available() else "cuda"

# ============================== TRAIN ==============================

useTripletMode = False
use_2phase_training = False

print(f"Running on 2-phase training: {use_2phase_training} and triplet mode: {useTripletMode}")

train_dataloader, test_dataloader = get_dataloaders("celebmask_245/images", bsize=32, useTriplets=useTripletMode)
print(f"Train dataset size: {len(train_dataloader.dataset)} images => {len(train_dataloader)} batches")
print(f"Test dataset size: {len(test_dataloader.dataset)} images => {len(test_dataloader)} batches")

# Train the model
model = Siamese().to(device)
multi_gpu_model = torch.nn.DataParallel(model)
trainer = Trainer(multi_gpu_model, train_dataloader, test_dataloader, useTripletMode, use_2phase_training, device)
trainer.train()