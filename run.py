import os
import torch

from .model import get_instance_segmentation_model, get_object_detection_model, Modified_VAE
from .dataset import AnimalsDataModule
from .evaluation import MOTS_prediction, MOT_prediction, COCO_prediction

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#Define data location, properties,...
data_params = { "PATH_TRAINING_SET" : "./WildlifeCrossingDataset_Training", #path to frame folder
                "PATH_TEST_SET": "./WildlifeCrossingDataset_Test",
                "PATH_ANNO": "./Annotation_WildlifeCrossing.csv",
                "NUM_CLASSES": 5,
                "CLASS_DICT": {"boar": 1, "deer": 2, "fox": 3, "hare": 4},
                "FRAME_SIZE": (720, 1280)}  #orignal frame size

#Define model threshold
model_params = {"DETECTION_THRESHOLD": 0.95,      #detecion threshold score
                "MATCHING_COST_THRESHOLD": 0.95,   #bbox matching cost threshold
                "FC_DIM": [4096, 2048],
                "RESNET_BLOCK_LAYERS": [2,2,2,2],
                "LATENT_DIM": 1024}  

#Define training parameters
training_params = {"LR": 0.00015,  #learning rate
        "weight_decay": 0.0005,
        "LR_down_step": 1,    
        "schedule_gamme": 0.75,
        'kld_weight' : 0.01,          #weighting parameters
        'recons_weight' : 10,
        'mask_weight': 0.5,
        'box_weight': 1,
        'class_weight': 1}

#If there are pre_trained_model
pre_trained_model = {"PATH_MASK_RCNN": '../TrainedParameters/MaskRCNN/MaskRCNN_WildlifeCrossing.pth',
                     "PATH_VAE": "../TrainedParameters/VAE/VAE_WildlifeCrossing.pth"}

if __name__ == "__main__":

        TRAIN_MODE = False

        #Get and load Mask-RCNN models      
        mask_rcnn_model = get_instance_segmentation_model(data_params["NUM_CLASSES"]).to(device) 
        mask_rcnn_model.load_state_dict(torch.load(pre_trained_model["PATH_MASK_RCNN"], map_location=device))
        mask_rcnn_model.eval()


        #Get VAE model
        vae_model = Modified_VAE(in_channels=7,
                                params = training_params,
                                layers = model_params["RESNET_BLOCK_LAYERS"],
                                fc_dims = model_params["FC_DIM"],
                                latent_dim= model_params["LATENT_DIM"],
                                n_classes = data_params["NUM_CLASSES"],
                                train_mode = TRAIN_MODE)
        vae_model.initialize_weights().to(device)

        if not TRAIN_MODE:

                #Loading trained VAE parameters
                vae_model.load_state_dict(torch.load(pre_trained_model["PATH_VAE"]))

                #Test video example
                test_videos = {1: "boar10"}

                #Give prediction in MOTS form
                MOTS_prediction(test_videos, vae_model)
    
        else:   
                #Saving training details in tensor board 
                LOG_SAVE_DIR = "./logger"

                logger = TensorBoardLogger(save_dir=LOG_SAVE_DIR)
                data_module = AnimalsDataModule()

                trainer = pl.Trainer(logger = logger, 
                     num_sanity_val_steps=0,
                     precision=16,
                     gpus=1, 
                     max_epochs = 20,
                     checkpoint_callback = False)

                #Training
                torch.cuda.empty_cache()
                trainer.fit(vae_model, data_module)

