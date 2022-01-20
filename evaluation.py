from dataset import Eval_Dataset, get_transform
from run import data_params
from utils import prediction, unmold_mask
from torch.utils.data import DataLoader
import numpy as np
from pycocotools import mask as coco
import json

TEST_SET = data_params["PATH_TEST_SET"]
ANNO = data_params["PATH_ANNO"]

def MOTS_prediction(videos, model):
    """
    Input: VAE model and a dictionaries of test video
    Output: Prediction MOT results in txt data 
    """
    for j in videos:
        video_name = videos[j]
        test_data = Eval_Dataset(TEST_SET, ANNO, transforms=get_transform(train=False), video_name=video_name + "-")
        data = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
        filename = video_name + ".txt"
        with open(filename, 'w') as file_object:
            
            for i, sample in enumerate(data):
                print(sample[1]["image_id"])
                pred = prediction(sample)
                boxes, cls, masks, tracks, score = pred
                if len(cls) == 0:
                    continue
                for j in range(len(boxes)):
                    if boxes[j][1][0] == 0 and boxes[j][1][1] == 0:
                            continue
                    full_mask = unmold_mask(masks[j], [boxes[j][0][0], boxes[j][0][1], boxes[j][1][0], boxes[j][1][1]])
                    full_mask = np.asfortranarray(full_mask)
                    file_object.write(str(i) + " " + #time frame
                            cls[j] + "00" + tracks[j] + " " +#class id + track id
                            cls[j] + #class id
                            " 720 1280 " + #
                            coco.encode(full_mask)["counts"].decode("utf-8") + "\n")


def MOT_prediction(videos, model):
    for j in videos:
        video_name = videos[j]
        test_data = Eval_Dataset(TEST_SET, ANNO, transforms=get_transform(train=False), video_name=video_name + "-")
        data = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
        filename = video_name + ".txt"
        with open(filename, 'w') as file_object:
            
            for i, sample in enumerate(data):
                print(sample[1]["image_id"])
                pred = prediction(sample)
                boxes, cls, masks, tracks, score = pred
                if len(cls) == 0:
                    continue
                for j in range(len(boxes)):

                    file_object.write(str(i+1) + "," \
                            + tracks[j] + "," \
                            + str(round(boxes[j][0][0],3)) + "," \
                            + str(round(boxes[j][0][1],3)) + "," + str(round(boxes[j][1][0]-boxes[j][0][0],3)) + "," + str(round(boxes[j][1][1]-boxes[j][0][1],3)) \
                            + ",-1,-1,-1,-1\n")

def COCO_prediction(videos, model):
    img_id = 0
    cls_id = 0  
    filename = "result_coco.json"
    with open(filename, 'w') as file_object:
        results = []
        for j in videos:
            video_name = videos[j]
            if video_name.startswith("deer"):
                cls_id = 1
            elif video_name.startswith("boar"):
                cls_id = 2
            elif video_name.startswith("fox"):
                cls_id = 3
            elif video_name.startswith("hare"):
                cls_id = 4
            else:
                cls_id = 0
            
            test_data = Eval_Dataset(TEST_SET, ANNO, transforms=get_transform(train=False), video_name=video_name + "-")
            data = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
                
            for i, sample in enumerate(data):
                    img_id = img_id + 1
                    
                    print(sample[1]["image_id"])
                    pred = prediction(sample)
                    boxes, cls, masks, tracks, score = pred
                    if len(cls) == 0:
                        continue
                    
                    for j in range(len(boxes)):
                        if boxes[j][1][0] == 0 and boxes[j][1][1] == 0:
                                continue

                        full_mask = unmold_mask(masks[j], [boxes[j][0][0], boxes[j][0][1], boxes[j][1][0], boxes[j][1][1]])
                        full_mask = np.asfortranarray(full_mask) 

                        result = {
                                    "image_id": img_id,
                                    "category_id": cls_id,
                                    "score": round(float(score[j]),2),
                                    "segmentation": {"size": [720,1280], 
                                                    "counts": coco.encode(full_mask)["counts"].decode("utf-8")}
                        }
                        results.append(result)
                        
        json.dump(results, file_object)


