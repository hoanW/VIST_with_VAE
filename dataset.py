import os
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import random

import torch
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from run import data_params
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#Dataset for Mask-RCNN
#Defining the dataset
class Mask_RCNN_Dataset(object):
    def __init__(self, images, annotations, transforms=None):
        self.imgpath = images
        self.annpath =annotations
        self.transforms = transforms
        # load the cocodata File for the annotations of each video and sorted
        self.csvdata = pd.read_csv(self.annpath).sort_values(by=['filename'])
        self.csvdata = self.csvdata.reset_index(drop=True)

        # rename the column, because it represents the video_number
        self.csvdata.rename(columns={'file_attributes':'video_number'}, inplace=True)
        self.csvdata.rename(columns={'region_attributes':'class'}, inplace=True)
        self.csvdata.rename(columns={'region_shape_attributes':'xpoints'}, inplace=True)

        self.classdict= data_params["CLASS_DICT"] 
        # for splitting attribut values, insert the new columns
        self.csvdata['track'] = pd.Series(np.random.randn(self.csvdata.shape[0]), index=self.csvdata.index)
        self.csvdata.insert(6,"ypoints",pd.Series(np.random.randn(self.csvdata.shape[0]), index=self.csvdata.index))

        self.csvdata.insert(3,"image_id",pd.Series(np.random.randn(self.csvdata.shape[0]), index=self.csvdata.index))
        # groupby the filenames for generating a dictionary of corresponding image ids
        filegroup= self.csvdata.groupby(self.csvdata["filename"])
        num=np.arange(filegroup.ngroups)  
        imgid_dict= dict(zip(filegroup.groups.keys(),num))
        
        # preprocess the data from the csv for better reading from the dataframe
        for i in range(self.csvdata.shape[0]):
            # write just the Video number int in the row for better accessing of the values
            if int(self.csvdata.loc[i,"region_count"])>0:
                p=self.csvdata.loc[i, "video_number"]
                val=[int(s) for s in p.split("\"") if s.isdigit()]
                self.csvdata.loc[i, "video_number"]=val[0]
                
                s=self.csvdata.loc[i,"xpoints"]
                sp= s.split("[")
                x_points= sp[1].split("]")[0]
                y_points= sp[2].split("]")[0]
                # concatenate the x and y points by just a ; for easier extraction later on 
            
                self.csvdata.loc[i,"xpoints"]=x_points
                self.csvdata.loc[i,"ypoints"]=y_points
                
                #prepare the region attributes column for better usage
                r=self.csvdata.loc[i,"class"]
                rs=r.split("\"")

                self.csvdata.loc[i,"class"]= self.classdict[rs[3]]
                self.csvdata.loc[i,"track"]=int(rs[7])
                
                # insert image ids
                self.csvdata.loc[i,"image_id"] =int(imgid_dict[self.csvdata.loc[i,"filename"]])
        
        # filter out the rows where are no annotations
        self.csvdata = self.csvdata[self.csvdata["region_count"] !=0]
        #get imgs path in an array
        self.imgs = list(sorted(os.listdir(self.imgpath)))
        #get len of this array
        self.len = len(self.imgs)
        
        

    def __getitem__(self, idx):
            #get image path
            imfile=os.path.join(self.imgpath,self.imgs[idx])
            framegroup = self.csvdata.loc[self.csvdata['filename'] == self.imgs[idx]] 
            
            img= Image.open(imfile)
            # convert to grayscale because of the night shots
            img =img.convert('L')

            # Get the number of objects / animals by extracting the region count value
            num_objs= framegroup["region_count"].max()

            # initialise list of boxes  
            boxes=[]        
            # generate the binary masks
            masks=np.zeros((num_objs,img.size[1],img.size[0]),dtype=np.uint8)
            # area of the segments and iscrowd attribute
            area=torch.zeros((num_objs,),dtype=torch.float32)
            iscrowd =torch.zeros((num_objs,), dtype=torch.int64)
            # save the labels
            labels=torch.zeros((num_objs,), dtype=torch.int64)
            
            # save the track number
            tracks=torch.zeros((num_objs,), dtype=torch.int64)

            count=0
            for _, frame in framegroup.iterrows():
                
                # extract the polygon points and split by defined marker ;
                xpoint_str=frame.loc["xpoints"]
                ypoint_str=frame.loc["ypoints"]
                
                # convert to int list
                xpoints=list(map(int, xpoint_str.split(',')))
                ypoints=list(map(int, ypoint_str.split(',')))
                
                # generate the mask from the polyline
                points=[]
                for j in range(len(xpoints)):
                    points.append(xpoints[j])
                    points.append(ypoints[j])
                # generate the mask with the poly line
                imgMask = Image.new('L', (img.size[0],img.size[1]), 0)
                ImageDraw.Draw(imgMask).polygon(points, outline=1, fill=1)
                masks[count] = np.array(imgMask)
                # get the area of the segment
                
                # is crowd always to 0, should indicate overlap, but is here not interesting for us
                iscrowd[count]=0
                
                
                # extract the bounding box information from the polyline
                xmin = min(xpoints)
                ymin = min(ypoints)
                xmax = max(xpoints)
                ymax = max(ypoints)
                boxes.append([xmin,ymin,xmax,ymax])
                
                # set the label
                labels[count]=frame.loc["class"]
                # set the track number
                tracks[count]=frame.loc["track"]
                
                count+=1
                
      

            # convert the np array to a tensor
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            # convet the bounding boxes to tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area =(boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
 
    
            # generate image id part, not really relevant    
            image_id=  torch.tensor([idx])#frame.loc["filename"] 
            
                
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = image_id 
            target["area"] = area
            target["iscrowd"] = iscrowd
            #target["track"]=tracks
    
            # convert image back to RGB, because the reid model and other models need it in this way
            img=img.convert("RGB")
            # in transforms the PIL image will be converted to a pytorch tensor
            if self.transforms is not None:
                img, target = self.transforms(img, target)
  
            #target_list.append(target)
            

            return img, target

    def __len__(self):
        return self.len

#VAE dataset
class VAE_Dataset(Dataset):
    def __init__(self, images, annotations, transforms):
        self.imgpath = images
        self.annpath = annotations
        self.transforms = transforms
        # load the cocodata File for the annotations of each video
        self.csvdata = pd.read_csv(self.annpath, engine='python')
        # rename the column, because it represents the video_number
        self.csvdata.rename(columns={'file_attributes':'video_number'}, inplace=True)
        self.csvdata.rename(columns={'region_attributes':'class'}, inplace=True)
        self.csvdata.rename(columns={'region_shape_attributes':'xpoints'}, inplace=True)

        self.classdict= data_params["CLASS_DICT"]
        # for splitting attribut values, insert the new columns
        self.csvdata['track'] = pd.Series(np.random.randn(self.csvdata.shape[0]), index=self.csvdata.index)
        self.csvdata.insert(6,"ypoints",pd.Series(np.random.randn(self.csvdata.shape[0]), index=self.csvdata.index))
        
        # preprocess the data from the csv for better reading from the dataframe
        for i in range(self.csvdata.shape[0]):
            # write just the Video number int in the row for better accessing of the values
            p=self.csvdata.loc[i, "video_number"]
            val=[int(s) for s in p.split("\"") if s.isdigit()]
            self.csvdata.loc[i, "video_number"]=val[0]
            
            s=self.csvdata.loc[i,"xpoints"]
            sp= s.split("[")
            x_points= sp[1].split("]")[0]
            y_points= sp[2].split("]")[0]
            # concatenate the x and y points by just a ; for easier extraction later on 
        
            self.csvdata.loc[i,"xpoints"]=x_points
            self.csvdata.loc[i,"ypoints"]=y_points
            
            # prepare the region attributes column for better usage
            r=self.csvdata.loc[i,"class"]
            rs=r.split("\"")

            self.csvdata.loc[i,"class"]= self.classdict[rs[3]]
            self.csvdata.loc[i,"track"]=int(rs[7])        
        
        #get imgs path in an array
        self.imgs = list(sorted(os.listdir(self.imgpath)))
        #get len of this array
        self.len = len(self.imgs)
        
        

    def __getitem__(self, idx):
        #get vid_idx
        filename = self.imgs[idx]
        #get all rows with same image name
        framegroup = self.csvdata.loc[self.csvdata['filename'] == self.imgs[idx]] 
        #get vid index for this image
        vid_idx = framegroup['video_number'].iloc[0]

        # extract the corresponding frames 
        vidlist= self.csvdata.loc[self.csvdata['video_number'] == vid_idx]
        # group by the image names, because there might be multiple rows when there is more than one object annotated in the video
        vidgrouped= vidlist.groupby(vidlist["filename"])
        # save here all the images and the targets
        filename_list = []
        vidgrouped_list = []
        #get list of frame name in the videos
        for name, group in vidgrouped:
            filename_list.append(name)
            vidgrouped_list.append(group)
        #get filename idx, but in the videos
        filename_idx = filename_list.index(filename)
        #if it's the fist or last frame of videos, then use the next frame to it
        #because these frame have no last or future frame for target data
        if filename_idx == 0:
            filename = filename_list[1]
            filename_idx = 1
        elif filename_idx == len(filename_list)-1:
            filename = filename_list[-2]
            filename_idx =  len(filename_list)-2

        i = filename_idx

        # construct the path to the image and load it
        #current image
        imfile=os.path.join(self.imgpath,filename_list[i])
        img= Image.open(imfile)
        #last image
        last_imfile=os.path.join(self.imgpath,filename_list[i-1])
        last_img= Image.open(last_imfile)
        #future image
        future_imfile=os.path.join(self.imgpath,filename_list[i+1])
        future_img= Image.open(future_imfile)

        # convert to RGB 
        img = img.convert('RGB')
        last_img = last_img.convert('RGB')
        future_img = future_img.convert('RGB')

        #PROCESSING INPUT DATA
        last_boxes = []
        last_tracks = []
        #last mask
        last_masks=np.zeros((1,img.size[1],img.size[0]),dtype=np.uint8)
        last_imgMask = Image.new('L', (img.size[0],img.size[1]), 0)

        for _, frame in vidgrouped_list[i-1].iterrows():
                xpoint_str=frame.loc["xpoints"]
                ypoint_str=frame.loc["ypoints"]
                
                # convert to int list
                xpoints=list(map(int, xpoint_str.split(',')))
                ypoints=list(map(int, ypoint_str.split(',')))
                
                # generate the mask from the polyline
                points=[]
                for j in range(len(xpoints)):
                    points.append(xpoints[j])
                    points.append(ypoints[j])
                # generate the mask with the poly line
                ImageDraw.Draw(last_imgMask).polygon(points, outline=1, fill=1)

                # extract the bounding box information from the polyline
                xmin = min(xpoints)
                ymin = min(ypoints)
                xmax = max(xpoints)
                ymax = max(ypoints)
                last_boxes.append([xmin,ymin,xmax,ymax])

                #get track id
                last_tracks.append(frame.loc["track"])

        last_masks[0] = np.array(last_imgMask)

        #PROCESSING TARGET DATA
        # Get the number of objects / animals by extracting the region count value
        num_objs= vidgrouped_list[i].iloc[0].loc["region_count"]
       
        boxes=[]        
        # generate the binary masks
        masks=np.zeros((num_objs ,img.size[1],img.size[0]),dtype=np.uint8)
        # area of the segments and iscrowd attribute
        iscrowd =torch.zeros((num_objs,), dtype=torch.int64)
        # save the labels
        labels=torch.zeros((num_objs,), dtype=torch.int64)   
        # save the track number
        tracks=torch.zeros((num_objs,), dtype=torch.int64)
            
        # count the segments
        count=0
        for _, frame in vidgrouped_list[i].iterrows():
                #print(frame)
                # extract the polygon points and split by defined marker ;
     
                xpoint_str=frame.loc["xpoints"]
                ypoint_str=frame.loc["ypoints"]
                
                # convert to int list
                xpoints=list(map(int, xpoint_str.split(',')))
                ypoints=list(map(int, ypoint_str.split(',')))
                
                # generate the mask from the polyline
                points=[]
                for j in range(len(xpoints)):
                    points.append(xpoints[j])
                    points.append(ypoints[j])
                # generate the mask with the poly line
                imgMask = Image.new('L', (img.size[0],img.size[1]), 0)
                ImageDraw.Draw(imgMask).polygon(points, outline=1, fill=1)
                masks[count] = np.array(imgMask)
                
                # is crowd always to 0, should indicate overlap, but is here not interesting for us
                iscrowd[count]=0
                
                
                # extract the bounding box information from the polyline
                xmin = min(xpoints)
                ymin = min(ypoints)
                xmax = max(xpoints)
                ymax = max(ypoints)
                boxes.append([xmin,ymin,xmax,ymax])
                
                # set the label
                labels[count]=frame.loc["class"]
                # set the track number
                tracks[count]=frame.loc["track"]
                
                count+=1

        #GET EVERYTHING TOGETHER
        # convert the np array to a tensor
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        # convet the bounding boxes to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # generate image id part, not really relevant    
        image_id=  frame.loc["filename"]     
        future_img = F.to_tensor(future_img)
        #target 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id 
        #target["area"] = area
        target["iscrowd"] = iscrowd
        target["track"]=tracks 
        target["recons_img"] = future_img
            

        #convert to Tensor 
        img = F.to_tensor(img)
        last_img = F.to_tensor(last_img)
        last_masks = torch.as_tensor(last_masks, dtype=torch.uint8)
        con_img = torch.cat((img, last_img, last_masks), 0)
        last_boxes = torch.as_tensor(last_boxes, dtype=torch.float32)
        last_tracks = torch.as_tensor(last_tracks, dtype=torch.int64)

        #input
        input = {}
        input["img"] = img
        input["con_img"] = con_img
        input["boxes"] = last_boxes
        input["track"] = last_tracks
            
        # in transforms the PIL image will be converted to a pytorch tensor
        if self.transforms is not None:
                input, target = self.transforms(input, target)
            
        return input, target

    def __len__(self):
        return self.len

#Transform, Data Augmentation, Resize
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input, target):
        for t in self.transforms:
            input, target = t(input, target)
        return input, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, input, target):
        if random.random() < self.prob:
            height, width = input["img"].shape[-2:]
            input["img"] = input["img"].flip(-1)
            input["con_img"] = input["con_img"].flip(-1)
            last_bbox = input["boxes"]
            last_bbox[:, [0, 2]] = width - last_bbox[:, [2, 0]]
            input["boxes"] = last_bbox 

            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            target["recons_img"] = target["recons_img"].flip(-1)
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
        return input, target

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, input, target):
        size = self.size
        height, width = input["img"].shape[-2:]
        input["img"] = F.resize(input["img"], size)
        input["con_img"] = F.resize(input["con_img"], size)
        last_bbox = input["boxes"]
        last_bbox[:, [0, 2]] = last_bbox[:, [0, 2]]*(size[1]/width)
        last_bbox[:, [1, 3]] = last_bbox[:, [1, 3]]*(size[0]/height)
 
        bbox = target["boxes"]
        bbox[:, [0, 2]] = bbox[:, [0, 2]]*(size[1]/width)
        bbox[:, [1, 3]] = bbox[:, [1, 3]]*(size[0]/height)
        target["recons_img"] = F.resize(target["recons_img"], size)
        if "masks" in target:
          target["masks"] = F.resize(target["masks"], size)
        return input, target

def get_transform(train):
    transforms = []
    #resize image + gt
    transforms.append(Resize([768, 1280]))
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)

class AnimalsDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage):
        self.training_data = VAE_Dataset(data_params["PATH_TRAINING_SET"], data_params["PATH_ANNO"], transforms=get_transform(train=True))
        self.test_data = VAE_Dataset(data_params["PATH_TEST_SET"], data_params["PATH_ANNO"] , transforms=get_transform(train=False))  

    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=1, 
                          shuffle=True, num_workers=4)


    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, 
                          shuffle=False, num_workers=4)


#Dataset for evaluation/testing
class Eval_Dataset(Dataset):
    def __init__(self, images, annotations, transforms, video_name):
        self.imgpath = images
        self.annpath =annotations
        self.transforms = transforms
        self.video_name = video_name
        # load the cocodata File for the annotations of each video
        self.csvdata = pd.read_csv(self.annpath)
        # rename the column, because it represents the video_number
        self.csvdata.rename(columns={'file_attributes':'video_number'}, inplace=True)
        self.csvdata.rename(columns={'region_attributes':'class'}, inplace=True)
        self.csvdata.rename(columns={'region_shape_attributes':'xpoints'}, inplace=True)

        self.classdict= {"deer":0, "boar":1, "fox":2, "hare":3}
        # for splitting attribut values, insert the new columns
        self.csvdata['track'] = pd.Series(np.random.randn(self.csvdata.shape[0]), index=self.csvdata.index)
        self.csvdata.insert(6,"ypoints",pd.Series(np.random.randn(self.csvdata.shape[0]), index=self.csvdata.index))
        
        # preprocess the data from the csv for better reading from the dataframe
        for i in range(self.csvdata.shape[0]):
            # write just the Video number int in the row for better accessing of the values
            p=self.csvdata.loc[i, "video_number"]
            val=[int(s) for s in p.split("\"") if s.isdigit()]
            self.csvdata.loc[i, "video_number"]=val[0]
            
            s=self.csvdata.loc[i,"xpoints"]
            sp= s.split("[")
            x_points= sp[1].split("]")[0]
            y_points= sp[2].split("]")[0]
            # concatenate the x and y points by just a ; for easier extraction later on 
        
            self.csvdata.loc[i,"xpoints"]=x_points
            self.csvdata.loc[i,"ypoints"]=y_points
            
            #prepare the region attributes column for better usage
            r=self.csvdata.loc[i,"class"]
            rs=r.split("\"")

            self.csvdata.loc[i,"class"]= self.classdict[rs[3]]
            self.csvdata.loc[i,"track"]=int(rs[7])
            #print(self.csvdata.loc[i,"region_attributes"])
            

        #print(self.csvdata)
        
        
        #get imgs path in an array
        self.imgs = list(sorted(i for i in os.listdir(self.imgpath) if i.startswith(self.video_name)))
        #get len of this array
        self.len = len(self.imgs)
        
        

    def __getitem__(self, idx):
        #get vid_idx
        filename = self.imgs[idx]
        #get image path
        #imfile=os.path.join(self.imgpath,self.imgs[idx])
        #get all rows with same image name
        framegroup = self.csvdata.loc[self.csvdata['filename'] == self.imgs[idx]] 
        #print(framegroup)
        #get vid index for this image
        vid_idx = framegroup['video_number'].iloc[0]

        # extract the corresponding frames 
        vidlist= self.csvdata.loc[self.csvdata['video_number'] == vid_idx]
        #print(vidlist)
        # group by the image names, because there might be multiple rows when there is more than one object annotated in the video
        vidgrouped= vidlist.groupby(vidlist["filename"])
        # safe here all the images and the targets
        #input_list =[]
        #target_list=[]
        filename_list = []
        vidgrouped_list = []
        #get list of frame name in the videos
        for name, group in vidgrouped:
            filename_list.append(name)
            vidgrouped_list.append(group)
        #print(filename_list)
        #get filename idx, but in the videos
        filename_idx = filename_list.index(filename)
        #if it's the fist or last frame of videos, then use the next frame to it
        #because these frame have no last or future frame for target data
        if filename_idx == 0:
            filename = filename_list[1]
            filename_idx = 1
        elif filename_idx == len(filename_list)-1:
            filename = filename_list[-2]
            filename_idx =  len(filename_list)-2

        i = filename_idx

        # construct the path to the image and load it
        #current image
        imfile=os.path.join(self.imgpath,filename_list[i])
        img= Image.open(imfile)
        #last image
        last_imfile=os.path.join(self.imgpath,filename_list[i-1])
        last_img= Image.open(last_imfile)
        #future image
        future_imfile=os.path.join(self.imgpath,filename_list[i+1])
        future_img= Image.open(future_imfile)

        # convert to RGB 
        img = img.convert('RGB')
        last_img = last_img.convert('RGB')
        future_img = future_img.convert('RGB')

        #PROCESSING INPUT DATA
        last_boxes = []
        last_tracks = []
        #last mask
        last_masks=np.zeros((1,img.size[1],img.size[0]),dtype=np.uint8)
        last_imgMask = Image.new('L', (img.size[0],img.size[1]), 0)

        for _, frame in vidgrouped_list[i-1].iterrows():
                xpoint_str=frame.loc["xpoints"]
                ypoint_str=frame.loc["ypoints"]
                
                # convert to int list
                xpoints=list(map(int, xpoint_str.split(',')))
                ypoints=list(map(int, ypoint_str.split(',')))
                
                # generate the mask from the polyline
                points=[]
                for j in range(len(xpoints)):
                    points.append(xpoints[j])
                    points.append(ypoints[j])
                # generate the mask with the poly line
                ImageDraw.Draw(last_imgMask).polygon(points, outline=1, fill=1)

                # extract the bounding box information from the polyline
                xmin = min(xpoints)
                ymin = min(ypoints)
                xmax = max(xpoints)
                ymax = max(ypoints)
                last_boxes.append([xmin,ymin,xmax,ymax])

                #get track id
                last_tracks.append(frame.loc["track"])

        last_masks[0] = np.array(last_imgMask)
        #PROCESSING TARGET DATA
        # Get the number of objects / animals by extracting the region count value
        num_objs= vidgrouped_list[i].iloc[0].loc["region_count"]
       
        boxes=[]        
        # generate the binary masks
        masks=np.zeros((num_objs ,img.size[1],img.size[0]),dtype=np.uint8)
        # area of the segments and iscrowd attribute
        #area=torch.zeros((num_objs,),dtype=torch.float32)
        iscrowd =torch.zeros((num_objs,), dtype=torch.int64)
        # save the labels
        labels=torch.zeros((num_objs,), dtype=torch.int64)
            
        # save the track number
        tracks=torch.zeros((num_objs,), dtype=torch.int64)
            
        # count the segments
        count=0
        for _, frame in vidgrouped_list[i].iterrows():
                #print(frame)
                # extract the polygon points and split by defined marker ;
     
                xpoint_str=frame.loc["xpoints"]
                ypoint_str=frame.loc["ypoints"]
                
                # convert to int list
                xpoints=list(map(int, xpoint_str.split(',')))
                ypoints=list(map(int, ypoint_str.split(',')))
                
                # generate the mask from the polyline
                points=[]
                for j in range(len(xpoints)):
                    points.append(xpoints[j])
                    points.append(ypoints[j])
                # generate the mask with the poly line
                imgMask = Image.new('L', (img.size[0],img.size[1]), 0)
                ImageDraw.Draw(imgMask).polygon(points, outline=1, fill=1)
                masks[count] = np.array(imgMask)
                
                # get the area of the segment
                #area[count]=cv.countNonZero(masks[count])
                # is crowd always to 0, should indicate overlap, but is here not interesting for us
                iscrowd[count]=0
                
                
                # extract the bounding box information from the polyline
                xmin = min(xpoints)
                ymin = min(ypoints)
                xmax = max(xpoints)
                ymax = max(ypoints)
                boxes.append([xmin,ymin,xmax,ymax])
                
                # set the label
                labels[count]=frame.loc["class"]
                # set the track number
                tracks[count]=frame.loc["track"]
                
                count+=1

        #GET EVERYTHING TOGETHER
        # convert the np array to a tensor
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        # convet the bounding boxes to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # generate image id part, not really relevant    
        image_id=  frame.loc["filename"]     
        future_img = F.to_tensor(future_img)
        #target 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id 
        #target["area"] = area
        target["iscrowd"] = iscrowd
        target["track"]=tracks 
        target["recons_img"] = future_img
            

        #convert to Tensor 
        img = F.to_tensor(img)
        last_img = F.to_tensor(last_img)
        last_masks = torch.as_tensor(last_masks, dtype=torch.uint8)
        con_img = torch.cat((img, last_img, last_masks), 0)
        last_boxes = torch.as_tensor(last_boxes, dtype=torch.float32)
        last_tracks = torch.as_tensor(last_tracks, dtype=torch.int64)

        #input
        input = {}
        input["img"] = img
        input["con_img"] = con_img
        input["boxes"] = last_boxes
        input["track"] = last_tracks
            
        # in transforms the PIL image will be converted to a pytorch tensor
        if self.transforms is not None:
                input, target = self.transforms(input, target)
            
        return input, target

    def __len__(self):
        return self.len