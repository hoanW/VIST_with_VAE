import torch, torchvision
from torch import nn, Tensor, optim
from torch.nn import functional as nnF
from torch.autograd import Variable
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from torchvision.models.resnet import conv1x1
from typing import List

from .run import model_params, mask_rcnn_model
from .utils import BaseVAE, BasicBlock, RoIAlign, TransBasicBlock, apply_box_deltas, aug_postprocess, batch_bb_matching, pro_postprocess
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#Mask R-CNN
def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

#Faster R-CNN
def get_object_detection_model(num_classes):
    # load an faster RCNN model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

#Dectector
detector = mask_rcnn_model
def bbox_detector(img):
        with torch.no_grad():
            pred = detector(img)

        bbox_mask_rcnn=[]
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        idx = [pred_score.index(x) for x in pred_score if x > model_params["DETECTION_THRESHOLD"]]
        bbox_mask_rcnn = pred[0]["boxes"][idx, :]
        
        return bbox_mask_rcnn.detach()

#VAE
class Modified_VAE(BaseVAE):
    def __init__(self,
                 params,               #training parameters
                 in_channels: int,   #channels of encoding input
                 layers: List = None,   #number of layers for each residual blocks
                 fc_dims: List = None,   #number of fc dims
                 latent_dim: int = None,  
                 n_classes: int = None,    
                 train_mode = False,              #training or not       
                 **kwargs) -> None:
        super(Modified_VAE, self).__init__()
        self.params = params
        self.train_mode = train_mode
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.initialize_weights()

        #Define ResNet's properties 
        block = BasicBlock
        transblock = TransBasicBlock
        if layers == None:
          layers = [3, 4, 6, 3]
        self.inplanes = 64    
        
        #Build Encoder
        #ResNet layer
        self.conv1 = nn.Conv2d(self.in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)                               #(64, 384, 640)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   #(64, 192, 320)
        self.layer1 = self._make_layer(block, 64, layers[0])              #(64, 192, 320)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)   #(128, 96, 160)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)   #(256, 48, 80)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  #(512, 24, 40)
        self.avgpool = nn.AdaptiveAvgPool2d((3, 5))                       #(512, 3, 5)

        #Fc layers
        fc_modules = []
        if fc_dims is None: 
            fc_dims = [1024, 512]
        fc_in_dim = 512*3*5

        for fc_dim in fc_dims:
            fc_modules.append(
                nn.Sequential(
                    nn.Linear(fc_in_dim, fc_dim),
                    nn.LeakyReLU())
            )
            fc_in_dim = fc_dim


        self.encoder_fc = nn.Sequential(*fc_modules)
        self.fc_mu = nn.Linear(fc_dims[-1], latent_dim)
        self.fc_var = nn.Linear(fc_dims[-1], latent_dim)


        # Build Decoder
        fc_modules = []
        self.inplanes = 512

        #Fc layers
        fc_dims.reverse()
        fc_dims.append(512*3*5)
        fc_in_dim = latent_dim

        for fc_dim in fc_dims:
            fc_modules.append(
                nn.Sequential(
                    nn.Linear(fc_in_dim, fc_dim),
                    nn.LeakyReLU())
            )
            fc_in_dim = fc_dim
        self.decoder_fc = nn.Sequential(*fc_modules)

        #Subpixel ResNet layer
        self.de_layer1 = self._make_transpose(transblock, 512, layers[-1], stride=2) #(512, 6, 10)
        self.de_layer2 = self._make_transpose(transblock, 512, layers[-2], stride=2)  #(512, 12, 20)
        self.de_layer3 = self._make_transpose(transblock, 512, layers[-3], stride=2)  #(512, 24, 40)
        self.de_layer4 = self._make_transpose(transblock, 256, layers[-4], stride=2)  #(256, 48, 80)
        self.de_layer5 = self._make_transpose(transblock, 128, 1, stride=2)  #(128, 96, 160)
        self.de_layer6 = self._make_transpose(transblock, 64, 1, stride=2)  #(64, 192, 320)
        self.de_layer7 = self._make_transpose(transblock, 64, 1, stride=2)  #(64, 384, 640)                                          
        

        #Final conv
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(64,
                                               8,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(8),
                            nn.LeakyReLU(),
                            nn.Conv2d(8, out_channels= 3,
                                      kernel_size= 3, padding =1),
                            nn.Sigmoid())
        
        #Define proposal branch properties
        #Original size (1280, 768)
        self.n_classes = n_classes
        #Construction
        self.pro_POOLING_SIZE = 14
        self.pro_SPATIAL_SCALE = 1.0/8.0  #pooling from DevCon5 (Conv x2) has size of DeConv5:  (1, 128, 96, 160) 
        self.pro_roi_align = RoIAlign(self.pro_POOLING_SIZE, self.pro_SPATIAL_SCALE, 2)

        pro_representation_size = 128 * self.pro_POOLING_SIZE * self.pro_POOLING_SIZE #64 is channel_size of pooled image (Deconv4)

        #Build FasterRCNNPredictor
        self.pro_fc1 = nn.Linear(pro_representation_size, 2048)
        self.pro_fc2 = nn.Linear(2048, 1024)
        self.pro_relu = nn.ReLU(inplace =True)
        self.pro_bbox_fc = nn.Linear(1024, 4) #self.n_classes*4

        #Define augment branch properties

        self.aug_POOLING_SIZE = 14
        self.aug_SPATIAL_SCALE = 1.0/16.0 #pooling from Conv x3 has size of (1, 256, 48, 80) 
        self.aug_roi_align = RoIAlign(self.aug_POOLING_SIZE, self.aug_SPATIAL_SCALE, 2)

        aug_representation_size = 256 * self.aug_POOLING_SIZE * self.aug_POOLING_SIZE #512 is channel_size of pooled image (x4)
        self.aug_fc1 = nn.Linear(aug_representation_size, 2048)
        self.aug_fc2 = nn.Linear(2048, 1024)
        self.aug_relu = nn.ReLU(inplace =True)
        self.aug_logits_fc = nn.Linear(1024, self.n_classes)
        self.aug_bbox_fc = nn.Linear(1024, self.n_classes*4)
        
        #FCN for generating mask
        self.mask_fcn1 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.mask_fcn2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.mask_fcn3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.mask_fcn4 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.mask_relu = nn.ReLU(inplace =True)
        self.mask_conv5 = nn.ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2))
        self.mask_fcn_logits = nn.Conv2d(256, 5, kernel_size=(1, 1), stride=(1, 1))
        self.mask_sigmoid = nn.Sigmoid()

        #Roi for loss
        self.loss_roi_align = RoIAlign(self.aug_POOLING_SIZE*2, 1, 2) #pool form original mask (1, 1280, 768)
        self.relu = nn.ReLU(inplace =True)


    def encode(self, input: Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        :return: (Tensor) List of multi-scale feature maps
        """
        #print("Input size:", input.size())
        #print("Encoding:....")
        #ResNet layer
        x = self.conv1(input)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)  #(64, 192, 320)
        x2 = self.layer2(x1) #(128, 96, 160)
        x3 = self.layer3(x2)  #(256, 48, 80)
        x4 = self.layer4(x3)   #(512, 24, 40)
        x5 = self.avgpool(x4)  #(512, 3, 5)

        #Fc layer
        x = torch.flatten(x5, start_dim=1)
        result = self.encoder_fc(x)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var], [x1, x2, x3, x4, x5]
    
    def de_auxiliary(self, z: Tensor, x) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        #print("Decoding Aux:....")
        x1, x2, x3, x4, x5 = x
        #Fc layer
        z = self.decoder_fc(z)
        z = z.view(-1,512, 3, 5)

        #Trans Conv ResNet layer
        z = self.de_layer1(z)
        z = self.de_layer2(z)
        z = self.de_layer3(z)
        z = self.de_layer4(z)
        z = self.de_layer5(z)
        z = self.de_layer6(z)
        z = self.de_layer7(z)
        recons = self.final_layer(z)
        return recons


    def de_proposal(self, z: Tensor, x: List[Tensor], last_bboxes: List[Tensor]) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :param x: List(Tensor) List of Skipping Connection Feature Maps
        :param last_bboxes: List(Tensor) List of RoIs (G, 5) [x1 ,y1 ,x2, y2, class] int
        :return bb_pred: (Tensor) (G,4)
        :return cls_prob (Tensor) (G,5)
        """
        #print("Decoding Pro:....")
        #Unpacking encoding multi-scale feature maps for skipping connection
        x1, x2, x3, x4, x5 = x
        #Fc layer
        z = self.decoder_fc(z)
        z = z.view(-1,512, 3, 5)

        #Trans Conv ResNet layer
        z = self.de_layer1(z+x5)    (512, 6,10)
        z = self.de_layer2(z)  (512, 12, 20)
        z = self.de_layer3(z)   (512, 24, 40)
        z = self.de_layer4(z + x4)    #(256, 48, 80)
        z = self.de_layer5(z + x3)    #(128, 96, 160)

        #RoI Align
        pooled_features = self.pro_roi_align(z, last_bboxes)
        z = pooled_features.view(pooled_features.size()[0], -1)
       
        #BBox regression
        z = self.pro_fc1(z)
        z = self.pro_relu(z)
        z = self.pro_fc2(z)
        z = self.pro_relu(z)
        bbox_pred = self.pro_bbox_fc(z)
        pro_rois = Variable(torch.cat(last_bboxes, 0).to(device), requires_grad=False)

        result ={}
        if self.train_mode:
             result["rois"] = pro_rois
             result["boxes"] = bbox_pred
        else:
             
             result["boxes"] = apply_box_deltas(pro_rois, bbox_pred)
        return result

    def de_augment(self, z: Tensor, x: List[Tensor], bboxes_pro: Tensor, bboxes_mask_rcnn: Tensor, last_id: List[Tensor]) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param x: List(Tensor)
        :return: (Tensor) dictionary of bboxes, mask, class
        """
        #print("Decoding Aug:....")
        ##Unpacking encoding multi-scale feature maps for skipping connection
        x1, x2, x3, x4, x5 = x
        #Fc layer
        z = self.decoder_fc(z)
        z = z.view(-1,512, 3, 5)

        #Trans Conv ResNet layer
        z = self.de_layer1(z + x5)   (512, 6,10)
        z = self.de_layer2(z)   (512, 12, 20)
        z = self.de_layer3(z)   (512, 24, 40)
        z = self.de_layer4(z + x4)    #(256, 48, 80)
        ft_maps = z + x3   #(256, 48, 80)
        #BBoxes matching cost with IoU, pairing with Hungarian algorithmn
        bboxes, ids = batch_bb_matching([bboxes_pro], [bboxes_mask_rcnn], last_id) 
        
        #RoI Align
        pooled_features = self.aug_roi_align(ft_maps, [bboxes])

        z = pooled_features.view(pooled_features.size()[0], -1)

        #FC
        z = self.aug_fc1(z)
        z = self.aug_relu(z)
        z = self.aug_fc2(z)
        z = self.aug_relu(z)
        #Classifier
        cls_logits = self.aug_logits_fc(z)
        cls_prob = nnF.softmax(cls_logits, 1)
        cls_scores, cls_labels = torch.max(cls_prob, 1)

        #Get corresponding label index
        idx = torch.arange(cls_labels.size()[0]).long()

        #BBox regression
        bbox_pred = self.aug_bbox_fc(z)
        bbox_pred = bbox_pred.view(bbox_pred.size()[0], -1, 4)
        bbox_pred = bbox_pred[idx, cls_labels]

        #Mask with FCN layer
        x = self.mask_fcn1(pooled_features)
        x = self.mask_relu(x)
        x = self.mask_fcn2(x)
        x = self.mask_relu(x)
        x = self.mask_fcn3(x)
        x = self.mask_relu(x)
        x = self.mask_fcn4(x)
        x = self.mask_relu(x)
        x = self.mask_conv5(x)
        x = self.mask_relu(x)
        x = self.mask_fcn_logits(x)
        

        result = {}
    
        if self.train_mode:
            masks = x[idx, cls_labels]
            labels = cls_logits
            result["rois"] = bboxes
        else:
            labels = cls_labels
            bbox_pred = apply_box_deltas(bboxes, bbox_pred)
            x = self.mask_sigmoid(x)
            masks = x[idx, cls_labels]
        
        #Final result dict       
        result["labels"] = labels
        result["score"] = cls_scores
        result["track"] = ids
        result["boxes"] = bbox_pred
        result["masks"] = masks
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        #Unpacking input data
        con_img, img, last_bbox, last_id = input
        #Encoding
        [mu, log_var], x = self.encode(con_img)
        #Reparameterization
        z = self.reparameterize(mu, log_var)
        #Decoding: Aux branch
        de_aux = self.de_auxiliary(z, x)
        #Decoding: Pro branch
        de_pro = self.de_proposal(z, x, last_bbox)
        if self.train_mode:
            bbox_pro = apply_box_deltas(de_pro["rois"], de_pro["boxes"])
        else:
            bbox_pro = de_pro["boxes"]
        #Mask RCNN
        bbox_mask_rcnn = bbox_detector(img)
        #Decoding: Aug branch
        de_aug = self.de_augment(z, x, bbox_pro, bbox_mask_rcnn, last_id)
        return  [de_aux, de_pro, de_aug, mu, log_var]

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.de_auxiliary(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.de_auxiliary(x)[0]


    def initialize_weights(self):
        """Initialize model weights.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
               m.weight.data.fill_(1)
               m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    #ResNet residual block layers
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                  kernel_size=2, stride=stride,
                                  padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride=stride),
                nn.BatchNorm2d(planes),
            )
     
        layers = []

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)


    def loss_function(self, pred, target,
                      mu, log_var) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        recons: predicted future image
        recons_gt : gt future image
      
        :param kwargs:
        :return:
        """

        alpha = self.params['kld_weight'] # Account for the minibatch samples from the dataset
        beta = self.params['recons_weight']

        #Auxiliary branch loss (reconstruction loss with MSE loss)
        recons, pro_pred_bbox, pro_rois, aug_pred_bbox, pred_prob_class, pred_masks, pred_tracking_id, aug_rois = pred
        target_recons, target_bbox, target_class, target_masks, target_track_ids = target
        recons_loss = nnF.mse_loss(recons, target_recons)

        #Latent variable loss (Distribution loss with KL Divergence)
        kld_loss = -0.5 * torch.mean(torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
      
        if target[1].size():
            pro_pred, pro_target  = pro_postprocess([pro_pred_bbox, pro_rois], target_bbox)  
            #Proposal branch loss (BBox regression with smooth L1 loss)      
            pro_boxes_loss = nnF.smooth_l1_loss(pro_pred, pro_target)
   
            #Augment branch
            aug_pred_bbox, pred_logits_class, pred_masks, \
              target_bbox, target_class, target_masks, roi_target_bbox = \
              aug_postprocess([aug_pred_bbox, pred_prob_class, pred_masks, aug_rois],[target_bbox, target_class, target_masks])
            aug_boxes_loss = nnF.smooth_l1_loss(aug_pred_bbox, target_bbox)
            if len(target_class) != 0:
                class_loss = nnF.cross_entropy(pred_logits_class, target_class)
                
                target_size = len(roi_target_bbox)                    
                matched_idxs = torch.tensor(range(target_size)).unsqueeze(1).to(device)
                roi_bbox = torch.cat([matched_idxs, roi_target_bbox], dim=-1)
                roi_target_masks = self.loss_roi_align(target_masks.unsqueeze(1).float(), roi_bbox) #unsqueeze(1) to add one channel to mask
                            
                masks_loss = nnF.binary_cross_entropy_with_logits(pred_masks, roi_target_masks.squeeze(1))
                
            else: 
                class_loss =  Variable(torch.FloatTensor([0]), requires_grad=False).to(device)
                masks_loss = Variable(torch.FloatTensor([0]), requires_grad=False).to(device)

            aug_loss = self.params["box_weight"] * aug_boxes_loss + class_loss + self.params["mask_weight"] * masks_loss

        else:
            pro_boxes_loss = Variable(torch.FloatTensor([0]), requires_grad=False).to(device)
            aug_loss = Variable(torch.FloatTensor([0]), requires_grad=False).to(device)

        
        #Total loss
        loss = beta * recons_loss + alpha * kld_loss + self.params["mask_weight"] * pro_boxes_loss + aug_loss

        return {'loss': loss, 'recons_loss':recons_loss, 'pro_loss': pro_boxes_loss, 
                'aug_loss': aug_loss, 'kld': -kld_loss}


    #Lightning module
    def training_step(self, batch, batch_idx):
        input, target = batch  
        input = [input["con_img"].to(device), input["img"].to(device), [input["boxes"].squeeze(0).to(device)], [input["track"].squeeze(0).to(device)]]
        
        #forward
        results = self.forward(input)
        
        #loss
        pred = [results[0], results[1]["boxes"], results[1]["rois"],
                results[2]["boxes"], results[2]["labels"], results[2]["masks"], results[2]["track"], results[2]["rois"]]
        pred = [i.to(device) for i in pred]
        target = [target["recons_img"], target["boxes"], target["labels"], target["masks"], target["track"]]
        target = [i.to(device) for i in target]
        train_loss = self.loss_function(pred, target, mu=results[3], log_var=results[4])
        
        self.log("Total loss/Train step", train_loss["loss"], on_step=True, on_epoch=True )
        self.log("Recons loss/Train step", train_loss["recons_loss"], on_step=False, on_epoch=True)
        self.log("Proposal boxes loss/Train step", train_loss["pro_loss"], on_step=False, on_epoch=True)
        self.log("Augment branch loss/Train step", train_loss["aug_loss"], on_step=False, on_epoch=True)
        self.log("KLD loss/Train step", train_loss["kld"], on_step=False, on_epoch=True)

        return train_loss
    
    def validation_step(self, batch, batch_idx):
        input, target = batch      
        input = [input["con_img"].to(device), input["img"].to(device), [input["boxes"].squeeze(0).to(device)], [input["track"].squeeze(0).to(device)]]

        #forward
        results = self.forward(input)
        
        pred = [results[0], results[1]["boxes"], results[1]["rois"],
                results[2]["boxes"], results[2]["labels"], results[2]["masks"], results[2]["track"], results[2]["rois"]]
        pred = [i.to(device) for i in pred]
        target = [target["recons_img"], target["boxes"], target["labels"], target["masks"], target["track"]]
        target = [i.to(device) for i in target]
        validation_loss = self.loss_function(pred, target, mu=results[3], log_var=results[4])

        self.log("Total loss/Validation epoch", validation_loss["loss"], on_step=True, on_epoch=True )
        return validation_loss
    
    def configure_optimizers(self):
        optims = []
        scheds = []
        #optim with Adam optimizer
        optimizer = optim.Adam(self.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)

        #a learning rate scheduler which decreases the learning rate by
        lr_scheduler = optim.lr_scheduler.StepLR(optims[0],
                                               step_size=self.params["LR_down_step"],
                                               gamma=self.params['schedule_gamme'])
        scheds.append(lr_scheduler)
        return optims, scheds