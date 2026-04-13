from ultralytics.nn.tasks import BaseModel, DetectionModel
from ultralytics.utils import LOGGER, RANK
import torch.nn as nn 
import torch
import torch.nn.functional as F
from ultralytics.nn.modules import Detect, Segment
from ultralytics.utils.loss import v8DetectionLoss
from typing import Tuple
from ultralytics.utils.tal import select_candidates_in_gts, make_anchors
from ultralytics.utils.torch_utils import (fuse_conv_and_bn, fuse_deconv_and_bn, initialize_weights, intersect_dicts,
                                           make_divisible, model_info, scale_img, time_sync)
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from typing import Optional

class GcBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv_mask_s = nn.ModuleList([nn.Conv2d(teacher_channels, 1, kernel_size=1) for teacher_channels in ch])
        self.conv_mask_t = nn.ModuleList([nn.Conv2d(teacher_channels, 1, kernel_size=1)  for teacher_channels in ch])
        self.channel_add_conv =nn.ModuleList([nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels//2, kernel_size=1),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels//2, teacher_channels, kernel_size=1)) for teacher_channels in ch])

    def forward(self, student_fm : list[torch.Tensor], teacher_fm : list[torch.Tensor]):
        teacher_gb = []
        student_gb = []
        for i in range(len(teacher_fm)):
            b,c,w,h = teacher_fm[i].shape
            tf = self.conv_mask_t[i](teacher_fm[i])
            tf = tf.view(b,1, w*h,1)
            tf = F.softmax(tf, dim = 2)
            tf = torch.matmul( teacher_fm[i].view(b,1,c,-1),tf).view(b,c,1,1)
            tf = self.channel_add_conv[i](tf) + teacher_fm[i]

            b,c,w,h = student_fm[i].shape
            sf = self.conv_mask_s[i](student_fm[i])
            sf = sf.view(b,1,w*h,1)
            sf = F.softmax(sf, dim = 2)
            sf = torch.matmul(student_fm[i].view(b,1,c,-1),sf ).view(b,c,1,1)
            sf = self.channel_add_conv[i](sf) + student_fm[i]

            teacher_gb.append(tf)
            student_gb.append(sf)

        return  student_gb ,teacher_gb

class TeacherModel(DetectionModel):
    def __init__(self, cfg, ch, nc, weight, verbose ):
        super().__init__(cfg,ch,nc,verbose)
        self.load(weight)

    def _predict_once(self, x, profile=False, visualize=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y = []
        with torch.no_grad():
            for m in self.model:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if isinstance(m, (Detect, Segment)):
                    teacher_fm = [_x.clone() for _x in x]
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output

        return teacher_fm
    
    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return lambda x, y : x

    

class DetectionDistillModel(BaseModel):
    def __init__(self, cfg, weights, tm: TeacherModel ,verbose):
        super().__init__()
        self.student  = DetectionModel(cfg, verbose=verbose and RANK == -1 )
        self.stride = self.student.stride
        self.model = self.student.model
        if weights:
            self.student.load(weights)
        teacher_ch = [ head[0].conv.weight.shape[1] for head in tm.model[-1].cv2]
        student_ch = [ head[0].conv.weight.shape[1] for head in self.student.model[-1].cv2]
        # Feature Adaption Layer
        self.fa_layer = nn.ModuleList([ nn.Conv2d(stu_in, tea_out, 1) 
                                       for stu_in, tea_out in zip(student_ch, teacher_ch)] )
        self.teacher_stride = tm.model[-1].stride
        self.global_loss = GcBlock(teacher_ch)
        self.ch = teacher_ch

        # Init weights, biases
        initialize_weights(self.student)
        if verbose:
            self.info()
            LOGGER.info('')

    def _predict_once(self, inputs, teacher_fm: Optional[list[torch.Tensor]] = None, profile=False, visualize=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            teacher_fm (list[Tensor] | None): The feature maps from the neck of the teacher.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if not self.training:
            return self.student._predict_once(inputs)
        y = []  # outputs
        # forward pass in student
        x = inputs
        for m in self.student.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if isinstance(m, (Detect, Segment)):
                _student_fm = [_x.clone() for _x in x]
            x = m(x)  # run
            y.append(x if m.i in self.student.save else None)  # save output

        student_result = x
        student_fm = [self.fa_layer[i](_student_fm[i]) for i in range(len(self.fa_layer))]
        student_gb, teacher_gb = self.global_loss(student_fm, teacher_fm)
        return student_fm, teacher_fm, student_gb, teacher_gb, student_result
    
    def forward(self, x, teacher_fm :  Optional[list[torch.Tensor]] = None,  *args, **kwargs):
        """
        Forward pass of the model on a single scale. Wrapper for `_forward_once` method.

        Args:
            x (torch.Tensor | dict): The input image tensor or a dict including image tensor and gt labels.
            teacher_fm (list[Tensor] | None): The feature maps from the neck of the teacher.

        Returns:
            (torch.Tensor): The output of the network.
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, teacher_fm, *args, **kwargs)
        return self.predict(x,  teacher_fm,*args, **kwargs)
    
    def predict(self, x, teacher_fm : Optional[list[torch.Tensor]] = None, profile=False, visualize=False, augment=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            teacher_fm (list[Tensor] | None): The feature maps from the neck of the teacher.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            augment (bool): Augment image during prediction, defaults to False.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, teacher_fm,profile, visualize)
    
    def loss(self, batch, teacher_fm : Optional[list[torch.Tensor]] = None, preds=None):
        """ 
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()

        preds = self.forward(batch['img'], teacher_fm) if preds is None else preds
        if self.training:
            return self.criterion(preds, batch)
        else:
            return self.criterion.detection_loss(preds, batch)
    
    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return DistillLoss(self)
    
class DistillLoss:
    """Implementation of <Focal and Global Knowledge Distillation for Detectors> (https://arxiv.org/abs/2111.11837)"""

    def __init__(self, model : DetectionDistillModel):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        self.detection_loss = v8DetectionLoss(model.student)
        m = model.student.model[-1]  # Detect() module
        self.stride = m.stride  # model strides
        self.device = m.stride.device
        self.xyxy2hw = torch.tensor([[-1,0],[0,-1],[1,0],[0,1]], device  = m.stride.device )
        self.mse = nn.MSELoss()
        self.alpha = 16e-3
        self.beta = 8e-4
        self.gamma = 8e-3
        self._lambda = 8e-6
        self.tau = 0.09
        self.channel_per_scale = model.ch
        self.anchor_per_scale = [ (((model.student.args.imgsz) // s)**2).type(torch.long) for s in model.stride ]

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out
    
    def get_mask(self, batch, fm : list[torch.Tensor], anchor_per_scale):
        """Get binary and scale mask from ground truth information

        Args:
            batch (dict): Batch including ground truth information.
            fm (list[torch.Tensor]): Feature maps from the neck of the architecture.
            anchor_per_scale (_type_): List of the number of anchors per each feature map.

        Returns:
            tuple(Tensor, Tensor): scale mask, binary mask (batch_size, 1, total_num_anchor)
        """
        anchor_points, stride_tensor = make_anchors(fm, self.stride, 0.5)
        dtype = fm[0].dtype
        batch_size = fm[0].shape[0]
        device = fm[0].device
        imgsz = torch.tensor(fm[0].shape[2:], device=device, dtype=dtype) * self.stride[0]  # image size (h,w)
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt : torch.Tensor = gt_bboxes.sum(2, keepdim=True).gt_(0).type(torch.bool) # (batch_num, max_num_box, 1)
        if gt_bboxes.shape[1] != 0:
            gt_projection = select_candidates_in_gts(anchor_points * stride_tensor,gt_bboxes) # (batch_num, max_num_box, anchor_num)
            gt_projection[~mask_gt.expand_as(gt_projection)] = 0 # padding box에 대해서는 0 처리
            binary_mask = gt_projection.sum(1, keepdim=True).gt_(0).type(torch.bool) # (batch_num, 1, anchor_num)
        else: 
            binary_mask = torch.zeros(batch_size, 1, (anchor_points).shape[0], dtype= torch.bool, device = device)
        _scale_mask = torch.zeros_like(binary_mask, dtype= dtype, device = device)
        wh = gt_bboxes@self.xyxy2hw.type(gt_bboxes.dtype)
        area = (wh[...,0:1] * wh[...,1:2])
        if gt_bboxes.shape[1] != 0:
            temp_mask = torch.nonzero(gt_projection)
            i,j,k = temp_mask.t()
            _scale_mask[i,0,k] = area[i,j,0]
        prev_anchor = 0
        for i, num_anchors in enumerate(anchor_per_scale):
            for j in range(batch_size):
                _scale_mask[j,0,prev_anchor :prev_anchor+num_anchors][~binary_mask[j,0,prev_anchor :prev_anchor+num_anchors]] = (~binary_mask[j,0,prev_anchor :prev_anchor+num_anchors]).sum().type(dtype)
            prev_anchor += num_anchors
        scale_mask = 1 / _scale_mask
        return scale_mask, binary_mask
    
    def get_attention(self, fm : list[torch.Tensor],anchor_per_scale, channel_per_scale ):
        """Get attention mask from feature map"""
        batch_size = fm[0].shape[0]

        scale_mean_list =  [(1/channel_per_scale[i]) * _fm.abs().mean(dim=(1)).view(batch_size, 1,  -1 ) for i, _fm in enumerate(fm)] # (batch_size, 1, anchor_num)
        channel_mean_list = [(1/anchor_per_scale[i]) * _fm.abs().mean(dim=(2,3)).view(batch_size, channel_per_scale[i], -1 ) for i, _fm in enumerate( fm)] #  (batch_size, channel_num, 1)
        scale_attention = torch.concat([anchor_per_scale[i] * F.softmax(scale_mean / self.tau, dim = 2) for i, scale_mean in enumerate(scale_mean_list)], dim = 2)
        channel_attention = [channel_per_scale[i] * F.softmax(channel_mean / self.tau, dim = 1) for i, channel_mean in enumerate(channel_mean_list)]
        return scale_attention, channel_attention

    def __call__(self, preds : Tuple[torch.Tensor], batch):

        student_fm, teacher_fm,  student_gb, teacher_gb, student_result = preds        
        anchor_per_scale = self.anchor_per_scale
        channel_per_scale = self.channel_per_scale
        batch_size ,c,h,w = teacher_fm[0].shape
        student_loss, loss_items = self.detection_loss(student_result, batch)
        scale_mask, binary_mask = self.get_mask(batch, teacher_fm, anchor_per_scale)
        t_scale_att, t_ch_att = self.get_attention(teacher_fm, anchor_per_scale, channel_per_scale)
        s_scale_att, s_ch_att = self.get_attention(student_fm, anchor_per_scale, channel_per_scale)
        loss_feature = torch.tensor(0, dtype = student_fm[0].dtype, device = student_fm[0].device)
        prev_anc = 0
        for i, (s_fm, t_fm) in enumerate(zip(student_fm, teacher_fm)):
            loss_feature += (
            self.alpha * (torch.square((s_fm - t_fm)).view(batch_size, channel_per_scale[i], -1) * binary_mask[...,prev_anc : prev_anc + anchor_per_scale[i]] * 
                scale_mask[...,prev_anc : prev_anc + anchor_per_scale[i]] * 
                t_scale_att[...,prev_anc : prev_anc + anchor_per_scale[i]] *  t_ch_att[i]).sum() 
            + self.beta * (torch.square((s_fm - t_fm)).view(batch_size, channel_per_scale[i], -1) * ~binary_mask[...,prev_anc : prev_anc + anchor_per_scale[i]] * 
                scale_mask[...,prev_anc : prev_anc + anchor_per_scale[i]] * 
                t_scale_att[...,prev_anc : prev_anc + anchor_per_scale[i]] * 
                t_ch_att[i]).sum()) 
            prev_anc += anchor_per_scale[i]
        loss_attention = self.gamma * (F.l1_loss(t_scale_att, s_scale_att) + F.l1_loss(torch.concat(t_ch_att,dim = 1), torch.concat(s_ch_att, dim = 1)))

        loss_focal = loss_attention + loss_feature
        loss_global = torch.tensor(0, dtype = student_fm[0].dtype, device = student_fm[0].device)
        for s, t in zip(student_gb, teacher_gb):
            loss_global += self.mse(s,t)

        loss_global *= self._lambda

        return student_loss + loss_global + loss_focal, loss_items
