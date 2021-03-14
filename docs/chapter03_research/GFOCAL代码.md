### 代码路径：`mmdet/models/detectors/single_stage.py`

```python
def forward_train(self,img,img_metas,gt_bboxes,gt_labels,gt_bboxes_ignore=None):
    x = self.extract_feat(img)  
    '''
    len(x)=5
    x[0].shape=[1,256,136,100]
    x[1].shape=[1,256,68,50]
    x[2].shape=[1,256,34,25]
    x[3].shape=[1,256,17,13]
    x[4].shape=[1,256,9,7]
    '''

    outs = self.bbox_head(x)
    '''
    len(outs)=2
    len(outs[0])=5,len(outs[1])=5
    outs[0][0].shape=[1,80,136,100],outs[0][0].shape=[1,80,136,100]
    outs[0][1].shape=[1,80,68,50],outs[1][1].shape=[1,80,68,50]
    outs[0][2].shape=[1,80,34,25],outs[1][2].shape=[1,80,34,25]
    outs[0][3].shape=[1,80,17,13],outs[1][3].shape=[1,80,17,13]
    outs[0][4].shape=[1,80,9,7],outs[1][4].shape=[1,80,9,7]
    '''
    loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
    '''
    gt_bboxes[0].shape=[1,4]  11个目标
    gt_labels[0].shape=[1]
    '''
    losses = self.bbox_head.loss(*loss_inputs,gt_bboxes_ignore=gt_bboxes_ignore)
    return losses
```

### 代码路径：`mmdet/models/anchor_heads/gfl_head.py` 
```python
def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, cfg, gt_bboxes_ignore=None):

    featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    '''
    featmap_sizes[0]=[136,100]
    featmap_sizes[1]=[68,50]
    featmap_sizes[2]=[34,25]
    featmap_sizes[3]=[17,13]
    featmap_sizes[2]=[9,7]
    '''
    
    anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
    '''
    anchor_list[0][0].shape=[13600,4]
    anchor_list[0][1].shape=[3400,4]
    anchor_list[0][2].shape=[850,4]
    anchor_list[0][3].shape=[221,4]
    anchor_list[0][4].shape=[63,4]
    '''
    label_channels = 80
    cls_reg_targets = self.gfl_target(
        anchor_list,
        valid_flag_list,
        gt_bboxes,
        img_metas,
        cfg,
        gt_bboxes_ignore_list=gt_bboxes_ignore,
        gt_labels_list=gt_labels,
        label_channels=label_channels)
    if cls_reg_targets is None:
        return None

    (anchor_list, labels_list, label_weights_list, bbox_targets_list,bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets    
    num_total_samples = reduce_mean(torch.tensor(num_total_pos).cuda()).item()
    num_total_samples = max(num_total_samples, 1.0)    
    losses_qfl, losses_bbox, losses_dfl,avg_factor = multi_apply( self.loss_single, anchor_list, cls_scores, bbox_preds, labels_list, label_weights_list, bbox_targets_list, self.anchor_strides, num_total_samples=num_total_samples, cfg=cfg)    
    avg_factor = sum(avg_factor)
    avg_factor = reduce_mean(avg_factor).item()
    losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
    losses_dfl  = list(map(lambda x: x / avg_factor, losses_dfl))
    return dict(loss_qfl=losses_qfl,loss_bbox=losses_bbox,loss_dfl=losses_dfl)
```