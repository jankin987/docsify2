nanodet/trainer/trainer.py

86行

### meta:

img:[1,3,320,320]

img_info:

gt_bboxes

gt_labels

wrap_matrix



标注：

ann['bboxes']
array([[225.15,  54.93, 351.26, 355.13],
       [191.78,   6.73, 334.01, 325.66],
       [224.6 , 123.23, 245.78, 153.88]], dtype=float32)
ann['labels']
array([17,  0, 24])





## 调试记录

grid坐标
```
0.,  16.,  32.,  48.,  64.,  80.,  96., 112., 128., 144., 160., 176., 192., 208., 224., 240., 256., 272., 288., 304.
```

### 代码1
```
断点顺序：
1. `tools/train.py` 
   - 11行
   - 125行 建立 model



`mmdet/models/detectors/single_stage.py`

- 27行 SingleStageDetector 初始化

`mmdet/models/builder.py`  15行 build函数



1. `mmdet/models/anchor_heads/gfl_head.py` 
  - 53行 GFLHead类初始化
  - 67行 self.loss_qfl和loss_dfl初始化 
3. `tools/train.py` 142行 train_detector()函数



3. `mmdet/apis/train.py`166行 runner.run()函数

`mmdet/datasets/samplers/group_sampler.py` GroupSampler类
  - 48行  __len__ 函数
  -  

`mmdet/apis/train.py`  （主要训练的函数）
  - 59行 batch_processor()函数 计算loss

`mmdet/core/fp16/decorators.py` 
  - 49行 auto_fp16_wrapper函数 暂时不知道什么用 

`mmdet/models/detectors/single_stage.py`

  -  71行 SingleStageDetector类的forward_train函数，有提取特征等操作

`mmdet/core/fp16/decorators.py` 
  - 127行 force_fp32函数 暂时不知道什么用 

`mmdet/models/anchor_heads/gfl_head.py`

  - 216行 进入GFLHead类 loss函数
  - 243行 GFLHead类 loss函数中 计算losses_qfl,losses_bbox, losses_dfl,avg_factor

`mmdet/core/utils/misc.py` 22行 multi_apply函数



`mmdet/ops/conv_module.py` ConvModule 类

5. `mmdet/models/anchor_heads/gfl_head.py`
  - 149行  loss_single()函数
  - 190行   loss_single()函数中的loss_dfl
6. `mmdet/models/losses/gfocal_loss.py` 
  - 108行 DistributionFocalLoss类的forward函数
  - 42行 distribution_focal_loss()函数
  - 75行 QualityFocalLoss类的forward函数
  - 16行 quality_focal_loss()函数
```


### 代码2
```
def loss_single(self, grid_cells, cls_score, bbox_pred, labels,
                    label_weights, bbox_targets, stride, num_total_samples):

        grid_cells = grid_cells.reshape(-1, 4)             （1600，4）  1600个锚框
        cls_score = cls_score.permute(0, 2, 3,1).reshape(-1, self.cls_out_channels)          （1600,80） 预测的分类的分数
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))         （1600,32）  计算得到的lftb
        bbox_targets = bbox_targets.reshape(-1, 4)                                              1600,4）  标注的bbox
        labels = labels.reshape(-1)                                                             (1600)标注的label
        label_weights = label_weights.reshape(-1)                                               (1600)全是1

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = torch.nonzero((labels >= 0)                                                  2  只有2个位置有效
                                 & (labels < bg_class_ind), as_tuple=False).squeeze(1)

        score = label_weights.new_zeros(labels.shape)                                            1600个0

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]                            （2,4）           取出两个位置标注的bbox，实际上是一样的
            pos_bbox_pred = bbox_pred[pos_inds]                                   （2,32）    取出两个位置预测的lftb
            pos_grid_cells = grid_cells[pos_inds]                            （2,4）           取出两个位置的锚框坐标
            pos_grid_cell_centers = self.grid_cells_to_center(pos_grid_cells) / stride    （2,2）    两个位置的锚框坐标的中心

            weight_targets = cls_score.detach().sigmoid()                       （1600,80）  分类预测的分数   
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]             先dim=1求最大值，得到1600，然后找pos的值，得到两个值
            pos_bbox_pred_corners = self.distribution_project(pos_bbox_pred)   根据预测的（2,32）的求积分结果，得到（2,4）,这里应该是lftb
            pos_decode_bbox_pred = distance2bbox(pos_grid_cell_centers,         根据选择的锚框的中心，和预测的的（2,4），得到（2,4）
                                                 pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride                 标注的对应在40*40特征图上的坐标，两个是一样的
            score[pos_inds] = bbox_overlaps(                                    求预测的和标注的iou，分别为0.0524,0.0504
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)            （2,32）变8,8
            target_corners = bbox2distance(pos_grid_cell_centers,                   标注的框和锚框的中心求lrtb
                                           pos_decode_bbox_targets,
                                           self.reg_max).reshape(-1)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,                                                      标注的框和锚框的中心求lrtb
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = torch.tensor(0).cuda()

        # qfl loss
        loss_qfl = self.loss_qfl(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples)

        return loss_qfl, loss_bbox, loss_dfl, weight_targets.sum()
```

### value

```
meta:
    img:[1,3,320,320]

    img_info:
        file_name: 302364.jpg
        height:359
        id:302364
        width:640
    gt_bboxes: 66.42354 , 131.44385 , 129.47855 , 281.54385
    gt_labels: 17,0,24


x = self.backbone(x)
'''
len(x)=3
x[0].shape=[1,116,40,40]
x[1].shape=[1, 232, 20, 20]
x[2].shape=[1, 464, 10, 10]
'''

x = self.fpn(x)
'''
len(x)=3
x[0].shape=[1, 96, 40, 40]
x[1].shape=[1, 96, 20, 20]
x[2].shape=[1, 96, 10, 10]
'''

x = self.head(x)
'''
len(x)=2
len(x[0])=3
len(x[1])=3
x[0][0].shape=[1, 80, 40, 40]
x[0][1].shape=[1, 80, 20, 20]
x[0][2].shape=[1, 80, 10, 10]

x[1][0].shape=[1, 32, 40, 40]
x[1][1].shape=[1, 32, 20, 20]
x[1][2].shape=[1, 32, 10, 10]
'''

cls_scores=x[0]
bbox_preds=x[1]

featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
featmap_sizes[0]=[40, 40], stride=8
featmap_sizes[1]=[20, 20], stride=16
featmap_sizes[2]=[10, 10], stride=32



cls_reg_targets=self.target_assign(batch_size, featmap_sizes, gt_bboxes,
                                             gt_bboxes_ignore, gt_labels, device=device)


[40,40],stride=8
cell_size=stride*5=40
[4.,  12.,  20.,  28.,  36.,  44.,  52.,  60.,  68.,  76.,  84.,  92.,
        100., 108., 116., 124., 132., 140., 148., 156., 164., 172., 180., 188.,
        196., 204., 212., 220., 228., 236., 244., 252., 260., 268., 276., 284.,
        292., 300., 308., 316]

锚点的个数：40*40，基本锚框的个数40*40,尺寸也是40

[20,20],stride=16
cell_size=stride*5=80
[8.,  24.,  40.,  56.,  72.,  88., 104., 120., 136., 152., 168., 184.,
        200., 216., 232., 248., 264., 280., 296., 312.,  ]
锚点的个数：20*20，基本锚框的个数20*20,尺寸是80

[10,10],stride=32
cell_size=stride*5=160
[16.,  48.,  80., 112., 144., 176., 208., 240., 272., 304 ]
锚点的个数：10*10，基本锚框的个数10*10,尺寸是160

mlvl_grid_cells_list[0][0].shape=[1600,4]
mlvl_grid_cells_list[0][1].shape=[400,4]
mlvl_grid_cells_list[0][2].shape=[100,4]

num_level_cells=[1600,400,100]=num_level_cells_list
拼接：
mlvl_grid_cells_list[0].shape=[2100,4]


(all_grid_cells, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
            self.target_assign_single_img,   head模型
            mlvl_grid_cells_list,  
            num_level_cells_list,  [1600,400,100]
            gt_bboxes_list,  66.42354 , 131.44385 , 129.47855 , 281.54385
            gt_bboxes_ignore_list,  none
            gt_labels_list)  17,0,24

assign_result = self.assigner.assign(grid_cells, num_level_cells,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

gt_bboxes
tensor([[ 66.4235, 131.4438, 129.4785, 281.5439],
        [ 49.7385, 107.3438, 120.8535, 266.8088],
        [ 66.1485, 165.5938,  76.7385, 180.9189]


中心位置候选：
[1012,  930,  848],
[1052,  931,  849],
[1011,  890,  888],
[1051,  970,  889],
[1013,  929,  808],
[ 972,  891,  809],
[1053,  971,  847],
[ 971,  889,  850],
[1092,  969,  887],

[1846, 1825, 1804],
[1866, 1845, 1824],
[1845, 1824, 1803],
[1865, 1844, 1805],
[1847, 1826, 1823],
[1826, 1805, 1825],
[1867, 1846, 1784],
[1825, 1804, 1783],
[1886, 1806, 1844],

[2063, 2052, 2052],
[2062, 2062, 2051],
[2053, 2053, 2042],
[2052, 2063, 2062],
[2073, 2051, 2041],
[2072, 2061, 2053],
[2064, 2042, 2061],
[2061, 2043, 2043],
[2054, 2072, 2063]]+


[0.1691, 0.1411, 0.1014],
[0.1691, 0.1411, 0.1014],
[0.1691, 0.1411, 0.1014],
[0.1691, 0.1411, 0.1014],
[0.1691, 0.1411, 0.1014],
[0.1691, 0.1411, 0.1014],
[0.1691, 0.1411, 0.1014],
[0.1691, 0.1411, 0.1014],
[0.1691, 0.1411, 0.1014],

[0.4662, 0.4721, 0.0254],
[0.4662, 0.4721, 0.0254],
[0.4503, 0.3904, 0.0254],
[0.4503, 0.3904, 0.0254],
[0.3325, 0.3448, 0.0254],
[0.4662, 0.4721, 0.0254],
[0.3325, 0.3448, 0.0254],
[0.4503, 0.3904, 0.0254],
[0.4662, 0.3448, 0.0254],

[0.3697, 0.4009, 0.0063],
[0.3697, 0.3647, 0.0063],
[0.2886, 0.4009, 0.0063],
[0.2886, 0.3647, 0.0063],
[0.2797, 0.4009, 0.0063],
[0.2797, 0.3647, 0.0063],
[0.3697, 0.2896, 0.0063],
[0.3579, 0.2896, 0.0063],
[0.2886, 0.2588, 0.0063]

mean：0.3072, 0.2973, 0.0444
std: 0.1164, 0.1227, 0.0419



[1012, 3030, 5048],
[1052, 3031, 5049],
[1011, 2990, 5088],
[1051, 3070, 5089],
[1013, 3029, 5008],
[ 972, 2991, 5009],
[1053, 3071, 5047],
[ 971, 2989, 5050],
[1092, 3069, 5087],
[1846, 3925, 6004],
[1866, 3945, 6024],
[1845, 3924, 6003],
[1865, 3944, 6005],
[1847, 3926, 6023],
[1826, 3905, 6025],
[1867, 3946, 5984],
[1825, 3904, 5983],
[1886, 3906, 6044],
[2063, 4152, 6252],
[2062, 4162, 6251],
[2053, 4153, 6242],
[2052, 4163, 6262],
[2073, 4151, 6241],
[2072, 4161, 6253],
[2064, 4142, 6261],
[2061, 4143, 6243],
[2054, 4172, 6263]]



[0, 0, 1], [    ,     , 5048], [1012,  930,  848],
[0, 0, 1], [    ,     , 5049], [1052,  931,  849],
[0, 0, 1], [          , 5088], [1011,  890,  888],
[0, 0, 1], [            5089], [1051,  970,  889],
[0, 0, 0], [                ], [1013,  929,  808],
[0, 0, 0], [                 , [ 972,  891,  809],
[0, 0, 0], [                ], [1053,  971,  847],
[0, 0, 0], [                 , [ 971,  889,  850],
[0, 0, 0], [                ], [1092,  969,  887],

[1, 1, 0], [1846, 3925,     ], [1846, 1825, 1804],
[1, 1, 0], [1866, 3945,     ], [1866, 1845, 1824],
[1, 0, 0], [1845,     ,     ], [1845, 1824, 1803],
[1, 0, 0], [1865,     ,     ], [1865, 1844, 1805],
[0, 0, 0], [    ,     ,     ], [1847, 1826, 1823],
[1, 1, 0], [1826, 3905,     ], [1826, 1805, 1825],
[0, 0, 0], [    ,     ,     ], [1867, 1846, 1784],
[1, 0, 0], [1825,     ,     ], [1825, 1804, 1783],
[1, 0, 0], [1886,     ,     ], [1886, 1806, 1844],

[0, 0, 0], [                ], [2063, 2052, 2052],
[0, 0, 0], [                ], [2062, 2062, 2051],
[0, 0, 0], [                ], [2053, 2053, 2042],
[0, 0, 0], [                ], [2052, 2063, 2062],
[0, 0, 0], [                ], [2073, 2051, 2041],
[0, 0, 0], [                ], [2072, 2061, 2053],
[0, 0, 0], [                ], [2064, 2042, 2061],
[0, 0, 0], [                ], [2061, 2043, 2043],
[0, 0, 0]],[                ]] [2054, 2072, 2063]]



848,  849,  888,  889, 1805, 1825, 1826, 1845, 1846, 1865, 1866, 1886],
       device='cuda:0')
assigned_gt_inds[pos_inds]
tensor([3, 3, 3, 3, 2, 2, 1, 2, 1, 1, 1, 1], device='cuda:0')
assigned_gt_inds[pos_inds]-1
tensor([2, 2, 2, 2, 1, 1, 0, 1, 0, 0, 0, 0], device='cuda:0')
        24 24 24 24 0  0  17 0  17 17 17 17



AssignResult(
num_gt=3, 
assigned_gt_inds：每个候选框对应哪个label 1,2,3

max_overlaps

, labels=assigned_labels对应的类别

)

pos_inds:848,  849,  888,  889, 1805, 1825, 1826, 1845, 1846, 1865, 1866, 1886
neg_inds:[  2088 个值]

pos_gt_bboxes:=pos_bbox_targets
[ 66.1485, 165.5938,  76.7385, 180.9189],
[ 66.1485, 165.5938,  76.7385, 180.9189],
[ 66.1485, 165.5938,  76.7385, 180.9189],
[ 66.1485, 165.5938,  76.7385, 180.9189],
[ 49.7385, 107.3438, 120.8535, 266.8088],
[ 49.7385, 107.3438, 120.8535, 266.8088],
[ 66.4235, 131.4438, 129.4785, 281.5439],
[ 49.7385, 107.3438, 120.8535, 266.8088],
[ 66.4235, 131.4438, 129.4785, 281.5439],
[ 66.4235, 131.4438, 129.4785, 281.5439],
[ 66.4235, 131.4438, 129.4785, 281.5439],
[ 66.4235, 131.4438, 129.4785, 281.5439]

pos_assigned_gt_inds:[2, 2, 2, 2, 1, 1, 0, 1, 0, 0, 0, 0]

num_cells =2100

grid_cells: 2100个锚框


label_weights：2100个0
bbox_weights： (2100,4) 全是0
bbox_target： (2100,4) 全是0
labels：2100个80


grid_cells: (2100,4)
tensor([[-16., -16.,  24.,  24.],
        [ -8., -16.,  32.,  24.],
        [  0., -16.,  40.,  24.],
        ...,
        [160., 224., 320., 384.],
        [192., 224., 352., 384.],
        [224., 224., 384., 384.]

转换成了mlvl_grid_cells：     又转换成 grid_cells_list
len(mlvl_grid_cells)=3
mlvl_grid_cells[0]:(1600,4)
mlvl_grid_cells[1]:(400,4)
mlvl_grid_cells[2]:(100,4)


labels: (2100)
tensor([80, 80, 80,  ..., 80, 80, 80]

848 24
849 24
888 24
889 24
1805 0
1825 0
1826 17
1845 0
1846 17
1865 17
1866 17
1886 17

转换成了mlvl_labels：   又转换成labels_list
len(mlvl_labels)=3
mlvl_labels[0]:(1600)
mlvl_labels[1]:(400)
mlvl_labels[2]:(100)


labes_weights: (2100)  正负样本全是1
tensor([1., 1., 1.,  ..., 1., 1., 1.]
848 1.0
849 1.0
888 1.0
889 1.0
1805 1.0
1825 1.0
1826 1.0
1845 1.0
1846 1.0
1865 1.0
1866 1.0
1886 1.0

转换成了mlvl_label_weights： 又转换成label_weights_list
len(mlvl_label_weights)=3
mlvl_label_weights[0]:(1600)
mlvl_label_weights[1]:(400)
mlvl_label_weights[2]:(100)


bbox_targets:(2100,4)
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        ...,
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]
848 tensor([ 66.1485, 165.5938,  76.7385, 180.9189], device='cuda:0')
849 tensor([ 66.1485, 165.5938,  76.7385, 180.9189], device='cuda:0')
888 tensor([ 66.1485, 165.5938,  76.7385, 180.9189], device='cuda:0')
889 tensor([ 66.1485, 165.5938,  76.7385, 180.9189], device='cuda:0')
1805 tensor([ 49.7385, 107.3438, 120.8535, 266.8088], device='cuda:0')
1825 tensor([ 49.7385, 107.3438, 120.8535, 266.8088], device='cuda:0')
1826 tensor([ 66.4235, 131.4438, 129.4785, 281.5439], device='cuda:0')
1845 tensor([ 49.7385, 107.3438, 120.8535, 266.8088], device='cuda:0')
1846 tensor([ 66.4235, 131.4438, 129.4785, 281.5439], device='cuda:0')
1865 tensor([ 66.4235, 131.4438, 129.4785, 281.5439], device='cuda:0')
1866 tensor([ 66.4235, 131.4438, 129.4785, 281.5439], device='cuda:0')
1886 tensor([ 66.4235, 131.4438, 129.4785, 281.5439]

转换成了mlvl_bbox_targets:  又转换成bbox_targets_list
len(mlvl_bbox_targets)=3
mlvl_bbox_targets[0]:(1600,4)
mlvl_bbox_targets[1]:(400,4)
mlvl_bbox_targets[2]:(100,4)


bbox_weights: (2100,4)  
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        ...,
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]

848 tensor([1., 1., 1., 1.], device='cuda:0')
849 tensor([1., 1., 1., 1.], device='cuda:0')
888 tensor([1., 1., 1., 1.], device='cuda:0')
889 tensor([1., 1., 1., 1.], device='cuda:0')
1805 tensor([1., 1., 1., 1.], device='cuda:0')
1825 tensor([1., 1., 1., 1.], device='cuda:0')
1826 tensor([1., 1., 1., 1.], device='cuda:0')
1845 tensor([1., 1., 1., 1.], device='cuda:0')
1846 tensor([1., 1., 1., 1.], device='cuda:0')
1865 tensor([1., 1., 1., 1.], device='cuda:0')
1866 tensor([1., 1., 1., 1.], device='cuda:0')
1886 tensor([1., 1., 1., 1.], device='cuda:0')

转换成了mlvl_bbox_weights:  又转换成bbox_weights_list
len(mlvl_bbox_weights)=3
mlvl_bbox_weights[0]:(1600,4)
mlvl_bbox_weights[1]:(400,4)
mlvl_bbox_weights[2]:(100,4)


pos_inds:848,  849,  888,  889, 1805, 1825, 1826, 1845, 1846, 1865, 1866, 1886

 neg_inds

num_total_pos=12
num_total_neg=2088




self.loss_single 后面head的模型结构




candidate_idxs              candidate_overlaps              is_pos                      candidate_idxs          is_pos
[ 745,  666,  588],       [0.1691, 0.1411, 0.1014],      [False, False,  True],      [ 745, 2766, 4788],      [False, False,  True],
[ 785,  667,  628],       [0.1691, 0.1411, 0.1014],      [False, False,  True],      [ 785, 2767, 4828],      [False, False,  True],
[ 744,  626,  589],       [0.1691, 0.1411, 0.1014],      [False, False,  True],      [ 744, 2726, 4789],      [False, False, False],
[ 784,  706,  587],       [0.1691, 0.1411, 0.1014],      [False, False,  True],      [ 784, 2806, 4787],      [False, False, False],
[ 746,  627,  548],       [0.1691, 0.1411, 0.1014],      [False, False,  True],      [ 746, 2727, 4748],      [False, False, False],
[ 705,  665,  629],       [0.1691, 0.1411, 0.1014],      [False, False,  True],      [ 705, 2765, 4829],      [False, False, False],
[ 786,  707,  627],       [0.1691, 0.1411, 0.1014],      [False, False,  True],      [ 786, 2807, 4827],      [False, False, False],
[ 704,  625,  549],       [0.1691, 0.1411, 0.1014],      [False, False,  True],      [ 704, 2725, 4749],      [False, False, False],
[ 825,  705,  547],       [0.1691, 0.1411, 0.1014],      [False, False,  True],      [ 825, 2805, 4747],      [False, False, False],
[1792, 1773, 1754],       [0.4662, 0.4721, 0.0254],      [ True,  True, False],      [1792, 3873, 5954],      [ True,  True, False],
[1793, 1753, 1753],       [0.4041, 0.4721, 0.0254],      [False,  True, False],      [1793, 3853, 5953],      [False,  True, False],
[1772, 1772, 1734],       [0.4662, 0.3826, 0.0254],      [ True, False, False],      [1772, 3872, 5934],      [ True, False, False],
[1812, 1752, 1733],       [0.4662, 0.3826, 0.0254],      [ True, False, False],      [1812, 3852, 5933],      [ True, False, False],
[1791, 1774, 1774],       [0.3741, 0.3522, 0.0254],      [False, False, False],      [1791, 3874, 5974],      [False, False, False],
[1773, 1793, 1755],       [0.4041, 0.4721, 0.0254],      [False,  True, False],      [1773, 3893, 5955],      [False,  True, False],
[1813, 1754, 1773],       [0.4041, 0.3522, 0.0254],      [False, False, False],      [1813, 3854, 5973],      [False, False, False],
[1771, 1792, 1735],       [0.3741, 0.3826, 0.0254],      [False, False, False],      [1771, 3892, 5935],      [False, False, False],
[1811, 1794, 1775],       [0.3741, 0.3522, 0.0254],      [False, False, False],      [1811, 3894, 5975],      [False, False, False],
[2046, 2046, 2037],       [0.3631, 0.3956, 0.0063],      [False, False, False],      [2046, 4146, 6237],      [False, False, False],
[2056, 2036, 2036],       [0.3050, 0.3698, 0.0063],      [False, False, False],      [2056, 4136, 6236],      [False, False, False],
[2045, 2047, 2047],       [0.3631, 0.3956, 0.0063],      [False, False, False],      [2045, 4147, 6247],      [False, False, False],
[2055, 2037, 2046],       [0.3050, 0.3698, 0.0063],      [False, False, False],      [2055, 4137, 6246],      [False, False, False],
[2047, 2045, 2027],       [0.3631, 0.3956, 0.0063],      [False, False, False],      [2047, 4145, 6227],      [False, False, False],
[2036, 2035, 2026],       [0.2639, 0.3698, 0.0063],      [False, False, False],      [2036, 4135, 6226],      [False, False, False],
[2057, 2056, 2038],       [0.3050, 0.2851, 0.0063],      [False, False, False],      [2057, 4156, 6238],      [False, False, False],
[2035, 2057, 2048],       [0.2639, 0.2851, 0.0063],      [False, False, False],      [2035, 4157, 6248],      [False, False, False],
[2037, 2026, 2035]]       [0.2639, 0.2632, 0.0063]]      [False, False, False]],     [2037, 4126, 6235]]      [False, False, False]]



overlaps_mean_per_gt: 0.2982, 0.2970, 0.0444
overlaps_std_per_gt:0.1078, 0.1222, 0.0419
overlaps_thr_per_gt:[0.4059, 0.4192, 0.0862]

588,  628, 1753, 1772, 1773, 1792, 1793, 1812
[     ,      ,  True],
[     ,      ,  True],
[     ,      ,      ],
[     ,      ,      ],
[     ,      ,      ],
[     ,      ,      ],
[     ,      ,      ],
[     ,      ,      ],
[     ,      ,      ],

[ True,  True,      ],
[     ,  True,      ],
[ True,      ,      ],
[ True,      ,      ],
[     ,      ,      ],
[     ,  True,      ],
[     ,      ,      ],
[     ,      ,      ],
[     ,      ,      ],

[     ,      ,      ],
[     ,      ,      ],
[     ,      ,      ],
[     ,      ,      ],
[     ,      ,      ],
[     ,      ,      ],
[     ,      ,      ],
[     ,      ,      ],
[     ,      ,      ]]



grid_cells: (2100,4)
tensor([[-16., -16.,  24.,  24.],
        [ -8., -16.,  32.,  24.],
        [  0., -16.,  40.,  24.],
        ...,
        [160., 224., 320., 384.],
        [192., 224., 352., 384.],
        [224., 224., 384., 384.]

转换成了mlvl_grid_cells：     又转换成 grid_cells_list
len(mlvl_grid_cells)=3
mlvl_grid_cells[0]:(1600,4)
mlvl_grid_cells[1]:(400,4)
mlvl_grid_cells[2]:(100,4)


labels: (2100)
tensor([80, 80, 80,  ..., 80, 80, 80]

588     24
628     24
1753    0
1772    17
1773    0
1792    17
1793    0
1812    17

labes_weights: (2100)  正负样本全是1
tensor([1., 1., 1.,  ..., 1., 1., 1.]

588   1  
628   1 
1753  1      
1772  1      
1773  1      
1792  1     
1793  1    
1812  1   

转换成了mlvl_label_weights： 又转换成label_weights_list
len(mlvl_label_weights)=3
mlvl_label_weights[0]:(1600)
mlvl_label_weights[1]:(400)
mlvl_label_weights[2]:(100)


bbox_targets:(2100,4)
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        ...,
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]
588 tensor([222.7531, 110.0244, 233.3431, 125.3494], device='cuda:0')
628 tensor([222.7531, 110.0244, 233.3431, 125.3494], device='cuda:0')
1753 tensor([178.6381,  51.7744, 249.7531, 211.2394], device='cuda:0')
1772 tensor([170.0131,  75.8744, 233.0681, 225.9744], device='cuda:0')
1773 tensor([178.6381,  51.7744, 249.7531, 211.2394], device='cuda:0')
1792 tensor([170.0131,  75.8744, 233.0681, 225.9744], device='cuda:0')
1793 tensor([178.6381,  51.7744, 249.7531, 211.2394], device='cuda:0')
1812 tensor([170.0131,  75.8744, 233.0681, 225.9744], device='cuda:0')

转换成了mlvl_bbox_targets:  又转换成bbox_targets_list
len(mlvl_bbox_targets)=3
mlvl_bbox_targets[0]:(1600,4)
mlvl_bbox_targets[1]:(400,4)
mlvl_bbox_targets[2]:(100,4)


bbox_weights: (2100,4)  
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        ...,
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]

588 tensor([1., 1., 1., 1.], device='cuda:0')
628 tensor([1., 1., 1., 1.], device='cuda:0')
1753 tensor([1., 1., 1., 1.], device='cuda:0')
1772 tensor([1., 1., 1., 1.], device='cuda:0')
1773 tensor([1., 1., 1., 1.], device='cuda:0')
1792 tensor([1., 1., 1., 1.], device='cuda:0')
1793 tensor([1., 1., 1., 1.], device='cuda:0')
1812 tensor([1., 1., 1., 1.], device='cuda:0')

转换成了mlvl_bbox_weights:  又转换成bbox_weights_list
len(mlvl_bbox_weights)=3
mlvl_bbox_weights[0]:(1600,4)
mlvl_bbox_weights[1]:(400,4)
mlvl_bbox_weights[2]:(100,4)





pos_inds:588,  628, 1753, 1772, 1773, 1792, 1793, 1812

 neg_inds

num_total_pos=8
num_total_neg=2010



743, 1790, 1791, 1809, 1810, 1811
labels: 24,  0,  0, 17,  0,  0     其他都是80
label_weights：1,1,1,1,1,1    其他都是1
bbox_targets   其他都是0
        [182.7431, 142.7573, 189.2332, 152.1493],
        
        [155.7069, 107.0585, 199.2902, 204.7875],
        [155.7069, 107.0585, 199.2902, 204.7875],
        [150.4211, 121.8283, 189.0647, 213.8179],
        [155.7069, 107.0585, 199.2902, 204.7875],
        [155.7069, 107.0585, 199.2902, 204.7875]

bbox_weights           其他都是0 
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]
```

