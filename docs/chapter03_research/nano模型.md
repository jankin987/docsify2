```
model
DataParallel(
  (module): GFL(
    (backbone): ShuffleNetV2(
      (conv1): Sequential(
        (0): Conv2d(3, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.1, inplace)
      )
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (stage2): Sequential(
        (0): ShuffleV2Block(
          (branch1): Sequential(
            (0): Conv2d(24, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=24, bias=False)
            (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Conv2d(24, 58, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): LeakyReLU(negative_slope=0.1, inplace)
          )
          (branch2): Sequential(
            (0): Conv2d(24, 58, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.1, inplace)
            (3): Conv2d(58, 58, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=58, bias=False)
            (4): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): Conv2d(58, 58, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (6): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (7): LeakyReLU(negative_slope=0.1, inplace)
          )
        )
        (1): ShuffleV2Block(
          (branch1): Sequential()
          (branch2): Sequential(
            (0): Conv2d(58, 58, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.1, inplace)
            (3): Conv2d(58, 58, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=58, bias=False)
            (4): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): Conv2d(58, 58, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (6): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (7): LeakyReLU(negative_slope=0.1, inplace)
          )
        )
        (2): ShuffleV2Block(
          (branch1): Sequential()
          (branch2): Sequential(
            (0): Conv2d(58, 58, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.1, inplace)
            (3): Conv2d(58, 58, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=58, bias=False)
            (4): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): Conv2d(58, 58, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (6): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (7): LeakyReLU(negative_slope=0.1, inplace)
          )
        )
        (3): ShuffleV2Block(
          (branch1): Sequential()
          (branch2): Sequential(
            (0): Conv2d(58, 58, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.1, inplace)
            (3): Conv2d(58, 58, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=58, bias=False)
            (4): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): Conv2d(58, 58, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (6): BatchNorm2d(58, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (7): LeakyReLU(negative_slope=0.1, inplace)
          )
        )
      )
      (stage3): Sequential(
        (0): ShuffleV2Block(
          (branch1): Sequential(
            (0): Conv2d(116, 116, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=116, bias=False)
            (1): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): LeakyReLU(negative_slope=0.1, inplace)
          )
          (branch2): Sequential(
            (0): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.1, inplace)
            (3): Conv2d(116, 116, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=116, bias=False)
            (4): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (6): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (7): LeakyReLU(negative_slope=0.1, inplace)
          )
        )
        (1): ShuffleV2Block(
          (branch1): Sequential()
          (branch2): Sequential(
            (0): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.1, inplace)
            (3): Conv2d(116, 116, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=116, bias=False)
            (4): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (6): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (7): LeakyReLU(negative_slope=0.1, inplace)
          )
        )
        (2): ShuffleV2Block(
          (branch1): Sequential()
          (branch2): Sequential(
            (0): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.1, inplace)
            (3): Conv2d(116, 116, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=116, bias=False)
            (4): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (6): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (7): LeakyReLU(negative_slope=0.1, inplace)
          )
        )
        (3): ShuffleV2Block(
          (branch1): Sequential()
          (branch2): Sequential(
            (0): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.1, inplace)
            (3): Conv2d(116, 116, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=116, bias=False)
            (4): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (6): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (7): LeakyReLU(negative_slope=0.1, inplace)
          )
        )
        (4): ShuffleV2Block(
          (branch1): Sequential()
          (branch2): Sequential(
            (0): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.1, inplace)
            (3): Conv2d(116, 116, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=116, bias=False)
            (4): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (6): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (7): LeakyReLU(negative_slope=0.1, inplace)
          )
        )
        (5): ShuffleV2Block(
          (branch1): Sequential()
          (branch2): Sequential(
            (0): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.1, inplace)
            (3): Conv2d(116, 116, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=116, bias=False)
            (4): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (6): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (7): LeakyReLU(negative_slope=0.1, inplace)
          )
        )
        (6): ShuffleV2Block(
          (branch1): Sequential()
          (branch2): Sequential(
            (0): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.1, inplace)
            (3): Conv2d(116, 116, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=116, bias=False)
            (4): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (6): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (7): LeakyReLU(negative_slope=0.1, inplace)
          )
        )
        (7): ShuffleV2Block(
          (branch1): Sequential()
          (branch2): Sequential(
            (0): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.1, inplace)
            (3): Conv2d(116, 116, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=116, bias=False)
            (4): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): Conv2d(116, 116, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (6): BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (7): LeakyReLU(negative_slope=0.1, inplace)
          )
        )
      )
      (stage4): Sequential(
        (0): ShuffleV2Block(
          (branch1): Sequential(
            (0): Conv2d(232, 232, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=232, bias=False)
            (1): BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): Conv2d(232, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): LeakyReLU(negative_slope=0.1, inplace)
          )
          (branch2): Sequential(
            (0): Conv2d(232, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.1, inplace)
            (3): Conv2d(232, 232, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=232, bias=False)
            (4): BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): Conv2d(232, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (6): BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (7): LeakyReLU(negative_slope=0.1, inplace)
          )
        )
        (1): ShuffleV2Block(
          (branch1): Sequential()
          (branch2): Sequential(
            (0): Conv2d(232, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.1, inplace)
            (3): Conv2d(232, 232, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=232, bias=False)
            (4): BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): Conv2d(232, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (6): BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (7): LeakyReLU(negative_slope=0.1, inplace)
          )
        )
        (2): ShuffleV2Block(
          (branch1): Sequential()
          (branch2): Sequential(
            (0): Conv2d(232, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.1, inplace)
            (3): Conv2d(232, 232, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=232, bias=False)
            (4): BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): Conv2d(232, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (6): BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (7): LeakyReLU(negative_slope=0.1, inplace)
          )
        )
        (3): ShuffleV2Block(
          (branch1): Sequential()
          (branch2): Sequential(
            (0): Conv2d(232, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.1, inplace)
            (3): Conv2d(232, 232, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=232, bias=False)
            (4): BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): Conv2d(232, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (6): BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (7): LeakyReLU(negative_slope=0.1, inplace)
          )
        )
      )
    )
    (fpn): PAN(
      (lateral_convs): ModuleList(
        (0): ConvModule(
          (conv): Conv2d(116, 96, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): ConvModule(
          (conv): Conv2d(232, 96, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): ConvModule(
          (conv): Conv2d(464, 96, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
    (head): NanoDetHead(
      (distribution_project): Integral()
      (loss_qfl): QualityFocalLoss()
      (loss_dfl): DistributionFocalLoss()
      (loss_bbox): GIoULoss()
      (cls_convs): ModuleList(
        (0): ModuleList(
          (0): DepthwiseConvModule(
            (depthwise): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
            (pointwise): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (dwnorm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pwnorm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): LeakyReLU(negative_slope=0.1, inplace)
          )
          (1): DepthwiseConvModule(
            (depthwise): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
            (pointwise): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (dwnorm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pwnorm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): LeakyReLU(negative_slope=0.1, inplace)
          )
        )
        (1): ModuleList(
          (0): DepthwiseConvModule(
            (depthwise): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
            (pointwise): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (dwnorm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pwnorm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): LeakyReLU(negative_slope=0.1, inplace)
          )
          (1): DepthwiseConvModule(
            (depthwise): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
            (pointwise): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (dwnorm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pwnorm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): LeakyReLU(negative_slope=0.1, inplace)
          )
        )
        (2): ModuleList(
          (0): DepthwiseConvModule(
            (depthwise): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
            (pointwise): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (dwnorm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pwnorm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): LeakyReLU(negative_slope=0.1, inplace)
          )
          (1): DepthwiseConvModule(
            (depthwise): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
            (pointwise): Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (dwnorm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pwnorm): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): LeakyReLU(negative_slope=0.1, inplace)
          )
        )
      )
      (reg_convs): ModuleList(
        (0): ModuleList()
        (1): ModuleList()
        (2): ModuleList()
      )
      (gfl_cls): ModuleList(
        (0): Conv2d(96, 112, kernel_size=(1, 1), stride=(1, 1))
        (1): Conv2d(96, 112, kernel_size=(1, 1), stride=(1, 1))
        (2): Conv2d(96, 112, kernel_size=(1, 1), stride=(1, 1))
      )
      (gfl_reg): ModuleList(
        (0): Conv2d(96, 32, kernel_size=(1, 1), stride=(1, 1))
        (1): Conv2d(96, 32, kernel_size=(1, 1), stride=(1, 1))
        (2): Conv2d(96, 32, kernel_size=(1, 1), stride=(1, 1))
      )
    )
  )
)

```





for i in dataset['annotations']:

  if i['image_id']==302364:

​    print(i)



for i in dataset['images']:

  if i['file_name']=="000000302364.jpg":

​    print(i)



sum{
$ P(y_i) * y_i $


