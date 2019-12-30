module_list.0.Conv2d conv0 CONV

module_list.2.Conv2d fire0/squeeze1x1 CONV
module_list.3.Conv2d fire0/expand1x1 CONV
module_list.5.Conv2d fire0/expand3x3 CONV

module_list.8.Conv2d fire1/squeeze1x1 CONV
module_list.9.Conv2d fire1/expand1x1 CONV
module_list.11.Conv2d fire1/expand3x3 CONV

module_list.14.Conv2d fire2/squeeze1x1 CONV
module_list.15.Conv2d fire2/expand1x1 CONV
module_list.17.Conv2d fire2/expand3x3 CONV

module_list.19.Conv2d conv10 CONV
module_list.19.BatchNorm2d conv10/bn BN
module_list.19.BatchNorm2d conv10/scale SCALE

module_list.20.Conv2d yolo_conv_m CONV
module_list.20.BatchNorm2d yolo_conv_m/bn BN
module_list.20.BatchNorm2d yolo_conv_m/scale SCALE

module_list.21.Conv2d yolo_out_m CONV


module_list.24.Conv2d conv11 CONV
module_list.24.BatchNorm2d conv11/bn BN
module_list.24.BatchNorm2d conv11/scale SCALE

module_list.27.Conv2d yolo_conv_s CONV
module_list.27.BatchNorm2d yolo_conv_s/bn BN
module_list.27.BatchNorm2d yolo_conv_s/scale SCALE

module_list.28.Conv2d yolo_out_s CONV