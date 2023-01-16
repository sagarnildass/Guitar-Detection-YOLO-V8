YOLO-V8 is here finally!

YOLO (You Only Look Once) is a popular real-time object detection system. YOLO-V8 is the latest version of the YOLO (You Only Look Once) object detection and image segmentation model developed by Ultralytics. It is designed to improve upon the previous versions of YOLO by increasing the accuracy and speed of object detection while also reducing the amount of computation required. Some of the key features of YOLO v8 include a new architecture, improved data augmentation techniques, and a focus on reducing the number of false positives.

The YOLOv8 model is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection and image segmentation tasks. It can be trained on large datasets and is capable of running on a variety of hardware platforms, from CPUs to GPUs.

I did not find any good documentation, particularly for YOLO-V8 (at the time of writing this post) training on a custom dataset. So, in this post, we will see how to use YOLO-V8 to train on a custom dataset to detect guitars!

A Brief History of YOLO

YOLO (You Only Look Once) is a popular object detection and image segmentation model developed by Joseph Redmon and Ali Farhadi at the University of Washington. The first version of YOLO was released in 2015 and quickly gained popularity due to its high speed and accuracy.

YOLOv2 was released in 2016 and improved upon the original model by incorporating batch normalization, anchor boxes, and dimension clusters. YOLOv3 was released in 2018 and further improved the model's performance by using a more efficient backbone network, adding a feature pyramid, and making use of focal loss.

In 2020, YOLOv4 was released which introduced a number of innovations such as the use of Mosaic data augmentation, a new anchor-free detection head, and a new loss function.

In 2021, Ultralytics released YOLOv5, which further improved the model's performance and added new features such as support for panoptic segmentation and object tracking.

YOLO has been widely used in a variety of applications, including autonomous vehicles, security and surveillance, and medical imaging. It has also been used to win several competitions, such as the COCO Object Detection Challenge and the DOTA Object Detection Challenge.

1. Download Dataset

We will use the OIDV4 Toolkit to download the images. We will not use the labels that OIDV4 provides as that will defeat the purpose of our learning. So we will annotate the images ourselves.

To download the guitar images, we have to perform the following steps:

a) Clone the repository

git clone https://github.com/EscVM/OIDv4_ToolKit

b) Go into the repository and open a command terminal

c) Run the following command

python3 main.py downloader --classes Guitar --type_csv all -y --limit 500 --noLabels

This command will download the guitar images (not more than 500), and subdivide them into three folders - train, validation, and test.

2. Use RoboFlow to annotate the images for YOLO-V8

I have gone through many tools like labelimg, but Roboflow makes it so much easier to annotate and use the dataset. Just go to their site and create an account to follow along.

Note - If you just want to download my annotated and labeled data, you can download it here. Just change the data.yaml file and change the paths. Also, change the roboflow workspace name to yours.

After you have created an account, it will prompt you to create a project. This is where you will have to mention the project name and your purpose.

Next, you will have to upload your images. To do that, click the upload button and select your folder containing the images. 

Note that for the purpose of this demo, I have annotated 66 images. I will use 50 of them for training and 16 for validation. Of course, you are welcome to annotate more images to obtain a better result.

After you have uploaded your images, go to the annotate menu, and just start drawing bounding boxes around your object of interest. In this case, we will annotate the guitars.

Annotation with Roboflow for YOLO-V8

This is of course a tedious task, but this is where you can create the perfect dataset of your liking and get your own custom model to detect your target object. Just annotate as many images as you can as the performance of your model will increase with larger data.

Once you have annotated your dataset, click the generate tab to prepare your dataset. 

First, choose your train-validation-split steps. I have used 50 images for training and 16 for validation. I did not use any test images as I previously had them from OIDV4. We will see how to use them in a bit.

Train-validation-test split in Roboflow

Next, you can add any preprocessing or augmentation steps if required. I did not do any additional preprocessing or augmentation steps as we will train the YOLO-V8 model on the dataset as is.

Once you are done with these steps, click the generate step and you will see a screen like this.

Roboflow Dataset Generation step for YOLO-V8

Now click on the Get Snippet button. You will see a snippet of code generated for you. You can run this code in a jupyter notebook to download your prepared dataset into your local computer.

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="your_api_key_here")
project = rf.workspace("sagarnil-das-zmptn").project("guitar-detection")
dataset = project.version(3).download("yolov8")

3. Train the YOLO-V8 model

So now, we will open a new jupyter notebook (or any IDE of your liking), and execute the above code. Once you have done that, you will see a new folder called Guitar-Detection got generated in your working directory. This directory has all the prepared dataset, their annotations, and the data.yaml file. 

Folder structure after Roboflow Data Download

Make sure that in the data.yaml file, the paths mentioned are correct. It should look something like this:

names:
- Guitar
nc: 1
roboflow:
  license: CC BY 4.0
  project: guitar-detection
  url: https://universe.roboflow.com/sagarnil-das-zmptn/guitar-detection/dataset/2
  version: 2
  workspace: sagarnil-das-zmptn
test: /home/sagarnildass/python_notebooks/YOLO_Experiments/guitar_detection/Guitar-Detection-2/test/images
train: /home/sagarnildass/python_notebooks/YOLO_Experiments/guitar_detection/Guitar-Detection-2/train/images
val: /home/sagarnildass/python_notebooks/YOLO_Experiments/guitar_detection/Guitar-Detection-2/valid/images

Now, we will train the YOLO-V8 model. To do this, run the following code in your notebook.

!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=25 imgsz=800

I trained it for 25 epochs, but you can increase it to a higher number.

You will see results like this:

Ultralytics YOLOv8.0.6 🚀 Python-3.9.15 torch-1.13.1+cu117 CUDA:0 (GeForce GTX 1080 Ti, 11175MiB)
yolo/engine/trainer: task=detect, mode=train, model=yolov8s.pt, data=/home/sagarnildass/python_notebooks/YOLO_Experiments/cello_guitar/Guitar-Detection-2/data.yaml, epochs=25, patience=50, batch=16, imgsz=800, save=True, cache=False, device=, workers=8, project=None, name=None, exist_ok=False, pretrained=False, optimizer=SGD, verbose=False, seed=0, deterministic=True, single_cls=False, image_weights=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, show=False, save_txt=False, save_conf=False, save_crop=False, hide_labels=False, hide_conf=False, vid_stride=1, line_thickness=3, visualize=False, augment=False, agnostic_nms=False, retina_masks=False, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=17, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, fl_gamma=0.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0, cfg=None, hydra={'output_subdir': None, 'run': {'dir': '.'}}, v5loader=False, save_dir=/home/sagarnildass/python_notebooks/pytorch_projects/runs/detect/train17
Overriding model.yaml nc=80 with nc=1

                   from  n    params  module                                       arguments                     
  0                  -1  1       928  ultralytics.nn.modules.Conv                  [3, 32, 3, 2]                 
  1                  -1  1     18560  ultralytics.nn.modules.Conv                  [32, 64, 3, 2]                
  2                  -1  1     29056  ultralytics.nn.modules.C2f                   [64, 64, 1, True]             
  3                  -1  1     73984  ultralytics.nn.modules.Conv                  [64, 128, 3, 2]               
  4                  -1  2    197632  ultralytics.nn.modules.C2f                   [128, 128, 2, True]           
  5                  -1  1    295424  ultralytics.nn.modules.Conv                  [128, 256, 3, 2]              
  6                  -1  2    788480  ultralytics.nn.modules.C2f                   [256, 256, 2, True]           
  7                  -1  1   1180672  ultralytics.nn.modules.Conv                  [256, 512, 3, 2]              
  8                  -1  1   1838080  ultralytics.nn.modules.C2f                   [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.SPPF                  [512, 512, 5]                 
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 11             [-1, 6]  1         0  ultralytics.nn.modules.Concat                [1]                           
 12                  -1  1    591360  ultralytics.nn.modules.C2f                   [768, 256, 1]                 
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 14             [-1, 4]  1         0  ultralytics.nn.modules.Concat                [1]                           
 15                  -1  1    148224  ultralytics.nn.modules.C2f                   [384, 128, 1]                 
 16                  -1  1    147712  ultralytics.nn.modules.Conv                  [128, 128, 3, 2]              
 17            [-1, 12]  1         0  ultralytics.nn.modules.Concat                [1]                           
 18                  -1  1    493056  ultralytics.nn.modules.C2f                   [384, 256, 1]                 
 19                  -1  1    590336  ultralytics.nn.modules.Conv                  [256, 256, 3, 2]              
 20             [-1, 9]  1         0  ultralytics.nn.modules.Concat                [1]                           
 21                  -1  1   1969152  ultralytics.nn.modules.C2f                   [768, 512, 1]                 
 22        [15, 18, 21]  1   2116435  ultralytics.nn.modules.Detect                [1, [128, 256, 512]]          
Model summary: 225 layers, 11135987 parameters, 11135971 gradients, 28.6 GFLOPs

Transferred 349/355 items from pretrained weights
optimizer: SGD(lr=0.01) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias
train: Scanning /home/sagarnildass/python_notebooks/YOLO_Experiments/cello_guita
train: New cache created: /home/sagarnildass/python_notebooks/YOLO_Experiments/cello_guitar/Guitar-Detection-2/train/labels.cache
val: Scanning /home/sagarnildass/python_notebooks/YOLO_Experiments/cello_guitar/
val: New cache created: /home/sagarnildass/python_notebooks/YOLO_Experiments/cello_guitar/Guitar-Detection-2/valid/labels.cache
Image sizes 800 train, 800 val
Using 8 dataloader workers
Logging results to /home/sagarnildass/python_notebooks/pytorch_projects/runs/detect/train17
Starting training for 25 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/25      5.68G      1.929      8.201       2.39          3        800: 1
/home/sagarnildass/anaconda3/envs/mlagents_unity/lib/python3.9/site-packages/torch/optim/lr_scheduler.py:138: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20     0.0489       0.15     0.0274     0.0103

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/25      6.04G      2.089      6.508       2.36          7        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20     0.0505        0.2     0.0318     0.0115

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/25      6.06G      1.977      6.271      2.174          7        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20      0.213        0.1     0.0824     0.0299

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/25      6.06G      1.575      3.446      1.939          4        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20      0.286       0.35      0.248      0.136

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/25      6.06G      1.552      2.581      1.889          5        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20      0.342      0.442      0.337      0.211

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/25      6.06G      1.516      2.266      1.738          8        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20      0.391       0.55      0.442      0.273

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/25      6.06G      1.603      1.975       1.86          5        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20      0.544        0.6      0.546      0.362

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/25      6.06G      1.269      1.788      1.716          3        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20      0.897        0.5      0.641      0.427

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/25      6.06G      1.551      1.914      1.853          4        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20      0.622        0.6      0.633      0.389

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/25      6.06G       1.36      1.588      1.704          7        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20      0.582        0.5      0.605      0.354

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/25      6.06G      1.325       1.47       1.84          2        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20      0.628        0.6      0.601      0.349

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/25      6.06G      1.232      1.525      1.597          7        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20      0.749      0.598      0.701       0.42

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/25      6.06G      1.257      1.455      1.636          5        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20      0.652      0.468      0.635      0.388

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/25      6.06G       1.42      1.463      1.719          5        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20      0.672        0.6      0.687      0.392

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/25      6.06G      1.109      1.299      1.547          5        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20      0.549       0.75      0.672      0.395
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/25      6.06G      1.167      1.351      1.564          4        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20      0.516        0.5      0.566      0.334

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/25      6.06G      0.959      1.122      1.389          2        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20      0.659        0.5      0.632      0.387

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      18/25      6.06G      1.107      1.201      1.515          3        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20      0.628      0.506      0.593      0.382

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      19/25      6.06G      1.237      1.144      1.576          2        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20      0.772       0.51      0.603      0.363

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      20/25      6.06G     0.8558     0.8933      1.257          2        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20        0.8       0.55      0.619      0.372

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      21/25      6.06G       1.06     0.9549      1.464          2        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20      0.786      0.552      0.639       0.39

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      22/25      6.06G     0.9368     0.8545      1.426          3        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20      0.908      0.496      0.645      0.395

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      23/25      6.06G      1.111      1.218      1.518          2        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20      0.726       0.53      0.626      0.401

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      24/25      6.06G      1.015      1.071      1.523          2        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20      0.802       0.45      0.616      0.389

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      25/25      6.06G     0.8639     0.8018      1.313          2        800: 1
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20      0.679        0.5      0.644      0.404

25 epochs completed in 0.033 hours.
Optimizer stripped from /home/sagarnildass/python_notebooks/pytorch_projects/runs/detect/train17/weights/last.pt, 22.6MB
Optimizer stripped from /home/sagarnildass/python_notebooks/pytorch_projects/runs/detect/train17/weights/best.pt, 22.6MB

Validating /home/sagarnildass/python_notebooks/pytorch_projects/runs/detect/train17/weights/best.pt...
Ultralytics YOLOv8.0.6 🚀 Python-3.9.15 torch-1.13.1+cu117 CUDA:0 (GeForce GTX 1080 Ti, 11175MiB)
Fusing layers... 
Model summary: 168 layers, 11125971 parameters, 0 gradients, 28.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         16         20      0.893        0.5      0.641      0.428
Speed: 0.3ms pre-process, 11.3ms inference, 0.0ms loss, 0.7ms post-process per image
Saving /home/sagarnildass/python_notebooks/pytorch_projects/runs/detect/train17/predictions.json...
Results saved to /home/sagarnildass/python_notebooks/pytorch_projects/runs/detect/train17

We see our final results got saved into the /home/sagarnildass/python_notebooks/pytorch_projects/runs/detect/train17 folder. Now we will take a look at some of them.

4. Evaluating our YOLO-V8 model

To make sure that you are in the right directory, we will define a HOME variable like this:

import os
HOME = os.getcwd()
print(HOME)

To see the confusion matrix, you can use the following code:

from IPython.display import display, Image
%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train17/confusion_matrix.png', width=600)

YOLO-V8 confusion matrix

We see that our model is not state-of-the-art. But keep in mind, that we only used 50 images for training and for just 25 epochs. With more data and more rounds of training, you can get a much superior result.

To see the losses, mAP, precision, recall etc, run the following code:

%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train17/results.png', width=600)

YOLO-V8 loss functions

To see our validation results, we can write the following code:

%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train17/val_batch0_pred.jpg', width=600)

YOLO-V8 validation results

5. Testing our saved model on unseen images

To test our best model, we will first need to copy the test image folder from our OIDv4 folder to our Guitar-Detection folder. So the folder structure for the test folder should be Guitar-Detection-2 >> test >> images

Now we can execute the following code to generate our results on the test data and save them.

%cd {HOME}
!yolo task=detect mode=predict model={HOME}/runs/detect/train17/weights/best.pt conf=0.25 source={dataset.location}/test/images

Just make sure to replace the train17 folder with your own folder that has the best model weights.

Once you run this code, YOLO-V8 will make a prediction step on your test images and save them in the runs >> detect >> predict folder. 

Here are some of the sample results.

YOLO-V8 test set results 1

YOLO-V8 test set results 2

Conclusion

In conclusion, YOLO-V8 is the latest version of the popular real-time object detection system YOLO. Developed by Ultralytics, it has been designed to improve upon previous versions by increasing accuracy and speed while reducing computation requirements. Some of the key features include a new architecture, improved data augmentation techniques, and a focus on reducing false positives. The YOLOv8 model is easy to use, fast, and accurate making it a great choice for a wide range of object detection and image segmentation tasks. Furthermore, it can be trained on large datasets and run on a variety of hardware platforms. However, due to the lack of good documentation for training on a custom dataset, this post aimed at showing how to use YOLO-V8 for training on a custom dataset specifically for detecting guitars.
