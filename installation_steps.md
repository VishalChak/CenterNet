Steps for installation:
1. Clone the repo
    export CenterNet_ROOT=/path/to/clone/CenterNet
    git clone https://github.com/VishalChak/CenterNet.git $CenterNet_ROOT
2. Create a virtual env [conda create --name CenterNet python=3.6]
3. activate virtual env
4. install requirnet.txt
5. install conda install -c conda-forge pycocotools
6. conda install pytorch=0.4.1 torchvision cuda92 -c pytorch
7. down grad pillow version install   pip install Pillow==6.2.1
8. Disable cudnn batch normalization [Secund part of step 1 from intallation.md]
    # PYTORCH=/path/to/pytorch # usually ~/anaconda3/envs/CenterNet/lib/python3.6/site-packages/
    # for pytorch v0.4.0
    sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
    # for pytorch v0.4.1
    sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
9. Install COCOAPI:[Step 2 from intallation.md]
    # COCOAPI=/path/to/clone/cocoapi
    git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
    cd $COCOAPI/PythonAPI
    make
    python setup.py install --user
10. complile deformable convolutional (from DCNv2).
    cd $CenterNet_ROOT/src/lib/models/networks/DCNv2
    ./make.sh
11. Download pertained models for detection or pose estimation and move them to CenterNet/models/. More models can be found in Model zoo.
12. Object detection on Images/video:
    python demo.py ctdet --demo /path/to/image/or/folder/or/video --load_model ../models/ctdet_coco_dla_2x.pth
13. Object Detection on webcam:
    python demo.py ctdet --demo webcam --load_model ../models/ctdet_coco_dla_2x.pth
