# Start from Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:19.08-py3

# Install dependencies (pip or conda)
# RUN pip install -U -r requirements.txt
# RUN conda update -n base -c defaults conda
# RUN conda install -y -c anaconda future numpy opencv matplotlib tqdm pillow
# RUN conda install -y -c conda-forge scikit-image tensorboard pycocotools
# conda install pytorch torchvision -c pytorch

# Install OpenCV with Gstreamer support
#WORKDIR /usr/src
#RUN pip uninstall -y opencv-python
#RUN apt-get update
#RUN apt-get install -y gstreamer1.0-python3-dbg-plugin-loader
## RUN apt-get install gstreamer1.0
## RUN apt install -y ubuntu-restricted-extras
#RUN apt install -y libgstreamer1.0-dev
#RUN apt install -y libgstreamer-plugins-base1.0-dev
#RUN git clone https://github.com/opencv/opencv.git && cd opencv && git checkout 4.1.1 && mkdir build
#RUN git clone https://github.com/opencv/opencv_contrib.git && cd opencv_contrib && git checkout 4.1.1
#RUN cd opencv/build && cmake ../ \
#    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
#    -D BUILD_OPENCV_PYTHON3=ON \
#    -D PYTHON3_EXECUTABLE=/opt/conda/bin/python \
#    -D PYTHON3_INCLUDE_PATH=/opt/conda/include/python3.6m \
#    -D PYTHON3_LIBRARIES=/opt/conda/lib/python3.6/site-packages \
#    -D WITH_GSTREAMER=ON \
#    -D WITH_FFMPEG=OFF \
#    && make && make install && ldconfig
#RUN cd /usr/local/lib/python3.6/site-packages/cv2/python-3.6/ && mv cv2.cpython-36m-x86_64-linux-gnu.so cv2.so
#RUN cd /opt/conda/lib/python3.6/site-packages/ && ln -s /usr/local/lib/python3.6/site-packages/cv2/python-3.6/cv2.so cv2.so
#RUN python3 -c "import cv2; print(cv2.getBuildInformation())"

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

# ---------------------------------------------------  Extras Below  ---------------------------------------------------

# Build container
# rm -rf yolov3  # Warning: remove existing
# git clone https://github.com/ultralytics/yolov3 && cd yolov3 && python3 detect.py
# sudo docker image prune -af && sudo docker build -t ultralytics/yolov3:v0 .

# Run container
# sudo nvidia-docker run --ipc=host ultralytics/yolov3:v0 python3 detect.py

# Run container with local directory access
# sudo nvidia-docker run --ipc=host --mount type=bind,source="$(pwd)"/coco,target=/usr/src/coco ultralytics/yolov3:v0 python3 train.py
# sudo nvidia-docker run --ipc=host --mount type=bind,source="$(pwd)"/coco,target=/usr/src/coco ultralytics/yolov3:v0 python3 train.py --batch-size 64 --accumulate 1 --img-size 320 --arc uFBCE --prebias --epochs 27

# Push container to https://hub.docker.com/u/ultralytics
# docker push ultralytics/yolov3:v0

# Build and Push
# export tag=ultralytics/yolov3:v0 && sudo docker build -t $tag . && docker push $tag