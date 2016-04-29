FROM matthieudelaro/work-on-caffe

# install pip3
RUN apt-get install -y python3-pip

# Install python deps
RUN sudo pip3 install numpy
