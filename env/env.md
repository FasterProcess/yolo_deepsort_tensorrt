# 1. install C++ env

```bash
sudo apt-get update
sudo add-apt-repository ppa:ubuntu-toolchain-r/test

sudo apt update
sudo apt install -y gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 --slave /usr/bin/g++ g++ /usr/bin/g++-11
sudo apt update
sudo apt install -y clang-11
```

# 2. install CUDA

> 参考网址：https://developer.nvidia.com/cuda-12-6-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local
> 

## 2.1 install CUDA

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.0-560.28.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-6-local_12.6.0-560.28.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
sudo apt-get -y install nvtop
```

## 2.2 add environment variables

```bash
cat >> ~/.bashrc <<EOF

# CUDA ENV
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$PATH:\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
export LIBRARY_PATH=\$PATH:\$CUDA_HOME/lib64:\$LIBRARY_PATH
export C_INCLUDE_PATH=\$C_INCLUDE_PATH:\$CUDA_HOME/include
export CPLUS_INCLUDE_PATH=\$CPLUS_INCLUDE_PATH:\$CUDA_HOME/include

EOF

source ~/.bashrc
```

## 2.3 install CUDA tools

- cuDNN

```bash
# install cuDNN
wget https://developer.download.nvidia.com/compute/cudnn/9.10.2/local_installers/cudnn-local-repo-ubuntu2204-9.10.2_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.10.2_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.10.2/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn-cuda-12
```

- tensorrt library

> refer to：https://developer.nvidia.com/tensorrt/download/10x
if no need for onnxruntime TensorRT-backend and TensorRT C++, this step can skip.
> 

```bash
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.4.0/tars/TensorRT-10.4.0.26.Linux.x86_64-gnu.cuda-12.6.tar.gz
sudo mkdir -p /usr/local/TensorRT 
sudo mv TensorRT-10.4.0.26.Linux.x86_64-gnu.cuda-12.6.tar.gz /usr/local/
cd /usr/local/TensorRT/
sudo tar -xavf ../TensorRT-10.4.0.26.Linux.x86_64-gnu.cuda-12.6.tar.gz
# sudo rm /usr/local/TensorRT-10.4.0.26.Linux.x86_64-gnu.cuda-12.6.tar.gz

cat >> ~/.bashrc <<EOF

# TensorRT ENV
export TENSORRT_HOME=/usr/local/TensorRT/TensorRT-10.4.0.26
export LD_LIBRARY_PATH=\$PATH:\$TENSORRT_HOME/lib:\$LD_LIBRARY_PATH
export LIBRARY_PATH=\$PATH:\$TENSORRT_HOME/lib:\$LIBRARY_PATH
export C_INCLUDE_PATH=\$C_INCLUDE_PATH:\$TENSORRT_HOME/include
export CPLUS_INCLUDE_PATH=\$CPLUS_INCLUDE_PATH:\$TENSORRT_HOME/include

EOF
source ~/.bashrc

```

- python
    - onnxruntime
    - tensorrt
    - pycuda
    - onnx