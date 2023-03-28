# gr-ieee80211cu
- CUDA DLC for [gr-ieee80211](https://github.com/cloud9477/gr-ieee80211).
- Need the gr-ieee80211 as the basic module to run this CUDA version.
- Replace the pre-processing, demodulation and decoding of the gr-ieee80211 to speed up the receptions.
- On Ubuntu 22.04 GNU Radio 3.10.
- For more information about how to use the IEEE 802.11 stack, please refer to [gr-ieee80211](https://github.com/cloud9477/gr-ieee80211) first.
- **Not finished yet, will be published and released in May 2023**

Installation
------------
- First install the Ubuntu and GNU Radio. I assuming you build this from scratch.
- When install the Ubuntu, I **didn't** click the **"Install third party drivers ..."**. You could ignore this if you could handle the CUDA installations.
- I usually put the GNU Radio modules in a folder named "sdr" in home folder, if you use a different path, please correct the paths in the codes. For example the python tools.

### 1. Install GNU Radio and UHD
```console
sdr@sdr:~$ sudo apt-get update
sdr@sdr:~$ sudo apt-get install gnuradio-dev uhd-host libuhd-dev cmake build-essential
sdr@sdr:~$ sudo cp /lib/uhd/utils/uhd-usrp.rules /etc/udev/rules.d/
sdr@sdr:~$ sudo udevadm control --reload-rules
sdr@sdr:~$ sudo udevadm trigger
sdr@sdr:~$ sudo uhd_images_downloader
```

### 2. Install CUDA
- Download and install the nvidia cuda toolkit, it will also install the driver. The toolkit version here is 12.0, it matches the driver version 525 (tested on 2022-12-9). The toolkit installation steps are from [NVIDIA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

```console
sdr@sdr:~$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sdr@sdr:~$ sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sdr@sdr:~$ wget https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda-repo-ubuntu2204-12-0-local_12.0.1-525.85.12-1_amd64.deb
sdr@sdr:~$ sudo dpkg -i cuda-repo-ubuntu2204-12-0-local_12.0.1-525.85.12-1_amd64.deb
sdr@sdr:~$ sudo cp /var/cuda-repo-ubuntu2204-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sdr@sdr:~$ sudo apt-get update
sdr@sdr:~$ sudo apt-get -y install cuda
sdr@sdr:~$ sudo reboot
sdr@sdr:~$ nvidia-smi
```
- After rebooting, use **nvidia-smi** to test the driver, it should show something like this:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.13    Driver Version: 525.60.13    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:01:00.0  On |                  N/A |
|  0%   50C    P8    12W / 200W |    613MiB /  8192MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1623      G   /usr/lib/xorg/Xorg                245MiB |
|    0   N/A  N/A      1834      G   /usr/bin/gnome-shell               83MiB |
|    0   N/A  N/A      2700      G   ...RendererForSitePerProcess      130MiB |
|    0   N/A  N/A     10102      G   ...1/usr/lib/firefox/firefox      149MiB |
+-----------------------------------------------------------------------------+
```

- Next, we config the path for nvidia tools. 
- Step 1: Add the PATH **"/usr/local/cuda-12.0/bin"** to the end of file **"/etc/environment"**, for example here is the content in my **"/etc/environment"** after adding the path:
```
PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/local/cuda-12.0/bin"
```
- Step 2: In folder "/etc/ld.so.conf.d/", create a new file (for example, I name it **"cloud_cuda.conf"**), and put the **"/usr/local/cuda-12.0/lib64"** in the created file for x64 system.
- Step 3: Reboot the system.
- These three steps add the path to the system permanently for Ubuntu 22.04.
- Now we could use the CUDA compiler **nvcc**, the following could test the driver and nvcc:
```console
sdr@sdr:~$ nvcc --version
sdr@sdr:~$ cd sdr
sdr@sdr:~$ git clone https://github.com/NVIDIA/cuda-samples.git
sdr@sdr:~$ cd cuda-samples/Samples/1_Utilities/deviceQuery
sdr@sdr:~$ make
sdr@sdr:~$ cd Samples/1_Utilities/deviceQuery
sdr@sdr:~$ ./deviceQuery
```
- If the make reports error, mostly it is because that the version is not correct (driver or cuda or anything), the cuda toolkit doesn't match the driver, or even other issues. I have met many.

### 2. Install gr-ieee80211 and gr-ieee80211cu
- Install the gr-ieee80211 module:
```console
sdr@sdr:~$ cd sdr
sdr@sdr:~$ git clone https://github.com/cloud9477/gr-ieee80211.git
sdr@sdr:~$ cd gr-ieee80211/
sdr@sdr:~$ mkdir build
sdr@sdr:~$ cd build
sdr@sdr:~$ cmake ../
sdr@sdr:~$ make
sdr@sdr:~$ sudo make install
sdr@sdr:~$ sudo ldconfig
```

- Install the gr-ieee80211cu module:
```console
sdr@sdr:~$ cd sdr
sdr@sdr:~$ git clone https://github.com/cloud9477/gr-ieee80211cu.git
sdr@sdr:~$ cd gr-ieee80211cu/
sdr@sdr:~$ mkdir build
sdr@sdr:~$ cd build
sdr@sdr:~$ cmake ../
sdr@sdr:~$ make
sdr@sdr:~$ sudo make install
sdr@sdr:~$ sudo ldconfig
```

CUDA Related Info
------------
- Some hardware tested by author.

### RTX 3070
- Zotac 3070 Gaming
- Device name: NVIDIA GeForce RTX 3070
- SM Number: 46
- SM Max Thread Number: 1536
- Block Share Mem Size: 49152
- Block Max Thread Number: 1024
- Memory Clock Rate (KHz): 7001000
- Memory Bus Width (bits): 256
- Peak Memory Bandwidth (GB/s): 448.064000

### GTX 1070
- MSI 1070 ARMOR
- Device name: NVIDIA GeForce GTX 1070
- SM Number: 15
- SM Max Thread Number: 2048
- Block Share Mem Size: 49152
- Block Max Thread Number: 1024
- Memory Clock Rate (KHz): 4004000
- Memory Bus Width (bits): 256
- Peak Memory Bandwidth (GB/s): 256.256000

Some Tips
------------
- Use watch and nvidia-smi to minitor GPU
```console
sdr@sdr:~$ watch -n 0.1 nvidia-smi
```

- Use top to record CPU usage, the -1 is usage by core, -d is interval, -n is sample number, -b is logging.
```console
sdr@sdr:~$ top -1 -d 0.1 -n 200 -b > toplog.txt
```
