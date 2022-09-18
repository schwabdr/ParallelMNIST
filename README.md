# ParallelMNIST
A simple example of how to parallelize MNIST training using PyTorch.

First create a new conda environment:

<pre>
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install matplotlib
conda install -c conda-forge torchinfo
</pre>

I'll take this tutorial and add parallel training to it.
https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118

Steps for parallelizing the code (running on multiple GPUs):
1) In your train loop, you must move the data to the GPU: <pre>img = img.to(device)</pre>, where device should be "cuda"
2) After you instantiate your model, move it to the GPU: <pre>cnn = torch.nn.DataParalle(cnn).cuda()</pre>
3) You can optionally set: <pre>cudnn.benchmark=True</pre> - this may give a performance boost

If everything works correctly, while your model is training, you can run <pre>nvidia-smi</pre> in a second terminal window. You should see something similar to this:

<pre>
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.141.03   Driver Version: 470.141.03   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:19:00.0 Off |                  N/A |
| 18%   35C    P2    60W / 250W |    929MiB / 11019MiB |      6%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce ...  Off  | 00000000:1A:00.0 Off |                  N/A |
| 18%   39C    P2    64W / 250W |    927MiB / 11019MiB |      4%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA GeForce ...  Off  | 00000000:67:00.0 Off |                  N/A |
| 18%   42C    P2    67W / 250W |    927MiB / 11019MiB |      4%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA GeForce ...  Off  | 00000000:68:00.0 Off |                  N/A |
| 18%   43C    P2    45W / 250W |    990MiB / 11016MiB |      4%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
</pre>

We can see every GPU has some memory allocated (almost 1GB per GPU in this example). You can increase the batch size and you should see the Memory-Usage on each GPU go up.

If you ever see an error like: <pre> RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! </pre> This error indicates that some of your data is on the CPU and some is on the GPU. I made this mistake making the tutorial when I forgot to move my TEST set to the GPU but I did all my training on the GPU.