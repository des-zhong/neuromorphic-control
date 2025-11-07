# Neuromorphic controller

Short README for this repository. Use this as a starting point to run, train, and deploy the models included in the workspace. Additional modules and documentation will be gradually released as part of ongoing updates

## Repository layout (key files)
- [snn_tutorial.md](snn_tutorial.md) — tutorial and helper functions such as [`snn_tutorial.get_lr_list`](snn_tutorial.md) and [`snn_tutorial.spike_net_forward`](snn_tutorial.md).

- [custom_op_in_pytorch/](custom_op_in_pytorch/) —Lynxi api

- [AllScene](AllScene)—MuJoCo model files and environment definition

- [demo_v3.py](demo_v3.py)—Main simulation script

  

## Requirements
Install dependencies listed in [requirements.txt](requirements.txt):

```
stable_baselines3==2.4.1
mujoco-py==2.1.2.14
mujoco-python-viewer==0.1.4
opencv-python==4.1.2.30
pygame==2.1.0
gym==0.9.1
gymnasium==0.29.1
torch  2.2.1+cu118
tensorflow==2.13.1
pickle==0.7.5
lynbidl==1.7.0
```

Compilation and inference on neuromorphic chip need additional package:

```sh
lyngor==1.17.0
pylynchipsdk==1.16.0
pylynsdkrdma==3.8
```

## Quickstart on Simulation

1. Prepare dataset and update dataset path in [data_loader_slice.py](data_loader_slice.py).
2. To run a simulation for inspection:

   if running on a remote server, set 'offscreen' for render mode and run 

   ```sh
   xvfb-run -a python demo_v3.py
   ```

   ​	This will generate a video showing the inspection process in recordings folder

   if running locally, do not set 'offscreen', the process will be rendered in real-time

   ```
   python demo_v3.py
   ```

​		This will also generate a video showing the inspection in recordings folder

3. Compile to Lynxi KA200 fomat (checkpoint in exp/mapping_output)

## Deployment on Apollo

Convert IDM and PE pth file to onnx:

```
python convert2onnx.py
```

Then compile onnx file to Lynxi format

```sh
model_compile -m model.onnx -t apu -f Onnx --input_nodes "input_1" --input_shapes "1,1,200,200" --output_nodes "output"
```



