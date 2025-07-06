## Adaptive Calibration: A Unified Conversion Framework of Spiking Neural Networks

 [[Paper]](https://arxiv.org/pdf/2412.16219) [[Appendix]](https://github.com/user-attachments/files/18434508/_AAAI_25__Adaptive_Calibration_appendix.pdf) [[Slides]](https://github.com/user-attachments/files/19058773/aaai_presentation_for_upload.pdf)

Checkpoints will be uploaded upon request.


![aaai_poster_final](https://github.com/user-attachments/assets/5fa7ec16-9f30-4ac2-a0a9-d47ddc0f4408)


### Key contributions

Our "***Adaptive Calibration***" framework leverages biologically inspired burst-firing mechanisms, already supported by commercial neuromorphic hardware like Intel's Loihi 2 and Synsense's Xylo. By developing a training-free optimization algorithm, we automatically determine optimal burst-firing configurations based on each layer's sensitivity characteristics, improving both efficiency and performance. Key features include:

- **Training-free ANN-to-SNN Conversion:** Outperforms existing methods with minimal setup time (0.09 hours), eliminating the need for costly retraining while maintaining superior accuracy-energy balance.
- **Energy-efficient converted SNNs:** Delivers remarkable energy reductions across benchmark datasets (70.1% on CIFAR-10, 60.3% on CIFAR-100, and 43.1% on ImageNet) while preserving high accuracy.
- **Comprehensive Tasks/Architecture Support:** Demonstrates exceptional versatility through extensive validation across diverse tasks (2D/3D classification, event-driven processing, object detection, and segmentation) and modern architectures (ResNet, VGG, and Vision Transformers).




<p align="center">
<img src=https://github.com/user-attachments/assets/8f809915-5ed0-4a6a-a333-d540d22c8819 width="500">
</p>


### Running the Code

Our codebase follows the same file structure. Taking the training and inference of the neuromorphic dataset as an example:

#### 1. Pre-training ANN on Neuromorphic Datasets:
```bash
python train_neuromorphic.py --dataset xxx --arch xxx
```
- `--dataset`: Specifies the dataset to be used, including `cifar10dvs, ncaltech101, ncars, action recognition`.


#### 2. Adaptive Calibration with Sensitivity Spike Compression:
```bash
python convert_neuromorphic.py --dataset cifar10dvs --arch resnet18 --T 8 --calib light \
--maxspike 4 --search --maxspike_ratio 1.0 --initialspike 8 --desired_spike 4 --minspike 1
```
- `--T`: timestep of SNN.
- `--calib`: calibration method, `light` as default .
- `--maxspike`: the maximum number of spikes allowed to be fired per time step.
- `--search`: Enables the search for optimal burst-firing pattern for each layer.
- `--maxspike_ratio`: the factor ratio of energy budget.
- `--initialspike`: the initial burst-firing setting.
- `--desired_spike 4`: the target burst firing pattern.
- `--minspike 1`: the minimum number of spikes allowed.

For some tasks, we also supports an advanced calibraton mode, see ImageNet experimental code for details.

#### 3. Adaptive Calibration with AdaFire Neuron:
```bash
python convert_neuromorphic.py --dataset xxx --arch xxx --T 8 --calib light \
--maxspike 4 --search_threshold --threshold_ratio 1.0
```
- `--search_threshold`: Activates the search for the 
- `--threshold_ratio 1.0`: the threshold ratio for spike compression.


#### 4. Adaptive Calibration with Input-Aware Adaptive Timestep:

The code for input-aware Adaptive Timestep inference is provided in the static dataset codebase and can also be applied to other tasks.

### TO-DO Main results and checkpoints

### Acknowledgement:
The code adopts some implementation in the following repositories:

3D Tasks: https://github.com/fxia22/pointnet.pytorch

Object Detection and Semantic Segmentation: https://github.com/zju-bmi-lab/Fast-SNN (modifications in the folder of tools and vedaseg)

Adaptive Timestep for SNN Inference: https://github.com/Intelligent-Computing-Lab-Yale/SEENN
