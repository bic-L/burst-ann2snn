## Adaptive Calibration: A Unified Conversion Framework of Spiking Neural Networks

Codes and checkpoints will be updated SOON! [[Paper]](https://arxiv.org/abs/2311.14265)

### Key contributions

Our "***Adaptive Calibration***" framework leverages biologically-inspired burst-firing mechanisms, already supported by commercial neuromorphic hardware like Intel's Loihi 2 and Synsense's Xylo. By developing a training-free optimization algorithm, we automatically determine optimal burst-firing configurations based on each layer's sensitivity characteristics,  improving both efficiency and performance. Key features includes:

- **Training-free ANN-to-SNN Conversion:** Outperforms existing methods with minimal setup time (0.09 hours), eliminating the need for costly retraining while maintaining superior accuracy-energy balance.
- **Energy-efficient converted SNNs:** Delivers remarkable energy reductions across benchmark datasets (70.1% on CIFAR-10, 60.3% on CIFAR-100, and 43.1% on ImageNet) while preserving high accuracy.
- **Comprehensive Tasks/Architecture Support:** Demonstrates exceptional versatility through extensive validation across diverse tasks (2D/3D classification, event-driven processing, object detection, and segmentation) and modern architectures (ResNet, VGG, and Vision Transformers).


*Hardware implementation code will be available in subsequent works.

<p align="center">
<img src=https://github.com/user-attachments/assets/8f809915-5ed0-4a6a-a333-d540d22c8819 width="500">
</p>


### Running the Code

#### 1. Pre-training ANN on Neuromorphic Datasets:
```bash
python main_train_cifardvs.py --dataset cifar10dvs --arch resnet18
```
- `--dataset`: Specifies the dataset to be used, including `cifar10dvs, ncaltech101, ncars, action recognition`.

#### 2. SNN Calibration with Burst-Spike Technique:
```bash
python main_train_cifardvs.py --dataset cifar10dvs --arch resnet18 --T 8 --calib light --maxspike 4
```
- `--T`: timestep of SNN.
- `--calib`: calibration method, `light` as default .
- `--maxspike`: maximum number of burst-spikes.

#### 3. SNN Calibration with Burst-Spike Reallocation Technique:
```bash
python main_train_cifardvs.py --dataset cifar10dvs --arch resnet18 --T 8 --calib light \
--maxspike 4 --search --maxspike_ratio 1.0 --initialspike 8 --desired_spike 4 --minspike 1
```
- `--search`: Enables the search for optimal burst-spike reallocation.
- `--maxspike_ratio`: the factor ratio of energy budget.
- `--initialspike`: the initial number of burst-spike.
- `--desired_spike 4`: the target number of burst-spikes.
- `--minspike 1`: the minimum number of spikes allowed.

#### 4. SNN Calibration with Sensitivity Spike Compression Technique:
```bash
python main_train_cifardvs.py --dataset cifar10dvs --arch resnet18 --T 8 --calib light \
--maxspike 4 --search_threshold --threshold_ratio 1.0
```
- `--search_threshold`: Activates the search for the optimal threshold based on sensitivity.
- `--threshold_ratio 1.0`: the threshold ratio for spike compression.

