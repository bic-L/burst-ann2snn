## Adaptive Calibration: A Unified Conversion Framework of Spiking Neural Networks

### Abstract
Spiking Neural Networks (SNNs) have emerged as a promising energy-efficient alternative to traditional Artificial Neural Networks (ANNs). Despite this, bridging the performance gap with ANNs in practical scenarios remains a significant challenge. This paper focuses on addressing the dual objectives of enhancing the performance and efficiency of SNNs through the established SNN Calibration conversion framework. Inspired by the biological nervous system, we propose a novel **Adaptive-Firing Neuron Model (AdaFire)** that dynamically adjusts firing patterns across different layers, substantially reducing conversion errors within limited timesteps. Moreover, to meet our efficiency objectives, we propose two novel strategies: an **Sensitivity Spike Compression (SSC)** technique and an **Input-aware Adaptive Timesteps (IAT)** technique. These techniques synergistically reduce both energy consumption and latency during the conversion process, thereby enhancing the overall efficiency of SNNs. Extensive experiments demonstrate our approach outperforms state-of-the-art SNNs methods, showcasing superior performance and efficiency in 2D, 3D, and event-driven classification, as well as object detection and segmentation tasks. 


![Main Figure](figures/main.png)


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

