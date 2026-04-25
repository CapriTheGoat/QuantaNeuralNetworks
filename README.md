# Quanta Neural Networks

**Authors:** Varun Sundar, Tianyi Zhang, Sacha Jungerman, Mohit Gupta.

**Abstract:** Quanta image sensors record individual photons, enabling capabilities like imaging in near-complete darkness and ultra-high-speed videography. 
Yet, most research on quanta sensors is limited to recovering image intensities. 
Can we go beyond just imaging and develop algorithms that can extract high-level scene information from quanta sensors? 
This could unlock new possibilities in vision systems, offering reliable operation in extreme conditions. 
The challenge: raw photon streams captured by quanta sensors have fundamentally different characteristics than conventional images, making them incompatible with vision models. 
One approach is to first transform raw photon streams to conventional-like images, but this is prohibitively expensive in terms of compute, memory, and latency.

We propose quanta neural networks (QNNs) that directly produce downstream task objectives from raw photon streams. 
Our core proposal is a trainable QNN layer that can seamlessly integrate with existing image- and video-based neural networks, producing quanta counterparts. 
By avoiding image reconstruction and allocating computational resources on a scene-adaptive basis, QNNs achieve 1 to 2 orders of magnitude improvements across all efficiency metrics (compute, latency, readout bandwidth) as compared to reconstruction-based quanta vision, while maintaining high task accuracy across a wide gamut of challenging scenarios, including low light and rapid motion.

## Install

You can use [`mamba`](https://mamba.readthedocs.io/en/latest/index.html) as a drop-in replacement for `conda` below.

### CPU Only

```bash
conda env create -f environment_cpu.yml
```
to create a new environment, and
```bash
conda env update -n qnn -f environment_cpu.yml --prune
```
to update.

### GPU Enabled

Use `environment_gpu` instead of `environment_cpu` in the above commands.

### CPU and GPU

```bash
pip install -e .
```

to install the `qnn` library.

## Data Folder

* Can be found [here](https://github.com/wision-lab/datasets/). Includes hot-pixel masks and photon-cube sequences.
* Place photon-cube sequences in `data/sequences`.
* Place hot-pixel masks in `data/hot_pixel_mask`.
* Place color-filter arrays in `data/color_filter_array`.