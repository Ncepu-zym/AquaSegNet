# AquaSegNet: A Real-time Dynamic Network for High-Precision Water Leakage Segmentation

[![Powered by Ultralytics](https://img.shields.io/badge/Powered%20by-Ultralytics%20YOLOv11-blue)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-AGPL%203.0-green)]()

This repository contains the official implementation of the core modules and configuration for **AquaSegNet**, as proposed in our EAAI paper: *"AquaSegNet: A real-time dynamic network for high-precision water leakage segmentation in shield tunnel lining"*.

AquaSegNet introduces three novel architectural units designed for irregular leakage patterns and complex backgrounds:
1.  **MSDIM (Multi-Scale Dynamic Inception Mixer)**: Replaces standard bottlenecks to capture multi-scale strip-like features.
2.  **SEFFN (Spectral Enhanced Feed-Forward Network)**: Integrates global frequency-domain perception to suppress noise.
3.  **DFSH (Dynamic Fusion Shared Head)**: Efficiently decouples detection and segmentation tasks.

## 📂 Repository Structure

The core code is designed to be "Plug-and-Play" within the [Ultralytics](https://github.com/ultralytics/ultralytics) framework.

```text
.
├── AquaSegNet.yaml        # Model configuration file (Architecture)
├── modules/
│   ├── C3K2-MSDIM.py      # Implementation of MSDIM & C3k2_MSDIM
│   ├── Spec-C2PSA.py      # Implementation of SEFFN & Spec_C2PSA
│   └── DFSH.py            # Implementation of the Dynamic Fusion Shared Head
└── README.md

## ⚠️ Usage Note (Important)

Since **AquaSegNet** introduces custom architectural modules (**MSDIM**, **SEFFN**, **DFSH**) that are not part of the standard YOLOv11 distribution, you **must** register them in the framework before training.

### Integration Steps:

1.  **Copy Source Files**:
    Move the `.py` files from the `modules/` folder of this repository into your local Ultralytics source directory:
    `ultralytics/nn/modules/`

2.  **Register Modules**:
    Open `ultralytics/nn/tasks.py` and modify the `parse_model` function to recognize the new classes.
    
    *Add the import at the top of `tasks.py`:*
    ```python
    from ultralytics.nn.modules.C3K2_MSDIM import C3k2_MSDIM
    from ultralytics.nn.modules.Spec_C2PSA import Spec_C2PSA
    from ultralytics.nn.modules.DFSH import Segment_DFSH
    ```

    *Add classes to the parsing loop:*
    ```python
    # Find the line: for m in (Conv, GhostConv, Bottleneck, ...):
    # Add your modules to the list:
    for m in (..., C3k2_MSDIM, Spec_C2PSA, Segment_DFSH):
    ```

3.  **Run Training**:
    ```bash
    yolo segment train model=AquaSegNet.yaml data=your_dataset.yaml epochs=300
    ```