
# Rubik's Cube Solver
![Project Status: In Development](https://img.shields.io/badge/STATUS-WIP-blue)

> A Rubik's Cube solver combining a Python computer vision scanner for state-capture with an Arduino stepper controller to physically resolve the puzzle.

## Main Pipeline
```mermaid
flowchart TB
    A[Camera Feed] --> B[Image Processing]
    B --> C[String of colors]
    C --> D[Solution Algorithm]
    D --> E[Solution String]
    E --> F[Microcontroller]
    F --> G[Physical Control]
```

## Work Breakdown Structure (WBS)
```mermaid
flowchart LR
    A([Autonomous Rubik's Cube Solver])
    
    A --> B[Computer Vision]
    A --> C[Solving Logic]
    A --> E[CAD & Mechanical Design]
    A --> F[Electronics & Power Systems]
    A --> D[Communication Interface]
    A --> G[Microcontroller Firmware]

    %% Computer Vision Breakdown
    B --> B1[Isolating Cube ✅]
    B --> B2[Face ROI Extraction 🟡]
    B --> B3[Color Classification ⬅️]

    %% Solve Logic Breakdown
    C --> C1[OOP Design ⬅️]
    C --> C2[Sorting Color Data]
    C --> C3[Solve Algorithm]

    %% CAD Breakdown
    E --> E1[NEMA 17 Motor Brackets]
    E --> E2[Core Face Couplers]
    E --> E3[Chassis Frame Design]

    %% Electronics Breakdown
    F --> F1[Stepper Driver Wiring]
    F --> F2[Vref Current Limiting]
    F --> F3[Power Bus Distribution]
    F --> F4[PCB Design]

    %% Firmware Breakdown
    G --> G1[Instruction Interpretation]
    G --> G2[Motor Execution Routines]
```
* ✅ **Completed** - The feature is fully functional and implemented.
* ⬅️ **WIP** - Currently being developed or under active work.
* 🟡 **In Testing** – The code/design is done and currently undergoing validation.
* (None) **Not Started** - Planned feature; development has not yet begun.

## Tech Stack & Hardware Components

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)
![Arduino](https://img.shields.io/badge/Arduino-00878F?style=for-the-badge&logo=arduino&logoColor=white)

| Domain | Technologies / Components |
| -------- | -------- |
| **Software & Computer Vision** | Python 3.13.3, OpenCV, NumPy |
| **Microcontroller**  | Arduino Mega |
| **Electronics** | DRV8825 Drivers, 12V DC Power Supply |
| **Actuators** | NEMA 17 Stepper Motors |
| **CAD & Mechanical** | Custom 3D Printed Motor Brackets, Core Couplers, Frame Chassis |


## Gallery

<figure>
    <img src="assets/Isolated_cube.png" alt="Isolation of the cube" width="600">
    <figcaption><i>Isolation of the cube by computer vision >> next step: extract ROIs (valid faces)</i></figcaption>
</figure>
  

<figure>
    <img src="assets/Rubiks_vision_v1.gif" alt="Cube faces detection and isolation" width="600">
    <figcaption><i>Face detection and Region of Interest (ROI) generation for each detected face</i></figcaption>
</figure>

<figure>
    <img src="assets/Old_prot.png" alt="Old Prototype" width="600">
    <figcaption><i>This is an old prototype of the project; a new design will be shared soon.</i></figcaption>
</figure>