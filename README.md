
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
    B --> B2[Face ROI Extraction ⬅️]
    B --> B3[Color Classification]

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

## Tech Stack & Hardware Components

| Domain | Technologies / Components |
| -------- | -------- |
| Software & Computer Vision | Python 3.13.3, OpenCV, NumPy |
| Microcontroller  | Arduino Mega |
| Electronics | DRV8825 Drivers, 12V DC Power Supply |
| Actuators | NEMA 17 Stepper Motors |
| CAD & Mechanical | Custom 3D Printed Motor Brackets, Core Couplers, Frame Chassis |

## Gallery

<figure>
    <img src="assets/Isolated_cube.png" alt="Isolation of the cube" width="600">
    <figcaption><i>Isolation of the cube by computer vision >> next step: extract ROIs (valid faces)</i></figcaption>
</figure>

<figure>
    <img src="assets/Old_prot.png" alt="Old Prototype" width="600">
    <figcaption><i>This is an old prototype of the project; a new design will be shared soon.</i></figcaption>
</figure>