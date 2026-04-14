# Simulation of Human Arm Movement in a Virtual Environment

Bachelor's Thesis (TFG) — Bachelor's Degree in Computer Engineering, Universitat de Barcelona.

A Virtual Reality environment built in Unity 6, integrated with an OptiTrack 6-camera optical tracking system, designed to study motor perception and the sense of body ownership. The system renders a first-person virtual arm driven by 3 physical markers (shoulder, elbow, hand) and enables controlled dissociation between real motor execution and visual feedback in VR.

---

## Repository structure

```
.
├── tfgadria/                  Unity 6 project
│   ├── Assets/
│   │   ├── Models/            Y Bot rigged mesh (Mixamo)
│   │   ├── Scripts/           C# scripts (XR logging, UDP receiver, calibration, arm controller)
│   │   ├── Scenes/            VR scene with table and XR Origin
│   │   ├── Settings/          URP and XR settings
│   │   └── ...                Materials, Textures, XR Toolkit assets
│   ├── Packages/              Unity package manifest
│   ├── ProjectSettings/       Unity project settings
│   └── Positions/             CSV logs from XR and OptiTrack sessions
├── PythonClient/              OptiTrack NatNet client + UDP forwarder
├── imgs/                      Reference images
├── docs/                      Theoretical / supporting documentation
├── tests/                     Experimental tests (to be added)
└── README.md
```

---

## Quick start

### Requirements
- Unity 6 (LTS) with the XR Interaction Toolkit
- A VR headset (Meta Quest / Oculus, currently used for development)
- Python 3.10+ on the OptiTrack PC
- Motive software with NatNet streaming enabled (for real tracking)

### Running the tracking pipeline
1. On the OptiTrack PC, run `python PythonClient/PythonSample.py` (or `PythonSample2.py` for cleaner output).
2. Open `tfgadria/` in Unity and press Play.
3. Place 4 markers on the table corners and press **C** to calibrate.
4. Place 3 markers on the arm (shoulder, elbow, hand) and press **O** to activate the virtual arm.

For testing without OptiTrack hardware, use `PythonClient/PythonSampleMouse.py` to simulate marker data with the mouse.

---

## Author

Adrià Gasull Rectoret · Bachelor's Thesis 2025–2026 · Universitat de Barcelona

Supervisor: Dr. Ignasi Cos Aguilera

## License

MIT
