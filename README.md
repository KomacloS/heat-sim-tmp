# Laser-Pad-Thermal

This repository contains a small Streamlit demo for heating a circular copper pad using a lumped thermal model (Milestone 1).

The pad properties are computed from its diameter and thickness. Temperature rise is integrated in time assuming all laser power is absorbed.

Run the interactive demo with:

```bash
poetry run demo-m1
```

For the spatially resolved transient solver (Milestone 2):

```bash
poetry run demo-m2
```

For the trace-aware multilayer model (Milestone 5):

```bash
poetry run demo-m5
```

When the app asks for a trace configuration, upload `sample_traces.json` to see
a simple demo.
