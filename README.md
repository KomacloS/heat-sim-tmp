# Laser-Pad-Thermal

This repository contains a small Streamlit demo for heating a circular copper pad using a lumped thermal model (Milestone 1).

The pad properties are computed from its diameter and thickness. Temperature rise is integrated in time assuming all laser power is absorbed.

Run the interactive demo with:

```bash
poetry run streamlit run demos/demo_m1.py
```

Running it directly with `poetry run demo-m1` executes in **bare mode**,
which still works but prints several warnings.

For the spatially resolved transient solver (Milestone&nbsp;2):

```bash
poetry run streamlit run demos/demo_m2.py
```

Milestones 3–5 extend the solver with beam-shape effects,
multilayer stacks, and PCB traces. Launch them via

```bash
poetry run streamlit run demos/demo_m3.py
poetry run streamlit run demos/demo_m4.py
poetry run streamlit run demos/demo_m5.py
```

These demos use an explicit time-marching scheme with a stability
limit on the time step. If you see an error like
"Time step ... exceeds stability limit ...", reduce the chosen step
size or enable the **Ignore stability limit** checkbox in the UI.

Run the test suite with:

```bash
poetry run pytest -q
```

For the trace-aware multilayer model (Milestone 5):

```bash
poetry run demo-m5
```

When the app asks for a trace configuration, upload `sample_traces.json` to see
a simple demo.

To terminate a running demo and launch another one, press `Ctrl+C` in the
terminal where Streamlit is running. This stops the server so you can start the
next demo without closing VS Code.
