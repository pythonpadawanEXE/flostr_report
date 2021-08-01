# flostr

**flo**w **str**ucture analysis

# Installation

## Miniconda

Download one of the [Miniconda Windows Installers](https://docs.conda.io/en/latest/miniconda.html) and run it.

(Anaconda is fine too.)

## conda env

In the *Anaconda PowerShell*:
```shell
conda create -n flostr python
conda activate flostr
```

The `conda env` doesn't have to be called "flostr"; call it whatever.

## Requirements

Download the [`requirements.txt`](./requirements.txt) file.

```shell
pip install -r requirements.txt
conda install pyamg
```

## ParaView

Install [ParaView](https://paraview.org).

# Getting started

Download the scripts:
* [`cylinder.py`](./cylinder.py)
* [`st08_navier_stokes_cylinder.py`](./st08_navier_stokes_cylinder.py)
* [`probe.py`](./probe.py)
* [`pod.py`](./pod.py)

## Simulation

Run the simulation:
```shell
python st08_navier_stokes_cylinder.py
```

It takes about five minutes on a modern desktop.

This produces an [XDMF](https://xdmf.org) solution file containing the mesh (about 1600 nodes) and the nodal velocity and pressure at 5000 time-steps.
* `st08_navier_stokes_cylinder.xdmf`
* `st08_navier_stokes_cylinder.h5`

It also calls [`cylinder.py`](./cylinder.py) to create a mesh and store it in `cylinder.json`.  If present, this is reread rather than regenerated.

## Postprocessing

These can be inspected with ParaView; just open the .xdmf file while the .h5 file is in the same directory.
```shell
paraview st08_navier_stokes_cylinder.xdmf
```

(Or launch ParaView from the MS-Windows start-bar and open the .xdmf file.)

Choose *Xdmf3ReaderS* from the list of readers.

## Point probe

Download [`probe.py`](./probe.py) and run it in the directory containing `st08_navier_stokes_cylinder.xdmf`.
```shell
python probe.py
```

It should produce `st08_navier_stokes_cylinder.png`, showing the history of the pressure at the nominal fore and aft stagnation points.

## Proper orthogonal decomposition

Download [`pod.py`](./pod.py) and run it in the directory containing `st08_navier_stokes_cylinder.xdmf`.
```shell
python pod.py
```

It should produce `pod.xdmf`, showing the first half-dozen POD modes.  It can be viewed in ParaView, as above.
