# TREMOR: Template Matching for Large Seismic Data Collections

**Manos Chatzakis**<sup>1</sup>, **Rodrigo Flores-Allende**<sup>2</sup>, **Mikael Freire**<sup>3</sup>, **Yoann Cano**<sup>3</sup>, **Léonard Seydoux**<sup>2</sup>, **Themis Palpanas**<sup>1</sup>

<sup>1</sup> Université Paris Cité, LIPADE &nbsp;&nbsp; <sup>2</sup> IPGP &nbsp;&nbsp; <sup>3</sup> CEA

📄 Paper: `<PAPER-LINK>` &nbsp;·&nbsp; 💾 Datasets: `<DATASET-LINK>`

## Abstract

Seismic station networks continuously record the ground velocity at several locations on earth in the form of waveform
time series. Seismologists analyze these waveforms to detect various kinds of geophysical events (including
earthquakes) that can then be located and characterized. Some of these events are of particular interest: they are
called *templates* and are used to search the seismic data collections for matching, similar events. This process,
known as *template matching*, is a fundamental task in seismology, serving as a backbone for various seismic analyses.
However, template matching requires extensive processing times, especially for seismic collections that exceed the
memory capacity of a single machine. This poses a significant challenge to seismologists, and is becoming worse as the
seismological datasets continue to grow in size. In this paper, we introduce TREMOR, a distributed data series
processing framework for template matching, designed to efficiently handle large waveform collections. We apply TREMOR
to two representative real-world seismic use cases for template matching and, through an extensive experimental
evaluation, we demonstrate its efficiency, with TREMOR being 2.3–6.2× faster than the best competing method on every
workload, while returning exactly the same results.


If you use TREMOR or our datasets, please cite:

```bibtex
@article{tremor,
  title   = {TREMOR: Template Matching for Large Seismic Data Collections},
  author  = {Chatzakis, Manos and Flores-Allende, Rodrigo and Freire, Mikael and Cano, Yoann and Seydoux, L{\'e}onard and Palpanas, Themis},
  journal = {TODO},
  year    = {TODO}
}

@misc{tremor-datasets,
  title  = {TREMOR datasets: TODO},
  author = {TODO},
  year   = {TODO},
  doi    = {TODO}
}

@article{duverger2021ceaseismic,
  title={A decade of seismicity in metropolitan France (2010--2019): the CEA/LDG methodologies and observations},
  author={Duverger, Clara and Mazet-Roux, Gilles and Bollinger, Laurent and Guilhem Trilla, Aurelie and Vallage, Amaury and Hernandez, Bruno and Cansi, Yves},
  journal={BSGF-Earth Sciences Bulletin},
  volume={192},
  number={1},
  pages={25},
  year={2021},
  publisher={EDP Sciences}
}

@article{flores2026fine,
  title={Fine-scale segmentation and spatiotemporal variability of the 2010 Mw 8.8 Maule aftershock sequence revealed by a deep-learning-based earthquake catalog},
  author={Flores-Allende, Rodrigo and Seydoux, L{\'e}onard and Beauc{\'e}, {\'E}ric and Bonilla, Luis Fabian and Gueguen, Philippe and Satriano, Claudio},
  journal={Journal of Geophysical Research: Solid Earth},
  volume={131},
  number={4},
  pages={e2026JB034262},
  year={2026},
  publisher={Wiley Online Library}
}
```

## Overview

TREMOR finds, for every template, the subsequences of a long seismic waveform that match it, with the z-normalized
Euclidean distance or, equivalently, the Pearson correlation coefficient. It supports both search variants used in
seismology: **k-nearest-neighbor search** (the k best matches of every template) and **threshold search** (every
subsequence whose correlation with the template exceeds a threshold). All answers are exact.

- **Distributed subsequence index.** Each node indexes the z-normalized subsequences of its part of the waveform in an
  iSAX index; the parts can be replicated over several nodes, which then share the templates (building on Odyssey,
  our framework for distributed data series similarity search).
- **Adaptive search.** For every template, TREMOR estimates from a small sample how well the index would prune. It
  answers the template through its index when the index prunes well, and otherwise with an early-abandoning scan of
  the waveform or with FFT-based distance computation, whichever its cost model predicts to be faster.
- **Built for HPC clusters.** C++17, MPI across nodes, and many threads with SIMD within a node.

## Repository structure

| Path | Contents |
|---|---|
| `tremor/`, `tremor_main.cpp` | TREMOR (index, distributed search, scan and FFT fallbacks), and the skip-sequential scan baseline |
| `dmass/` | DMASS, our distributed versions of the MASS algorithm (`DMASS_V1.c`, `DMASS_V3.c`) used as baselines |
| `dataprep/` | Scripts that build the binary waveforms and templates from the raw seismic recordings, and the metadata (plots and statistics) of the channels used in the paper (`dataprep/paper_datasets_metadata/`) |
| `experiments/` | Scripts that run the experiments of the paper on a Slurm cluster |
| `notebooks/` | Analysis notebook: produces the figures and tables of the paper, and checks every number of the paper against the runs |
| `Makefile` | Builds TREMOR |

## Requirements

- An MPI C/C++ compiler (`mpicxx`, `mpicc`; e.g., Intel MPI or Open MPI) with OpenMP support
- [FFTW 3](https://www.fftw.org/) (with its threads library, for DMASS)
- An x86-64 CPU with AVX2 and FMA
- For the data preparation and the notebook: Python 3 with NumPy, pandas, Matplotlib, Jupyter, and (for `dataprep/`) ObsPy

## Building

```bash
# TREMOR -> bin/tremor_main (set FFTW to your FFTW installation)
make -j FFTW=/path/to/fftw

# DMASS (optional, the MASS baselines) -> bin/dmass_main and bin/dmass_v3_main
mpicc -O3 -std=gnu11 -fopenmp -I/path/to/fftw/include -o bin/dmass_main    dmass/DMASS_V1.c -L/path/to/fftw/lib -lfftw3 -lfftw3_threads -lpthread -lm
mpicc -O3 -std=gnu11 -fopenmp -I/path/to/fftw/include -o bin/dmass_v3_main dmass/DMASS_V3.c -L/path/to/fftw/lib -lfftw3 -lfftw3_threads -lpthread -lm
```

## Quick start

Download the datasets (`<DATASET-LINK>`), then run TREMOR with one MPI process per node. With the configuration of the
paper (adaptive search, full replication, 128 threads per node):

```bash
mkdir -p results

# Threshold search on a Maule channel: every match with correlation >= 0.9
mpirun -n 4 ./bin/tremor_main \
    --dataset  maule/waveforms/ZE_G50S__HHN.len440844139.bin \
    --queries  maule/templates/ZE_G50S__HHN.q50.bin --window-length 992 \
    --mode threshold --threshold 0.9 --merge-offset 248 \
    --refinement adaptive --replication-groups 1 \
    --index-threads 128 --search-workers 128 --leaf-size 2000 --paa-segments 16 \
    --output results/ZE_G50S__HHN.t0.9

# k-NN search on a SeiFR channel: the 5 best matches of every template
mpirun -n 4 ./bin/tremor_main \
    --dataset  seifr/waveforms/RD_ROSF__BHZ.len1576800001.bin \
    --queries  seifr/templates/RD_ROSF__BHZ.q50.bin --window-length 3000 \
    --mode knn --k 5 --merge-offset 750 \
    --refinement adaptive --replication-groups 1 \
    --index-threads 128 --search-workers 128 --leaf-size 2000 --paa-segments 8 \
    --output results/RD_ROSF__BHZ.k5.csv
```

- `--refinement adaptive` enables TREMOR's adaptive search (index, scan, or FFTs per template), as in the paper.
- `--replication-groups G` splits the waveform into `G` parts, each held by `#processes / G` nodes
  (`1`: full replication; `0`: no replication).
- `--merge-offset` reports nearby matches of the same event once (a quarter of the template length in the paper).
- `--method skip-sequential` runs the skip-sequential scan baseline instead of TREMOR.
- `./bin/tremor_main --help` lists all options.

**Outputs.** Threshold search writes one CSV per process (`<output>_<rank>.csv`: template, match counter, position, and
correlation of every match); k-NN search writes one CSV (`template, k, position, correlation`). A timings CSV
(`<output>.timings.csv`) gives the index construction and query answering time of every process.

## Datasets

The datasets of the paper are available at `<DATASET-LINK>`:

| | **Maule** | **SeiFR** |
|---|---|---|
| Origin | International Maule Aftershock Deployment (IMAD), Chile, 2010 | CEA/LDG network, metropolitan France, 2020–2021 |
| Channels | 20 | 7 (vertical) |
| Sampling rate | 50 or 100 Hz | 25 Hz |
| Points | 13.9 billion (56 GB) | 10.5 billion (42 GB) |
| Templates used | 972 (50 per channel; all 22 for one channel) | 350 (50 per channel) |
| Template length | 496, 992, or 1,000 points (10 s) | 3,000 points (120 s) |
| Search in the paper | threshold search | k-NN search |

Waveforms (`<channel>.len<N>.bin`) and templates are raw arrays of 32-bit little-endian floats, band-pass filtered
between 1 and 10 Hz; a template file stores its templates one after the other. The `.q50.json` files record how the 50
templates of every channel were sampled. Plots and statistics of every waveform and template set are in
`dataprep/paper_datasets_metadata/`.


