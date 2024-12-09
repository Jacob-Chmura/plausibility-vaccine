<a id="readme-top"></a>

<div align="center">
  <h1 style="font-size:3vw;padding:0;margin:0;display:inline">Plausibility Vaccine</h3>
  <h3 style="margin:0">Injecting LLM Knowledge for Event Plausibility</h3>
  <a href="https://github.com/Jacob-Chmura/plausibility-vaccine"><strong>Read the paper»</strong></a>
</div>

<br />

<div align="center">

<a href="">[![Contributors][contributors-shield]][contributors-url]</a>
<a href="">[![Issues][issues-shield]][issues-url]</a>
<a href="">[![MIT License][license-shield]][license-url]</a>

</div>

<div align="center">

<a href="">![example workflow](https://github.com/Jacob-Chmura/plausibility-vaccine/actions/workflows/ruff.yml/badge.svg)</a>
<a href="">![example workflow](https://github.com/Jacob-Chmura/plausibility-vaccine/actions/workflows/mypy.yml/badge.svg)</a> <a href="">![example workflow](https://github.com/Jacob-Chmura/plausibility-vaccine/actions/workflows/testing.yml/badge.svg)</a>

</div>

## About The Project

_Plausibility Vaccine_ is a library that investigates parameter-efficient finetuning and modular transfer of prompted physical property knowledge for modelling event plausibility.

## Getting Started

### Prerequisites

The project uses [uv](https://docs.astral.sh/uv/) to manage and lock project dependencies for a consistent and reproducible environment. If you do not have `uv` installed on your system, visit [this page](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

**Note**: If you have `pip` you can just invoke:

```sh
pip install uv
```

### Installation

```sh
# Clone the repo
git clone https://github.com/Jacob-Chmura/plausibility-vaccine.git

# Enter the repo directory
cd plausibility-vaccine

# Install dependencies into an isolated environment
uv sync
```

## Usage

### Running Plausibility Vaccine

_Full End-to-End Experiments_

```sh
./run_plausibility_vaccine.sh
```

_Baseline Experiments Only_

```sh
./run_plausibility_vaccine.sh config/baseline.yaml
```

_Pre-training Adapters Only_

```sh
./run_plausibility_vaccine.sh config/pretraining.yaml
```

### Running Analysis of Results

_All Analytics_

**Note**: requires that you have previously ran `plausibility_vaccine.sh` and have generated results

```sh
./run_analytics.sh
```

_Non-result Dependent Analytics_

```sh
./run_analytics.sh --no-results
```

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Jacob Chmura - jacobpaul.chmura@gmail.com

## Citation

```
@article{chmura-etal-2024-plausibility,
  title   = "Plausibility Vaccine: Injecting LLM Knowledge for Event Plausibility",
  author  = "Chmura, Jacob and Dauvet, Jonah, and Sabry, Sebastian"
  journal = "arXiv preprint arXiv:TODO.TODO",
  url     = "TODO"
  year    = "2024",
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[contributors-shield]: https://img.shields.io/github/contributors/Jacob-Chmura/plausibility-vaccine.svg?style=for-the-badge
[contributors-url]: https://github.com/Jacob-Chmura/plausibility-vaccine/graphs/contributors
[issues-shield]: https://img.shields.io/github/issues/Jacob-Chmura/plausibility-vaccine.svg?style=for-the-badge
[issues-url]: https://github.com/Jacob-Chmura/plausibility-vaccine/issues
[license-shield]: https://img.shields.io/github/license/Jacob-Chmura/plausibility-vaccine.svg?style=for-the-badge
[license-url]: https://github.com/Jacob-Chmura/plausibility-vaccine/blob/master/LICENSE.txt
