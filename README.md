# Latent Mamba Operator for Partial Differential Equations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/M3RG-IITD/LaMO/blob/main/LICENSE)
[![ArXiv](https://img.shields.io/static/v1?&logo=arxiv&label=Paper&message=Arxiv:LaMO&color=B31B1B)](https://arxiv.org/abs/2505.19105)

This repository contains the official implementation of our [Latent Mamba Operator for Partial Differential Equations](https://arxiv.org/abs/2505.19105).

> [**Latent Mamba Operator for Partial Differential Equations**](https://arxiv.org/abs/2505.19105)   
> Karn Tiwari, Niladri Dutta, N M Anoop Krishnan, Prathosh A P

<p align="center">
<img src=".\assets\Architecture.png" height = "200" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overview of LaMO.
</p>


## Codebase for Reproducibility: Getting Started

1. Install Python 3.8. For convenience, please go ahead and execute the following command.

```bash
pip install -r requirements.txt
```

2. Prepare Data. You can obtain experimental datasets from the following links (Download).

| Dataset       | Task                                    | Geometry        | Link                                                         |
| ------------- | --------------------------------------- | --------------- | ------------------------------------------------------------ |
| Elasticity    | Estimate material inner stress          | Point Cloud     | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Plasticity    | Estimate material deformation over time | Structured Mesh | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Navier-Stokes | Predict future fluid velocity           | Regular Grid    | [[Google Cloud]](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) |
| Darcy         | Estimate fluid pressure through medium  | Regular Grid    | [[Google Cloud]](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) |
| AirFoil       | Estimate airﬂow velocity around airfoil | Structured Mesh | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Pipe          | Estimate fluid velocity in a pipe       | Structured Mesh | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |

3. Train and evaluate the model. We provide the experiment scripts of all benchmarks under the folder `./scripts/.` You can reproduce the experiment results as the following examples:

```bash
bash scripts/LaMO_Elas.sh # for Elasticity
bash scripts/LaMO_Plas.sh # for Plasticity
bash scripts/LaMO_NS.sh # for Navier-Stokes
bash scripts/LaMO_Darcy.sh # for Darcy
bash scripts/LaMO_Airfoil.sh # for Airfoil
bash scripts/LaMO_Pipe.sh # for Pipe
```
Note: You must change the argument `--data-path` in the above script files to your dataset path.

4. Develop your own model. Here are the instructions:

   - Add the model file under folder `./models/`.
   - Add the model name into `./model_dict.py`.
   - Add a script file under folder `./scripts/` and change the argument `--model`.

## Main Results

<p align="center">
<img src=".\assets\result.png" height = "400" alt="" align=center />
<br><br>
<b>Figure 2.</b> Main Results Compared with SOTA Operator. L2 Error is reported.
</p>

**LaMO achieves 32.3% averaged relative increment over the previous second-best model i.e, Transolver.**

## Showcases

<p align="center">
<img src=".\assets\showcase.png" height = "400" alt="" align=center />
<br><br>
<b>Figure 3.</b> Showcases. Error heatmap is plotted for Transolver and LaMO for Darcy, Plasticity, and Navier-Stokes.
</p>

## Citation

Please consider citing our paper if you find it helpful. Thank you!

```
@misc{tiwari2025latentmambaoperatorpartial,
      title={Latent Mamba Operator for Partial Differential Equations}, 
      author={Karn Tiwari and Niladri Dutta and N M Anoop Krishnan and Prathosh A P},
      year={2025},
      eprint={2505.19105},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.19105}, 
}
```
## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code base or datasets on which we have built our code:

1) https://github.com/neuraloperator/neuraloperator

2) https://github.com/thuml/Transolver/tree/main

3) https://github.com/state-spaces/mamba/tree/main

4) https://github.com/goombalab/hydra/tree/main

