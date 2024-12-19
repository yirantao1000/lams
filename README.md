
<div align="center">
  <img width="200px" src="imgs/logo_new.png"/>
  
  # LAMS: LLM-Driven Automatic Mode Switching for Assistive Teleoperation
  ### HRI 2025
</div>

<div align="center">
  <img src="imgs/teaser.png"/>
</div> 

<p align="left">
    <a href=''>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://lams-assistance.github.io/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
</p>
This is the official repo for the paper:  


**[LAMS: LLM-Driven Automatic Mode Switching for Assistive Teleoperation](https://lams-assistance.github.io/)**  
[Yiran Tao*](https://yirantao1000.github.io/), [Jehan Yang*](https://jehanyang.github.io/), [Dan Ding](https://www.shrs.pitt.edu/people/dan-ding), [Zackory Erickson](https://zackory.com/)

accepted to HRI 2025 

Teleoperating robotic manipulators via interfaces like joysticks often requires frequent switching between control modes, where each mode maps joystick movements to specific robot actions. LAMS (LLM-Driven Automatic Mode Switching) is a novel approach that leverages Large Language Models (LLMs) to automatically switch control modes based on task context. LAMS requires no prior task demonstrations and incrementally improves by integrating user-generated mode-switching examples. 

This repository provides the official implementation of LAMS, designed for experiments with a Kinova robotic arm.

<!-- ## Table of Contents
- [Setup](#setup) -->
  <!-- - [RoboGen](#RoboGen)
  - [OMPL](#Open-Motion-Planning-Library)
  - [Dataset](#dataset) -->
<!-- - [Run LAMS](#lams) -->
  <!-- - [Automated Task Generation & Skill Learning](#One-click-for-all) -->



## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yirantao1000/lams.git
   ```

2. **Set up a conda environment**:
   We recommend using a conda environment to manage dependencies.
   ```bash
   conda env create -f environment.yaml
   conda activate lams
   ```
   If installing from the `environment.yaml` file fails, manually install any missing packages.

3. **Set up OpenAI API**:
   Add your OpenAI API key in `gpt_api.py`:
   ```python
   self.client = OpenAI(api_key="your_api_key")
   ```


## Run LAMS

Book storage task:
```
python main.py --config_path configs/book.json 
```
Water Pouring task:
```
python main.py --config_path configs/water.json 
```
Simply run the same code multiple times for incremental improvement. LAMS will automatically save user generated examples in examples folder, and the summarized rules in rules folder. "example_index" and "interact_index" in the config file will also update automatically. 

### Incremental Improvement

- Simply run the same code multiple times to enable incremental improvement.
- LAMS automatically:
  - Saves user-generated examples in the `examples` folder.
  - Updates summarized rules in the `rules` folder.
  - Modifies `example_index` and `interact_index` in the config file.



<!-- ## Acknowledgements
- The interface between OMPL and pybullet is based on [pybullet_ompl](https://github.com/lyfkyle/pybullet_ompl).
- Part of the objaverse annotations are from [Scalable 3D Captioning with Pretrained Models](https://arxiv.org/abs/2306.07279) -->

## Citation
If you find this codebase/paper useful for your research, please consider citing:
<!-- ```
@article{wang2023robogen,
  title={Robogen: Towards unleashing infinite data for automated robot learning via generative simulation},
  author={Wang, Yufei and Xian, Zhou and Chen, Feng and Wang, Tsun-Hsuan and Wang, Yian and Fragkiadaki, Katerina and Erickson, Zackory and Held, David and Gan, Chuang},
  journal={arXiv preprint arXiv:2311.01455},
  year={2023}
}
``` -->