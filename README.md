# Dynamic Gradient Balancing for Enhanced Adversarial Attacks on Multi-Task Models 
This is the website for our paper "Dynamic Gradient Balancing for Enhanced Adversarial Attacks on Multi-Task Models". 
The arXiv version can be found [here](https://arxiv.org/abs/2305.12066.pdf).

### Abstract
Multi-task learning (MTL) creates a single machine learning model called multi-task model, to simultaneously perform multiple tasks.  
Although the security of single task classifiers has been extensively studied, there are several critical security research questions for multi-task models including 1) How secure are multi-task models to single task adversarial attacks, 2) Can adversarial attacks be designed to attack multiple tasks simultaneously, and 3) Does task sharing and adversarial training increase multi-task model robustness to adversarial attacks? In this paper, we answer these questions through careful analysis and rigorous experimentation. First, we develop naïve adaptation of single-task white-box attacks and analyze their inherent drawbacks. We then propose a novel attack framework, Dynamic Gradient Balancing Attack (DGBA). Our framework poses the problem of attacking a multi-task model as an optimization problem based on averaged relative loss change across tasks and solves the problem by approximating it as an integer linear programming problem. Extensive evaluation on two popular MTL benchmarks, NYUv2 and Tiny-Taxonomy, demonstrates the effectiveness of DGBA compared to naïve multi-task attack baselines, on both clean and adversarially trained multi-task models. Our results also reveal a fundamental trade-off between improving task accuracy via parameter sharing across tasks and undermining model robustness due to increased attack transferability from parameter sharing.


### Cite
Welcome to cite our work if you find it is helpful to your research.
```
@misc{zhang2023dynamic,
      title={Dynamic Gradient Balancing for Enhanced Adversarial Attacks on Multi-Task Models}, 
      author={Lijun Zhang and Xiao Liu and Kaleel Mahmood and Caiwen Ding and Hui Guan},
      year={2023},
      eprint={2305.12066},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# How To Use
*Under Construction*