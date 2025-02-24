
# [ICLR 2025] Graph Assisted Offline-Online Deep Reinforcement Learning for Dynamic Workflow Scheduling

This repository contains the implementation of [**Graph Assisted Offline-Online Deep Reinforcement Learning (GOODRL) for Dynamic Workflow Scheduling (DWS)**](https://openreview.net/forum?id=4PlbIfmX9o), which addresses the dynamic workflow scheduling problem. The paper introduces three key innovations in <font color="green"><b>dynamic graph representations</b></font>, <font color="green"><b>decoupled Actor-Critic encoders for RL stability</b></font>, and <font color="green"><b>training methods for unpredictable changes</b></font>.

## Citation

If you find GOODRL helpful for your research or applied projects:
  ```bibtex
  @InProceedings{
      yang2025graph,
      title={Graph Assisted Offline-Online Deep Reinforcement Learning for Dynamic Workflow Scheduling},
      author={Yang, Yifan and Chen, Gang and Ma, Hui and Zhang, Cong and Cao, Zhiguang and Zhang, Mengjie},
      booktitle={International Conference on Learning Representations},
      year={2025}
  }
  ```

## Requirements

- Python 3.11.5
- PyTorch 2.4.1
- PyTorch Geometric 2.5.3

Install dependencies:
```bash
pip install --upgrade pip
pip install rl-zoo3 deap torch_geometric gym joblib openpyxl
```

> **Note**: The current implementation is **CPU-based**. GPU adaptations may require modifications. Multi-CPU parallel execution is supported by adjusting the following:
>
> **Non-parallel version**:
> ```python
> meanFlowTimes = []
> for t in range(configs.valid_num):
>     meanFlowTime = validation_H(t, configs)
>     meanFlowTimes.append(meanFlowTime)
> ```
>
> **Parallel version**:
> ```python
> meanFlowTimes = Parallel(n_jobs=-1)(delayed(validation_H)(t, configs) for t in range(configs.valid_num))
> ```

## How to Use the Code

### Offline Phase – Imitation Learning
Run the following command for offline imitation learning:
```bash
python step1.py --vm_types 6 --each_vm_type_num 4 --arr_rate 5.4 --lr_a 0.0001 --log_interval 1 --max_updates 10
```
- Execute multiple independent runs and select the best-performing actor for the next stage: offline PPO.
- The trained actors from this stage can be found in `./validation_data/step1`.

### Offline Phase – PPO
Run the following command for offline PPO training:
```bash
python step2.py --vm_types 6 --each_vm_type_num 4 --arr_rate 5.4 --lr_a 0.0003 --lr_c 0.001 --warmup_critic 200
```
- Execute multiple independent runs and select the best-performing actor for offline testing and as the initialization for the online PPO phase.
- The trained actor and critic models from this stage are saved in `./validation_data/step2`.

### Online Phase – Online PPO
Run the following command for online PPO training:
```bash
python step3.py --vm_types 6 --each_vm_type_num 4 --arr_rate 9 --online_start_ac 5_5_5.4 --wf_num 10000 --max_updates 500 --warmup_steps 50000 --lr_a 0.00005 --lr_c 0.0001 --n_epochs 5
```
- In each run, the final result represents the online performance after processing `wf_num` workflows.

Additionally:
- `mainGP.py` is the training script for GPHH.
- `mainESRL.py` is the training script for ESRL.

## Future Updates

The code and documentation will be continuously updated, including multi-objective versions and applications in FJSS environments. For any inquiries, feel free to contact us via [💌](mailto:yifanyang@ecs.vuw.ac.nz).

## Acknowledgements

The implementation references the following works:
- **L2D**: [Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning](https://github.com/zcaicaros/L2D)
- **FJSP**: [Flexible Job Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning](https://github.com/songwenas12/fjsp-drl)
- **L2S**: [Deep Reinforcement Learning Guided Improvement Heuristic for Job Shop Scheduling](https://github.com/zcaicaros/L2S)
