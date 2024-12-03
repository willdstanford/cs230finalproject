# CS230 Final Project: ARC Transformer

This repository contains code for my CS230 Final Project: ARC Transformer.

- `'arc_final.py'` contains all project code

- `'re_arc/tasks/0a938d79.json'` contains sample data: 1000 instances of ARC task 0a938d79 generated using RE-ARC

The last line of `'arc_final.py'` is `train_task_model('0a938d79',1000,10)`. This code trains a demo model: an Independent Task-Specific Model for task 0a938d79. To run the example, just run `pytyhon arc_final.py`. 

Note: Line 46 includes the constant `TORCH_DEVICE` which is set to `'cuda'`. You may need to set it to `'cpu'` to run locally. 
