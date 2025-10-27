# Weakly Supervised Affordance Grounding with Dual Semantic Guidance
Code implementation for our paper.

## Requirements

- Python 3.10
- Pytorch 2.0.1
- opencv
- A TESLA V100s GPU


## Datasets

- AGD20K: You can find it [here](https://github.com/lhc1224/Cross-View-AG/tree/main/code/cvpr).

Download the dataset and place it in the dataset/AGD20K

And you need to process the data to get the label data used for test:

```python preprocessing.py --data_root dataset/AGD20K --divide Unseen```

```python preprocessing.py --data_root dataset/AGD20K --divide Seen```

## Train

Download the pretrained CLIP model [here](https://drive.google.com/file/d/1hom1CUtOmu9ePjJcTUNRlP5qLjOSf6IN/view?usp=drive_link). 

Clone DINOv2 and it will be downloaded automatically in the training code.

```git clone https://github.com/facebookresearch/dinov2.git```


```python train.py --data_root dataset/AGD20K --divide Unseen --save_root save_models_unseen --model_name WAGDS```

```python train.py --data_root dataset/AGD20K --divide Seen --save_root save_models_seen --model_name WAGDS```

## Test

Replace the model_path with the path of the trained model. There are **num_exo** models for each divide any one of these will work with similar results. 

```python test.py --data_root dataset/AGD20K --divide Unseen --save_path pred_results --model_path model_path```

```python test.py --data_root dataset/AGD20K --divide Seen --save_path pred_results --model_path model_path```

## Acknowledgements

We would like to express our gratitude to the following repositories for their contributions and inspirations, and we borrowed some code from them: [WSMA](https://github.com/xulingjing88/WSMA), [Cross-View-AG](https://github.com/lhc1224/Cross-View-AG), [LOCATE](https://github.com/Reagan1311/LOCATE), [OOAL](https://github.com/Reagan1311/OOAL), [CLIP](https://github.com/openai/CLIP), [DINOv2](https://github.com/facebookresearch/dinov2).