# HNN-Inference


## Within Each Directory
`models/`: Contains PyTorch DNN models for each node of the MNN-Tree.

`trained_weights/`: Contains Trained PyTorch weights and parameters for each node.

`run.py`: Performs inference.

`hierarhy.py`: Contains hierarchy structure for the MNN-Tree.


## Running MNN-Tree

`python run.py --data <path where dataset is stored/should be downloaded> --pretrained_path <path where trained models are saved>`
