# Deep-Neural-Network-Retrieval
This repo is initial implementation for the paper "Deep Neural Network Retrieval"
## How to run
Take five-tasks (MNIST) as an example (containing three steps)
```
cd MNIST_five_tasks
```
First: preprocess dataset
```
python processdata.py
```
Second: train models
```
bash trainmodels.sh
```
Third: <br>

Vector-based semantic feature extraction
```
python main_vector.py
```
Matrix-based semantic feature extraction
```
bash generatematrix.sh
python main_matrix.py
```
