## Prototypical Network Experiment

The code for the experiment of the prototypical network few-shot learning on the Plant Doc dataset is in ProtoNet.ipynb. 
The manually created few shot version of the Plant Doc is in the directory PlantDoc_FewShot. We have a train and test subdirectory in it. 
One can change the path of the read_images to be of the Plant Village to test on it. 
The model.pth is a saved model on the Plant Doc dataset that can be used for testing.


## Zero-shot with CLIP Experiment
The file clip_fwd.py contains the code for performing the zero-shot clip experiment with the Plant Village dataset to classify healthy and disease infected plant. The ./plant_images folder consists the images of healthy and infected plants/leaves on which this zero-shot experiment is done.

## CLIP Experiments

The code for this has been adopted from the official github repo of [Learning to Prompt for Vision-Language Models](https://github.com/KaiyangZhou/CoOp.git). The instructions for running the code are in [README.md](https://github.com/KaiyangZhou/CoOp/blob/main/README.md).

### Dataset
Refer to DATASETS.md on setting the appropriate path for the plant_village dataset. The Plant Village dataset directory should be named "plant_village" with a "images" subdirectory in it which contains all the subdirectories of classes of plant diseases.

### Linear Probe on CLIP
Refer ./CoOp/lpclip/README.md

### Few-shot learning using CoOp on Plant Village
Refer COOP.md in ./CoOp directory

### Generalization to Unseen classes using CoOp and CoOp
Refer COCOOP.md in ./CoOp directory


