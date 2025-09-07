# BiSMRec:Bidirectional Sequence Modeling for Multi-modal Recommendation

Dynamic modeling of users’ historical behavioral preferences remains a central challenge in recommender systems. Traditional multi-modal recommendation approaches rely on unidirectional neural networks to encode user interaction sequences, which frequently suffer from inadequate representation of implicit features and issues related to feature degradation.The unidirectional information propagation mechanism restricts the expressive capacity of implicit features, making it challenging to extract complex dependency relationships within user behavior sequences. Furthermore, as the sequence length increases, the influence of early behavioral feature information diminishes during the propagation process, resulting in the tail items of the sequence overshadowing the final representation.To this end, this study proposes a bidirectional processing multi-modal recommendation model, BiSMRec, which integrates bidirectional sequence information via a hierarchical bidirectional encoding architecture and combines Graph Convolutional Networks (GCN) to extract long-term and short-term interest patterns.This study introduces a multi-modal adaptive gating mechanism designed to dynamically fuse cross-modal features, complemented by self-supervised learning to enhance feature robustness. Furthermore, residual-connected filtering layers are employed to mitigate the issue of gradient vanishing. Extensive experiments conducted on e-commerce datasets demonstrate that BiSMRec significantly outperforms existing baseline models in terms of recommendation accuracy, thereby validating its effectiveness in capturing complex user intent and modeling multi-modal feature interactions.

# Enviroment Requirement
```text
python 3.8
Pytorch 1.12
numpy>=1.24
pandas>=2.1
scipy>=1.10
PyYAML>=6.0
matplotlib>=3.7
Pillow>=10.0
lmdb>=1.4    

# Dataset
Experiments were conducted on the three datasets of baby, sports and clothing in the Amazon Review Dataset. This dataset contains product descriptions and image information, which are processed respectively as text and visual features. In the experiment, five core users and five core projects were retained, ensuring that each user or project was associated with at least five intersections. This is a widely adopted standard practice in existing research.

| Dataset  | #Users | #Items | #Interaction | Sparsity | Density |
|---------|-------:|-------:|-------------:|---------:|--------:|
| baby     | 19,445 | 7,050  | 160,792 | 99.88% | 0.117% |
| sports   | 35,598 | 18,357 | 296,337 | 99.95% | 0.045% |
| clothing | 39,387 | 23,033 | 278,677 | 99.97% | 0.031% |

We provide three processed datasets: Baby, Sports and Clothing.
Download from Google Drive: [Baby/Sports/Clothing](https://drive.google.com/file/d/1tpP-IQtUubSlVvYpkA61bffPKkhvT62T/view)

# Training
cd ./src
python main.py

# Directory Structure
```text
├─ common/          # trainer, initialization, loss functions
├─ utils/           # configuration, datasets & DataLoader, evaluation, logging
├─ src/
│  ├─ main.py       # entry point
│  └─ model/
│     └─ BiSMRec.py # model implementation
├─ data/            # place datasets (see data format)
└─ configs/         # runtime configs (overall / model / dataset)
 
