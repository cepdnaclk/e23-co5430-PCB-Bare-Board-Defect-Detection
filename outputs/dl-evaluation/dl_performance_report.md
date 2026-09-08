# Deep Learning (YOLOv11) Performance Evaluation

Tested on 366 high-resolution images from the PKU-Market-PCB dataset using Sliding-Window Inference.

## Global Metrics
| Metric | Score |
|---|---|
| **Macro Precision** | 0.8717 |
| **Macro Recall** | 0.9962 |
| **Macro F1-Score** | 0.9296 |
| **Overall Accuracy** | 0.8674 |
| **Average Speed** | 0.73 FPS |

## Per-Class Metrics
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Missing_hole | 0.8677 | 1.0000 | 0.9292 |
| Mouse_bite | 0.8770 | 1.0000 | 0.9345 |
| Open_circuit | 0.8952 | 1.0000 | 0.9447 |
| Short | 0.8992 | 0.9907 | 0.9427 |
| Spur | 0.8750 | 1.0000 | 0.9333 |
| Spurious_copper | 0.8162 | 0.9867 | 0.8934 |


*(Note: Metrics calculated at IoU threshold = 0.50)*