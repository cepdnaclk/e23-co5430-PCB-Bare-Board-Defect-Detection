# YOLO Output Files Explained
**Directory:** `runs/deep_learning/microinspect_v2_medium/`

When the YOLO training script (`train.py`) executes, it automatically generates a massive suite of analytics to help audit the model's performance. Below is a complete breakdown of every file generated in this directory, grouped by their purpose.

---

## 1. The Core Model Files
These files are the actual deliverables of your training run.

* **`weights/` (Directory):** The most important folder. 
  * `best.pt`: The weights from the epoch with the highest accuracy (mAP). **You will use this file for inference in `predict.py`.**
  * `last.pt`: The weights from the final epoch of training. Used if you need to resume a crashed training run.
* **`args.yaml`**: A configuration file that recorded the exact hyperparameters used for this training run (learning rate, mosaic, focal loss, etc.) so you can perfectly reproduce it later.

---

## 2. High-Level Metrics (Start Here)
These files provide a macroscopic view of how successful your training was.

* **`results.png`**: The master graph. It visually plots every metric (Precision, Recall, mAP50, and all the Loss types) across all 100 epochs. You want to see the loss curves going down smoothly and the metric curves going up and flattening out.
* **`results.csv`**: The exact same data as `results.png`, but in raw numbers if you want to pull it into Excel or Python for custom graphing.
* **`confusion_matrix.png` & `confusion_matrix_normalized.png`**: A heat-map grid showing how often the model gets confused. If the model incorrectly thought 50 "Spurs" were actually "Shorts", this chart will show a bright spot exactly at the intersection of those two classes. (Normalized just shows percentages instead of raw numbers).

---

## 3. The Analytics Curves
These graphs help you decide what confidence threshold you should set for your final application.

* **`BoxPR_curve.png`** (Precision-Recall Curve): Shows the trade-off between Precision (being correct) and Recall (finding everything). A perfect model hugs the top-right corner. It helps you see if pushing for higher Precision causes a massive drop in Recall.
* **`BoxF1_curve.png`**: The F1 score is a balanced average of Precision and Recall. This curve peaks at the specific confidence threshold where your model performs the best overall. 
* **`BoxP_curve.png`** & **`BoxR_curve.png`**: Individual curves showing how Precision and Recall react as you raise or lower the confidence threshold.

---

## 4. Visualizing the Dataset & Augmentations
These files help you debug if your data loading and augmentations are working properly.

* **`labels.jpg`**: Analytics about your raw data! It shows graphs of where bounding boxes typically appear on the images and how large they generally are.
* **`train_batch0.jpg`, `train_batch1.jpg`, `train_batch2.jpg`**: These show the very first images fed into the neural network during epoch 1. If you look closely, you will see the effects of the `mosaic` and `mixup` augmentations we configured (multiple images stitched together).
* **`train_batch124110.jpg`** (etc): These show the very last images fed into the network at the end of training.

---

## 5. Visualizing Model Performance (Auditing)
These files allow you to visually audit the model's accuracy without running any code.

* **`val_batch0_labels.jpg` vs. `val_batch0_pred.jpg`**: Open these side-by-side! 
  * The `_labels.jpg` file is the **Ground Truth** (what the defect actually is). 
  * The `_pred.jpg` file is what your model **actually guessed**. 
  * This is the absolute best way to visually "audit" the model with your own eyes to see if its bounding boxes are tight and accurate on unseen data.
