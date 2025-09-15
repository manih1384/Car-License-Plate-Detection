# Car License Plate Detection

This repository contains the **Final Project** of the **Data Science** course at the **University of Tehran**.  
We developed a complete workflow for **car license plate detection** — covering dataset preparation, exploratory visualization, feature engineering, and the training & evaluation of **deep learning models**.

The project was carried out in **three structured phases**, implemented in a Jupyter notebook (`car_plate_detection.ipynb`), supported by modular Python scripts, and summarized in a formal [Presentation](./Presentation.pdf).  


## Project Overview

### Phase 1 — Data Collection & Preparation
    
-   Integrated **annotations** and **image metadata** into a unified schema for easy querying.    
-   Built an interactive **Power BI** dashboard for early insights.
    <p align="center">
    <img width="2026" height="1077" alt="Image" src="https://github.com/user-attachments/assets/ac6d0ee2-db15-4f46-9ec3-64d89924d2d4" />
</p>

  _For More Tools and Interactive Filtering Visit the File [Initial Dataset Visualization.pbix](https://github.com/manih1384/FINAL_DS_PROJECT/blob/main/Initial%20Dataset%20Visualization.pbix)_


### Phase 2 — Feature Engineering & Preprocessing

-   Designed a **SQLite** schema to manage images, annotations, engineered features, and predictions.
-   Implemented feature engineering:
	-  **bbox geometry**, **aspect ratio**, **area fraction**, **normalized centers**, **margins**, and **blur detection** (variance of Laplacian).
-   Wrote modular scripts and a **pipeline** to import annotations, compute features, persist them to DB, and prepare data for modeling.
    
   > #### Database Design
```sql
-- Raw annotations (one row per image; bbox describes the plate)
CREATE  TABLE  IF  NOT  EXISTS image_annotations (
id INTEGER  PRIMARY  KEY  AUTOINCREMENT,
filename TEXT NOT  NULL,
folder TEXT NOT  NULL,
width INTEGER  NOT  NULL,
height INTEGER  NOT  NULL,
xmin INTEGER  NOT  NULL,
ymin INTEGER  NOT  NULL,
xmax INTEGER  NOT  NULL,
ymax INTEGER  NOT  NULL,
image_path TEXT NOT  NULL,
UNIQUE(filename)
);

-- Engineered features (one-to-one with image_annotations by filename)
CREATE  TABLE  IF  NOT  EXISTS engineered_plate_features (
id INTEGER  PRIMARY  KEY  AUTOINCREMENT,
filename TEXT NOT  NULL,
bbox_width REAL  NOT  NULL,
bbox_height REAL  NOT  NULL,
bbox_area REAL  NOT  NULL,
aspect_ratio REAL  NOT  NULL,
area_fraction REAL  NOT  NULL,
center_x_norm REAL  NOT  NULL,
center_y_norm REAL  NOT  NULL,
margin_left REAL  NOT  NULL,
margin_top REAL  NOT  NULL,
margin_right REAL  NOT  NULL,
margin_bottom REAL  NOT  NULL,
blur_score REAL,
is_blurry INTEGER,
FOREIGN  KEY (filename) REFERENCES image_annotations(filename)
);

-- Detector predictions (written back for evaluation & auditing)
CREATE TABLE IF NOT EXISTS predictions (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  filename   TEXT NOT NULL,
  x_min      REAL,
  y_min      REAL,
  x_max      REAL,
  y_max      REAL,
  score      REAL,
  label      TEXT,
  model_ver  TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### Phase 3 — Modeling & Evaluation

-   Implemented and compared two approaches:
	-   **Traditional CNN (Baseline)**
A custom Convolutional Neural Network was built as a baseline to classify cropped plate regions for OCR readiness (usable vs. noisy/blurry).  
		- **Architecture:** 3 convolutional layers + pooling, followed by fully connected layers.  
		- **Training:** Implemented in PyTorch with cross-entropy loss and Adam optimizer.  
		- **Purpose:** Served as a lightweight approach for preprocessing plates before OCR.

	-   **Faster R-CNN (Detection)**
We used Faster R-CNN (via `torchvision.models.detection`) to localize license plates in raw car images.  
		- **Backbone:** ResNet-based CNN pre-trained on COCO, fine-tuned on our annotated dataset.  
		- **Outputs:** Bounding boxes with confidence scores for plate regions.  
		- **Integration:** Detected plate crops were passed downstream for CNN classification and OCR. 
		
-   Evaluated on detection rate, IoU, accuracy, and OCR-readiness.
-   Benchmarked against OCR testing tools with EasyOCR and TrOCR.
        

## Results
* **Faster R-CNN** achieved high detection performance:
	* Detection Rate: \~94% @ IoU ≥ 0.5
	* Mean IoU: \~0.81
* **CNN Classifier** for OCR-readiness:
	* Accuracy: \~88%
	* F1-score: \~0.93
* **OCR Tools**:
	 - **EasyOCR**: Exact Match 0.13, Char. Accuracy 0.41
     - **TrOCR**: Exact Match 0.26, Char. Accuracy 0.67
  	 - TrOCR consistently outperformed EasyOCR, but both struggled with **blurry or oblique plates**.  


 >   Training loss/accuracy curves

   <p align="center">
              <img width="1774" height="728" alt="Image" src="https://github.com/user-attachments/assets/46c23e98-de17-46c7-996c-32b36d730ec6" />
</p>

 >   Detected plates overlay

<p align="center">
     <img width="404" height="400" alt="Image" src="https://github.com/user-attachments/assets/a0658f21-5428-45be-a2af-91fe10ec9894" />

</p>

 >   OCR examples
     
   <p align="center">
<img width="746" height="239" alt="Image" src="https://github.com/user-attachments/assets/fd47e81a-0c8f-42f0-bef0-e1d1d8e8817f" />
	   
</p>


## How to Reproduce

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    _(Ensure PyTorch & Torchvision match your environment)_
2.  Initialize / update the database:
    
    ```bash
    python scripts/import_to_db.py
    python scripts/feature_engineering.py
    ```
3.  Run the full pipeline:
    ```bash
    python pipeline.py
    ```
4.  Explore:
    -   `car_plate_detection.ipynb` → end-to-end workflow.
    -   `train_model.py` → training script.
    -   `make_prediction.py` → inference.


## Repository Structure
```
.
├─ contents/
│  ├─ Project Instructions
│  ├─ annotations/
│  ├─ images/
│  └─ outputs/
├─ database/
│  └─ dataset.db
├─ scripts/
│  ├─ database_connection.py
│  ├─ load_data.py
│  ├─ preprocess.py
│  ├─ feature_engineering.py
│  ├─ train_model.py
│  ├─ make_prediction.py
│  └─ import_to_db.py
├─ car_plate_detection.ipynb
├─ pipeline.py
├─ Presentation.pdf
├─ Initial Dataset Visualization.pbix
├─ requirements.txt
└─ README.md
```

----------

## Contributors


<div align="center">
<a href="https://github.com/ParsaBukani">Parsa Bukani</a> &nbsp;&nbsp;|&nbsp;&nbsp;
<a href="https://github.com/manih1384">Mani Hosseini</a> &nbsp;&nbsp;|&nbsp;&nbsp;
<a href="https://github.com/erfan-f">Erfan Falahati</a>

</div>

    

## Join the Project  

Contributions are welcome!   
Whether it’s refining preprocessing, testing new OCR/detection models, or expanding the dataset — feel free to open an issue or submit a PR.  

## License

This project is licensed under the **MIT License**.


