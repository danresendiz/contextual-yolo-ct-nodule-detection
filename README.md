# contextual-yolo-ct-nodule-detection
Anatomical-Contextual YOLOv8–YOLOv12 Framework for Pulmonary Nodule Detection in CT &amp; Multi-Organ Segmentation (Published in MDPI Diagnostics, 2026).
## Key Utilities
1. **Automated Nodule Annotation Curation (`filter_nodule_annotations.py`):** Parses raw LabelMe JSON files, isolates confirmed pulmonary nodule annotations, and automatically prunes non-nodule slices and unannotated image pairs.
2. **ROI Cropping & Coordinate Remapping (`crop_nodules_roi.py`):** Extracts bounding regions around annotated pulmonary nodules with configurable pixel padding and mathematically remaps polygon coordinates to the local cropped reference frame.
3. **YOLOv8–YOLOv12 Benchmark Visualization (`plot_yolo_benchmarks.py`):** Automates the generation of 300-DPI comparative training and validation curves across all five YOLO generations (Precision, Recall, mAP@50, mAP@50-95, and Box Loss).

## Requirements
```bash
pip install numpy pandas matplotlib pillow
