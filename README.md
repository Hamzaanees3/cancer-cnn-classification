# Cancer Image Classification (Educational CNN Demo)

## Files needed
- `app.py`
- `requirements.txt`
- `cancer_cnn_model.keras` (copy your trained model from Colab)
- `class_names.json` (create this using the exact class order from training)

## Important: verify class order
The example file `class_names.example.json` shows `benign` then `malignant`, but this order is only an example.
Use the exact label order from your training pipeline. For a Keras directory iterator, check `train_generator.class_indices`.
For a dataset loader, check the class names / class order it reported.

Example `class_names.json`:
```json
[
  "benign",
  "malignant"
]
```

## Important: verify preprocessing
The app currently resizes to the model's input dimensions and divides pixel values by 255.
This must match the preprocessing used during training. If your model already contains a Rescaling layer,
remove the `/ 255.0` operation in `app.py` to avoid scaling twice.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload the files above to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Select `app.py` as the main file.
4. Deploy.

This is an educational image-classification demonstration, not a medical diagnostic tool.
Do not use predictions for clinical decisions.
