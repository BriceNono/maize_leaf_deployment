# Model Directory

Place your trained model file here:

    model/
    └── vgg16_maize_best.h5    ← copy from Google Colab SAVE_DIR

## How to export from Colab

After training Cell 8 (ModelTrainer), the best checkpoint is saved to:
    /content/drive/MyDrive/MaizeCNN_Results/vgg16_maize_best.h5

Download it and place it in this folder.

## Critical: verify class index order

Run this in Colab BEFORE copying the model:

    print(aug_pipeline.train_gen.class_indices)
    # Must output: {'Blight': 0, 'Common_Rust': 1, 'Gray_Leaf_Spot': 2, 'Healthy': 3}

This MUST match CLASS_ORDER_KEYS in utils/disease_data.py:
    CLASS_ORDER_KEYS = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

If your folder names differ (e.g. Corn___Common_Rust), update
CLASS_ORDER_KEYS to match the alphabetical order of your folder names.

## Preprocessing reminder

Training used: ImageDataGenerator(rescale=1.0/255)
Inference uses: arr = np.array(resized, dtype=np.float32) / 255.0

Do NOT use keras.applications.vgg16.preprocess_input — it would
cause misclassification because the value ranges would not match.
