"""
utils/disease_data.py
=====================
Disease metadata registry.

CLASS_ORDER_KEYS is the single source of truth that maps softmax output
indices to disease keys. It MUST match train_gen.class_indices from Colab.

From the notebook (Cell 3):
    CLASS_KEYS = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

Keras ImageDataGenerator assigns indices alphabetically from folder names.
With folders Blight / Common_Rust / Gray_Leaf_Spot / Healthy:
    0 → Blight
    1 → Common_Rust
    2 → Gray_Leaf_Spot
    3 → Healthy

Verify after training in Colab:
    print(aug_pipeline.train_gen.class_indices)
    # Expected: {'Blight': 0, 'Common_Rust': 1, 'Gray_Leaf_Spot': 2, 'Healthy': 3}
"""

# ── Must match Colab notebook CLASS_KEYS ─────────────────────────
CLASS_ORDER_KEYS = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# ── Disease registry — mirrors Colab Cell 13 exactly ─────────────
DISEASE_REGISTRY = {
    "Blight": {
        "label":          "Northern Leaf Blight",
        "scientific":     "Exserohilum turcicum",
        "severity":       "High",
        "severity_level": 3,
        "spread":         "Airborne spores — spreads very rapidly",
        "color":          "#E24B4A",
        "icon":           "🔴",
        "description": (
            "Large, cigar-shaped, grayish-green to tan necrotic lesions "
            "running parallel to leaf veins (2.5–15 cm). Yield losses of "
            "30–80% when infection occurs before tasseling."
        ),
        "actions": [
            "Isolate affected rows immediately to limit airborne spread.",
            "Apply mancozeb + metalaxyl systemic fungicide every 7 days.",
            "Remove and destroy heavily infected leaves.",
            "Report outbreak to agricultural extension officer.",
            "Consider early harvest on plots with >50% canopy affected.",
        ],
        "prevention": (
            "Plant Ht-gene resistant hybrids. Two-year crop rotation with "
            "soybean or legumes. Preventive fungicide at V6 stage in "
            "high-risk seasons."
        ),
    },
    "Common_Rust": {
        "label":          "Common Rust",
        "scientific":     "Puccinia sorghi",
        "severity":       "Moderate",
        "severity_level": 2,
        "spread":         "Wind-borne urediniospores — spreads rapidly",
        "color":          "#EF9F27",
        "icon":           "🟡",
        "description": (
            "Small (0.2–2 mm), circular to elongate, powdery, brick-red to "
            "dark-brown pustules on both leaf surfaces. Thrives at 16–23 degrees C "
            "with >95% relative humidity."
        ),
        "actions": [
            "Apply triazole fungicide (propiconazole/tebuconazole) within 48 h.",
            "Remove leaves with heaviest pustule load.",
            "Increase inter-row spacing to improve airflow.",
            "Scout neighbouring plots — rust spreads by wind rapidly.",
            "Apply second treatment if new pustules appear after 10 days.",
        ],
        "prevention": (
            "Select rust-resistant varieties. Apply preventive fungicide "
            "at V6–V8 stage. Maintain adequate soil potassium >100 ppm."
        ),
    },
    "Gray_Leaf_Spot": {
        "label":          "Gray Leaf Spot",
        "scientific":     "Cercospora zeae-maydis",
        "severity":       "High",
        "severity_level": 3,
        "spread":         "Residue-borne — moderate to high in humid conditions",
        "color":          "#888780",
        "icon":           "⚫",
        "description": (
            "Rectangular, tan-to-gray lesions with sharply defined parallel "
            "margins bounded by leaf veins. Favoured by >12 h leaf wetness "
            "and dense plant populations."
        ),
        "actions": [
            "Apply strobilurin + triazole fungicide immediately (delay risks 30-40% yield loss).",
            "Switch to drip irrigation to reduce leaf wetness duration.",
            "Remove lower canopy leaves to improve air circulation.",
            "Deep-plow infected residue after harvest.",
            "Rotate to sorghum, legumes, or cassava next season.",
        ],
        "prevention": (
            "Crop rotation is most effective. Use certified disease-free seed. "
            "Avoid minimum-till in fields with GLS history."
        ),
    },
    "Healthy": {
        "label":          "Healthy",
        "scientific":     "No pathogen detected",
        "severity":       "None",
        "severity_level": 0,
        "spread":         "Not applicable",
        "color":          "#639922",
        "icon":           "🟢",
        "description": (
            "Leaf shows no evidence of fungal infection, lesions, or disease "
            "stress. The plant appears in good physiological condition."
        ),
        "actions": [
            "Continue routine monitoring every 3-4 days.",
            "Maintain soil nitrogen at 50-70 kg/ha.",
            "Ensure proper field drainage.",
            "Scout neighbouring plots for early signs of outbreak.",
            "Record observation in farm field diary.",
        ],
        "prevention": (
            "Continue current practices. Monitor weather — warm, humid "
            "conditions increase disease risk. Consider preventive fungicide "
            "at V6 if seasonal forecast indicates prolonged wet conditions."
        ),
    },
}
