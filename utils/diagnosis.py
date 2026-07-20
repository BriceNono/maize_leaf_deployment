"""
utils/diagnosis.py
==================
DiagnosisResult — transforms raw softmax probs into a full diagnosis.
Mirrors Colab Cell 13 DiagnosisResult class exactly, with confidence
score prominently included in build_response().
"""

import numpy as np


class DiagnosisResult:
    """
    Encapsulates a single CNN disease prediction and formats
    it for display in result.html.

    Mirrors Colab Cell 13 DiagnosisResult exactly:
        self._pred_idx   = int(np.argmax(probs))
        self._pred_key   = class_order[self._pred_idx]
        self._confidence = float(probs[self._pred_idx])
        self._info       = registry[self._pred_key]

    Attributes:
        probs        (np.ndarray) : softmax array (4,)
        class_order  (list)       : ['Blight','Common_Rust','Gray_Leaf_Spot','Healthy']
        registry     (dict)       : DISEASE_REGISTRY
        leaf_image                : LeafImage (optional)
    """

    # Confidence thresholds for the UI badge
    HIGH_CONF   = 0.80   # green badge
    MEDIUM_CONF = 0.55   # amber badge
    # below MEDIUM_CONF → red "low confidence" warning

    def __init__(self, probs, class_order, registry, leaf_image=None):
        self.probs       = np.array(probs, dtype=float)
        self.class_order = class_order
        self.registry    = registry
        self.leaf_image  = leaf_image

        # Core prediction — mirrors Colab Cell 13 __init__ exactly
        self._pred_idx   = int(np.argmax(self.probs))
        self._pred_key   = class_order[self._pred_idx]
        self._confidence = float(self.probs[self._pred_idx])
        self._info       = registry[self._pred_key]

    # ── Public methods ────────────────────────────────────────────

    def format_confidence(self) -> str:
        """Return confidence as 'XX.X%' string. Mirrors Colab Cell 13."""
        return f"{self._confidence * 100:.1f}%"

    def confidence_level(self) -> str:
        """
        Return a human-readable confidence level label.
        Used by result.html to colour the confidence badge.
        """
        if self._confidence >= self.HIGH_CONF:
            return "High"
        if self._confidence >= self.MEDIUM_CONF:
            return "Medium"
        return "Low"

    def confidence_badge_color(self) -> str:
        """Return hex color for the confidence level badge."""
        if self._confidence >= self.HIGH_CONF:
            return "#4caf1f"
        if self._confidence >= self.MEDIUM_CONF:
            return "#EF9F27"
        return "#E24B4A"

    def get_all_probs_sorted(self) -> list:
        """
        Return list of (label, probability) tuples sorted descending.
        Mirrors Colab Cell 13 get_all_probs_sorted().
        """
        return sorted(
            [
                (self.registry[k]['label'], float(p))
                for k, p in zip(self.class_order, self.probs)
            ],
            key=lambda x: x[1],
            reverse=True,
        )

    def build_response(self) -> dict:
        """
        Build the complete prediction response dictionary.
        Mirrors Colab Cell 13 build_response() and adds confidence
        score fields for the result.html dial and badge.

        Keys guaranteed present (used by result.html):
            prediction, class_key, confidence, confidence_pct,
            confidence_level, confidence_color,
            confidence_bar_pct (0-100 int for CSS width),
            scientific, severity, severity_level, spread,
            color, icon, description, actions, prevention,
            all_probs (list of dicts, sorted desc),
            is_healthy, is_high_severity, low_confidence_warning
        """
        # Build all_probs list (sorted descending, winner first)
        all_probs = sorted(
            [
                {
                    "label": self.registry[k]["label"],
                    "key":   k,
                    "prob":  round(float(p), 4),
                    "pct":   f"{float(p) * 100:.1f}%",
                    "color": self.registry[k]["color"],
                    "icon":  self.registry[k].get("icon", ""),
                    "bar_width": int(round(float(p) * 100)),
                }
                for k, p in zip(self.class_order, self.probs)
            ],
            key=lambda x: x["prob"],
            reverse=True,
        )

        return {
            # ── Core prediction ───────────────────────────────────
            "prediction":       self._info["label"],
            "class_key":        self._pred_key,

            # ── Confidence score (main addition) ─────────────────
            "confidence":       round(self._confidence, 4),
            "confidence_pct":   self.format_confidence(),
            "confidence_level": self.confidence_level(),        # High/Medium/Low
            "confidence_color": self.confidence_badge_color(),  # hex
            "confidence_bar_pct": int(round(self._confidence * 100)),  # 0-100

            # ── Disease metadata ──────────────────────────────────
            "scientific":       self._info["scientific"],
            "severity":         self._info["severity"],
            "severity_level":   self._info.get("severity_level", 0),
            "spread":           self._info["spread"],
            "color":            self._info["color"],
            "icon":             self._info.get("icon", ""),
            "description":      self._info["description"],
            "actions":          self._info["actions"],
            "prevention":       self._info["prevention"],

            # ── All 4 class probabilities (for bar chart) ─────────
            "all_probs":        all_probs,

            # ── Template logic helpers ────────────────────────────
            "is_healthy":           self._pred_key == "Healthy",
            "is_high_severity":     self._info.get("severity_level", 0) >= 3,
            "low_confidence_warning": self._confidence < self.MEDIUM_CONF,
        }
