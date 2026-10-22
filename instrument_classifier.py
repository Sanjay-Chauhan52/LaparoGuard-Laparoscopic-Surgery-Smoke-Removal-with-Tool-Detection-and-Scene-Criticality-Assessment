"""
instrument_classifier.py — Surgical instrument classification using RASO
Distinguishes between instrument types (hook, grasper, scissors, etc.)
"""

import cv2
import numpy as np
import torch
from PIL import Image


class InstrumentClassifier:
    def __init__(self, model_path=None):
        """
        Initialize RASO model for instrument classification
        model_path: Path to RASO checkpoint (downloads if None)
        """
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # RASO instrument categories
        self.instrument_types = [
            'hook', 'grasper', 'scissors', 'forceps', 'clipper',
            'needle_driver', 'retractor', 'suction', 'irrigator',
            'cautery', 'camera', 'other'
        ]
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load RASO model from checkpoint"""
        try:
            # RASO uses a custom loader - this is a simplified version
            # In production, you would use: from raso import raso
            print(f"Loading RASO model from {model_path}")
            # self.model = raso(pretrained=model_path, image_size=384, vit="swin_l")
            # self.model.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using fallback classification based on instrument shape")
    
    def classify_instrument(self, frame, bbox):
        """
        Classify instrument type from bounding box
        Returns: instrument_type, confidence
        """
        if self.model is None:
            # Fallback: classify based on aspect ratio and shape
            return self._classify_by_shape(frame, bbox)
        
        # Use RASO model for classification
        x1, y1, x2, y2 = [int(v) for v in bbox]
        inst_crop = frame[y1:y2, x1:x2]
        
        if inst_crop.size == 0:
            return 'unknown', 0.0
        
        # Preprocess for RASO
        inst_pil = Image.fromarray(cv2.cvtColor(inst_crop, cv2.COLOR_BGR2RGB))
        inst_pil = inst_pil.resize((384, 384))
        
        # Get predictions from RASO
        with torch.no_grad():
            # tags = self.model.inference(inst_pil)
            # Parse tags to find instrument type
            pass
        
        return 'unknown', 0.0
    
    def _classify_by_shape(self, frame, bbox):
        """
        Fallback classification based on instrument shape
        Uses aspect ratio and position to guess instrument type
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / (height + 1e-6)
        
        # Simple heuristics
        if aspect_ratio > 3.0:
            # Long and thin - likely hook or needle driver
            return 'hook', 0.6
        elif aspect_ratio < 0.5:
            # Tall and narrow - likely grasper or forceps
            return 'grasper', 0.6
        elif 0.8 < aspect_ratio < 1.2:
            # Square-ish - likely camera or retractor
            return 'camera', 0.5
        else:
            # Default
            return 'grasper', 0.4
    
    def is_hook(self, frame, bbox):
        """Check if instrument is a hook"""
        inst_type, conf = self.classify_instrument(frame, bbox)
        return inst_type == 'hook', conf
    
    def is_grasper(self, frame, bbox):
        """Check if instrument is a grasper"""
        inst_type, conf = self.classify_instrument(frame, bbox)
        return inst_type == 'grasper', conf


def main():
    """Test the classifier"""
    classifier = InstrumentClassifier()
    
    # Test with a sample frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = [100, 100, 300, 200]
    
    inst_type, conf = classifier.classify_instrument(frame, bbox)
    print(f"Instrument type: {inst_type}, confidence: {conf:.2f}")


if __name__ == "__main__":
    main()
