import os
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO


class YOLOInferenceService:
    """
    Service class for YOLO inference with bounding boxes and attention visualization
    """

    def __init__(self, model_path: str):
        """
        Initialize YOLO model

        Args:
            model_path: Path to YOLO model weights (.pt file)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"🔄 Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        self.model.model.eval()
        print("✅ Model loaded successfully!")

    def predict(self, image_path: str, conf_threshold: float = 0.25) -> Dict:
        """
        Run inference and return bounding boxes

        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections

        Returns:
            Dictionary containing detection results
        """
        print(f"\n🔍 Running inference on: {os.path.basename(image_path)}")

        # Run inference
        results = self.model(image_path, conf=conf_threshold, verbose=False)

        # Parse results
        detections = {
            "boxes": [],
            "labels": [],
            "confidences": [],
            "class_ids": [],
            "num_detections": 0,
        }

        if len(results[0].boxes) == 0:
            print("⚠️  No objects detected!")
            return detections

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = results[0].names[cls_id]

            detections["boxes"].append([x1, y1, x2, y2])
            detections["labels"].append(label)
            detections["confidences"].append(conf)
            detections["class_ids"].append(cls_id)

        detections["num_detections"] = len(detections["boxes"])

        print(f"✅ Found {detections['num_detections']} objects")
        for i, label in enumerate(detections["labels"]):
            conf = detections["confidences"][i]
            print(f"   • {label}: {conf:.2%}")

        return detections

    def generate_attention(
        self,
        image_path: str,
        save_dir: str = "gradcam_results",
        target_layers: Optional[List[str]] = None,
    ) -> str:
        """
        Generate YOLO attention visualization

        Args:
            image_path: Path to input image
            save_dir: Directory to save visualization
            target_layers: Specific layers to visualize (optional)

        Returns:
            Path to saved visualization image
        """
        print(f"\n{'=' * 70}")
        print("🔥 GENERATING ATTENTION VISUALIZATION")
        print(f"{'=' * 70}\n")

        os.makedirs(save_dir, exist_ok=True)

        # Read image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Get predictions
        results = self.model(image_path, verbose=False)

        if len(results[0].boxes) == 0:
            print("⚠️  No objects detected!")
            return None

        # Get feature maps from backbone
        feature_maps = {}

        def hook_fn(name):
            def hook(module, input, output):
                feature_maps[name] = output.detach()

            return hook

        # Register hooks on key layers
        hooks = []
        layer_names = []

        # Default target layers if not specified
        if target_layers is None:
            target_layers = ["model.9", "model.12", "model.15", "model.18", "model.21"]

        for name, module in self.model.model.named_modules():
            if any(x in name for x in target_layers):
                if isinstance(module, (torch.nn.Conv2d, torch.nn.modules.conv.Conv2d)):
                    hooks.append(module.register_forward_hook(hook_fn(name)))
                    layer_names.append(name)
                    if len(layer_names) >= 3:  # Limit to 3 layers
                        break

        # Forward pass to get feature maps
        _ = self.model(image_path, verbose=False)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        if not feature_maps:
            print("⚠️  No feature maps captured!")
            return None

        # Create visualization
        num_layers = len(feature_maps)
        fig, axes = plt.subplots(2, num_layers + 1, figsize=(6 * (num_layers + 1), 12))

        if num_layers == 1:
            axes = axes.reshape(2, -1)

        fig.suptitle(
            f"YOLO Attention Maps - {os.path.basename(image_path)}",
            fontsize=16,
            fontweight="bold",
        )

        # Original image with detections
        axes[0, 0].imshow(img_rgb)
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{results[0].names[cls_id]} {conf:.2f}"

            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, color="lime", linewidth=3
            )
            axes[0, 0].add_patch(rect)
            axes[0, 0].text(
                x1,
                y1 - 10,
                label,
                bbox=dict(boxstyle="round", facecolor="lime", alpha=0.8),
                fontsize=10,
                fontweight="bold",
            )
        axes[0, 0].set_title("Original + Detections", fontsize=12, fontweight="bold")
        axes[0, 0].axis("off")

        axes[1, 0].imshow(img_rgb)
        axes[1, 0].set_title("Reference Image", fontsize=12, fontweight="bold")
        axes[1, 0].axis("off")

        # Process each feature map
        for idx, (layer_name, fmap) in enumerate(feature_maps.items(), 1):
            # Average across channels
            fmap_avg = fmap[0].mean(dim=0).cpu().numpy()

            # Normalize
            fmap_normalized = (fmap_avg - fmap_avg.min()) / (
                fmap_avg.max() - fmap_avg.min() + 1e-8
            )

            # Resize to original image size
            fmap_resized = cv2.resize(fmap_normalized, (w, h))

            # Show heatmap
            axes[0, idx].imshow(fmap_resized, cmap="jet")
            axes[0, idx].set_title(
                f"Layer: {layer_name.split('.')[-1]}\nAttention Map",
                fontsize=10,
                fontweight="bold",
            )
            axes[0, idx].axis("off")

            # Overlay on image
            img_normalized = img_rgb.astype(np.float32) / 255.0
            heatmap = cv2.applyColorMap(np.uint8(255 * fmap_resized), cv2.COLORMAP_JET)
            heatmap = (
                cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            )
            overlay = 0.6 * img_normalized + 0.4 * heatmap
            overlay = np.clip(overlay, 0, 1)

            axes[1, idx].imshow(overlay)

            # Add boxes to overlay
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                rect = plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False, color="lime", linewidth=2
                )
                axes[1, idx].add_patch(rect)

            axes[1, idx].set_title(
                f"Overlay + Detections", fontsize=10, fontweight="bold"
            )
            axes[1, idx].axis("off")

        plt.tight_layout()

        # Save
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(save_dir, f"attention_{base_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved: {save_path}")

        return save_path
