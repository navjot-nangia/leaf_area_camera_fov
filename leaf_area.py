import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

class LeafSegmenter:
    def __init__(self, sam_checkpoint: str, model_type: str = "vit_h"):
        """
        Initialize SAM model and predictor.
        """
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.eval()
        self.predictor = SamPredictor(self.sam)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    def segment_leaf(self, image_path: str):
        """
        Segment the largest leaf in the image.
        Returns the leaf mask (boolean array) and original image (RGB).
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError("Image not found or path is incorrect.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Generate all masks
        masks = self.mask_generator.generate(img_rgb)
        if not masks:
            raise ValueError("No masks detected in image.")
        
        # Choose the largest mask (assumed leaf)
        largest_mask = max(masks, key=lambda x: x['area'])['segmentation']
        return largest_mask, img_rgb

    def calculate_area(self, mask: np.ndarray, fov_x_deg: float, fov_y_deg: float, camera_height_cm: float):
        """
        Convert leaf pixel area to real-world area using camera FOV and height.
        """
        leaf_pixels = np.count_nonzero(mask)
        H, W = mask.shape
        total_pixels = H * W

        percent_coverage = 100 * leaf_pixels / total_pixels

        fov_x_rad = np.deg2rad(fov_x_deg)
        fov_y_rad = np.deg2rad(fov_y_deg)

        real_width = 2 * camera_height_cm * np.tan(fov_x_rad / 2)
        real_height = 2 * camera_height_cm * np.tan(fov_y_rad / 2)
        real_area = real_width * real_height

        cm2_per_pixel = real_area / total_pixels
        leaf_area_cm2 = leaf_pixels * cm2_per_pixel

        return {
            "pixel_area": leaf_pixels,
            "percent_coverage": percent_coverage,
            "real_leaf_area_cm2": leaf_area_cm2
        }

    def visualize(self, img_rgb: np.ndarray, mask: np.ndarray, overlay_alpha: float = 0.3):
        """
        Display the original image with leaf mask overlay.
        """
        overlay = img_rgb.copy()
        overlay[mask] = [0, 255, 0]  # green overlay
        combined = cv2.addWeighted(img_rgb, 1 - overlay_alpha, overlay, overlay_alpha, 0)

        plt.figure(figsize=(8,8))
        plt.imshow(combined)
        plt.axis('off')
        plt.show()

# --- Example usage ---
if __name__ == "__main__":
    # Path to SAM checkpoint
    checkpoint_path = "sam_vit_h_4b8939.pth"

    # Initialize leaf segmenter
    segmenter = LeafSegmenter(checkpoint_path)

    # Segment leaf
    leaf_mask, image_rgb = segmenter.segment_leaf("leaf_2.jpg")

    # Calculate area
    area_result = segmenter.calculate_area(
        leaf_mask,
        fov_x_deg=60,
        fov_y_deg=40,
        camera_height_cm=30
    )

    # Print results
    print("Leaf pixel area:", area_result["pixel_area"])
    print("Leaf covers %.2f%% of image" % area_result["percent_coverage"])
    print("Estimated real leaf area: %.2f cm²" % area_result["real_leaf_area_cm2"])

    # Visualize
    segmenter.visualize(image_rgb, leaf_mask)
