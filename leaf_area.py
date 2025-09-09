import cv2
import numpy as np

class LeafAnalyzer:
    def __init__(self, threshold: int = 30):
        """
        Initialize analyzer with a grayscale threshold.
        Pixels above this threshold are considered leaf.
        """
        self.threshold = threshold

    def segment_leaf(self, image_path: str):
        """
        Segment the leaf from a black background using threshold.
        Returns the mask (boolean array) and original image.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or path is incorrect.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)

        # Morphology to remove small noise (optional)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask, img

    def calculate_area(self, mask: np.ndarray, fov_x_deg: float, fov_y_deg: float, camera_height_cm: float):
        """
        Calculate leaf area in pixels and convert to real-world area using camera FOV.
        """
        leaf_pixels = np.count_nonzero(mask)
        total_pixels = mask.size
        percent_coverage = 100 * leaf_pixels / total_pixels

        # Convert to real-world area
        fov_x_rad = np.deg2rad(fov_x_deg)
        fov_y_rad = np.deg2rad(fov_y_deg)

        real_width = 2 * camera_height_cm * np.tan(fov_x_rad / 2)
        real_height = 2 * camera_height_cm * np.tan(fov_y_rad / 2)
        real_area = real_width * real_height  # cm² of full image

        cm2_per_pixel = real_area / total_pixels
        leaf_area_cm2 = leaf_pixels * cm2_per_pixel

        return {
            "pixel_area": leaf_pixels,
            "percent_coverage": percent_coverage,
            "real_leaf_area_cm2": leaf_area_cm2
        }

    def visualize(self, img: np.ndarray, mask: np.ndarray, alpha: float = 0.3):
        """
        Visualize leaf mask overlay on original image.
        """
        overlay = img.copy()
        overlay[mask == 255] = [0, 255, 0]  # green overlay for leaf
        combined = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

        cv2.imshow("Original Image", img)
        cv2.imshow("Leaf Mask", mask)
        cv2.imshow("Leaf Overlay", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# --- Example usage ---
if __name__ == "__main__":
    analyzer = LeafAnalyzer(threshold=30)
    mask, img = analyzer.segment_leaf("/Users/navjotsingh/Documents/leaf_area_camera_fov/images/leaf.jpg")
    area_result = analyzer.calculate_area(mask, fov_x_deg=60, fov_y_deg=40, camera_height_cm=30)

    print("Leaf pixel area:", area_result["pixel_area"])
    print("Leaf covers %.2f%% of image" % area_result["percent_coverage"])
    print("Estimated real leaf area: %.2f cm²" % area_result["real_leaf_area_cm2"])

    analyzer.visualize(img, mask)
