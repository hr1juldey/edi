import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime



def visualize_yolo_detections(
    image_path, 
    model_path="yolo11n-seg.pt",
    conf_threshold=0.35,
    iou_threshold=0.45,
    show_window=True
):
    """
    Maximize YOLO detections with bounding boxes for VLM processing.
    
    Args:
        image_path: Path to input image
        model_path: Path to YOLO model weights
        conf_threshold: Confidence threshold (lower = more detections)
        iou_threshold: IoU threshold for NMS (lower = more boxes)
        show_window: Show CV2 window or just save (default: True)
    """
    # Load model
    model = YOLO(model_path)
    
    # Run inference with optimized parameters for maximum detections
    results = model(
        image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        max_det=300,  # Maximum detections per image
        agnostic_nms=False,  # Class-specific NMS
        verbose=True
    )
    
    # Process each result
    for result in results:
        # Get original image
        img = result.orig_img.copy()
        img_height, img_width = img.shape[:2]
        
        # Check if we have detections
        if result.boxes is None or len(result.boxes) == 0:
            print("No objects detected")
            continue
        
        boxes = result.boxes
        
        # Check for masks
        has_masks = result.masks is not None
        
        # Create overlay for masks if available
        if has_masks:
            overlay = img.copy()
        
        # Statistics
        detected_objects = {}
        total_detections = len(boxes)
        
        print(f"\n{'='*60}")
        print(f"Total Detections: {total_detections}")
        print(f"{'='*60}")
        
        # Process each detection
        for idx in range(len(boxes)):
            # Get bounding box
            box = boxes.xyxy[idx].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            
            # Get class and confidence
            cls_id = int(boxes.cls[idx].cpu().numpy())
            conf = float(boxes.conf[idx].cpu().numpy())
            class_name = result.names[cls_id]
            
            # Track statistics
            if class_name not in detected_objects:
                detected_objects[class_name] = 0
            detected_objects[class_name] += 1
            
            # Generate unique color for each class
            np.random.seed(cls_id * 100)
            color = tuple(map(int, np.random.randint(50, 255, 3)))
            
            # Draw mask if available
            if has_masks and idx < len(result.masks.data):
                mask_data = result.masks.data[idx].cpu().numpy()
                mask_resized = cv2.resize(
                    mask_data,
                    (img_width, img_height)
                )
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # Apply semi-transparent colored mask
                colored_mask = np.zeros_like(img)
                colored_mask[mask_binary == 1] = color
                overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.4, 0)
            
            # Draw thick bounding box for VLM visibility
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            # Create detailed label
            label = f"#{idx+1} {class_name} {conf:.2f}"
            
            # Calculate label size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (label_w, label_h), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Draw label background with padding
            padding = 5
            label_y1 = max(y1 - label_h - padding * 2, 0)
            label_y2 = y1
            
            cv2.rectangle(
                img,
                (x1, label_y1),
                (x1 + label_w + padding * 2, label_y2),
                color,
                -1
            )
            
            # Draw white border for label
            cv2.rectangle(
                img,
                (x1, label_y1),
                (x1 + label_w + padding * 2, label_y2),
                (255, 255, 255),
                1
            )
            
            # Draw label text
            cv2.putText(
                img,
                label,
                (x1 + padding, label_y2 - padding),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        # Blend with masks if available
        if has_masks:
            result_img = cv2.addWeighted(img, 0.8, overlay, 0.2, 0)
        else:
            result_img = img
        
        # Add detection summary to image
        summary_y = 30
        summary_lines = [
            f"Total Objects: {total_detections}",
            f"Confidence Threshold: {conf_threshold}",
        ]
        
        for line in summary_lines:
            cv2.rectangle(
                result_img, (10, summary_y - 20), 
                (400, summary_y + 5), (0, 0, 0), -1
            )
            cv2.putText(
                result_img, line, (15, summary_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            summary_y += 30
        
        # Print detection summary
        print("\nDetected Objects by Class:")
        print(f"{'-'*60}")
        for class_name in sorted(detected_objects.keys()):
            count = detected_objects[class_name]
            print(f"  {class_name}: {count}")
        print(f"{'='*60}\n")
        
        # Display
        # Resize for display if image is too large
        display_img = result_img.copy()
        max_display_width = 1920
        if display_img.shape[1] > max_display_width:
            scale = max_display_width / display_img.shape[1]
            new_width = int(display_img.shape[1] * scale)
            new_height = int(display_img.shape[0] * scale)
            display_img = cv2.resize(
                display_img, (new_width, new_height)
            )
        
        # Display (press any key to close, or it auto-closes after 5 seconds)
        if show_window:
            cv2.imshow("YOLO Maximum Detections", display_img)
            key = cv2.waitKey(5000)  # 5 second timeout
            cv2.destroyAllWindows()
        
        # Save result with datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"YOLO_detections_{timestamp}.jpg"
        cv2.imwrite(output_path, result_img)
        print(f"Result saved to: {output_path}")
        print("Ready for VLM processing!\n")


if __name__ == "__main__":
    # Example usage with maximum detections
    image_path = r"/home/riju279/Documents/Code/Zonko/EDI/edi/work/edi_vision_tui/test_image.jpeg"
    
    # For quality detections - higher confidence to reduce false positives
    visualize_yolo_detections(
        image_path,
        conf_threshold=0.35,   # Higher = fewer false positives
        iou_threshold=0.45,
        show_window=False      # Set True to display window
    )
    
    # For maximum detections (may include false positives)
    # visualize_yolo_detections(
    #     image_path,
    #     conf_threshold=0.25,
    #     iou_threshold=0.40,
    #     show_window=False
    # )