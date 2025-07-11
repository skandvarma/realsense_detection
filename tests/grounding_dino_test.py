#!/usr/bin/env python3
"""
Simple Grounding DINO test script with webcam.
"""

import cv2
import torch
import numpy as np
from PIL import Image
import time

# Check if Grounding DINO is available
try:
    from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection

    GROUNDING_DINO_AVAILABLE = True
    print("✅ Grounding DINO available")
except ImportError as e:
    print(f"❌ Grounding DINO not available: {e}")
    print("Install with: pip install transformers[grounding-dino]")
    GROUNDING_DINO_AVAILABLE = False
    exit(1)


class SimpleGroundingDino:
    def __init__(self):
        print("🚀 Initializing Grounding DINO...")

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model and processor
        model_name = "IDEA-Research/grounding-dino-tiny"  # Smaller, faster model

        try:
            self.processor = GroundingDinoProcessor.from_pretrained(model_name)
            self.model = GroundingDinoForObjectDetection.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("Trying alternative model...")
            try:
                model_name = "IDEA-Research/grounding-dino-base"
                self.processor = GroundingDinoProcessor.from_pretrained(model_name)
                self.model = GroundingDinoForObjectDetection.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
                print("✅ Alternative model loaded successfully")
            except Exception as e2:
                print(f"❌ Both models failed: {e2}")
                exit(1)

        # Current prompt
        self.current_prompt = "person."

        # Pre-defined prompts for cycling
        self.prompts = [
            "person",
            "dog . cat . bird . animal",
            "phone . laptop . book . bag",
            "car . truck . bus . bicycle",
            "person with glasses . person with hat",
            "red car . blue car . white car",
            "food . drink . plate . spoon",
            "tree . flower . plant . grass"
        ]
        self.prompt_index = 0

        print(f"🎯 Current prompt: '{self.current_prompt}'")

    def detect(self, image, confidence_threshold=0.3):
        """Detect objects in image using Grounding DINO."""
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            # Process inputs
            inputs = self.processor(
                images=pil_image,
                text=self.current_prompt,
                return_tensors="pt"
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Process outputs
            target_sizes = torch.tensor([image.shape[:2][::-1]]).to(self.device)  # [width, height]
            results = self.processor.post_process_grounded_object_detection(
                outputs, target_sizes=target_sizes, threshold=confidence_threshold
            )[0]

            detections = []
            if len(results["boxes"]) > 0:
                for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    detections.append({
                        'box': [int(x1), int(y1), int(x2), int(y2)],
                        'score': float(score.cpu().numpy()),
                        'label': label
                    })

            return detections

        except Exception as e:
            print(f"Detection error: {e}")
            return []

    def cycle_prompt(self):
        """Cycle to next prompt."""
        self.prompt_index = (self.prompt_index + 1) % len(self.prompts)
        self.current_prompt = self.prompts[self.prompt_index]
        print(f"🔄 Switched to prompt {self.prompt_index + 1}/{len(self.prompts)}: '{self.current_prompt}'")

    def set_custom_prompt(self, prompt):
        """Set custom prompt."""
        self.current_prompt = prompt
        print(f"🎯 Custom prompt: '{self.current_prompt}'")


def draw_detections(image, detections):
    """Draw detection boxes and labels on image."""
    vis_image = image.copy()

    # Colors for different detections
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0)]

    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = detection['box']
        score = detection['score']
        label = detection['label']

        # Choose color
        color = colors[i % len(colors)]

        # Draw box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label_text = f"{label}: {score:.2f}"
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        # Background for text
        cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0] + 5, y1), color, -1)

        # Text
        cv2.putText(vis_image, label_text, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return vis_image


def main():
    print("🎥 Simple Grounding DINO Webcam Test")
    print("=" * 40)

    # Initialize Grounding DINO
    detector = SimpleGroundingDino()

    # Open webcam
    print("📹 Opening webcam...")
    cap = cv2.VideoCapture(0)  # /dev/video0

    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("✅ Webcam opened successfully")
    print("\n🎮 Controls:")
    print("  'p' - Cycle through prompts")
    print("  'c' - Enter custom prompt")
    print("  's' - Save screenshot")
    print("  'q' - Quit")
    print(f"\n🎯 Current prompt: '{detector.current_prompt}'")
    print("\nStarting detection...")

    frame_count = 0
    detection_times = []

    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break

            frame_count += 1

            # Run detection every few frames to maintain FPS
            if frame_count % 3 == 0:  # Detect every 3rd frame
                start_time = time.time()
                detections = detector.detect(frame, confidence_threshold=0.25)
                detection_time = time.time() - start_time
                detection_times.append(detection_time)

                # Show stats every 30 frames
                if frame_count % 30 == 0:
                    avg_time = np.mean(detection_times[-10:]) if detection_times else 0
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                    print(f"📊 Frame {frame_count}: {len(detections)} objects, {fps:.1f} FPS")

                    # Show detections
                    for det in detections:
                        print(f"  - {det['label']}: {det['score']:.3f}")
            else:
                # Use previous detections for smooth display
                pass

            # Draw detections
            if 'detections' in locals():
                vis_frame = draw_detections(frame, detections)
            else:
                vis_frame = frame.copy()

            # Add info overlay
            cv2.putText(vis_frame, f"Objects: {len(detections) if 'detections' in locals() else 0}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(vis_frame, f"Prompt: {detector.current_prompt[:50]}...",
                        (10, vis_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(vis_frame, f"Frame: {frame_count}",
                        (10, vis_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display
            cv2.imshow("Grounding DINO Detection", vis_frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('p'):
                detector.cycle_prompt()
            elif key == ord('c'):
                print("\n🎯 Enter custom prompt (separate objects with ' . '):")
                custom_prompt = input("Prompt: ").strip()
                if custom_prompt:
                    detector.set_custom_prompt(custom_prompt)
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"grounding_dino_capture_{timestamp}.jpg"
                cv2.imwrite(filename, vis_frame)
                print(f"💾 Screenshot saved: {filename}")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        # Final stats
        if detection_times:
            avg_time = np.mean(detection_times)
            print(f"\n📊 Final Stats:")
            print(f"  Total frames: {frame_count}")
            print(f"  Average detection time: {avg_time * 1000:.1f}ms")
            print(f"  Average FPS: {1.0 / avg_time:.1f}")

        print("👋 Done!")


if __name__ == "__main__":
    main()