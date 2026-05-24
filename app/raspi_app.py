#!/usr/bin/env python3
"""
================================================================================
🔬 MALARIASENSE - RASPBERRY PI EDITION
================================================================================
Research malaria screening prototype for Raspberry Pi 5 with Camera Module 3

Features:
- Live camera preview with autofocus
- Automatic image capture when focused
- AI-powered parasite detection
- Parasitemia calculation
- Parasitaemia estimation (research use only)
- Touchscreen interface

Requirements:
    pip install customtkinter pillow numpy opencv-python ultralytics
    pip install picamera2  # For Raspberry Pi camera

Hardware:
    - Raspberry Pi 5 (recommended) or Pi 4
    - Raspberry Pi Camera Module 3 (with autofocus)
    - 7" touchscreen display (optional)

Usage:
    python malariasense_pi.py

================================================================================
"""

import customtkinter as ctk
import cv2
import numpy as np
import threading
import queue
import os
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional
from PIL import Image, ImageTk
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    # Model path - update this to your exported NCNN model folder
    MODEL_PATH = "./best_ncnn_model"  # UPDATE: path to your exported NCNN model folder
    
    # Camera settings
    CAMERA_RESOLUTION = (2048, 1536)  # 3MP for good quality
    PREVIEW_SIZE = (640, 480)
    CAPTURE_DELAY = 1.0  # Seconds to wait after focus before capture
    
    # UI settings
    WINDOW_SIZE = "1024x600"  # Good for 7" display
    FULLSCREEN_ON_PI = True
    
    # Detection settings
    CONFIDENCE_THRESHOLD = 0.15
    TILE_SIZE = 640
    TILE_OVERLAP = 0.15
    
    # Colors (Medical theme)
    COLOR_BG = "#0a0a0a"
    COLOR_PANEL = "#1a1a2e"
    COLOR_ACCENT = "#00d4aa"
    COLOR_DANGER = "#ff4757"
    COLOR_WARNING = "#ffa502"
    COLOR_SUCCESS = "#2ed573"
    COLOR_TEXT = "#ffffff"
    COLOR_TEXT_DIM = "#888888"


# ==============================================================================
# CAMERA HANDLER (Raspberry Pi Camera Module 3)
# ==============================================================================

class CameraHandler:
    """
    Handles Raspberry Pi Camera Module 3 with autofocus.
    Falls back to USB webcam on non-Pi systems for testing.
    """
    
    def __init__(self):
        self.camera = None
        self.is_pi_camera = False
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.focus_state = "searching"  # searching, focused, locked
        self.last_focus_time = 0
        
    def initialize(self) -> bool:
        """Initialize camera - tries Pi camera first, falls back to USB."""
        try:
            # Try Raspberry Pi Camera Module 3
            from picamera2 import Picamera2
            from libcamera import controls
            
            self.camera = Picamera2()
            
            # Configure for preview + capture
            preview_config = self.camera.create_preview_configuration(
                main={"size": Config.CAMERA_RESOLUTION, "format": "RGB888"},
                lores={"size": Config.PREVIEW_SIZE, "format": "RGB888"},
                display="lores"
            )
            self.camera.configure(preview_config)
            
            # Enable continuous autofocus (Camera Module 3 feature)
            self.camera.set_controls({
                "AfMode": controls.AfModeEnum.Continuous,
                "AfSpeed": controls.AfSpeedEnum.Fast,
            })
            
            self.camera.start()
            self.is_pi_camera = True
            print("✓ Raspberry Pi Camera Module 3 initialized with autofocus")
            return True
            
        except ImportError:
            print("⚠ picamera2 not found, trying USB webcam...")
        except Exception as e:
            print(f"⚠ Pi camera error: {e}, trying USB webcam...")
        
        # Fallback to USB webcam
        try:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.PREVIEW_SIZE[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.PREVIEW_SIZE[1])
                self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                self.is_pi_camera = False
                print("✓ USB webcam initialized")
                return True
        except Exception as e:
            print(f"✗ USB camera error: {e}")
        
        return False
    
    def get_focus_state(self) -> str:
        """Get current autofocus state."""
        if not self.is_pi_camera:
            return "focused"  # USB cams handle this internally
        
        try:
            from libcamera import controls
            metadata = self.camera.capture_metadata()
            af_state = metadata.get("AfState", 0)
            
            # AfState values: 0=Idle, 1=Scanning, 2=Focused, 3=Failed
            if af_state == 2:
                return "focused"
            elif af_state == 1:
                return "searching"
            else:
                return "idle"
        except:
            return "unknown"
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame."""
        if self.is_pi_camera:
            try:
                frame = self.camera.capture_array("main")
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Capture error: {e}")
                return None
        else:
            ret, frame = self.camera.read()
            return frame if ret else None
    
    def capture_preview(self) -> Optional[np.ndarray]:
        """Capture preview frame (lower resolution for speed)."""
        if self.is_pi_camera:
            try:
                frame = self.camera.capture_array("lores")
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except:
                return None
        else:
            ret, frame = self.camera.read()
            return frame if ret else None
    
    def capture_high_res(self) -> Optional[np.ndarray]:
        """Capture high resolution image for analysis."""
        if self.is_pi_camera:
            try:
                # Trigger autofocus and wait
                from libcamera import controls
                self.camera.set_controls({"AfTrigger": controls.AfTriggerEnum.Start})
                time.sleep(0.5)  # Wait for focus
                
                frame = self.camera.capture_array("main")
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"High-res capture error: {e}")
                return None
        else:
            # For USB, just capture current frame
            ret, frame = self.camera.read()
            return frame if ret else None
    
    def release(self):
        """Release camera resources."""
        if self.camera:
            if self.is_pi_camera:
                self.camera.stop()
                self.camera.close()
            else:
                self.camera.release()


# ==============================================================================
# DETECTION ENGINE
# ==============================================================================

@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    
    @property
    def bbox(self): 
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))
    
    @property
    def area(self): 
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    @property
    def center(self):
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class DetectionEngine:
    """Optimized malaria detection for Raspberry Pi."""
    
    # Class groupings (WHO: count asexual forms only for parasitaemia)
    MALARIA_ASEXUAL_CLASSES = {'ring', 'trophozoite', 'schizont', 'Parasitized'}
    GAMETOCYTE_CLASS = {'gametocyte'}
    MIMIC_CLASSES = {'Babesia'}
    ALL_ABNORMAL_CLASSES = MALARIA_ASEXUAL_CLASSES | GAMETOCYTE_CLASS | MIMIC_CLASSES
    
    def __init__(self, model_path: str):
        from ultralytics import YOLO
        self.model = YOLO(model_path, task='detect')
        self.names = self.model.names
        
    def analyze_image(self, image: np.ndarray) -> dict:
        """Quick image analysis for preprocessing decisions."""
        small = cv2.resize(image, (64, 64))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        blue_mask = (h > 75) & (h < 145) & (s > 20)
        
        return {
            'is_blue': np.sum(blue_mask) / blue_mask.size > 0.3,
            'mean_brightness': np.mean(v),
        }
    
    def get_variants(self, image: np.ndarray, analysis: dict) -> List[Tuple[str, np.ndarray, float]]:
        """Generate image variants for ensemble detection."""
        variants = [('original', image, 1.0)]
        
        # Contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        variants.append(('enhanced', enhanced, 1.0))
        
        # Blue stain correction
        if analysis['is_blue']:
            for shift in [18, 28]:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
                shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                variants.append((f'hue_{shift}', shifted, 1.1))
            
            # Enhanced + shifted
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + 20) % 180
            combo = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            variants.append(('combo', combo, 1.15))
        
        return variants
    
    def detect(self, image: np.ndarray, progress_callback=None) -> List[Detection]:
        """Run full detection pipeline."""
        
        def update(p):
            if progress_callback:
                progress_callback(p)
        
        # Resize if too large
        h, w = image.shape[:2]
        scale = 1.0
        if w > 2048:
            scale = 2048 / w
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        analysis = self.analyze_image(image)
        variants = self.get_variants(image, analysis)
        
        all_dets = []
        h_img, w_img = image.shape[:2]
        
        tile_size = Config.TILE_SIZE
        stride = int(tile_size * (1 - Config.TILE_OVERLAP))
        
        total_variants = len(variants)
        
        for v_idx, (name, variant, weight) in enumerate(variants):
            update((v_idx / total_variants) * 0.9)
            
            # Generate tiles
            x_steps = list(range(0, max(1, w_img - tile_size + 1), stride))
            if not x_steps or x_steps[-1] + tile_size < w_img:
                x_steps.append(max(0, w_img - tile_size))
            
            y_steps = list(range(0, max(1, h_img - tile_size + 1), stride))
            if not y_steps or y_steps[-1] + tile_size < h_img:
                y_steps.append(max(0, h_img - tile_size))
            
            for y in y_steps:
                for x in x_steps:
                    tile = variant[y:y+tile_size, x:x+tile_size]
                    
                    # Pad if needed
                    if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                        padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                        padded[:tile.shape[0], :tile.shape[1]] = tile
                        tile = padded
                    
                    results = self.model(tile, imgsz=640, conf=0.08, verbose=False)
                    
                    for r in results:
                        if r.boxes is None:
                            continue
                        
                        boxes = r.boxes.xyxy.cpu().numpy()
                        confs = r.boxes.conf.cpu().numpy()
                        classes = r.boxes.cls.cpu().numpy()
                        
                        for box, conf, cls_id in zip(boxes, confs, classes):
                            # Map to original coordinates
                            x1 = (box[0] + x) / scale
                            y1 = (box[1] + y) / scale
                            x2 = (box[2] + x) / scale
                            y2 = (box[3] + y) / scale
                            
                            class_name = self.names[int(cls_id)]
                            
                            all_dets.append(Detection(
                                x1=x1, y1=y1, x2=x2, y2=y2,
                                confidence=float(conf) * weight,
                                class_id=int(cls_id),
                                class_name=class_name,
                            ))
        
        update(0.95)
        
        # NMS
        final_dets = self._nms(all_dets)
        
        update(1.0)
        return final_dets
    
    def _nms(self, dets: List[Detection], iou_thresh: float = 0.4) -> List[Detection]:
        """Fast NMS implementation."""
        if not dets:
            return []
        
        # Sort by confidence
        dets = sorted(dets, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while dets:
            curr = dets.pop(0)
            if curr.confidence < Config.CONFIDENCE_THRESHOLD:
                continue
            keep.append(curr)
            
            # Remove overlapping
            new_dets = []
            for d in dets:
                xx1 = max(curr.x1, d.x1)
                yy1 = max(curr.y1, d.y1)
                xx2 = min(curr.x2, d.x2)
                yy2 = min(curr.y2, d.y2)
                
                inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                iou = inter / (curr.area + d.area - inter + 1e-6)
                
                if iou < iou_thresh:
                    new_dets.append(d)
            
            dets = new_dets
        
        return keep


# ==============================================================================
# RESULTS CALCULATOR
# ==============================================================================

class ResultsCalculator:
    """Calculate parasitaemia from detections.

    Counting follows WHO malaria microscopy guidance:
      - Parasitaemia numerator: asexual forms (ring, trophozoite, schizont,
        generic 'Parasitized') only.
      - Gametocytes: reported separately, excluded from parasitaemia.
      - Babesia mimics: excluded from parasitaemia numerator, retained in
        detected-RBC denominator, and reported separately for review.
    Note: this is a research estimate, not a clinical determination.
    """

    ASEXUAL_CLASSES = {'ring', 'trophozoite', 'schizont', 'Parasitized'}
    GAMETOCYTE_CLASS = {'gametocyte'}
    MIMIC_CLASSES = {'Babesia'}

    @staticmethod
    def calculate(detections: List[Detection]) -> dict:
        asexual = sum(1 for d in detections if d.class_name in ResultsCalculator.ASEXUAL_CLASSES)
        gametocytes = sum(1 for d in detections if d.class_name in ResultsCalculator.GAMETOCYTE_CLASS)
        babesia = sum(1 for d in detections if d.class_name in ResultsCalculator.MIMIC_CLASSES)
        uninfected = sum(1 for d in detections if d.class_name == 'Uninfected')

        # Parasitaemia: asexual parasites / total detected RBCs
        # Babesia-like detections excluded from numerator but retained in denominator
        total_rbc = asexual + gametocytes + babesia + uninfected
        parasitemia = (asexual / total_rbc * 100) if total_rbc > 0 else 0

        # Risk flag — NOT a clinical severity grade
        if parasitemia == 0:
            risk_flag = "No parasites detected"
            risk_color = "success"
        elif parasitemia < 1:
            risk_flag = f"Estimated parasitaemia: {parasitemia:.2f}%"
            risk_color = "warning"
        elif parasitemia < 10:
            risk_flag = f"Estimated parasitaemia: {parasitemia:.2f}%"
            risk_color = "warning"
        else:
            risk_flag = f"Hyperparasitaemia alert: {parasitemia:.1f}% — clinical interpretation required"
            risk_color = "danger"

        return {
            'asexual_parasites': asexual,
            'gametocytes': gametocytes,
            'babesia_mimics': babesia,
            'parasites': asexual,  # backward compat
            'uninfected': uninfected,
            'total_rbc': total_rbc,
            'parasitemia': parasitemia,
            'severity': risk_flag,      # backward compat key
            'severity_color': risk_color,
        }


# ==============================================================================
# MAIN APPLICATION UI
# ==============================================================================

class MalariaSenseApp(ctk.CTk):
    """
    Research screening interface for MalariaSense.
    """
    
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.title("MalariaSense — Research Screening Prototype")
        self.geometry(Config.WINDOW_SIZE)
        self.configure(fg_color=Config.COLOR_BG)
        
        # Fullscreen on Pi
        if Config.FULLSCREEN_ON_PI and os.path.exists('/proc/device-tree/model'):
            self.attributes('-fullscreen', True)
        
        # State
        self.camera = None
        self.detector = None
        self.is_previewing = False
        self.captured_image = None
        self.current_detections = None
        self.preview_thread = None
        self.auto_capture_enabled = True
        self.focus_stable_time = 0
        
        # Build UI
        self._setup_fonts()
        self._build_ui()
        
        # Initialize in background
        threading.Thread(target=self._initialize_system, daemon=True).start()
        
        # Bind escape to exit
        self.bind('<Escape>', lambda e: self._exit_app())
    
    def _setup_fonts(self):
        """Setup custom fonts."""
        self.font_title = ctk.CTkFont(family="Helvetica", size=28, weight="bold")
        self.font_heading = ctk.CTkFont(family="Helvetica", size=18, weight="bold")
        self.font_body = ctk.CTkFont(family="Helvetica", size=14)
        self.font_small = ctk.CTkFont(family="Helvetica", size=12)
        self.font_large = ctk.CTkFont(family="Helvetica", size=48, weight="bold")
    
    def _build_ui(self):
        """Build the main UI layout."""
        # Main container
        self.main_container = ctk.CTkFrame(self, fg_color=Config.COLOR_BG)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Header
        self._build_header()
        
        # Content area (will switch between screens)
        self.content_frame = ctk.CTkFrame(self.main_container, fg_color=Config.COLOR_BG)
        self.content_frame.pack(fill="both", expand=True, pady=10)
        
        # Footer / Status bar
        self._build_footer()
        
        # Show loading screen
        self._show_loading_screen()
    
    def _build_header(self):
        """Build header with logo and title."""
        header = ctk.CTkFrame(self.main_container, fg_color=Config.COLOR_PANEL, height=60, corner_radius=10)
        header.pack(fill="x", pady=(0, 10))
        header.pack_propagate(False)
        
        # Logo/Icon
        logo_frame = ctk.CTkFrame(header, fg_color=Config.COLOR_ACCENT, width=40, height=40, corner_radius=8)
        logo_frame.pack(side="left", padx=15, pady=10)
        logo_frame.pack_propagate(False)
        ctk.CTkLabel(logo_frame, text="🔬", font=ctk.CTkFont(size=20)).place(relx=0.5, rely=0.5, anchor="center")
        
        # Title
        ctk.CTkLabel(
            header, 
            text="MALARIASENSE", 
            font=self.font_title,
            text_color=Config.COLOR_TEXT
        ).pack(side="left", padx=5)
        
        ctk.CTkLabel(
            header,
            text="Raspberry Pi Edition",
            font=self.font_small,
            text_color=Config.COLOR_TEXT_DIM
        ).pack(side="left", padx=10)
        
        # Time
        self.time_label = ctk.CTkLabel(header, text="", font=self.font_body, text_color=Config.COLOR_TEXT_DIM)
        self.time_label.pack(side="right", padx=15)
        self._update_time()
    
    def _build_footer(self):
        """Build footer status bar."""
        footer = ctk.CTkFrame(self.main_container, fg_color=Config.COLOR_PANEL, height=40, corner_radius=10)
        footer.pack(fill="x", pady=(10, 0))
        footer.pack_propagate(False)
        
        self.status_label = ctk.CTkLabel(
            footer,
            text="● Initializing...",
            font=self.font_small,
            text_color=Config.COLOR_WARNING
        )
        self.status_label.pack(side="left", padx=15, pady=10)
        
        # Version
        ctk.CTkLabel(
            footer,
            text="v1.0.0",
            font=self.font_small,
            text_color=Config.COLOR_TEXT_DIM
        ).pack(side="right", padx=15)
    
    def _update_time(self):
        """Update time display."""
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.configure(text=current_time)
        self.after(1000, self._update_time)
    
    def _set_status(self, text: str, color: str = Config.COLOR_TEXT_DIM):
        """Update status bar."""
        self.status_label.configure(text=f"● {text}", text_color=color)
    
    # ==========================================================================
    # SCREENS
    # ==========================================================================
    
    def _clear_content(self):
        """Clear content area."""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
    
    def _show_loading_screen(self):
        """Show loading/initialization screen."""
        self._clear_content()
        
        frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        frame.place(relx=0.5, rely=0.5, anchor="center")
        
        # Spinner animation (simple)
        self.loading_label = ctk.CTkLabel(
            frame,
            text="⏳",
            font=ctk.CTkFont(size=48)
        )
        self.loading_label.pack(pady=20)
        
        ctk.CTkLabel(
            frame,
            text="Initializing System...",
            font=self.font_heading,
            text_color=Config.COLOR_TEXT
        ).pack(pady=10)
        
        self.loading_detail = ctk.CTkLabel(
            frame,
            text="Loading AI model and camera",
            font=self.font_body,
            text_color=Config.COLOR_TEXT_DIM
        )
        self.loading_detail.pack(pady=5)
    
    def _show_camera_screen(self):
        """Show live camera preview screen."""
        self._clear_content()
        
        # Left side - Camera preview
        left_frame = ctk.CTkFrame(self.content_frame, fg_color=Config.COLOR_PANEL, corner_radius=15)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        # Preview header
        preview_header = ctk.CTkFrame(left_frame, fg_color="transparent", height=40)
        preview_header.pack(fill="x", padx=15, pady=(15, 5))
        
        ctk.CTkLabel(
            preview_header,
            text="📷 LIVE PREVIEW",
            font=self.font_heading,
            text_color=Config.COLOR_TEXT
        ).pack(side="left")
        
        self.focus_indicator = ctk.CTkLabel(
            preview_header,
            text="● FOCUSING",
            font=self.font_small,
            text_color=Config.COLOR_WARNING
        )
        self.focus_indicator.pack(side="right")
        
        # Preview canvas
        self.preview_label = ctk.CTkLabel(
            left_frame,
            text="",
            fg_color="#000000",
            corner_radius=10
        )
        self.preview_label.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Right side - Controls
        right_frame = ctk.CTkFrame(self.content_frame, fg_color=Config.COLOR_PANEL, corner_radius=15, width=280)
        right_frame.pack(side="right", fill="y", padx=(5, 0))
        right_frame.pack_propagate(False)
        
        # Controls header
        ctk.CTkLabel(
            right_frame,
            text="⚙️ CONTROLS",
            font=self.font_heading,
            text_color=Config.COLOR_TEXT
        ).pack(pady=20)
        
        # Auto capture toggle
        self.auto_capture_var = ctk.BooleanVar(value=True)
        auto_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        auto_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            auto_frame,
            text="Auto Capture",
            font=self.font_body,
            text_color=Config.COLOR_TEXT
        ).pack(side="left")
        
        ctk.CTkSwitch(
            auto_frame,
            text="",
            variable=self.auto_capture_var,
            onvalue=True,
            offvalue=False,
            progress_color=Config.COLOR_ACCENT
        ).pack(side="right")
        
        # Manual capture button
        self.capture_btn = ctk.CTkButton(
            right_frame,
            text="📸 CAPTURE NOW",
            font=self.font_heading,
            height=60,
            fg_color=Config.COLOR_ACCENT,
            hover_color="#00b894",
            command=self._manual_capture
        )
        self.capture_btn.pack(fill="x", padx=20, pady=20)
        
        # Load from file button
        ctk.CTkButton(
            right_frame,
            text="📁 Load Image",
            font=self.font_body,
            height=45,
            fg_color="#2d3436",
            hover_color="#636e72",
            command=self._load_from_file
        ).pack(fill="x", padx=20, pady=5)
        
        # Instructions
        instructions = ctk.CTkFrame(right_frame, fg_color="#1e272e", corner_radius=10)
        instructions.pack(fill="x", padx=20, pady=20)
        
        ctk.CTkLabel(
            instructions,
            text="📋 Instructions",
            font=self.font_body,
            text_color=Config.COLOR_ACCENT
        ).pack(anchor="w", padx=15, pady=(10, 5))
        
        for step in ["1. Position blood smear", "2. Wait for autofocus", "3. Image captures automatically"]:
            ctk.CTkLabel(
                instructions,
                text=step,
                font=self.font_small,
                text_color=Config.COLOR_TEXT_DIM
            ).pack(anchor="w", padx=15, pady=2)
        
        ctk.CTkLabel(instructions, text="").pack(pady=5)  # Spacer
        
        # Start preview
        self._start_preview()
    
    def _show_analyzing_screen(self, image: np.ndarray):
        """Show analysis in progress screen."""
        self._clear_content()
        
        frame = ctk.CTkFrame(self.content_frame, fg_color=Config.COLOR_PANEL, corner_radius=15)
        frame.place(relx=0.5, rely=0.5, anchor="center")
        
        # Show captured image thumbnail
        thumb = cv2.resize(image, (320, 240))
        thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(thumb_rgb)
        ctk_img = ctk.CTkImage(pil_img, size=(320, 240))
        
        ctk.CTkLabel(frame, image=ctk_img, text="").pack(padx=30, pady=(30, 15))
        
        ctk.CTkLabel(
            frame,
            text="🔬 Analyzing Sample...",
            font=self.font_heading,
            text_color=Config.COLOR_TEXT
        ).pack(pady=10)
        
        self.analysis_progress = ctk.CTkProgressBar(frame, width=280, progress_color=Config.COLOR_ACCENT)
        self.analysis_progress.pack(pady=15)
        self.analysis_progress.set(0)
        
        self.analysis_status = ctk.CTkLabel(
            frame,
            text="Detecting parasites...",
            font=self.font_body,
            text_color=Config.COLOR_TEXT_DIM
        )
        self.analysis_status.pack(pady=(5, 30))
    
    def _show_results_screen(self, image: np.ndarray, detections: List[Detection], results: dict):
        """Show detection results."""
        self._clear_content()
        
        # Left - Image with detections
        left_frame = ctk.CTkFrame(self.content_frame, fg_color=Config.COLOR_PANEL, corner_radius=15)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        ctk.CTkLabel(
            left_frame,
            text="🔬 DETECTION RESULTS",
            font=self.font_heading,
            text_color=Config.COLOR_TEXT
        ).pack(pady=15)
        
        # Draw detections on image
        vis_image = self._draw_detections(image, detections)
        
        # Resize for display
        h, w = vis_image.shape[:2]
        max_h, max_w = 380, 550
        scale = min(max_w / w, max_h / h)
        display_size = (int(w * scale), int(h * scale))
        
        vis_resized = cv2.resize(vis_image, display_size)
        vis_rgb = cv2.cvtColor(vis_resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(vis_rgb)
        ctk_img = ctk.CTkImage(pil_img, size=display_size)
        
        ctk.CTkLabel(left_frame, image=ctk_img, text="").pack(pady=10)
        
        # Legend
        legend = ctk.CTkFrame(left_frame, fg_color="transparent")
        legend.pack(pady=10)
        
        for label, color in [("Parasitized", "#ff4757"), ("Uninfected", "#a4a4a4")]:
            item = ctk.CTkFrame(legend, fg_color="transparent")
            item.pack(side="left", padx=15)
            ctk.CTkLabel(item, text="■", text_color=color, font=ctk.CTkFont(size=16)).pack(side="left")
            ctk.CTkLabel(item, text=label, font=self.font_small, text_color=Config.COLOR_TEXT_DIM).pack(side="left", padx=5)
        
        # Right - Results panel
        right_frame = ctk.CTkFrame(self.content_frame, fg_color=Config.COLOR_PANEL, corner_radius=15, width=300)
        right_frame.pack(side="right", fill="y", padx=(5, 0))
        right_frame.pack_propagate(False)
        
        ctk.CTkLabel(
            right_frame,
            text="📊 ANALYSIS",
            font=self.font_heading,
            text_color=Config.COLOR_TEXT
        ).pack(pady=20)
        
        # Severity indicator (large)
        severity_colors = {
            "success": Config.COLOR_SUCCESS,
            "warning": Config.COLOR_WARNING,
            "danger": Config.COLOR_DANGER
        }
        severity_color = severity_colors.get(results['severity_color'], Config.COLOR_TEXT)
        
        severity_frame = ctk.CTkFrame(right_frame, fg_color="#1e272e", corner_radius=15)
        severity_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            severity_frame,
            text=results['severity'],
            font=self.font_large,
            text_color=severity_color
        ).pack(pady=20)
        
        # Stats
        stats_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        stats_frame.pack(fill="x", padx=20, pady=10)
        
        stats = [
            ("Parasites", str(results['parasites']), Config.COLOR_DANGER),
            ("Uninfected", str(results['uninfected']), Config.COLOR_TEXT_DIM),
            ("Total RBC", str(results['total_rbc']), Config.COLOR_TEXT),
            ("Parasitemia", f"{results['parasitemia']:.2f}%", severity_color),
        ]
        
        for label, value, color in stats:
            row = ctk.CTkFrame(stats_frame, fg_color="transparent")
            row.pack(fill="x", pady=5)
            ctk.CTkLabel(row, text=label, font=self.font_body, text_color=Config.COLOR_TEXT_DIM).pack(side="left")
            ctk.CTkLabel(row, text=value, font=self.font_heading, text_color=color).pack(side="right")
        
        # Action buttons
        ctk.CTkButton(
            right_frame,
            text="📷 New Scan",
            font=self.font_heading,
            height=50,
            fg_color=Config.COLOR_ACCENT,
            hover_color="#00b894",
            command=self._show_camera_screen
        ).pack(fill="x", padx=20, pady=(30, 10))
        
        ctk.CTkButton(
            right_frame,
            text="💾 Save Report",
            font=self.font_body,
            height=40,
            fg_color="#2d3436",
            hover_color="#636e72",
            command=lambda: self._save_report(image, detections, results)
        ).pack(fill="x", padx=20, pady=5)
    
    # ==========================================================================
    # CAMERA & DETECTION LOGIC
    # ==========================================================================
    
    def _initialize_system(self):
        """Initialize camera and AI model."""
        # Initialize camera
        self.loading_detail.configure(text="Connecting to camera...")
        self._set_status("Connecting to camera...", Config.COLOR_WARNING)
        
        self.camera = CameraHandler()
        if not self.camera.initialize():
            self._set_status("Camera failed! Using file mode.", Config.COLOR_DANGER)
            self.after(2000, self._show_file_only_screen)
            return
        
        # Load AI model
        self.loading_detail.configure(text="Loading AI model...")
        self._set_status("Loading AI model...", Config.COLOR_WARNING)
        
        try:
            self.detector = DetectionEngine(Config.MODEL_PATH)
        except Exception as e:
            self._set_status(f"Model error: {str(e)[:30]}", Config.COLOR_DANGER)
            return
        
        # Ready!
        self._set_status("System Ready", Config.COLOR_SUCCESS)
        self.after(500, self._show_camera_screen)
    
    def _start_preview(self):
        """Start camera preview loop."""
        self.is_previewing = True
        self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self.preview_thread.start()
    
    def _stop_preview(self):
        """Stop camera preview."""
        self.is_previewing = False
    
    def _preview_loop(self):
        """Camera preview loop with autofocus detection."""
        last_focus_state = None
        focus_stable_start = None
        
        while self.is_previewing:
            frame = self.camera.capture_preview()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Check focus state
            focus_state = self.camera.get_focus_state()
            
            # Update focus indicator
            if focus_state == "focused":
                self.after(0, lambda: self.focus_indicator.configure(
                    text="● FOCUSED", text_color=Config.COLOR_SUCCESS))
                
                # Track how long we've been focused
                if last_focus_state != "focused":
                    focus_stable_start = time.time()
                elif focus_stable_start and self.auto_capture_var.get():
                    # Auto capture after stable focus
                    if time.time() - focus_stable_start > Config.CAPTURE_DELAY:
                        self.after(0, self._auto_capture)
                        focus_stable_start = None
            else:
                self.after(0, lambda: self.focus_indicator.configure(
                    text="● FOCUSING...", text_color=Config.COLOR_WARNING))
                focus_stable_start = None
            
            last_focus_state = focus_state
            
            # Update preview
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Resize to fit preview area
            ctk_img = ctk.CTkImage(pil_img, size=(560, 420))
            
            self.after(0, lambda img=ctk_img: self.preview_label.configure(image=img))
            
            time.sleep(0.033)  # ~30 FPS
    
    def _auto_capture(self):
        """Auto capture when focused."""
        if not self.is_previewing:
            return
        
        self._set_status("Capturing...", Config.COLOR_ACCENT)
        self._stop_preview()
        
        # Capture high-res image
        image = self.camera.capture_high_res()
        if image is not None:
            self._run_analysis(image)
        else:
            self._set_status("Capture failed", Config.COLOR_DANGER)
            self._show_camera_screen()
    
    def _manual_capture(self):
        """Manual capture button pressed."""
        self._stop_preview()
        self._set_status("Capturing...", Config.COLOR_ACCENT)
        
        image = self.camera.capture_high_res()
        if image is not None:
            self._run_analysis(image)
        else:
            self._set_status("Capture failed", Config.COLOR_DANGER)
            self._show_camera_screen()
    
    def _load_from_file(self):
        """Load image from file."""
        self._stop_preview()
        
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        
        if path:
            image = cv2.imread(path)
            if image is not None:
                self._run_analysis(image)
            else:
                self._set_status("Failed to load image", Config.COLOR_DANGER)
                self._show_camera_screen()
        else:
            self._show_camera_screen()
    
    def _run_analysis(self, image: np.ndarray):
        """Run AI analysis on captured image."""
        self.captured_image = image
        self._show_analyzing_screen(image)
        
        def progress_callback(p):
            self.after(0, lambda: self.analysis_progress.set(p))
            if p < 0.3:
                status = "Preprocessing image..."
            elif p < 0.7:
                status = "Detecting parasites..."
            elif p < 0.95:
                status = "Running ensemble..."
            else:
                status = "Finalizing..."
            self.after(0, lambda s=status: self.analysis_status.configure(text=s))
        
        def run():
            try:
                detections = self.detector.detect(image, progress_callback)
                results = ResultsCalculator.calculate(detections)
                
                self.current_detections = detections
                self.after(0, lambda: self._show_results_screen(image, detections, results))
                self._set_status("Analysis complete", Config.COLOR_SUCCESS)
            except Exception as e:
                print(f"Analysis error: {e}")
                self._set_status(f"Error: {str(e)[:30]}", Config.COLOR_DANGER)
        
        threading.Thread(target=run, daemon=True).start()
    
    def _draw_detections(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detection boxes on image."""
        vis = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            if det.class_name == 'Parasitized':
                color = (87, 71, 255)  # Red (BGR)
                thickness = 2
            else:
                color = (164, 164, 164)  # Gray
                thickness = 1
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        
        return vis
    
    def _save_report(self, image: np.ndarray, detections: List[Detection], results: dict):
        """Save analysis report."""
        from tkinter import filedialog
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"malaria_report_{timestamp}"
        
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
        )
        
        if path:
            # Draw detections and save
            vis = self._draw_detections(image, detections)
            
            # Add results overlay
            h, w = vis.shape[:2]
            cv2.rectangle(vis, (10, 10), (300, 120), (0, 0, 0), -1)
            cv2.putText(vis, f"Parasites: {results['parasites']}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, f"Parasitemia: {results['parasitemia']:.2f}%", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, f"Severity: {results['severity']}", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imwrite(path, vis)
            self._set_status(f"Saved: {Path(path).name}", Config.COLOR_SUCCESS)
    
    def _show_file_only_screen(self):
        """Show screen when camera is not available."""
        self._clear_content()
        
        frame = ctk.CTkFrame(self.content_frame, fg_color=Config.COLOR_PANEL, corner_radius=15)
        frame.place(relx=0.5, rely=0.5, anchor="center")
        
        ctk.CTkLabel(
            frame,
            text="📁",
            font=ctk.CTkFont(size=64)
        ).pack(pady=(40, 20))
        
        ctk.CTkLabel(
            frame,
            text="Camera Not Available",
            font=self.font_heading,
            text_color=Config.COLOR_TEXT
        ).pack(pady=10)
        
        ctk.CTkLabel(
            frame,
            text="Load images from file instead",
            font=self.font_body,
            text_color=Config.COLOR_TEXT_DIM
        ).pack(pady=5)
        
        ctk.CTkButton(
            frame,
            text="📁 Load Image",
            font=self.font_heading,
            height=50,
            width=200,
            fg_color=Config.COLOR_ACCENT,
            command=self._load_from_file
        ).pack(pady=30)
    
    def _exit_app(self):
        """Clean exit."""
        self._stop_preview()
        if self.camera:
            self.camera.release()
        self.destroy()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # Set appearance
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    # Run app
    app = MalariaSenseApp()
    app.mainloop()


if __name__ == "__main__":
    main()