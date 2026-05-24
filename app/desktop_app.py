import customtkinter as ctk
import cv2
import numpy as np
import threading
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image
from tkinter import filedialog

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# ⚠️ REPLACE WITH YOUR ACTUAL MODEL PATH
MODEL_PATH = './runs/train/best.pt'  # UPDATE: path to your trained model

# COLORS
COLOR_BG = "#1e1e1e"
COLOR_PANEL = "#2b2b2b"
COLOR_ACCENT = "#3B8ED0"
COLOR_SUCCESS = "#2CC985"
COLOR_DANGER = "#FF4747"
COLOR_TEXT_MAIN = "#FFFFFF"
COLOR_TEXT_SUB = "#AAAAAA"

# ==============================================================================
# BACKEND ENGINE (Hidden Complexity)
# ==============================================================================

CLASS_NAMES = {
    0: 'ring', 1: 'trophozoite', 2: 'schizont', 3: 'gametocyte',
    4: 'Uninfected', 5: 'Parasitized', 6: 'Babesia',
}
# Class groupings for parasitaemia calculation (WHO: count asexual forms only)
MALARIA_ASEXUAL_CLASSES = {'ring', 'trophozoite', 'schizont', 'Parasitized'}
GAMETOCYTE_CLASS = {'gametocyte'}
MIMIC_CLASSES = {'Babesia'}
ALL_PARASITE_CLASSES = MALARIA_ASEXUAL_CLASSES | GAMETOCYTE_CLASS  # excludes Babesia
ALL_ABNORMAL_CLASSES = ALL_PARASITE_CLASSES | MIMIC_CLASSES

@dataclass
class Detection:
    x1: float; y1: float; x2: float; y2: float
    confidence: float; class_id: int; class_name: str
    source: str = ''; votes: int = 1
    
    @property
    def bbox(self): return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))
    @property
    def center(self): return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    @property
    def area(self): return (self.x2 - self.x1) * (self.y2 - self.y1)
    @property
    def width(self): return self.x2 - self.x1
    @property
    def height(self): return self.y2 - self.y1

class ImageAnalyzer:
    @staticmethod
    def analyze(image: np.ndarray) -> dict:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, s, v = cv2.split(hsv)
        cell_mask = s > 20
        return {
            'mean_hue': float(np.mean(h[cell_mask])) if np.any(cell_mask) else float(np.mean(h)),
            'mean_sat': float(np.mean(s[cell_mask])) if np.any(cell_mask) else float(np.mean(s)),
            'is_blue': 75 < float(np.mean(h[cell_mask]) if np.any(cell_mask) else np.mean(h)) < 145,
            'is_soft': float(cv2.Laplacian(gray, cv2.CV_64F).var()) < 250,
        }

class ImageTransforms:
    @staticmethod
    def hue_shift(image: np.ndarray, shift: float) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        mask = ((h > 70) & (h < 150) & (s > 12)).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        h_new = h + shift * mask
        h_new = np.where(h_new > 180, h_new - 180, h_new)
        h_new = np.where(h_new < 0, h_new + 180, h_new)
        return cv2.cvtColor(cv2.merge([h_new.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)]), cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def saturation_boost(image: np.ndarray, factor: float = 1.15) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        s = np.clip(s * factor, 0, 255)
        return cv2.cvtColor(cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)]), cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def contrast_clahe(image: np.ndarray, clip: float = 2.0) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def sharpen(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (0, 0), 2)
        return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    
    @classmethod
    def get_all_variants(cls, image: np.ndarray, analysis: dict) -> List[Tuple[str, np.ndarray, float]]:
        variants = []
        variants.append(('original', image.copy(), 1.0))
        enhanced = cls.contrast_clahe(image, clip=2.0)
        variants.append(('enhanced', enhanced, 1.0))
        sharpened = cls.sharpen(image, strength=0.5)
        variants.append(('sharpened', sharpened, 0.95))
        enh_sharp = cls.sharpen(enhanced, strength=0.4)
        variants.append(('enh_sharp', enh_sharp, 0.95))
        
        if analysis['is_blue']:
            for shift in [10, 18, 25, 32]:
                shifted = cls.hue_shift(image, shift)
                weight = 1.1 if shift in [18, 25] else 1.0
                variants.append((f'hue_{shift}', shifted, weight))
            for shift in [15, 22]:
                combo = cls.hue_shift(enhanced, shift)
                variants.append((f'enh_hue_{shift}', combo, 1.15))
            sat = cls.saturation_boost(image, 1.12)
            sat_shift = cls.hue_shift(sat, 18)
            variants.append(('sat_shift', sat_shift, 1.05))
        return variants

class MaximumDetector:
    def __init__(self, model_path):
        self.device = '0'
        # EXACT Parameters from blue.py
        self.tile_sizes = [640, 512, 480]
        self.tile_overlap = 0.40
        self.base_conf = 0.03
        self.parasite_threshold = 0.055
        self.uninfected_threshold = 0.18
        
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        
        global CLASS_NAMES
        if hasattr(self.model, 'names') and self.model.names:
            CLASS_NAMES = self.model.names

    def _generate_tiles(self, w, h, tile_size):
        stride = int(tile_size * (1 - self.tile_overlap))
        positions = []
        y_pos = list(range(0, max(1, h - tile_size + 1), stride))
        x_pos = list(range(0, max(1, w - tile_size + 1), stride))
        if not y_pos or y_pos[-1] + tile_size < h: y_pos.append(max(0, h - tile_size))
        if not x_pos or x_pos[-1] + tile_size < w: x_pos.append(max(0, w - tile_size))
        for y in y_pos:
            for x in x_pos:
                positions.append((x, y))
        return list(set(positions))

    def _extract_tile(self, image, x, y, size):
        h, w = image.shape[:2]
        tile = image[y:min(y + size, h), x:min(x + size, w)]
        if tile.shape[0] < size or tile.shape[1] < size:
            padded = np.zeros((size, size, 3), dtype=np.uint8)
            padded[:tile.shape[0], :tile.shape[1]] = tile
            tile = padded
        return tile

    def _run_on_image(self, image, scale, source, weight):
        all_dets = []
        augments = [('', image), ('_hflip', cv2.flip(image, 1)), ('_vflip', cv2.flip(image, 0))]
        h_img, w_img = image.shape[:2]

        for aug_name, aug_img in augments:
            for tile_size in self.tile_sizes:
                tiles = self._generate_tiles(aug_img.shape[1], aug_img.shape[0], tile_size)
                for tx, ty in tiles:
                    tile = self._extract_tile(aug_img, tx, ty, tile_size)
                    tile_input = cv2.resize(tile, (640, 640)) if tile_size != 640 else tile
                    tile_scale = tile_size / 640.0
                    
                    results = self.model.predict(tile_input, conf=self.base_conf, iou=0.5, device=self.device, verbose=False, augment=True)
                    
                    for result in results:
                        if result.boxes is None: continue
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy().astype(int)
                        
                        for box, conf, cls in zip(boxes, confs, classes):
                            bx1 = box[0] * tile_scale + tx
                            by1 = box[1] * tile_scale + ty
                            bx2 = box[2] * tile_scale + tx
                            by2 = box[3] * tile_scale + ty
                            
                            if '_hflip' in aug_name: bx1, bx2 = w_img - bx2, w_img - bx1
                            if '_vflip' in aug_name: by1, by2 = h_img - by2, h_img - by1
                            
                            x1, y1 = bx1 / scale, by1 / scale
                            x2, y2 = bx2 / scale, by2 / scale
                            
                            class_name = CLASS_NAMES.get(cls, f'class_{cls}')
                            all_dets.append(Detection(x1, y1, x2, y2, float(conf) * weight, cls, class_name, source + aug_name))
        return all_dets

    def _is_on_cell(self, image, det):
        x1, y1, x2, y2 = det.bbox
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1: return False
        region = image[y1:y2, x1:x2]
        if region.size == 0: return False
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        return np.mean(hsv[:, :, 1]) > 15

    def _merge_detections(self, all_dets):
        if not all_dets: return []
        dets = sorted(all_dets, key=lambda x: x.confidence, reverse=True)
        merged = []
        used = [False] * len(dets)
        
        for i, det in enumerate(dets):
            if used[i]: continue
            group = [det]
            sources = {det.source.split('_')[0]}
            for j in range(i + 1, len(dets)):
                if used[j]: continue
                other = dets[j]
                xx1, yy1 = max(det.x1, other.x1), max(det.y1, other.y1)
                xx2, yy2 = min(det.x2, other.x2), min(det.y2, other.y2)
                inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                iou = inter / (det.area + other.area - inter + 1e-6)
                dist = np.sqrt((det.center[0] - other.center[0])**2 + (det.center[1] - other.center[1])**2)
                avg_size = (det.width + det.height) / 2
                if iou > 0.2 or dist < avg_size * 0.6:
                    group.append(other)
                    sources.add(other.source.split('_')[0])
                    used[j] = True
            
            votes = len(sources)
            boost = 0.35 if votes >= 6 else (0.28 if votes >= 5 else (0.22 if votes >= 4 else (0.15 if votes >= 3 else (0.08 if votes >= 2 else 0.0))))
            best = max(group, key=lambda x: x.confidence)
            if best.class_name in ALL_ABNORMAL_CLASSES: boost += 0.05
            final_conf = min(best.confidence + boost, 0.99)
            
            avg_x1 = np.mean([d.x1 for d in group])
            avg_y1 = np.mean([d.y1 for d in group])
            avg_x2 = np.mean([d.x2 for d in group])
            avg_y2 = np.mean([d.y2 for d in group])
            merged.append(Detection(avg_x1, avg_y1, avg_x2, avg_y2, final_conf, best.class_id, best.class_name, f"{votes}v", votes))
            used[i] = True
        return merged

    def _filter_and_nms(self, detections, image, orig_w, orig_h):
        scale = orig_w / 2048.0
        min_size = 8 * scale; max_size = 220 * scale
        filtered = []
        for det in detections:
            if det.width < min_size or det.height < min_size: continue
            if det.width > max_size or det.height > max_size: continue
            aspect = max(det.width, det.height) / (min(det.width, det.height) + 1e-6)
            if aspect > 3.0: continue
            if det.x1 < 2 or det.y1 < 2 or det.x2 > orig_w - 2 or det.y2 > orig_h - 2: continue
            if not self._is_on_cell(image, det): continue
            thresh = self.parasite_threshold if det.class_name in ALL_ABNORMAL_CLASSES else self.uninfected_threshold
            if det.confidence < thresh: continue
            filtered.append(det)
            
        dets = sorted(filtered, key=lambda x: (x.class_name != 'Parasitized', -x.confidence))
        keep = []
        for det in dets:
            should_keep = True
            for kept in keep:
                xx1, yy1 = max(det.x1, kept.x1), max(det.y1, kept.y1)
                xx2, yy2 = min(det.x2, kept.x2), min(det.y2, kept.y2)
                inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                iou = inter / (det.area + kept.area - inter + 1e-6)
                if iou > 0.35:
                    if det.class_name in ALL_ABNORMAL_CLASSES and kept.class_name == 'Uninfected':
                        keep.remove(kept)
                        break
                    else:
                        should_keep = False
                        break
            if should_keep: keep.append(det)
        return keep

    def predict(self, image_path, progress_callback=None):
        def update(p):
            if progress_callback: progress_callback(p)

        original = cv2.imread(image_path)
        if original is None: raise ValueError("Could not load image")
        orig_h, orig_w = original.shape[:2]
        
        update(0.05) # "Analyzing..."
        
        analysis = ImageAnalyzer.analyze(original)
        variants = ImageTransforms.get_all_variants(original, analysis)
        
        scale = 1.0
        if orig_w < 2000:
            scale = 2.5 if orig_w < 1200 else 2.0
            
        all_dets = []
        total_variants = len(variants)
        
        # Inference Loop with Progress
        for i, (name, variant, weight) in enumerate(variants):
            # Calculate progress from 10% to 90%
            current_progress = 0.10 + (i / total_variants) * 0.80
            update(current_progress)
            
            if scale > 1:
                variant = cv2.resize(variant, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            dets = self._run_on_image(variant, scale, name, weight)
            all_dets.extend(dets)
            
        update(0.92) # "Merging..."
        merged = self._merge_detections(all_dets)
        
        update(0.96) # "Filtering..."
        final = self._filter_and_nms(merged, original, orig_w, orig_h)
        
        update(1.0) # Done
        return final, original

    def visualize(self, image, detections):
        vis = image.copy()
        asexual = 0
        gametocytes = 0
        babesia = 0
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            label = det.class_name
            if det.class_name in MALARIA_ASEXUAL_CLASSES:
                asexual += 1
                color = (0, 0, 255)  # red
            elif det.class_name in GAMETOCYTE_CLASS:
                gametocytes += 1
                color = (255, 0, 255)  # magenta
            elif det.class_name in MIMIC_CLASSES:
                babesia += 1
                color = (0, 165, 255)  # orange
            else:
                color = (255, 200, 0)  # cyan (Uninfected)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 4)
            cv2.putText(vis, f"{label} {det.confidence:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return vis, asexual

# ==============================================================================
# APPLICATION GUI
# ==============================================================================

class MalariaProUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Window Configuration
        self.title("MalariaSense — Research Prototype")
        self.geometry("480x320")
        self.resizable(False, False)
        ctk.set_appearance_mode("Dark")
        
        # State Data
        self.model_path = MODEL_PATH
        self.detector = None
        self.selected_path = None
        self.is_running = False
        
        # Initialize Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.main_frame = ctk.CTkFrame(self, fg_color=COLOR_BG, corner_radius=0)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Show Loading Screen first, then Home
        self.build_loading_view()
        threading.Thread(target=self.init_system, daemon=True).start()

    def clear_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    # --- VIEWS ---

    def build_loading_view(self):
        self.clear_frame()
        ctk.CTkLabel(self.main_frame, text="MALARIASENSE", font=("Roboto", 24, "bold"), text_color=COLOR_ACCENT).pack(pady=(110, 10))
        ctk.CTkLabel(self.main_frame, text="Initializing System...", font=("Arial", 12), text_color="gray").pack()
        self.spinner = ctk.CTkProgressBar(self.main_frame, width=200, height=4, progress_color=COLOR_ACCENT)
        self.spinner.pack(pady=20)
        self.spinner.set(0)
        self.spinner.start()

    def build_home_view(self):
        self.clear_frame()
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        
        # Title Area
        ctk.CTkLabel(self.main_frame, text="READY FOR ANALYSIS", font=("Roboto", 16, "bold"), text_color="white").place(relx=0.5, rely=0.15, anchor="center")
        
        # Load Button
        self.btn_load = ctk.CTkButton(self.main_frame, text="SELECT IMAGE", 
                                    font=("Roboto", 14), height=50, width=180,
                                    fg_color=COLOR_PANEL, hover_color="#3a3a3a",
                                    command=self.select_image)
        self.btn_load.place(relx=0.5, rely=0.4, anchor="center")
        
        # Filename Label
        self.lbl_file = ctk.CTkLabel(self.main_frame, text="No File Selected", font=("Arial", 12), text_color="gray")
        self.lbl_file.place(relx=0.5, rely=0.55, anchor="center")
        
        # Run Button
        self.btn_run = ctk.CTkButton(self.main_frame, text="START ANALYSIS", 
                                   font=("Roboto", 14, "bold"), height=50, width=180,
                                   fg_color=COLOR_ACCENT, state="disabled",
                                   command=self.start_process)
        self.btn_run.place(relx=0.5, rely=0.75, anchor="center")

    def build_progress_view(self):
        self.clear_frame()
        
        ctk.CTkLabel(self.main_frame, text="ANALYZING...", font=("Roboto", 18, "bold"), text_color=COLOR_ACCENT).pack(pady=(80, 20))
        
        self.progress_bar = ctk.CTkProgressBar(self.main_frame, width=300, height=12, corner_radius=6, progress_color=COLOR_ACCENT)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)
        
        self.lbl_status = ctk.CTkLabel(self.main_frame, text="Deep Scanning Cell Morphology...", font=("Arial", 12), text_color="gray")
        self.lbl_status.pack(pady=10)
        
        ctk.CTkLabel(self.main_frame, text="Please Wait. Do Not Power Off.", font=("Arial", 10), text_color="#555").pack(side="bottom", pady=20)

    def build_result_view(self, pil_img, count):
        self.clear_frame()
        
        # Left Side: Image
        self.main_frame.grid_columnconfigure(0, weight=2)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # Resize image for display
        aspect = pil_img.width / pil_img.height
        disp_h = 300
        disp_w = int(disp_h * aspect)
        if disp_w > 300:
            disp_w = 300
            disp_h = int(disp_w / aspect)
            
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(disp_w, disp_h))
        
        img_label = ctk.CTkLabel(self.main_frame, image=ctk_img, text="")
        img_label.grid(row=0, column=0, sticky="ns", padx=10)
        
        # Right Side: Stats
        panel = ctk.CTkFrame(self.main_frame, fg_color=COLOR_PANEL, corner_radius=0)
        panel.grid(row=0, column=1, sticky="nsew")
        
        status_color = COLOR_DANGER if count > 0 else COLOR_SUCCESS
        status_text = "POSITIVE" if count > 0 else "NEGATIVE"
        
        ctk.CTkLabel(panel, text="RESULT", font=("Arial", 12), text_color="gray").pack(pady=(40, 5))
        ctk.CTkLabel(panel, text=status_text, font=("Roboto", 22, "bold"), text_color=status_color).pack(pady=0)
        
        ctk.CTkLabel(panel, text="COUNT", font=("Arial", 12), text_color="gray").pack(pady=(30, 5))
        ctk.CTkLabel(panel, text=str(count), font=("Roboto", 30, "bold"), text_color="white").pack(pady=0)
        
        ctk.CTkButton(panel, text="DONE", fg_color="#444", height=40, command=self.build_home_view).pack(side="bottom", pady=20, padx=10)

    # --- LOGIC ---

    def init_system(self):
        try:
            self.detector = MaximumDetector(self.model_path)
            self.after(0, self.build_home_view)
        except Exception as e:
            self.after(0, lambda: self.show_error(f"Model Error:\n{str(e)}"))

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.bmp;*.jpg;*.png;*.tif")])
        if path:
            self.selected_path = path
            self.lbl_file.configure(text=os.path.basename(path))
            self.btn_run.configure(state="normal")

    def start_process(self):
        self.build_progress_view()
        threading.Thread(target=self.run_inference, daemon=True).start()

    def update_progress_ui(self, val):
        self.after(0, lambda: self.progress_bar.set(val))
        
        # Dynamic friendly text based on progress
        txt = "Initializing..."
        if val > 0.1: txt = "Generating Diagnostics..."
        if val > 0.3: txt = "Scanning Cellular Patterns..."
        if val > 0.6: txt = "Analyzing High-Res Tiles..."
        if val > 0.9: txt = "Verifying Results..."
        
        self.after(0, lambda: self.lbl_status.configure(text=txt))

    def run_inference(self):
        try:
            dets, original = self.detector.predict(self.selected_path, progress_callback=self.update_progress_ui)
            vis, count = self.detector.visualize(original, dets)
            
            # Convert for UI
            img_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            self.after(0, lambda: self.build_result_view(pil_img, count))
            
        except Exception as e:
            print(e)
            self.after(0, lambda: self.show_error("Analysis Failed"))

    def show_error(self, msg):
        self.clear_frame()
        ctk.CTkLabel(self.main_frame, text="ERROR", text_color=COLOR_DANGER, font=("Bold", 20)).pack(pady=50)
        ctk.CTkLabel(self.main_frame, text=msg).pack()

if __name__ == "__main__":
    app = MalariaProUI()
    app.mainloop()