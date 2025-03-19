from __future__ import annotations
import tkinter as tk
from PIL import ImageGrab, ImageTk, Image
import time
import numpy as np
import cv2
import mss
from functools import lru_cache
from loguru import logger
from pybind11_pixelmatch import write_image, read_image
import json
import sys
import os


class ScreenMonitor:
    def __init__(self, root, config_path: str = None):
        self.root = root
        self.root.title("屏幕对比查看器")
        self.root.geometry("600x500")

        # bring to front
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.focus_force()
        self.root.after(100, lambda: self.root.attributes("-topmost", False))

        self.config_path: str | None = config_path

        # 控制面板
        control_panel = tk.Frame(root)
        control_panel.pack(pady=5, fill=tk.X)
        self.select_btn = tk.Button(
            control_panel,
            text="选择区域(可选两次)",
            command=self.select_area,
        )
        self.select_btn.pack(side=tk.LEFT, padx=10)
        self.is_monitoring = tk.BooleanVar(value=True)
        self.is_monitoring_btn = tk.Checkbutton(
            control_panel,
            text="监听",
            variable=self.is_monitoring,
        )
        self.is_monitoring_btn.pack(side=tk.LEFT, padx=10)
        self.show_original = tk.BooleanVar(value=True)
        self.show_original_btn = tk.Checkbutton(
            control_panel,
            text="原图",
            variable=self.show_original,
        )
        self.show_original_btn.pack(side=tk.LEFT, padx=10)

        self.split_x = None
        self.warp_matrix = None

        self.align_y = tk.BooleanVar(value=True)
        self.align_y_btn = tk.Checkbutton(
            control_panel,
            text="y-补齐",
            variable=self.align_y,
        )
        self.align_y_btn.pack(side=tk.LEFT, padx=10)

        self.save_btn = tk.Button(
            control_panel,
            text="保存",
            command=self.save,
        )
        self.save_btn.pack(side=tk.LEFT, padx=10)
        self.save_img: bool = True

        self.shift_limit: int = 30
        self.shift_x = tk.IntVar(value=0)
        self.shift_x_slider = tk.Scale(
            control_panel,
            from_=-self.shift_limit,
            to=self.shift_limit,
            orient=tk.HORIZONTAL,
            variable=self.shift_x,
            label="横向微调",
        )
        self.shift_x_slider.pack(side=tk.LEFT, padx=10)
        self.shift_y = tk.IntVar(value=0)
        self.shift_y_slider = tk.Scale(
            control_panel,
            from_=-self.shift_limit,
            to=self.shift_limit,
            orient=tk.HORIZONTAL,
            variable=self.shift_y,
            label="纵向微调",
        )
        self.shift_y_slider.pack(side=tk.LEFT, padx=10)

        self.threshold = tk.DoubleVar(value=0.1)
        self.threshold_slider = tk.Scale(
            control_panel,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            digits=2,
            orient=tk.HORIZONTAL,
            variable=self.threshold,
            label="差异阈值",
        )
        self.threshold_slider.pack(side=tk.LEFT, padx=10)

        self.include_aa = tk.BooleanVar(value=True)
        self.include_aa_btn = tk.Checkbutton(
            control_panel,
            text="统计抗锯齿",
            variable=self.include_aa,
        )
        self.include_aa_btn.pack(side=tk.LEFT, padx=10)
        self.split_ins_del = tk.BooleanVar(value=True)
        self.split_ins_del_btn = tk.Checkbutton(
            control_panel,
            text="区分增删",
            variable=self.split_ins_del,
        )
        self.split_ins_del_btn.pack(side=tk.LEFT, padx=10)

        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.label = tk.Label(self.frame)
        self.label.pack(expand=True, fill=tk.BOTH)

        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.monitor_areas: list[tuple[int, int, int, int]] = []

        self.inputs = []  # 输入的屏幕截图
        self.output = None  # 输出显示

        # 监听窗口大小变化
        self.root.bind("<Configure>", self.on_window_resize)
        self.last_width = root.winfo_width()
        self.last_height = root.winfo_height()

        self.overlay = None
        self.active_canvas = None

        self.sleep_ms: int = 20
        self.debug: bool = False

        self.update_monitor()

    def load_config(self):
        path = self.config_path
        if not path or not os.path.exists(path):
            return False
        logger.info(f"加载配置 {path}")
        with open(path, "r") as f:
            config = json.load(f)
        self.monitor_areas = config.get("monitor_areas", [])
        self.show_original.set(config.get("show_original", True))
        self.shift_x.set(config.get("shift_x", 0))
        self.shift_y.set(config.get("shift_y", 0))
        self.threshold.set(config.get("threshold", 0.1))
        self.include_aa.set(config.get("include_aa", True))
        self.split_ins_del.set(config.get("split_ins_del", True))
        return True

    def save(self):
        if self.save_img and self.inputs:
            for idx, img in enumerate(self.inputs):
                path = f"build/img{idx+1}.png"
                write_image(path, img)
                logger.info(f"保存图片 {path}")
        if self.config_path:
            path = os.path.abspath(self.config_path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(
                    {
                        "monitor_areas": self.monitor_areas,
                        "show_original": self.show_original.get(),
                        "shift_x": self.shift_x.get(),
                        "shift_y": self.shift_y.get(),
                        "threshold": self.threshold.get(),
                        "include_aa": self.include_aa.get(),
                        "split_ins_del": self.split_ins_del.get(),
                    },
                    f,
                    indent=4,
                    ensure_ascii=False,
                )
            logger.info(f"保存配置到 {path}")

    def __destroy_overlay(self):
        if self.overlay is not None:
            self.overlay.canvas.destroy()
            self.overlay.destroy()
            self.overlay = None

    def select_area(self):
        self.__destroy_overlay()
        curr_geom = self.root.winfo_geometry()
        curr_geom = self.root.winfo_geometry()
        x, y = [int(x) for x in curr_geom.split("+")[-2:]]
        w, h, x, y = which_monitor(x, y)

        overlay = tk.Toplevel(self.root)
        overlay.geometry(f"{w}x{h}+{x}+{y}")
        overlay.overrideredirect(True)
        overlay.attributes("-topmost", True)
        overlay.attributes("-alpha", 0.7)
        overlay.configure(bg="grey")
        canvas = tk.Canvas(overlay, bg="grey", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.monitor = w, h, x, y
        canvas.bind("<ButtonPress-1>", self.on_press)
        canvas.bind("<B1-Motion>", self.on_drag)
        canvas.bind("<ButtonRelease-1>", self.on_release)
        canvas.create_text(
            w // 2,
            h // 2,
            text="点击并拖动鼠标选择监听区域",
            fill="red",
            font=("Arial", 24),
        )
        self.overlay = overlay
        overlay.canvas = canvas

    def on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.active_canvas = event.widget
        self.selection_rect = self.active_canvas.create_rectangle(
            self.start_x,
            self.start_y,
            self.start_x,
            self.start_y,
            outline="red",
            width=2,
        )

    def on_drag(self, event):
        if self.active_canvas != event.widget:
            return
        self.end_x = event.x
        self.end_y = event.y
        self.active_canvas.coords(
            self.selection_rect,
            self.start_x,
            self.start_y,
            self.end_x,
            self.end_y,
        )

    def on_release(self, event):
        if self.active_canvas != event.widget or self.overlay is None:
            return
        self.end_x = event.x
        self.end_y = event.y
        ax, ay = self.overlay.canvas.monitor[-2:]
        x0, x1 = sorted([self.start_x, self.end_x])
        y0, y1 = sorted([self.start_y, self.end_y])
        if x1 - x0 < 40 or y1 - y0 < 40:
            logger.warning(f"窗口太小，忽略: {x1 - x0}x{y1 - y0}")
            return
        area = (ax + x0, ay + y0, ax + x1, ay + y1)
        self.monitor_areas.append(area)
        self.__normalize_monitor_areas()

        logger.info(f"anchor: ({ax},{ay})")
        logger.info(f"window: ({x0},{y0}) -- ({x1},{y1})")
        logger.info(f"monitor area: {area}")
        self.__destroy_overlay()

    def __normalize_monitor_areas(self):
        """
        如果上下要对齐
        """
        if len(self.monitor_areas) <= 1:
            return
        if len(self.monitor_areas) > 2:
            self.monitor_areas = self.monitor_areas[-2:]
        if not self.align_y.get():
            return
        ymin, ymax = self.monitor_areas[0][1::2]
        for _, y0, _, y1 in self.monitor_areas[1:]:
            ymin = min(ymin, y0)
            ymax = max(ymax, y1)
        self.monitor_areas = [
            (x0, ymin, x1, ymax) for x0, _, x1, _ in self.monitor_areas
        ]

    def on_window_resize(self, event):
        if event.widget == self.root and (
            self.last_width != self.root.winfo_width()
            or self.last_height != self.root.winfo_height()
        ):
            self.last_width = self.root.winfo_width()
            self.last_height = self.root.winfo_height()
            self.resize_and_display_images()

    def resize_and_display_images(self):
        if len(self.inputs) == 0:
            return
        if len(self.inputs) == 1:
            img = self.inputs[0]
            if self.split_x is None:
                self.split_x, _, _ = split_image(img)
            img1 = img[:, : self.split_x, ...]
            img2 = img[:, self.split_x :, ...]
        else:
            assert len(self.inputs) == 2
            img1, img2 = self.inputs

        if self.warp_matrix is None:
            self.warp_matrix, img1 = align_image(img1, img2)
        else:
            img1 = cv2.warpAffine(
                img1,
                self.warp_matrix,
                (img2.shape[1], img2.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            )

        sx, sy = self.shift_x.get(), self.shift_y.get()
        if sx != 0 or sy != 0:
            img1, img2 = self.shift_image(img1, img2, sx, sy)

        if img1.shape[0] % 2 != 0:
            img1 = img1[1:, ...]
            img2 = img2[1:, ...]

        _, diff = diff_image(
            img1,
            img2,
            threshold=self.threshold.get(),
            include_aa=self.include_aa.get(),
            diff_color_alt="rgba(0,255,0,255)" if self.split_ins_del.get() else None,
        )

        bw = 5
        img1 = cv2.copyMakeBorder(
            img1, bw, bw, bw, bw, cv2.BORDER_CONSTANT, value=[255, 0, 0, 255]
        )
        img2 = cv2.copyMakeBorder(
            img2, bw, bw, bw, bw, cv2.BORDER_CONSTANT, value=[0, 255, 0, 255]
        )
        diff = cv2.copyMakeBorder(
            diff, bw, bw, bw, bw, cv2.BORDER_CONSTANT, value=[0, 0, 0, 255]
        )
        w, h = self.last_width - 10, self.last_height - 80
        if self.show_original.get():
            stacked_img = np.hstack((img1, img2))
            stacked_img = cv2.resize(
                stacked_img, (stacked_img.shape[1] // 2, stacked_img.shape[0] // 2)
            )
            stacked_img = np.vstack((stacked_img, diff))
            # TODO，横着放
        else:
            stacked_img = diff
        original_height, original_width = stacked_img.shape[:2]
        aspect_ratio = original_width / original_height
        if w / h > aspect_ratio:
            new_height = h
            new_width = int(h * aspect_ratio)
        else:
            new_width = w
            new_height = int(w / aspect_ratio)
        resized_img = cv2.resize(stacked_img, (new_width, new_height))

        self.output = Image.fromarray(resized_img)
        self.photo = ImageTk.PhotoImage(self.output)
        self.label.config(image=self.photo)

    def update_monitor(self):
        if self.overlay is not None:
            self.root.after(self.sleep_ms, self.update_monitor)
            return
        try:
            if self.is_monitoring.get() and self.monitor_areas:
                self.inputs = [
                    np.array(ImageGrab.grab(bbox=bbox)) for bbox in self.monitor_areas
                ]
            self.resize_and_display_images()
        except Exception as e:
            logger.error(f"截图错误: {e}")
        self.root.after(self.sleep_ms, self.update_monitor)

    def shift_image(self, img1, img2, sx, sy):
        """
        如果 sx, sy 为 0，说明 img1 和 img2 是对齐的
        现在需要 img 偏移 sx,sy 能够和 img2 对齐
        请对齐并 crop 两个图片
        """
        assert img1.shape == img2.shape
        assert sx != 0 or sy != 0
        h, w = img1.shape[:2]
        if sx > 0:  # img1 needs to shift right
            x1_start, x1_end = sx, w
            x2_start, x2_end = 0, w - sx
        else:  # img1 needs to shift left (sx < 0)
            x1_start, x1_end = 0, w + sx
            x2_start, x2_end = -sx, w
        if sy > 0:  # img1 needs to shift down
            y1_start, y1_end = sy, h
            y2_start, y2_end = 0, h - sy
        else:  # img1 needs to shift up (sy < 0)
            y1_start, y1_end = 0, h + sy
            y2_start, y2_end = -sy, h
        img1_cropped = img1[y1_start:y1_end, x1_start:x1_end, ...].copy()
        img2_cropped = img2[y2_start:y2_end, x2_start:x2_end, ...].copy()
        assert (
            img1_cropped.shape == img2_cropped.shape
        ), f"Shapes don't match after shifting: {img1_cropped.shape} vs {img2_cropped.shape}"
        return img1_cropped, img2_cropped


def which_monitor(x, y) -> tuple[int, int, int, int]:
    with mss.mss() as sct:
        for m in sct.monitors[1:]:
            x0 = m["left"]
            w = m["width"]
            y0 = m["top"]
            h = m["height"]
            if x0 <= x <= x0 + w and y0 <= y <= y0 + h:
                return w, h, x0, y0
    raise Exception("无法确认窗口所在显示器")


def img2gray(img: np.ndarray) -> np.ndarray:
    """
    Convert an image (gray, RGB, RGBA) to grayscale.
    """
    if len(img.shape) != 3:
        return img
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)


def img2rgb(img: np.ndarray) -> np.ndarray:
    assert len(img.shape) == 3
    if img.shape[2] == 3:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)


def split_image(img: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """
    img 是左右两张图片拼接而成，需要找到左右的分界线，使得图片能较好地分割为两部分
    Split an image that contains two images side by side.

    This function attempts to find the boundary between two images that are
    concatenated horizontally. It looks for vertical lines with minimal variance
    or sharp transitions that might indicate a boundary.

    Args:
        img (np.ndarray): The input image, which is a concatenation of two images.

    Returns:
        tuple[np.ndarray, np.ndarray]: Left and right images after splitting.
    """
    _, width = img.shape[:2]

    # Convert to grayscale if the image is in color
    gray = img2gray(img)

    # Compute vertical variance (variance along each column)
    # Lower variance might indicate a boundary between images
    var_profile = np.var(gray, axis=0)

    # Apply a median filter to smooth the variance profile
    var_profile = cv2.medianBlur(var_profile.astype(np.float32), 5)

    # Find potential split points (look for local minima in the middle section)
    # Only consider the middle portion (e.g., 40%-60% of width) to avoid edges
    middle_start = int(width * 0.4)
    middle_end = int(width * 0.6)
    middle_section = var_profile[middle_start:middle_end]

    # Find local minima in the middle section
    # A local minimum is a point where the value is smaller than its neighbors
    candidates = []
    for i in range(1, len(middle_section) - 1):
        if (
            middle_section[i] < middle_section[i - 1]
            and middle_section[i] < middle_section[i + 1]
        ):
            candidates.append((middle_start + i, middle_section[i]))

    # If no candidates found, try another approach: look for sharp transitions
    if not candidates:
        # Compute horizontal gradient to find vertical edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.abs(sobelx)

        # Sum along columns to get a profile of vertical edge strength
        edge_profile = np.sum(abs_sobelx, axis=0)

        # Find the strongest edge in the middle section
        middle_edges = edge_profile[middle_start:middle_end]
        split_idx = middle_start + np.argmax(middle_edges)
    else:
        # Sort candidates by variance value (ascending)
        candidates.sort(key=lambda x: x[1])
        # Choose the candidate with lowest variance
        split_idx = candidates[0][0]

    # If all else fails, split in the middle
    if not candidates and np.all(edge_profile == 0):
        split_idx = width // 2

    # Split the image
    left_img = img[:, :split_idx, ...]
    right_img = img[:, split_idx:, ...]
    return split_idx, left_img, right_img


def align_image(
    img1: np.ndarray,
    img2: np.ndarray,
    *,
    motion: int = cv2.MOTION_TRANSLATION,
) -> tuple[np.ndarray, np.ndarray]:
    """
    align img1 to img2, return aligned image of img1
    """
    scale_factor = 0.25
    gray1_orig = img2gray(img1)
    gray2_orig = img2gray(img2)
    gray1 = cv2.resize(
        gray1_orig, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA
    )
    gray2 = cv2.resize(
        gray2_orig, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA
    )

    try:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            50,
            1e-4,
        )
        _, warp_matrix = cv2.findTransformECC(
            gray2,
            gray1,
            warp_matrix,
            motion,
            criteria,
            None,
            5,
        )
        warp_matrix[0, 2] /= scale_factor
        warp_matrix[1, 2] /= scale_factor
        if True:  # 如果需要微调则设为True
            criteria_fine = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-9)
            _, warp_matrix = cv2.findTransformECC(
                gray2_orig, gray1_orig, warp_matrix, motion, criteria_fine, None, 1
            )
    except Exception:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    aligned = cv2.warpAffine(
        img1,
        warp_matrix,
        (img2.shape[1], img2.shape[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )
    return warp_matrix, aligned


@lru_cache(maxsize=None)
def diff_image_options(
    threshold: float = 0.1,
    include_aa: bool = False,
    alpha: float = 0.1,
    aa_color: str = "rgba(255,255,0,255)",
    diff_color: str = "rgba(255,0,0,255)",
    diff_color_alt: str | None = None,
    diff_mask: bool = False,
):
    """
    online doc: https://github.com/jwmcglynn/pixelmatch-cpp17?tab=readme-ov-file#pixelmatchimg1-img2-output-width-height-strideinpixels-options
    Create options for image difference visualization.

    Args:
        threshold: Threshold for considering pixels different
        include_aa: Whether to include anti-aliasing in the comparison
        alpha: Alpha value for visualization
        aa_color: Color for anti-aliasing differences
        diff_color: Color for differences
        diff_color_alt: Alternative color for differences
        diff_mask: Whether to create a mask for differences

    Returns:
        dict: Options for image difference visualization
    """
    from pybind11_pixelmatch import normalize_color, Options

    options = Options()
    options.threshold = threshold
    options.includeAA = include_aa
    options.alpha = alpha
    options.aaColor = normalize_color(aa_color)
    options.diffColor = normalize_color(diff_color)
    options.diffColorAlt = normalize_color(diff_color_alt)
    options.diffMask = diff_mask
    return options


def diff_image(
    img0: np.ndarray | str,
    img1: np.ndarray | str,
    *,
    with_label: bool = False,
    verbose: bool = False,
    output: str | None = None,
    **options,
) -> tuple[int, np.ndarray]:
    """
    Calculate and visualize the difference between two images.

    Args:
        img0: First image
        img1: Second image
        with_label: Whether to include labels in the visualization
        verbose: Whether to print verbose information
        options: Options for image difference visualization

    Returns:
        tuple[int, np.ndarray]: Number of different pixels and difference image
    """
    assert img0.ndim == img1.ndim == 3
    assert img0.shape == img1.shape
    assert img0.shape[-1] == img1.shape[-1] == 4
    if verbose:
        logger.info(f"image shape: {img0.shape}")
    from pybind11_pixelmatch import pixelmatch

    options = diff_image_options(**options)
    if verbose:
        logger.info(f"options: {options}")

    diff = np.zeros(img1.shape, dtype=img1.dtype)
    num = pixelmatch(img0, img1, output=diff, options=options)
    if verbose:
        h, w = img0.shape[:2]
        logger.info(f"unmatched_pixels: {num:,} (total: {w * h:,})")
    if with_label:
        cv2.putText(
            diff,
            f"#unmatched_pixels: {num:,}",
            (10, 40),
            cv2.FONT_HERSHEY_COMPLEX,
            1.0,
            color=(255, 0, 0),
            thickness=1,
        )
    if output:
        write_image(output, diff)
    return num, diff


if __name__ == "__main__":
    """
    安装：
        pip install loguru mss opencv-python pillow>=8.0.0 pybind11-pixelmatch

    使用
        打开界面后，有三种使用流程：
        1.  画框监听
            -   点击选择区域，在当前显示器画一个框（会自动分为左右两块区域，做 diff）
            -  点击选择区域，画一个框，操作两次（会 diff 这两块区域）
        2.  传入图片查看 diff（和上面的框类似，直接传入图片）
            -   python3 main.py img.png
            -   python3 main.py img1.png img2.png
        3.  加载配置启动
            -   python3 main.py config.json，点击保存则会存储配置
    """
    root = tk.Tk()
    root.iconbitmap("favicon.ico")
    config_path = None
    if len(sys.argv) >= 3:
        app = ScreenMonitor(root)
        app.inputs = [read_image(p) for p in sys.argv[1:3]]
        app.save_img = False
    elif len(sys.argv) > 1:
        path = sys.argv[1]
        if path.endswith(".json"):
            app = ScreenMonitor(root, config_path)
        else:
            app = ScreenMonitor(root)
            app.inputs = [read_image(path)]
            app.save_img = False
    else:
        app = ScreenMonitor(root)
    root.mainloop()
