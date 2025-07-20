import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import json
import os
import onnxruntime as rt
from PIL import Image, ImageTk
from collections import deque
import threading
import queue
import logging
import time
from tensorflow.keras.models import load_model


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

class SignLanguageTranslator:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuroTalk")
        self.root.geometry("1920x1080")
        self.root.minsize(1024, 600)
        
        self.current_model_type = 'RSL'  
        self.current_config = None
        self.processing_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=5)
        self.cap = None
        self.keras_model = None
        self.session = None
        
        self.colors = {
            'background': '#2C3E50',
            'primary': '#3498DB',
            'secondary': '#34495E',
            'text': '#ECF0F1',
            'accent': '#2ECC71',
            'warning': '#E74C3C'
        }
        
        self.setup_ui()
        self.load_configs()
        self.init_video_processing()
        self.setup_bindings()
        
        self.periodic_checks()

    def setup_ui(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.control_panel = ttk.Frame(self.main_frame, width=300)
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=15, pady=15)
        
        self.video_panel = ttk.Frame(self.main_frame)
        self.video_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=15, pady=15)

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        
        self.create_device_selector()
        self.create_settings_controls()
        self.create_model_switcher()
        self.create_gesture_display()
        self.create_status_bar()

    def configure_styles(self):
        self.style.configure('TFrame', background=self.colors['background'])
        self.style.configure(
            'Header.TLabel',
            font=('Helvetica', 14, 'bold'),
            foreground=self.colors['primary'],
            background=self.colors['background']
        )
        self.style.configure(
            'Primary.TButton',
            font=('Helvetica', 10),
            foreground=self.colors['text'],
            background=self.colors['primary'],
            borderwidth=0,
            padding=8
        )
        self.style.map(
            'Primary.TButton',
            background=[('active', self.colors['primary']), ('disabled', '#7F8C8D')]
        )
        self.style.configure(
            'TCombobox',
            fieldbackground=self.colors['secondary'],
            background=self.colors['secondary'],
            foreground=self.colors['text']
        )

    def create_device_selector(self):
        container = ttk.Frame(self.control_panel)
        container.pack(fill=tk.X, pady=10)
        
        ttk.Label(container, 
                text="Выберите камеру:", 
                style='Header.TLabel').pack(anchor=tk.W)
        
        self.camera_selector = ttk.Combobox(container, state='readonly')
        self.camera_selector.pack(fill=tk.X, pady=5)
        
        ttk.Button(container,
                 text="Обновить список",
                 command=self.refresh_devices,
                 style='Primary.TButton').pack(pady=5)

    def create_settings_controls(self):
        container = ttk.Frame(self.control_panel)
        container.pack(fill=tk.X, pady=10)

        ttk.Label(container, text="Разрешение:").pack(anchor=tk.W)
        self.resolution_selector = ttk.Combobox(
            container,
            values=['640x480', '1280x720', '1920x1080'],
            state='readonly'
        )
        self.resolution_selector.current(1)
        self.resolution_selector.pack(fill=tk.X, pady=5)

        ttk.Label(container, text="Частота кадров:").pack(anchor=tk.W)
        self.fps_selector = ttk.Combobox(
            container,
            values=['15', '30', '60'],
            state='readonly'
        )
        self.fps_selector.current(1)
        self.fps_selector.pack(fill=tk.X, pady=5)

        btn_frame = ttk.Frame(container)
        btn_frame.pack(pady=10)
        self.start_btn = ttk.Button(
            btn_frame,
            text="Старт",
            style='Primary.TButton',
            command=self.start_processing
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(
            btn_frame,
            text="Стоп",
            style='Primary.TButton',
            state=tk.DISABLED,
            command=self.stop_processing
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

    def create_model_switcher(self):
        container = ttk.Frame(self.control_panel)
        container.pack(fill=tk.X, pady=10)
        
        self.model_switch = ttk.Combobox(
            container,
            values=['Русский ЖЯ', 'Американский ЖЯ'],
            state='readonly'
        )
        self.model_switch.current(0)
        self.model_switch.pack(side=tk.LEFT, padx=5)
        self.model_switch.bind('<<ComboboxSelected>>', self.switch_model)
        
        ttk.Button(container,
                 text="Настройки",
                 command=self.open_settings,
                 style='Primary.TButton').pack(side=tk.LEFT, padx=5)

    def create_gesture_display(self):
        self.video_canvas = tk.Canvas(
            self.video_panel,
            bg=self.colors['background'],
            bd=0,
            highlightthickness=0
        )
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.current_gesture_label = ttk.Label(
            self.video_panel,
            text="🖐️ Текущий жест: -",
            font=('Helvetica', 16),
            foreground=self.colors['accent'],
            background=self.colors['background']
        )
        self.current_gesture_label.pack(pady=10)
        
        self.history_label = ttk.Label(
            self.video_panel,
            text="📜 История: -",
            font=('Helvetica', 12),
            foreground=self.colors['text'],
            background=self.colors['background']
        )
        self.history_label.pack()

    def create_status_bar(self):
        self.status_bar = ttk.Label(
            self.root,
            text="Готов к работе",
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=('Helvetica', 9),
            foreground=self.colors['text'],
            background=self.colors['secondary']
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_bindings(self):
        self.root.bind('<Configure>', self.on_window_resize)
        self.root.bind('<Escape>', lambda e: self.on_close())
        self.root.bind('<F5>', lambda e: self.refresh_devices())

    def on_window_resize(self, event):
        if self.processing_event.is_set() and not self.frame_queue.empty():
            self.update_gui()

    def init_video_processing(self):
        self.cap = None
        self.video_thread = None
        self.buffer = deque(maxlen=32)
        self.gesture_history = deque(maxlen=10)

    def load_configs(self):
        config_paths = {
            'RSL': r'configs/config.json',
            'ASL': r'configs/lastconfig.json'
        }
        
        try:
            for path in config_paths.values():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Конфиг не найден: {path}")
            
            with open(config_paths['RSL'], 'r') as f:
                self.config_rsl = json.load(f)
            with open(config_paths['ASL'], 'r') as f:
                self.config_asl = json.load(f)
            
            self.current_config = (
                self.config_rsl 
                if self.current_model_type == 'RSL' 
                else self.config_asl
            )
            
            self.validate_configs()
            self.detect_devices()
            self.init_model()
            
        except Exception as e:
            self.show_error(f"Ошибка конфигурации: {str(e)}")
            self.root.destroy()

    def validate_configs(self):
        required_keys = {
            'RSL': ['path_to_model', 'threshold', 'topk', 
                   'path_to_class_list', 'window_size', 'provider'],
            'ASL': ['path_to_model', 'threshold', 'topk', 
                   'path_to_class_list', 'window_size']
        }
        
        for model_type in ['RSL', 'ASL']:
            config = self.config_rsl if model_type == 'RSL' else self.config_asl
            for key in required_keys[model_type]:
                if key not in config:
                    raise KeyError(f"Отсутствует ключ {key} в {model_type} конфиге")

    def init_model(self):
        try:
            self.current_config = (
                self.config_rsl 
                if self.current_model_type == 'RSL' 
                else self.config_asl
            )
            
            if self.current_model_type == 'RSL':
                if not os.path.exists(self.config_rsl['path_to_model']):
                    raise FileNotFoundError("Файл модели RSL не найден")

                options = rt.SessionOptions()
                options.intra_op_num_threads = 4
                options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

                self.session = rt.InferenceSession(
                    self.config_rsl['path_to_model'],
                    sess_options=options,
                    providers=[self.config_rsl['provider']]
                )
                self.input_name = self.session.get_inputs()[0].name
                logging.info(f"ONNX модель загружена. Входное имя: {self.input_name}")

            else:
                if not os.path.exists(self.config_asl['path_to_model']):
                    raise FileNotFoundError("Файл модели ASL не найден")

                self.keras_model = load_model(self.config_asl['path_to_model'])
                self.keras_model.make_predict_function()
                logging.info("Keras модель успешно загружена")

            self.labels = self.load_labels()
            self.buffer = deque(maxlen=self.current_config['window_size'])
            self.status_bar.config(text="Модель инициализирована")

        except Exception as e:
            self.show_error(f"Ошибка инициализации модели: {str(e)}")
            self.root.destroy()

    def load_labels(self):
        label_path = self.current_config['path_to_class_list']
        labels = {}
        
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split('\t' if self.current_model_type == 'RSL' else None)
                    if len(parts) < 2:
                        raise ValueError(f"Ошибка формата в строке {line_num}")
                    labels[int(parts[0])] = parts[1]
            return labels
        except Exception as e:
            self.show_error(f"Ошибка загрузки меток: {str(e)}")
            return {}
    def detect_devices(self):
        self.cameras = []
        max_checks = 4
        timeout = 2
        
        for index in range(max_checks):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                ret, _ = cap.read()
                if ret:
                    self.cameras.append(f"Камера {index}")
                    cap.release()
                    break
            else:
                cap.release()
                
        self.camera_selector['values'] = self.cameras
        if self.cameras:
            self.camera_selector.current(0)
            self.start_btn.state(['!disabled'])
        else:
            self.start_btn.state(['disabled'])
            self.show_warning("Камеры не обнаружены!")

    def refresh_devices(self):
        self.status_bar.config(text="Сканирование устройств...")
        self.detect_devices()
        self.status_bar.config(text="Готов к работе")
        logging.info("Список камер обновлен")

    def periodic_checks(self):
        if self.cap and not self.cap.isOpened() and self.processing_event.is_set():
            self.stop_processing()
            self.show_error("Потеряно соединение с камерой!")
            
        if self.frame_queue.qsize() > 3:
            logging.warning("Высокая загрузка буфера кадров")
            
        self.root.after(3000, self.periodic_checks)

    def start_processing(self):
        try:
            if not self.cameras:
                raise RuntimeError("Нет доступных камер")

            self.processing_event.set()
            camera_index = int(self.camera_selector.get().split()[-1])
            
            width, height = self.get_best_resolution(camera_index)
            self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, int(self.fps_selector.get()))
            
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

            self.video_thread = threading.Thread(
                target=self.video_processing_loop,
                daemon=True
            )
            self.video_thread.start()

            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_bar.config(text=f"Обработка {width}x{height} @ {self.fps_selector.get()} FPS")
            self.update_gui()

        except Exception as e:
            self.show_error(f"Ошибка запуска: {str(e)}")
            self.stop_processing()

    def video_processing_loop(self):
        last_frame_time = time.time()
        target_fps = int(self.fps_selector.get()) or 30
        frame_interval = 1.0 / target_fps
        
        while self.processing_event.is_set():
            try:
                current_time = time.time()
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)
                    continue
                
                ret, frame = self.cap.read()
                last_frame_time = current_time
                
                if not ret:
                    logging.warning("Пропущен кадр")
                    continue

                display_frame = cv2.resize(frame, (640, 480))
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                if self.frame_queue.qsize() < 3:
                    self.frame_queue.put((display_frame, frame))

                threading.Thread(
                    target=self.process_for_model, 
                    args=(frame.copy(),),
                    daemon=True
                ).start()

            except Exception as e:
                logging.error(f"Ошибка обработки: {str(e)}")
                self.stop_processing()

    def process_for_model(self, frame):
        try:
            if self.current_model_type == 'RSL':
                model_input = cv2.resize(frame, (224, 224))
                model_input = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)
                model_input = model_input.astype(np.float32) / 255.0
            else:
                model_input = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                model_input = cv2.resize(model_input, (28, 28))
                model_input = np.expand_dims(model_input, axis=-1)
                model_input = model_input.astype(np.float32) / 255.0

            self.buffer.append(model_input)

            if len(self.buffer) >= self.current_config['window_size']:
                if self.current_model_type == 'RSL':
                    clip = np.array(self.buffer)
                    clip = np.transpose(clip, (3, 0, 1, 2))
                    outputs = self.session.run(
                        None, 
                        {self.input_name: np.expand_dims(clip, axis=0)}
                    )
                    probabilities = self.softmax(outputs[0][0])
                else:
                    clip = np.array(self.buffer)
                    probabilities = self.keras_model.predict(clip, verbose=0)[0]

                top_indices = np.argsort(probabilities)[-self.current_config['topk']:][::-1]
                valid_gestures = [
                    (self.labels.get(idx, "Unknown"), prob)
                    for idx, prob in zip(top_indices, probabilities[top_indices])
                    if prob > self.current_config['threshold']
                ]

                if valid_gestures:
                    best_gesture = f"{valid_gestures[0][0]} ({valid_gestures[0][1]:.2f})"
                    self.update_gesture_history(best_gesture)

        except Exception as e:
            logging.error(f"Ошибка фоновой обработки: {str(e)}")

    def update_gesture_history(self, gesture):
        timestamp = time.strftime("%H:%M:%S")
        if self.gesture_history:
            last_gesture, last_time = self.gesture_history[-1]
            if gesture == last_gesture and (time.time() - self.last_gesture_time) < 2:
                return
        self.gesture_history.append((gesture, timestamp))
        self.last_gesture_time = time.time()
        logging.info(f"Распознан жест: {gesture}")

    def update_gui(self):
        if self.processing_event.is_set():
            start_time = time.time()
            
            try:
                display_frame, _ = self.frame_queue.get_nowait()
                
                img = Image.fromarray(display_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.video_canvas.delete("all")
                self.video_canvas.create_image(
                    self.video_canvas.winfo_width()//2,
                    self.video_canvas.winfo_height()//2,
                    image=imgtk,
                    anchor=tk.CENTER
                )
                self.video_canvas.image = imgtk

                current_text = "🖐️ Текущий жест: " + (
                    self.gesture_history[-1][0] if self.gesture_history else "-"
                )
                self.current_gesture_label.config(text=current_text)

                history_text = "\n".join(
                    [f"{g[0]} ({g[1]})" for g in self.gesture_history]
                )
                self.history_label.config(text=f"📜 История:\n{history_text}")

            except queue.Empty:
                pass
            
            processing_time = time.time() - start_time
            target_fps = int(self.fps_selector.get()) or 30
            interval = max(1, int(1000 / target_fps - processing_time * 1000))
            
            self.root.after(interval, self.update_gui)

    def stop_processing(self):
        self.processing_event.clear()
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
        self.buffer.clear()
        self.gesture_history.clear()
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_bar.config(text="Обработка остановлена")
        logging.info("Видеопоток корректно остановлен")

    def get_best_resolution(self, camera_index):
        try:
            test_cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            resolutions = [
                (1920, 1080), 
                (1280, 720), 
                (800, 600),
                (640, 480), 
                (320, 240)
            ]
            
            for w, h in resolutions:
                test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                actual_w = test_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_h = test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if actual_w == w and actual_h == h:
                    test_cap.release()
                    return (int(w), int(h))
            
            test_cap.release()
            return (640, 480)
        except:
            return (640, 480)

    def switch_model(self, event=None):
        new_model = 'RSL' if self.model_switch.get() == 'Русский ЖЯ' else 'ASL'
        if new_model != self.current_model_type:
            self.current_model_type = new_model
            self.stop_processing()
            self.load_configs()
            self.init_model()
            logging.info(f"Активирована модель: {new_model}")
            self.status_bar.config(text=f"Модель {new_model} активирована")

    def open_settings(self):
        settings_win = tk.Toplevel(self.root)
        settings_win.title("Расширенные настройки")
        settings_win.geometry("500x400")
        
        ttk.Label(settings_win, text="Макс. размер буфера:").grid(row=0, column=0)
        self.buffer_size_entry = ttk.Entry(settings_win)
        self.buffer_size_entry.insert(0, str(self.frame_queue.maxsize))
        self.buffer_size_entry.grid(row=0, column=1)
        
        ttk.Label(settings_win, text="Порог уверенности:").grid(row=1, column=0)
        self.threshold_entry = ttk.Entry(settings_win)
        self.threshold_entry.insert(0, str(self.current_config['threshold']))
        self.threshold_entry.grid(row=1, column=1)
        
        ttk.Button(settings_win, 
                 text="Сохранить", 
                 command=self.save_advanced_settings).grid(row=5, columnspan=2)

    def save_advanced_settings(self):
        try:
            new_size = int(self.buffer_size_entry.get())
            self.frame_queue = queue.Queue(maxsize=new_size)
            
            new_threshold = float(self.threshold_entry.get())
            self.current_config['threshold'] = new_threshold
            
            messagebox.showinfo("Успех", "Настройки сохранены!")
        except ValueError:
            self.show_error("Некорректные значения настроек")

    def show_error(self, message):
        self.status_bar.config(
            text=f"Ошибка: {message}", 
            foreground=self.colors['warning'],
            font=('Helvetica', 9, 'bold')
        )
        messagebox.showerror("Ошибка", message)
        logging.error(message)

    def show_warning(self, message):
        self.status_bar.config(
            text=f"Внимание: {message}", 
            foreground=self.colors['accent'],
            font=('Helvetica', 9, 'italic')
        )
        logging.warning(message)

    def softmax(self, x):
        max_x = np.max(x)
        e_x = np.exp(x - max_x)
        return e_x / e_x.sum(axis=0)

    def on_close(self):
        self.stop_processing()
        if self.session:
            del self.session
        if self.keras_model:
            del self.keras_model
        self.root.destroy()
        logging.info("Приложение закрыто корректно")

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageTranslator(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
