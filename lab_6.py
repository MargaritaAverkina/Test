# Лабораторна робота №6
# Метод Ейлера та метод Рунге–Кутта 4-го порядку
# З можливістю зміни функції та кроку h

import sys
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ================== Безпечне обчислення функції ==================
def compute_f(expr, x, y):
    denominator = y - x**2 * y
    if abs(denominator) < 1e-6:
        raise ZeroDivisionError

    return eval(expr, {
        "__builtins__": None,
        "x": x,
        "y": y,
        "np": np
    })


# ================== Метод Ейлера ==================
def euler_method(expr, x0, y0, h, steps):
    x, y = [x0], [y0]

    for _ in range(steps):
        try:
            y_next = y[-1] + h * compute_f(expr, x[-1], y[-1])
        except:
            break
        x.append(x[-1] + h)
        y.append(y_next)

    return np.array(x), np.array(y)


# ================== Метод Рунге–Кутта 4 ==================
def runge_kutta_4(expr, x0, y0, h, steps):
    x, y = [x0], [y0]

    for _ in range(steps):
        xi, yi = x[-1], y[-1]
        try:
            k1 = h * compute_f(expr, xi, yi)
            k2 = h * compute_f(expr, xi + h/2, yi + k1/2)
            k3 = h * compute_f(expr, xi + h/2, yi + k2/2)
            k4 = h * compute_f(expr, xi + h, yi + k3)
        except:
            break

        y.append(yi + (k1 + 2*k2 + 2*k3 + k4) / 6)
        x.append(xi + h)

    return np.array(x), np.array(y)


# ================== Головне вікно ==================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Лабораторна №6 — Чисельні методи")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # -------- Ввід функції та початкових значень --------
        input_layout = QHBoxLayout()

        input_layout.addWidget(QLabel("y' ="))
        self.func_input = QLineEdit("(x*y**2 + x)/(y - x**2*y)")
        input_layout.addWidget(self.func_input)

        input_layout.addWidget(QLabel("x₀:"))
        self.x0_input = QLineEdit("1.2")
        input_layout.addWidget(self.x0_input)

        input_layout.addWidget(QLabel("y₀:"))
        self.y0_input = QLineEdit("0.5")
        input_layout.addWidget(self.y0_input)

        layout.addLayout(input_layout)

        # -------- Ввід похибок --------
        h_layout = QHBoxLayout()

        h_layout.addWidget(QLabel("h₁:"))
        self.h1_input = QLineEdit("0.1")
        h_layout.addWidget(self.h1_input)

        h_layout.addWidget(QLabel("h₂:"))
        self.h2_input = QLineEdit("0.2")
        h_layout.addWidget(self.h2_input)

        layout.addLayout(h_layout)

        # -------- Кнопки --------
        button_layout = QHBoxLayout()

        self.btn_h1 = QPushButton("Побудувати h₁")
        self.btn_h2 = QPushButton("Побудувати h₂")
        self.btn_all = QPushButton("Побудувати разом")

        self.btn_h1.clicked.connect(lambda: self.plot_graph("h1"))
        self.btn_h2.clicked.connect(lambda: self.plot_graph("h2"))
        self.btn_all.clicked.connect(lambda: self.plot_graph("all"))

        button_layout.addWidget(self.btn_h1)
        button_layout.addWidget(self.btn_h2)
        button_layout.addWidget(self.btn_all)

        layout.addLayout(button_layout)

        # -------- Графік --------
        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.grid(alpha=0.3)

        self.canvas.draw()

    def plot_graph(self, mode):
        self.ax.clear()

        expr = self.func_input.text()
        x0 = float(self.x0_input.text())
        y0 = float(self.y0_input.text())
        h1 = float(self.h1_input.text())
        h2 = float(self.h2_input.text())
        steps = 25

        def draw(h, color_e, color_r, marker_e, marker_r, label_h):
            xe, ye = euler_method(expr, x0, y0, h, steps)
            xr, yr = runge_kutta_4(expr, x0, y0, h, steps)

            self.ax.plot(
                xe, ye, linestyle='--', marker=marker_e,
                color=color_e, linewidth=2,
                label=f"Ейлер h={label_h}"
            )

            self.ax.plot(
                xr, yr, linestyle='-', marker=marker_r,
                color=color_r, linewidth=3,
                label=f"Рунге–Кутта h={label_h}"
            )

        if mode == "h1":
            draw(h1, "tab:blue", "navy", "o", "^", h1)
        elif mode == "h2":
            draw(h2, "orange", "red", "s", "D", h2)
        else:
            draw(h1, "tab:blue", "navy", "o", "^", h1)
            draw(h2, "orange", "red", "s", "D", h2)

        self.ax.set_title("Чисельний розв'язок диференціального рівняння")
        self.ax.legend()
        self.ax.grid(alpha=0.3)

        self.ax.relim()
        self.ax.autoscale()

        self.canvas.draw()


# ================== Запуск ==================
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())
