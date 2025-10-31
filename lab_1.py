import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, 
    QVBoxLayout, QHBoxLayout, QGridLayout, QStatusBar, QComboBox, QTextEdit
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# --- функція рівняння ---
def f(x):
    return x**3 + x**2 - 5*x + 2


# --- похідна для методу Ньютона ---
def f_derivative(x):
    return 3*x**2 + 2*x - 5


# --- метод дихотомії (бісекції) ---
def bisection(a, b, eps):
    if f(a) * f(b) > 0:
        return None
    while abs(b - a) > eps:
        c = (a + b) / 2
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2


# --- метод простої ітерації ---
def iteration_method(x0, eps):
    def phi(x):
        return (5*x - 2 - x**2)**(1/3)
    x_prev = x0
    x_next = phi(x_prev)
    while abs(x_next - x_prev) > eps:
        x_prev = x_next
        x_next = phi(x_prev)
    return x_next


# --- метод Ньютона ---
def newton_method(x0, eps):
    x = x0
    while abs(f(x)) > eps:
        x = x - f(x) / f_derivative(x)
    return x


# --- головне вікно ---
class RootSolverApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Root Solver — x³ + x² − 5x + 2 = 0")
        self.setWindowIcon(QIcon())
        self.resize(950, 600)
        self.initUI()

    def initUI(self):
        # Поля введення
        self.label_a = QLabel("xmin:")
        self.input_a = QLineEdit("-5")

        self.label_b = QLabel("xmax:")
        self.input_b = QLineEdit("5")

        self.label_eps = QLabel("tolerance (ε):")
        self.input_eps = QLineEdit("1e-4")

        self.label_guess = QLabel("Initial guess (x₀):")
        self.input_guess = QLineEdit("1.0")

        self.method_label = QLabel("Method:")
        self.method_box = QComboBox()
        self.method_box.addItems(["Bisection", "Iteration", "Newton"])

        self.solve_btn = QPushButton("Solve / Scan roots")
        self.solve_btn.clicked.connect(self.solve)

        # Вивід результатів
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)

        # Статус
        self.status = QStatusBar()
        self.status.showMessage("Graph plotted. Select method and click Solve.")

        # Полотно для графіка
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        # --- одразу малюємо графік ---
        self.plot_initial_graph()

        # Макет
        left_layout = QGridLayout()
        left_layout.addWidget(self.label_a, 0, 0)
        left_layout.addWidget(self.input_a, 0, 1)
        left_layout.addWidget(self.label_b, 1, 0)
        left_layout.addWidget(self.input_b, 1, 1)
        left_layout.addWidget(self.label_eps, 2, 0)
        left_layout.addWidget(self.input_eps, 2, 1)
        left_layout.addWidget(self.label_guess, 3, 0)
        left_layout.addWidget(self.input_guess, 3, 1)
        left_layout.addWidget(self.method_label, 4, 0)
        left_layout.addWidget(self.method_box, 4, 1)
        left_layout.addWidget(self.solve_btn, 5, 0, 1, 2)
        left_layout.addWidget(QLabel("Output:"), 6, 0, 1, 2)
        left_layout.addWidget(self.output_box, 7, 0, 1, 2)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 2)
        main_layout.addWidget(self.canvas, 5)

        layout = QVBoxLayout()
        layout.addLayout(main_layout)
        layout.addWidget(self.status)
        self.setLayout(layout)

    def plot_initial_graph(self):
        """Малює базовий графік функції без позначок коренів"""
        try:
            a = float(self.input_a.text())
            b = float(self.input_b.text())
        except ValueError:
            a, b = -5, 5

        x_vals = np.linspace(a, b, 400)
        y_vals = f(x_vals)

        self.ax.clear()
        self.ax.plot(x_vals, y_vals, label="f(x) = x³ + x² − 5x + 2")
        self.ax.axhline(0, color="black", linewidth=1)
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

    def solve(self):
        try:
            a = float(self.input_a.text())
            b = float(self.input_b.text())
            eps = float(self.input_eps.text())
            x0 = float(self.input_guess.text())
        except ValueError:
            self.status.showMessage("Invalid input values!")
            return

        method = self.method_box.currentText()
        roots = []

        if method == "Bisection":
            x_vals = np.linspace(a, b, 1000)
            for i in range(len(x_vals) - 1):
                if f(x_vals[i]) * f(x_vals[i + 1]) < 0:
                    r = bisection(x_vals[i], x_vals[i + 1], eps)
                    if r is not None:
                        roots.append(round(r, 5))

        elif method == "Iteration":
            try:
                r = iteration_method(x0, eps)
                roots.append(round(r, 5))
            except Exception:
                self.output_box.setPlainText("Iteration method diverged or invalid φ(x).")
                return

        elif method == "Newton":
            try:
                r = newton_method(x0, eps)
                roots.append(round(r, 5))
            except Exception:
                self.output_box.setPlainText("Newton method failed (division by zero).")
                return

        # --- оновлення графіка ---
        self.plot_initial_graph()
        if roots:
            for r in roots:
                self.ax.plot(r, f(r), "ro")
                self.ax.text(r, f(r), f"{r:.2f}", color="red", ha="center", va="bottom")

        self.canvas.draw()

        # Вивід результатів
        if roots:
            self.output_box.setPlainText(f"Method: {method}\nFound roots:\n" + "\n".join(map(str, roots)))
            self.status.showMessage(f"{method}: {len(roots)} roots found.")
        else:
            self.output_box.setPlainText("No roots found.")
            self.status.showMessage("No roots detected.")


# --- запуск ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RootSolverApp()
    window.show()
    sys.exit(app.exec())
