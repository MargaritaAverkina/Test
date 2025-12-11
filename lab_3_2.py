# Averkina Margarita
# Лабораторна робота №3. Інтерполяція, екстраполяція, апроксимація, регресія
# Завдання 2: Параболічна регресія (y = a*x^2 + b)

import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class RegressionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Параболічна регресія — Лабораторна №3")
        self.setGeometry(200, 200, 700, 500)

        # === Дані варіанту 1 ===
        self.x = np.array([0, 1, 1.5, 2.5, 3, 4.5, 5, 6], dtype=float)
        self.y = np.array([0, 67, 101, 168, 202, 310, 334, 404], dtype=float)

        # === Прапорці, щоб не повторювалось ===
        self.points_shown = False
        self.regression_shown = False

        # === Графік ===
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.ax.set_title("Графік")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.grid(True)

        # === Кнопки ===
        self.show_points_btn = QPushButton("Показати точки")
        self.show_points_btn.clicked.connect(self.show_points)

        self.show_regression_btn = QPushButton("Показати регресію")
        self.show_regression_btn.setEnabled(False)
        self.show_regression_btn.clicked.connect(self.show_regression)

        self.result_label = QLabel("")
        self.result_label.setStyleSheet("font-size: 14px; margin-top: 8px;")

        # === Розмітка ===
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.show_points_btn)
        layout.addWidget(self.show_regression_btn)
        layout.addWidget(self.result_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def show_points(self):
        """Показати точки (лише один раз)."""
        if not self.points_shown:
            self.ax.scatter(self.x, self.y, color="blue", label="Експериментальні точки")
            self.ax.legend()
            self.ax.set_title("Експериментальні дані")
            self.canvas.draw()
            self.points_shown = True
            self.show_regression_btn.setEnabled(True)

    def show_regression(self):
        """Показати параболічну регресію (y = a*x^2 + b), лише один раз."""
        if not self.regression_shown:
            # Створюємо масив для фіктивної змінної (x^2)
            X = np.vstack([self.x ** 2, np.ones(len(self.x))]).T
            a, b = np.linalg.lstsq(X, self.y, rcond=None)[0]

            # Прогнозовані значення
            y_pred = a * self.x ** 2 + b

            # Коефіцієнт детермінації R^2
            ss_res = np.sum((self.y - y_pred) ** 2)
            ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
            r2 = 1 - ss_res / ss_tot

            # Побудова лінії регресії
            x_line = np.linspace(min(self.x), max(self.x), 200)
            y_line = a * x_line ** 2 + b
            self.ax.plot(x_line, y_line, color="orange", label="Параболічна регресія")
            self.ax.legend()

            self.result_label.setText(
                f"Рівняння: y = {a:.3f}x² + {b:.3f}    (R² = {r2:.4f})"
            )

            self.ax.set_title("Параболічна регресія методом найменших квадратів")
            self.canvas.draw()
            self.regression_shown = True


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RegressionApp()
    window.show()
    sys.exit(app.exec())
