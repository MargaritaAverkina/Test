import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QStatusBar,
    QComboBox, QTextEdit
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# === ВАРІАНТИ ФУНКЦІЙ ДЛЯ ЛАБОРАТОРНОЇ №2 ===
def f1(x):
    return 1 / np.sqrt(2 * x**2 + 1)

def f2(x):
    return np.log10(x + 2) / x

def f3(x):
    return 1 / np.sqrt(x**2 + 2.3)


# === ЧИСЕЛЬНІ МЕТОДИ ОБЧИСЛЕННЯ ІНТЕГРАЛІВ ===
def midpoint_rule(f, a, b, n):
    """Метод прямокутників (середніх)"""
    h = (b - a) / n
    x = np.linspace(a + h/2, b - h/2, n)
    return h * np.sum(f(x)), x, f(x)

def trapezoid_rule(f, a, b, n):
    """Метод трапецій"""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h * (np.sum(y) - 0.5 * (y[0] + y[-1])), x, y

def monte_carlo(f, a, b, n):
    """Метод Монте-Карло (з випадковими точками по площині)"""
    # Генеруємо випадкові точки по осі X
    xs = np.random.uniform(a, b, n)
    # Визначаємо верхню межу області
    x_dense = np.linspace(a, b, 1000)
    ymax = np.max(f(x_dense))
    ys = np.random.uniform(0, ymax, n)

    # Перевіряємо, які точки потрапили під криву
    under_curve = ys <= f(xs)
    ratio = np.sum(under_curve) / n

    # Площа прямокутника * частка точок під кривою
    area = (b - a) * ymax * ratio
    return area, xs, ys, under_curve, ymax


# === ОСНОВНЕ ВІКНО ПРОГРАМИ ===
class IntegralApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lab 2 — Числові методи обчислення інтегралів")
        self.setWindowIcon(QIcon())  # Можна вставити власну піктограму .ico
        self.resize(1000, 650)
        self.initUI()

    def initUI(self):
        """Ініціалізація інтерфейсу"""
        # Поля введення
        self.label_func = QLabel("Варіант функції:")
        self.combo_func = QComboBox()
        self.combo_func.addItems([
            "1) ∫[0.8,1.6] dx / √(2x²+1)",
            "2) ∫[1.2,2.0] lg(x+2)/x dx",
            "3) ∫[0.32,0.66] dx / √(x²+2.3)"
        ])

        self.label_a = QLabel("a:")
        self.input_a = QLineEdit("0.8")

        self.label_b = QLabel("b:")
        self.input_b = QLineEdit("1.6")

        self.label_N = QLabel("N (розбиття):")
        self.input_N = QLineEdit("50")

        self.label_method = QLabel("Метод:")
        self.combo_method = QComboBox()
        self.combo_method.addItems(["Прямокутників", "Трапецій", "Монте-Карло"])

        self.calc_btn = QPushButton("Обчислити")
        self.calc_btn.clicked.connect(self.calculate)

        # Вивід результатів
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)

        # Полотно для графіка
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        # Рядок стану
        self.status = QStatusBar()
        self.status.showMessage("Введіть параметри та натисніть 'Обчислити'.")

        # Макет лівої панелі
        left = QGridLayout()
        left.addWidget(self.label_func, 0, 0)
        left.addWidget(self.combo_func, 0, 1)
        left.addWidget(self.label_a, 1, 0)
        left.addWidget(self.input_a, 1, 1)
        left.addWidget(self.label_b, 2, 0)
        left.addWidget(self.input_b, 2, 1)
        left.addWidget(self.label_N, 3, 0)
        left.addWidget(self.input_N, 3, 1)
        left.addWidget(self.label_method, 4, 0)
        left.addWidget(self.combo_method, 4, 1)
        left.addWidget(self.calc_btn, 5, 0, 1, 2)
        left.addWidget(QLabel("Результати:"), 6, 0, 1, 2)
        left.addWidget(self.output_box, 7, 0, 1, 2)

        # Основний макет
        main_layout = QHBoxLayout()
        main_layout.addLayout(left, 3)
        main_layout.addWidget(self.canvas, 5)

        layout = QVBoxLayout()
        layout.addLayout(main_layout)
        layout.addWidget(self.status)
        self.setLayout(layout)

        # Початковий графік
        self.plot_function()

    # === ВИБІР ФУНКЦІЇ ===
    def get_selected_function(self):
        idx = self.combo_func.currentIndex()
        if idx == 0:
            return f1
        elif idx == 1:
            return f2
        else:
            return f3

    # === ГРАФІК БЕЗ ІЛЮСТРАЦІЇ ===
    def plot_function(self):
        """Малює базовий графік без фігур"""
        self.ax.clear()
        try:
            a = float(self.input_a.text())
            b = float(self.input_b.text())
        except ValueError:
            a, b = 0, 1
        f = self.get_selected_function()

        x = np.linspace(a, b, 400)
        y = f(x)
        self.ax.plot(x, y, color="blue", linewidth=2)
        self.ax.axhline(0, color="black", linewidth=1)
        self.ax.grid(True)
        self.canvas.draw()

    # === РОЗРАХУНОК ТА ВІЗУАЛІЗАЦІЯ ===
    def calculate(self):
        try:
            a = float(self.input_a.text())
            b = float(self.input_b.text())
            N = int(self.input_N.text())
        except ValueError:
            self.status.showMessage("Некоректні дані!")
            return

        f = self.get_selected_function()
        method = self.combo_method.currentText()

        # Виклик обраного методу
        if method == "Прямокутників":
            result, x, y = midpoint_rule(f, a, b, N)
        elif method == "Трапецій":
            result, x, y = trapezoid_rule(f, a, b, N)
        else:
            result, xs, ys, under_curve, ymax = monte_carlo(f, a, b, N)

        # --- Візуалізація ---
        self.ax.clear()
        X = np.linspace(a, b, 400)
        Y = f(X)
        self.ax.plot(X, Y, color="blue", linewidth=2)
        self.ax.axhline(0, color="black", linewidth=1)

        if method == "Прямокутників":
            h = (b - a) / N
            for xi, yi in zip(x, y):
                self.ax.add_patch(plt.Rectangle((xi - h/2, 0), h, yi,
                                                color="lightgreen", alpha=0.5,
                                                edgecolor="darkgreen"))
        elif method == "Трапецій":
            for i in range(len(x) - 1):
                xs_ = [x[i], x[i], x[i + 1], x[i + 1]]
                ys_ = [0, y[i], y[i + 1], 0]
                self.ax.fill(xs_, ys_, color="orange", alpha=0.4, edgecolor="brown")
        else:
            # --- Візуалізація Монте-Карло ---
            self.ax.fill_between(X, 0, Y, color="lightgray", alpha=0.3)
            self.ax.scatter(xs[under_curve], ys[under_curve],
                            color="green", s=12, alpha=0.6, label="під кривою")
            self.ax.scatter(xs[~under_curve], ys[~under_curve],
                            color="red", s=12, alpha=0.6, label="над кривою")
            self.ax.set_ylim(0, ymax * 1.1)
            self.ax.legend()

        self.ax.grid(True)
        self.canvas.draw()

        # Вивід результатів
        self.output_box.setPlainText(
            f"Метод: {method}\n"
            f"Інтервал: [{a}, {b}]\n"
            f"N = {N}\n"
            f"Результат ≈ {result:.8f}"
        )
        self.status.showMessage("Обчислення виконано успішно.")


# === ЗАПУСК ПРОГРАМИ ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IntegralApp()
    window.show()
    sys.exit(app.exec())
