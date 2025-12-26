import sys
import math

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ===============================
#     ЛАГРАНЖ
# ===============================
def lagrange_polynomial(x, xs, ys):
    total = 0
    n = len(xs)
    for i in range(n):
        term = ys[i]
        for j in range(n):
            if i != j:
                term *= (x - xs[j]) / (xs[i] - xs[j])
        total += term
    return total


# ===============================
#     GUI
# ===============================
class InterpolationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Інтерполяція поліномом Лагранжа")
        self.setFixedSize(900, 600)

        self.init_ui()

    def init_ui(self):
        main = QVBoxLayout()

        # ---------- controls ----------
        controls = QHBoxLayout()

        self.func_input = QLineEdit("math.sin(x)")
        self.xmin_input = QLineEdit("0")
        self.xmax_input = QLineEdit("6.28")

        controls.addWidget(QLabel("f(x) ="))
        controls.addWidget(self.func_input)
        controls.addWidget(QLabel("xmin"))
        controls.addWidget(self.xmin_input)
        controls.addWidget(QLabel("xmax"))
        controls.addWidget(self.xmax_input)

        self.btn = QPushButton("Побудувати")
        self.btn.clicked.connect(self.build_graph)

        # ---------- plot ----------
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        main.addLayout(controls)
        main.addWidget(self.btn)
        main.addWidget(self.canvas)

        self.setLayout(main)

        self.ax.set_title("Очікування побудови графіка...")
        self.canvas.draw()

    # ===============================
    #     BUILD GRAPH
    # ===============================
    def build_graph(self):
        self.ax.clear()

        try:
            xmin = float(self.xmin_input.text())
            xmax = float(self.xmax_input.text())
            f = lambda x: eval(self.func_input.text())
        except:
            self.ax.set_title("❌ Помилка у введених даних")
            self.canvas.draw()
            return

        # вузли інтерполяції
        h = (xmax - xmin) / 10
        xs = [xmin + i * h for i in range(11)]
        ys = [f(x) for x in xs]

        # значення полінома
        xs_dense = [xmin + i * (xmax - xmin) / 500 for i in range(501)]
        ys_poly = [lagrange_polynomial(x, xs, ys) for x in xs_dense]

        # графік
        self.ax.plot(xs_dense, ys_poly, label="Поліном Лагранжа", color="blue")
        self.ax.scatter(xs, ys, color="red", zorder=5, label="Вузли інтерполяції")

        self.ax.set_title("Інтерполяція поліномом Лагранжа")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.legend()
        self.ax.grid(True)

        self.canvas.draw()


# ===============================
#     RUN
# ===============================
app = QApplication(sys.argv)
window = InterpolationApp()
window.show()
sys.exit(app.exec())
