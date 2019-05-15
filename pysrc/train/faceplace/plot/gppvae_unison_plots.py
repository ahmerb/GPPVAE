import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

matplotlib.use("Qt5Agg")


class Plotter():
    def __init__(self, file_path="../out/gppvae_unison/history/history.pkl", interactive_mode=False):
        self.file_path = file_path
        self.history = self.load_history()
        self.interactive_mode = False

    def load_history(self):
        with open(self.file_path, "rb") as f:
            history = pickle.load(f)
            history['vs_1'] = [h[0] for h in history['vs']]
            history['vs_2'] = [h[1] for h in history['vs']]
            del history['vs']
            history['vars_1'] = [h[0] for h in history['vars']]
            history['vars_2'] = [h[1] for h in history['vars']]
            del history['vars']
            return history

    def _show(self, filename=None):
        if self.interactive_mode:
            plt.show()
        else:
            plt.savefig(filename)

    def plot_loss(self, filename="gppvae_unison_loss.eps"):
        fig, ax = plt.subplots()
        ys = self.history['loss']
        xs = np.arange(0, len(ys), 1)
        ax.plot(xs, ys)
        ax.set_ylabel('loss')
        ax.set_xlabel('# epochs')
        ax.grid(which='both', axis='both')
        self._show(filename)

    def plot_mse(self, filename="gppvae_unison_mse.eps"):
        fig, ax = plt.subplots()
        ys = self.history['mse']
        xs = np.arange(0, len(ys), 1)
        ax.plot(xs, ys, 'b')
        ys = self.history['mse_out']
        ax.plot(xs, ys, 'r')
        ax.set_ylabel('mean squared error')
        ax.set_xlabel('# epochs')
        ax.grid(which='both', axis='both')
        self._show(filename)

    def plot_gp_nll(self, filename="gppvae_unison_gp_nll.eps"):
        fig, ax = plt.subplots()
        ys = self.history['gp_nll']
        xs = np.arange(0, len(ys), 1)
        ax.plot(xs, ys)
        ax.set_ylabel('GP negative log likelihood')
        ax.set_xlabel('# epochs')
        ax.grid(which='both', axis='both')
        self._show(filename)


if __name__ == "__main__":
    plotter = Plotter(interactive_mode=False)
    plotter.plot_loss()
    plotter.plot_mse()
    plotter.plot_gp_nll()
