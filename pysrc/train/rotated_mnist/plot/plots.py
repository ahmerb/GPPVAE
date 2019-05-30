import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

matplotlib.use("Qt5Agg")


class Plotter():
    def __init__(
        self,
        cvae_file_path="../out/cvae/history/history.pkl",
        casale_gppvae_unison_file_path="../out/casale_gppvae_unison/history/history.pkl",
        casale_gppvae_separate_file_path="../out/casale_gppvae_separate/history/history.pkl",
        fitc_gppvae_unison_file_path="../out/fitc_gppvae_unison_old_fail/history/history.pkl",
        interactive_mode=False
    ):
        self.fitc_gppvae_unison_file_path = fitc_gppvae_unison_file_path
        # dict_keys(['mse_out', 'mse_val', 'gp_nll', 'mse', 'recon_term', 'pen_term', 'loss', 'vs_1', 'vs_2', 'vars_1', 'vars_2'])
        self.fitc_gppvae_unison_history = self.load_fitc_gppvae_unison_history()
        self.N_fitc_gppvae_unison = len(self.fitc_gppvae_unison_history['mse'])

        self.casale_gppvae_unison_file_path = casale_gppvae_unison_file_path
        self.casale_gppvae_unison_history = self.load_casale_gppvae_unison_history()
        self.N_casale_gppvae_unison = len(self.casale_gppvae_unison_history['mse'])

        self.casale_gppvae_separate_file_path = casale_gppvae_separate_file_path
        self.casale_gppvae_separate_history = self.load_casale_gppvae_separate_history()
        self.N_casale_gppvae_separate = len(self.casale_gppvae_separate_history['mse'])

        self.cvae_file_path = cvae_file_path
        self.cvae_history = self.load_cvae_history()
        self.N_cvae = len(self.cvae_history['mse'])

        self.interactive_mode = False

    def load_cvae_history(self):
        with open(self.cvae_file_path, "rb") as f:
            history = pickle.load(f)
            return history

    def load_casale_gppvae_unison_history(self):
        with open(self.casale_gppvae_unison_file_path, "rb") as f:
            history = pickle.load(f)
            history['vs_1'] = [h[0] for h in history['vs']]
            history['vs_2'] = [h[1] for h in history['vs']]
            del history['vs']
            history['vars_1'] = [h[0] for h in history['vars']]
            history['vars_2'] = [h[1] for h in history['vars']]
            del history['vars']
            return history

    def load_casale_gppvae_separate_history(self):
        with open(self.casale_gppvae_separate_file_path, "rb") as f:
            history = pickle.load(f)
            history['vs_1'] = [h[0] for h in history['vs']]
            history['vs_2'] = [h[1] for h in history['vs']]
            del history['vs']
            history['vars_1'] = [h[0] for h in history['vars']]
            history['vars_2'] = [h[1] for h in history['vars']]
            del history['vars']
            return history

    def load_fitc_gppvae_unison_history(self):
        with open(self.fitc_gppvae_unison_file_path, "rb") as f:
            history = pickle.load(f)
            return history

    def _show(self, filename=None):
        if self.interactive_mode:
            plt.show()
        else:
            plt.savefig(filename)

    def plot_mse_bars_casale_only(self):
        # valid plots
        fig, ax = plt.subplots()
        x_index = np.arange(0, 3)
        bar_casale_gppvae_separate, bar_casale_gppvae_unison, bar_cvae = ax.bar(
            x_index, (self.casale_gppvae_separate_history['mse_out'][-1],
                      np.min(self.casale_gppvae_unison_history['mse_out']), # mse_out=gppvae recon, mse_val=vae recon
                      self.cvae_history['mse_val'][-1])
        )
        ax.set_xticks(x_index)
        ax.set_xticklabels(['GPPVAE-separate', 'GPPVAE-unison', 'CVAE'])
        ax.set_ylabel('Mean Squared Error (test)')
        self._show("mse_test_bar_casale_only.eps")

        # train plots
        fig, ax = plt.subplots()
        x_index = np.arange(0, 3)
        bar_casale_gppvae_separate, bar_casale_gppvae_unison, bar_cvae = ax.bar(
            x_index, (self.casale_gppvae_separate_history['mse'][-1],
                      self.casale_gppvae_unison_history['mse'][-1],
                      self.cvae_history['mse'][-1])
        )
        ax.set_xticks(x_index)
        ax.set_xticklabels(['GPPVAE-separate', 'GPPVAE-unison', 'CVAE'])
        ax.set_ylabel('Mean Squared Error (train)')
        self._show("mse_train_bar_casale_only.eps")

        print(
            self.casale_gppvae_separate_history['mse_out'][-1],
                      np.min(self.casale_gppvae_unison_history['mse_out']), # mse_out=gppvae recon, mse_val=vae recon
                      self.cvae_history['mse_val'][-1]
        )
        print(
            self.casale_gppvae_separate_history['mse'][-1],
                      self.casale_gppvae_unison_history['mse'][-1],
                      self.cvae_history['mse'][-1]
        )

    def plot_mse_bars(self):
        # valid plots
        fig, ax = plt.subplots()
        x_index = np.arange(0, 4)
        bar_casale_gppvae_separate, bar_casale_gppvae_unison, bar_fitc_gppvae_unison, bar_cvae = ax.bar(
            x_index, (self.casale_gppvae_separate_history['mse_out'][-1],
                      np.min(self.casale_gppvae_unison_history['mse_out']), # mse_out=gppvae recon, mse_val=vae recon
                      self.fitc_gppvae_unison_history['mse_val'][-1],
                      self.cvae_history['mse_val'][-1])
        )
        ax.set_xticks(x_index)
        ax.set_xticklabels(['Casale-GPPVAE-separate', 'Casale-GPPVAE-unison', 'FITC-GPPVAE-unison', 'CVAE'],
                           rotation=60)
        ax.set_ylabel('Mean Squared Error (test)')
        plt.tight_layout()
        self._show("mse_test_bar.eps")

        # train plots
        fig, ax = plt.subplots()
        x_index = np.arange(0, 4)
        bar_casale_gppvae_separate, bar_casale_gppvae_unison, bar_fitc_gppvae_unison, bar_cvae = ax.bar(
            x_index, (self.casale_gppvae_separate_history['mse'][-1],
                      self.casale_gppvae_unison_history['mse'][-1],
                      self.fitc_gppvae_unison_history['mse'][-1],
                      self.cvae_history['mse'][-1])
        )
        ax.set_xticks(x_index)
        ax.set_xticklabels(['Casale-GPPVAE-separate', 'Casale-GPPVAE-unison', 'FITC-GPPVAE-unison', 'CVAE'],
                           rotation=60)
        ax.set_ylabel('Mean Squared Error (train)')
        plt.tight_layout()
        self._show("mse_train_bar.eps")

    def plot_cvae_curves(
        self,
        filenames={
            "mse": "cvae_mse.eps",
            "nll": "cvae_nll.eps",
            "kld": "cvae_kld.eps",
            "loss": "cvae_loss.eps",
            "mse_val": "cvae_mse_val.eps",
            "nll_val": "cvae_nll_val.eps",
            "kld_val": "cvae_kld_val.eps",
            "loss_val": "cvae_loss_val.eps"
        },
        ylabels={
            "mse": "Mean squared error (train)",
            "nll": "Negative log likelihood (train)",
            "kld": "KL divergence (train)", # \(q_{\bm{\varphi}}(\mathbf{z}|\mathbf{y})\)
            "loss": "Loss (train)",
            "mse_val": "Mean squared error (validation)",
            "nll_val": "Negative los likelihood (validation)",
            "kld_val": "KL divergence (validation)",
            "loss_val": "Loss (validation)"
        }
    ):
        xs = np.arange(0, self.N_cvae, 1)
        plot_keys = filenames.keys()
        for key in plot_keys:
            fig, ax = plt.subplots()
            ys = self.cvae_history[key]
            ax.plot(xs, ys)
            ax.set_xlabel('# Epochs')
            ax.set_ylabel(ylabels[key])
            ax.grid(which='both', axis='both')
            self._show(filenames[key])

    def plot_casale_gppvae_unison_curves(
        self,
        filenames={
            "mse": "casale_gppvae_unison_mse.eps",
            "mse_out": "casale_gppvae_unison_mse_out.eps",
            "gp_nll": "casale_gppvae_unison_gp_nll.eps",
            "recon_term": "casale_gppvae_unison_recon.eps",
            "pen_term": "casale_gppvae_unison_pen.eps",
            "loss": "casale_gppvae_unison_loss.eps"
        },
        ylabels={
            "mse": "Mean squared error (train)",
            "mse_out": "Mean squared error (validation)",
            "gp_nll": "GP negative log likelihood (train)",
            "recon_term": "Reconstruction term (train)",
            "pen_term": "Penalisation term (train)",
            "loss": "Loss (train)"
        }
    ):
        # plot_keys = {"mse", "mse_out", "gp_nll", "recon_term", "pen_term", "loss"}
        xs = np.arange(0, self.N_casale_gppvae_unison, 1)
        plot_keys = filenames.keys()
        for key in plot_keys:
            fig, ax = plt.subplots()
            ys = self.casale_gppvae_unison_history[key]
            ax.plot(xs, ys)
            ax.set_xlabel('# Epochs')
            ax.set_ylabel(ylabels[key])
            ax.grid(which='both', axis='both')
            self._show(filenames[key])

    def plot_casale_gppvae_separate_curves(
        self,
        filenames={
            "mse": "casale_gppvae_separate_mse.eps",
            "mse_out": "casale_gppvae_separate_mse_out.eps",
            "gp_nll": "casale_gppvae_separate_gp_nll.eps",
            "recon_term": "casale_gppvae_separate_recon.eps",
            "pen_term": "casale_gppvae_separate_pen.eps",
            "loss": "casale_gppvae_separate_loss.eps"
        },
        ylabels={
            "mse": "Mean squared error (train)",
            "mse_out": "Mean squared error (validation)",
            "gp_nll": "GP negative log likelihood (train)",
            "recon_term": "Reconstruction term (train)",
            "pen_term": "Penalisation term (train)",
            "loss": "Loss (train)"
        }
    ):
        # plot_keys = {"mse", "mse_out", "gp_nll", "recon_term", "pen_term", "loss"}
        xs = np.arange(0, self.N_casale_gppvae_separate, 1)
        plot_keys = filenames.keys()
        for key in plot_keys:
            fig, ax = plt.subplots()
            ys = self.casale_gppvae_separate_history[key]
            ax.plot(xs, ys)
            ax.set_xlabel('# Epochs')
            ax.set_ylabel(ylabels[key])
            ax.grid(which='both', axis='both')
            self._show(filenames[key])

    def plot_fitc_gppvae_unison_curves(
        self,
        filenames={
            "mse": "fitc_gppvae_unison_mse.eps",
            "gp_nll": "fitc_gppvae_unison_gp_nll.eps",
            "recon_term": "fitc_gppvae_unison_recon.eps",
            "pen_term": "fitc_gppvae_unison_pen.eps",
            "loss": "fitc_gppvae_unison_loss.eps"
        },
        ylabels={
            "mse": "Mean squared error (train)",
            "gp_nll": "GP negative log likelihood (train)",
            "recon_term": "Reconstruction term (train)",
            "pen_term": "Penalisation term (train)",
            "loss": "Loss (train)"
        }
    ):
        # NOTE temporarily (because we were accidetally writing both recon_term and recon_term_val to the same key)
        # thus remove every other entry from recon term (its a 6000 size array instead of 3000)
        self.fitc_gppvae_unison_history['recon_term'] = self.fitc_gppvae_unison_history['recon_term'][0::2]

        # plot_keys = {"mse", "mse_out", "gp_nll", "recon_term", "pen_term", "loss"}
        xs = np.arange(0, self.N_fitc_gppvae_unison, 1)
        plot_keys = filenames.keys()
        for key in plot_keys:
            fig, ax = plt.subplots()
            print(key)
            ys = self.fitc_gppvae_unison_history[key]
            ax.plot(xs, ys)
            ax.set_xlabel('# Epochs')
            ax.set_ylabel(ylabels[key])
            ax.grid(which='both', axis='both')
            self._show(filenames[key])


if __name__ == "__main__":
    plotter = Plotter(interactive_mode=False)
    # plotter.plot_casale_gppvae_unison_curves()
    # plotter.plot_fitc_gppvae_unison_curves()
    # plotter.plot_cvae_curves()
    # plotter.plot_mse_bars()
    # plotter.plot_casale_gppvae_separate_curves()
    plotter.plot_mse_bars_casale_only()
