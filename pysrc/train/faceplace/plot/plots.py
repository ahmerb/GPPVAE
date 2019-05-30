import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

matplotlib.use("Qt5Agg")


class Plotter():
    def __init__(
        self,
        vae_file_path="../out/vae/history/history.pkl",
        gppvae_file_path="../out/gppvae/history/history.pkl",
        gppvae_unison_file_path="../out/gppvae_unison/history/history.pkl",
        cvae_file_path="../out/cvae/history/history.pkl",
        svi_gppvae_unison_file_path="../out/svi_gppvae_unison/history/history.pkl",
        interactive_mode=False):

        self.svi_gppvae_unison_file_path = svi_gppvae_unison_file_path
        self.svi_gppvae_unison_history = self.load_svi_gppvae_unison_history()
        self.N_svi_gppvae_unison = len(self.svi_gppvae_unison_history['mse'])

        self.gppvae_unison_file_path = gppvae_unison_file_path
        # dict_keys(['mse_out', 'mse_val', 'gp_nll', 'mse', 'recon_term',
        #            'pen_term', 'loss', 'vs_1', 'vs_2', 'vars_1', 'vars_2'])
        self.gppvae_unison_history = self.load_gppvae_unison_history()
        self.N_gppvae_unison = len(self.gppvae_unison_history['mse'])

        self.gppvae_file_path = gppvae_file_path
        self.gppvae_history = self.load_gppvae_history()
        self.N_gppvae = len(self.gppvae_history['mse'])

        self.vae_file_path = vae_file_path
        self.vae_history = self.load_vae_history()
        self.N_vae = len(self.vae_history['mse'])

        # print(self.N_gppvae_unison, self.N_gppvae, self.N_vae)

        # self.cvae_file_path = cvae_file_path
        # self.cvae_history = self.load_cvae_history()
        # self.N_cvae = len(self.cvae_history['mse'])

        self.interactive_mode = interactive_mode

    def load_vae_history(self):
        with open(self.vae_file_path, "rb") as f:
            history = pickle.load(f)
            return history

    def load_cvae_history(self):
        with open(self.cvae_file_path, "rb") as f:
            history = pickle.load(f)
            return history

    def load_svi_gppvae_unison_history(self):
        with open(self.svi_gppvae_unison_file_path, "rb") as f:
            history = pickle.load(f)
            return history

    def load_gppvae_history(self):
        with open(self.gppvae_file_path, "rb") as f:
            history = pickle.load(f)
            history['vs_1'] = [h[0] for h in history['vs']]
            history['vs_2'] = [h[1] for h in history['vs']]
            del history['vs']
            history['vars_1'] = [h[0] for h in history['vars']]
            history['vars_2'] = [h[1] for h in history['vars']]
            del history['vars']
            return history

    def load_gppvae_unison_history(self):
        with open(self.gppvae_unison_file_path, "rb") as f:
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

    def plot_mse_bars(self):
        # NOTE: also plot GPPVAE-unison after 5k epochs (same as VAE runtime) ??
        # valid plots
        fig, ax = plt.subplots()
        x_index = np.arange(0, 4)
        bar_svi_gppvae_unison, bar_gppvae_unison, bar_gppvae, bar_vae = ax.bar(x_index, (
                                                                  self.svi_gppvae_unison_history['mse_val'][-1],
                                                                  self.gppvae_unison_history['mse_val'][-1],
                                                                  self.gppvae_history['mse_val'][-1],
                                                                  self.vae_history['mse_val'][-1]))
        ax.set_xticks(x_index)
        print(
            (
                                                                  self.svi_gppvae_unison_history['mse_val'][-1],
                                                                  self.gppvae_unison_history['mse_val'][-1],
                                                                  self.gppvae_history['mse_val'][-1],
                                                                  self.vae_history['mse_val'][-1])
        )
        # ax.set_xticklabels(['VAE', 'GPPVAE-separate', 'GPPVAE-unison'])
        ax.set_xticklabels(['SVI-GPPVAE-unison', 'Casale-GPPVAE-unison', 'Casale-GPPVAE-separate', 'VAE'], rotation=60)
        ax.set_ylabel('Mean Squared Error (test)')
        plt.tight_layout()
        self._show("mse_valid_bar.eps")

        # train plots
        fig, ax = plt.subplots()
        x_index = np.arange(0, 4)
        bar_svi_gppvae_unison, bar_gppvae_unison, bar_gppvae, bar_vae = ax.bar(x_index, (
                                                                  self.svi_gppvae_unison_history['mse'][-1],
                                                                  self.gppvae_unison_history['mse'][-1],
                                                                  self.gppvae_history['mse'][-1],
                                                                  self.vae_history['mse'][-1]))
        print(
            (
                                                                  self.svi_gppvae_unison_history['mse'][-1],
                                                                  self.gppvae_unison_history['mse'][-1],
                                                                  self.gppvae_history['mse'][-1],
                                                                  self.vae_history['mse'][-1])
        )
        ax.set_xticks(x_index)
        # ax.set_xticklabels(['VAE', 'GPPVAE-separate', 'GPPVAE-unison'])
        ax.set_xticklabels(['SVI-GPPVAE-unison', 'Casale-GPPVAE-unison', 'Casale-GPPVAE-separate', 'VAE'], rotation=60)
        ax.set_ylabel('Mean Squared Error (train)')
        plt.tight_layout()
        self._show("mse_train_bar.eps")

    def plot_vae_curves(
        self,
        filenames={
            "mse": "vae_mse.eps",
            "nll": "vae_nll.eps",
            "kld": "vae_kld.eps",
            "loss": "vae_loss.eps",
            "mse_val": "vae_mse_val.eps",
            "nll_val": "vae_nll_val.eps",
            "kld_val": "vae_kld_val.eps",
            "loss_val": "vae_loss_val.eps"
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
        xs = np.arange(0, self.N_vae, 1)
        plot_keys = filenames.keys()
        for key in plot_keys:
            fig, ax = plt.subplots()
            ys = self.vae_history[key]
            ax.plot(xs, ys)
            ax.set_xlabel('# Epochs')
            ax.set_ylabel(ylabels[key])
            ax.grid(which='both', axis='both')
            self._show(filenames[key])

    def plot_svi_gppvae_unison_curves(
        self,
        filenames={
            "mse": "svi_gppvae_unison_mse.eps",
            # "mse_val": "svi_gppvae_unison_mse_out.eps",
            "gp_nll": "svi_gppvae_unison_gp_nll.eps",
            "recon_term": "svi_gppvae_unison_recon.eps",
            "pen_term": "svi_gppvae_unison_pen.eps",
            "loss": "svi_gppvae_unison_loss.eps"
        },
        ylabels={
            "mse": "Mean squared error (train)",
            # "mse_val": "Mean squared error (test)",
            "gp_nll": "GP negative log likelihood (train)",
            "recon_term": "Reconstruction term (train)",
            "pen_term": "Penalisation term (train)",
            "loss": "Loss (train)"
        }
    ):
        xs = np.arange(0, self.N_svi_gppvae_unison, 1)
        plot_keys = filenames.keys()
        for key in plot_keys:
            fig, ax = plt.subplots()
            ys = self.svi_gppvae_unison_history[key]
            ax.plot(xs, ys)
            ax.set_xlabel('# Epochs')
            ax.set_ylabel(ylabels[key])
            ax.grid(which='both', axis='both')
            self._show(filenames[key])

    def plot_gppvae_unison_curves(
        self,
        filenames={
            "mse": "gppvae_unison_mse.eps",
            "mse_out": "gppvae_unison_mse_out.eps",
            "gp_nll": "gppvae_unison_gp_nll.eps",
            "recon_term": "gppvae_unison_recon.eps",
            "pen_term": "gppvae_unison_pen.eps",
            "loss": "gppvae_unison_loss.eps"
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
        xs = np.arange(0, self.N_gppvae_unison, 1)
        plot_keys = filenames.keys()
        for key in plot_keys:
            fig, ax = plt.subplots()
            ys = self.gppvae_unison_history[key]
            ax.plot(xs, ys)
            ax.set_xlabel('# Epochs')
            ax.set_ylabel(ylabels[key])
            ax.grid(which='both', axis='both')
            self._show(filenames[key])

    def plot_gppvae_curves(
        self,
        filenames={
            "mse": "gppvae_mse.eps",
            "mse_out": "gppvae_mse_out.eps",
            "gp_nll": "gppvae_gp_nll.eps",
            "recon_term": "gppvae_recon.eps",
            "pen_term": "gppvae_pen.eps",
            "loss": "gppvae_loss.eps"
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
        xs = np.arange(0, self.N_gppvae, 1)
        plot_keys = filenames.keys()
        for key in plot_keys:
            fig, ax = plt.subplots()
            ys = self.gppvae_history[key]
            ax.plot(xs, ys)
            ax.set_xlabel('# Epochs')
            ax.set_ylabel(ylabels[key])
            ax.grid(which='both', axis='both')
            self._show(filenames[key])

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

    def plot_mse_bars_for_failed_cvae(self):
        N = 2334 # last epoch before cvae failed
        cvae_train_mse = 0.001447
        cvae_test_mse = 0.002585

        # valid plots
        fig, ax = plt.subplots()
        x_index = np.arange(0, 3)
        bar_gppvae_unison, bar_vae, bar_cvae = ax.bar(x_index, (self.gppvae_unison_history['mse_val'][N],
                                                                self.vae_history['mse_val'][N],
                                                                cvae_test_mse))
        ax.set_xticks(x_index)
        ax.set_xticklabels(['GPPVAE-unison', 'VAE', 'CVAE'])
        ax.set_ylabel('Mean Squared Error (validation) after 2334 epochs')
        self._show("mse_valid_bar_with_failed_cvae.eps")

        # train plots
        fig, ax = plt.subplots()
        x_index = np.arange(0, 3)
        bar_gppvae_unison, bar_vae, bar_cvae = ax.bar(x_index, (self.gppvae_unison_history['mse'][N],
                                                                self.vae_history['mse'][N],
                                                                cvae_train_mse))
        ax.set_xticks(x_index)
        ax.set_xticklabels(['GPPVAE-unison', 'VAE', 'CVAE'])
        ax.set_ylabel('Mean Squared Error (train) after 2334 epochs')
        self._show("mse_train_bar_with_failed_cvae.eps")


if __name__ == "__main__":
    plotter = Plotter(interactive_mode=False)
    # plotter.plot_gppvae_unison_curves()
    # plotter.plot_vae_curves()
    # plotter.plot_gppvae_curves()
    # plotter.plot_cvae_curves()
    plotter.plot_mse_bars()
    # plotter.plot_mse_bars_for_failed_cvae()
    plotter.plot_svi_gppvae_unison_curves()
