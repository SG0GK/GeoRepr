import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge


def draw_crossplot(real, pred, fig, ax, xmin=None, xmax=None, cbar=True, **kwargs):
    if xmin is None:
        xmin = np.min([np.nanmin(real), np.nanmin(pred)])
    if xmax is None:
        xmax = np.max([np.nanmax(real), np.nanmax(pred)])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.plot((xmin, xmax), (xmin, xmax), 'k--', alpha=0.5, zorder=2)
    sc = ax.scatter(real, pred, **kwargs)
    if cbar:
        fig.colorbar(sc, ax=ax, label='log kernel value', fraction=0.045, pad=0.04)
    ax.set_aspect('equal')
    return xmin


def perform_tsne(X, d, dataset_names, path):
    y = list()
    for i in range(len(dataset_names) * d):
        y.append(i // d)
    y = np.array(y)

    print('Applying Logistic Regression on classes...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    print('Starting T-SNE...')
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    np.save(os.path.join(path, 'tsne_all.npy'), X_tsne)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.set_title(f'TSNE ROC-AUC={test_score}')
    sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, s=5, cmap='jet', zorder=0)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.ax.set_yticks([0, 1, 2, 3], dataset_names)
    ax.set_box_aspect(1)
    ax.set_rasterization_zorder(1)
    fig.savefig(os.path.join(path, 'tsne.pdf'), dpi=100)
    plt.close('all')
    print('T-SNE done!')
    return X_tsne, y


def regress_to_global_statistics(X, X_tsne, y, d, dataset_names, path, data_path, stats_fname):
    stats = list()
    for dataset_name in dataset_names:
        stats.append(pd.read_csv(
            os.path.join(os.path.join(data_path, dataset_name), stats_fname), index_col=0
        ))
    stats = pd.concat(stats, axis=0)
    titles = list(stats.keys())
    Y = stats.values
    n = d
    regs = list()
    super_regs = list()
    for i in tqdm(range(Y.shape[1])):
        regs.append(list())
        for d in [0, 1, 2, 3]:
            X_ = X[n * d:n * d + n]
            Y_ = Y[n * d:n * d + n]
            X_train, X_test, y_train, y_test = train_test_split(X_, Y_[:, i], test_size=0.2, shuffle=True)
            reg = Ridge(alpha=1.0)
            reg.fit(X_train, y_train)
            regs[i].append(reg)
        X_train, X_test, y_train, y_test = train_test_split(X, Y[:, i], test_size=0.2, shuffle=True)
        super_reg = Ridge(alpha=1.0)
        super_reg.fit(X_train, y_train)
        super_regs.append(super_reg)
    save_fname = stats_fname.replace('.csv', '').replace('/', '_')

    with PdfPages(os.path.join(path, f"repesentation_vs_{save_fname}.pdf")) as pdf:

        for i in range(Y.shape[1]):

            fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 4.5))

            fig.suptitle(f'{titles[i]}')

            markers = ['o', 'X', 'D', '^']
            colors = ['r', 'g', 'b', 'm']
            datasets = dataset_names

            ax0.set_title('TSNE vs sedimentation type')
            for d in [0, 1, 2, 3]:
                mask = y == d
                ax0.scatter(
                    X_tsne[mask, 0], X_tsne[mask, 1],
                    c=colors[d], s=1, label=datasets[d], zorder=0
                )
            ax0.legend()
            ax0.set_box_aspect(1)

            vmin, vmax = Y[:, i].min(), Y[:, i].max()

            ax1.set_title(f'TSNE vs {titles[i]}')
            for d in [0, 1, 2, 3]:
                mask = y == d
                sc = ax1.scatter(
                    X_tsne[mask, 0], X_tsne[mask, 1],
                    c=Y[mask, i], marker=markers[d], s=1, vmin=vmin, vmax=vmax,
                    cmap='jet', zorder=0
                )
            fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)

            ax2.set_title('Качество отображения\nв репрезентациях')
            for d in [0, 1, 2, 3]:
                mask = y == d
                y_real = Y[mask, i]
                y_pred = regs[i][d].predict(X[mask])
                label = f'{datasets[d]}, ' + r'$R^2_{\mathrm{local}}$=' + \
                    f'{regs[i][d].score(X[mask], y_real):.2}' + \
                    ' | ' + r'$R^2_{\mathrm{global}}$=' + \
                    f'{super_regs[i].score(X[mask], y_real):.2}'
                draw_crossplot(
                    y_real, y_pred, fig, ax2,
                    c=colors[d], marker=markers[d], s=1, cbar=False,
                    xmin=vmin, xmax=vmax, label=label, zorder=0
                )
                ax2.set_xlabel(f'{titles[i]} (Реальная)')
                ax2.set_ylabel(f'{titles[i]} (Linear Model)')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            for ax in (ax0, ax1, ax2):
                ax.set_box_aspect(1)
                ax.set_rasterization_zorder(1)
            fig.tight_layout()
            pdf.savefig(fig, dpi=100, bbox_inches='tight')

        plt.close('all')


def regress_to_variables(X, X_tsne, d_, dataset_names, path, data_path):
    for d, dataset_name in tqdm(enumerate(dataset_names)):
        print(f'regressing to variables of {dataset_name}')

        variables = pd.read_csv(
            os.path.join(os.path.join(data_path, dataset_name), 'variables_full_.csv'), index_col=0
        )
        Y = variables.values
        titles = list(variables.keys())

        n = d_

        X_ = X[n*d:n*d+n]
        X_tsne_ = X_tsne[n*d:n*d+n]

        regs = list()
        for i in tqdm(range(Y.shape[1])):
            if len(np.unique(Y[:, i])) < 21:
                reg = LogisticRegression()
                Y[:, i] = Y[:, i] - Y[:, i].min()
                X_train, X_test, y_train, y_test = train_test_split(X_, Y[:, i], test_size=0.2, shuffle=True)
                reg.fit(X_train, y_train)
            else:
                reg = Ridge()
                X_train, X_test, y_train, y_test = train_test_split(X_, Y[:, i], test_size=0.2, shuffle=True)
                reg.fit(X_train, y_train)

            regs.append(reg)

        scores = list()
        for i in range(Y.shape[1]):
            scores.append(regs[i].score(X_, Y[:, i]))

        qual_df = pd.DataFrame()
        qual_df['Контекст'] = titles
        qual_df['Качество'] = np.round(scores, 2)
        qual_df.sort_values(by='Качество')[::-1].to_csv(os.path.join(path, f'quality_{dataset_name}.csv'), sep=',')
        qual_df.sort_values(by='Качество')[::-1]
        index = qual_df.sort_values(by='Качество')[::-1].index.values

        with PdfPages(os.path.join(path, f'repesentation_vs_variables_{dataset_name}.pdf')) as pdf:

            for i in index:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
                fig.suptitle(f'{titles[i]}, dataset {dataset_name}')
                markers = ['o', 'X', 'D', '^']
                colors = ['r', 'g', 'b', 'm']
                datasets = dataset_names
                vmin, vmax = Y[:, i].min(), Y[:, i].max()
                ax1.set_title('Карта свойства')
                sc = ax1.scatter(
                    X_tsne_[:, 0], X_tsne_[:, 1],
                    c=Y[:, i], marker=markers[d], s=10,
                    vmin=vmin, vmax=vmax, cmap='jet', zorder=0
                )
                fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
                ax2.set_title('Качество отображения\nв репрезентациях')
                y_real = Y[:, i]
                y_pred = regs[i].predict(X_)
                label = f'{datasets[d]}, ' + r'$R^2$=' + f'{regs[i].score(X_, y_real):.2}'
                draw_crossplot(
                    y_real, y_pred, fig, ax2, c=colors[d], marker=markers[d],
                    s=1, cbar=False, xmin=vmin, xmax=vmax, label=label, zorder=0
                )
                ax2.set_xlabel(f'{titles[i]} (Реальная)')
                ax2.set_ylabel(f'{titles[i]} (Linear Model)')
                ax2.legend()
                for ax in (ax1, ax2):
                    ax.set_box_aspect(1)
                    ax.set_rasterization_zorder(1)
                fig.tight_layout()
                pdf.savefig(fig, dpi=100, bbox_inches='tight')

            plt.close('all')


def analyze_representations(X, d, dataset_names, path, data_path, stats_fnames):
    X_tsne, y = perform_tsne(X, d, dataset_names, path)
    for stats_fname in stats_fnames:
        print(f'Doing regression on {stats_fname}')
        regress_to_global_statistics(X, X_tsne, y, d, dataset_names, path, data_path, stats_fname)
    regress_to_variables(X, X_tsne, d, dataset_names, path, data_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='Path to representations')
    parser.add_argument('-o', '--order', nargs='+', default=['Bar', 'Tidal', 'Shelf', 'WD'], help='dataset names')
    parser.add_argument('-d', '--data_path', type=str, default='../../data/', help='path to data storage')
    parser.add_argument(
        '-s', '--stats_fnames', nargs='+',
        default=[
            'stats.csv', 'lorenz.csv',
            'derivatives_clipped_log.csv',
            'dykstra-parsons.csv',
            'dykstra-parsons-improved.csv'
        ],
        help='names of dataframes of global context'
    )
    parser.add_argument('--n_samples_per_dataset', type=int, default=5001, help='number of samples per dataset')
    args = parser.parse_args()

    X = np.load(args.path)
    # X = X.reshape(-1, 384)
    path = os.path.dirname(args.path)
    dataset_names = args.order
    d = args.n_samples_per_dataset
    data_path = args.data_path
    stats_fnames = args.stats_fnames

    analyze_representations(X, d, dataset_names, path, data_path, stats_fnames)


if __name__ == '__main__':
    main()
