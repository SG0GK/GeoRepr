import os
import pandas as pd
import numpy as np
from scipy import stats


df = list()
n = 5001
d = 4

for path in [
    '../../data/bar_tiny/',
    '../../data/tidal_tiny/',
    '../../data/shelf_tiny/',
    '../../data/wd_tiny/',
    '../../data/Bar/',
    '../../data/Tidal/',
    '../../data/Shelf/',
    '../../data/WD/',
]:
    print(f'Path: {path}')
    df = pd.read_csv(os.path.join(path, 'variables_full.csv'), index_col=0)
    columns = {f'variables/{key}': key.lower() for key in df.keys()}
    df = df.rename(columns=columns)
    columns = [key for key in df.keys() if 'seed' not in key]
    df.drop(columns=columns)
    df.to_csv(os.path.join(path, 'variables_full_.csv'))

#     df.append(pd.read_csv(os.path.join(path, 'variograms_full.csv'), index_col=0))

# df = pd.concat(df, axis=0)

# for key in ['poro_X', 'poro_Y', 'poro_Z', 'perm_X', 'perm_Y', 'perm_Z']:
#     keys = [key_ for key_ in df.keys() if key in key_]
#     min_ = np.percentile(df[keys].values, 1)
#     max_ = np.percentile(df[keys].values, 100)
#     print(key, min_, max_)
#     df[keys] = df[keys].clip(lower=min_, upper=max_)

# for d, path in enumerate([
#     # '../../data/bar_tiny/',
#     # '../../data/tidal_tiny/',
#     # '../../data/shelf_tiny/',
#     # '../../data/wd_tiny/',
#     '../../data/Bar/',
#     '../../data/Tidal/',
#     '../../data/Shelf/',
#     '../../data/WD/',
# ]):
#     print(df[n*d:n*d+n].shape)

#     df[n*d:n*d+n].to_csv(os.path.join(path, 'variograms_full_clipped.csv'))

#     df[n*d:n*d+n] = np.log(df[n*d:n*d+n])
#     df[n*d:n*d+n].to_csv(os.path.join(path, 'variograms_full_clipped_log.csv'))


# variograms = {
#     'poro_X': np.load(os.path.join(path, 'variograms_poro_0.npy')),
#     'poro_Y': np.load(os.path.join(path, 'variograms_poro_1.npy')),
#     'poro_Z': np.load(os.path.join(path, 'variograms_poro_2.npy')),
#     'perm_X': np.load(os.path.join(path, 'variograms_perm_0.npy')),
#     'perm_Y': np.load(os.path.join(path, 'variograms_perm_1.npy')),
#     'perm_Z': np.load(os.path.join(path, 'variograms_perm_2.npy'))
# }
# dfs = list()
# for key in variograms.keys():
#     columns = [f'v/{key}/{i}' for i in range(variograms[key].shape[1])]
#     data = variograms[key]
#     df = pd.DataFrame(columns=columns, data=data)
#     assert (df == df.dropna()).all
#     dfs.append(df)
# dfs = pd.concat(dfs, axis=1)
# dfs.to_csv(os.path.join(path, 'variograms_full.csv'))

# for key in variograms.keys():
#     keys = [key_ for key_ in dfs.keys() if key in key_]
#     min_ = np.percentile(dfs[keys].values, 1)
#     max_ = np.percentile(dfs[keys].values, 100)
#     print(key, min_, max_)
#     dfs[keys] = dfs[keys].clip(lower=min_, upper=max_)
# dfs.to_csv(os.path.join(path, 'variograms_full_clipped.csv'))

# dfs = np.log(dfs)
# dfs.to_csv(os.path.join(path, 'variograms_full_clipped_log.csv'))
# # df = pd.read_csv(os.path.join(path, 'stats.csv'), index_col=0)
# columns = {
#     'Доля коллектора': 'stats/frac',
#     'Прерывистость вдоль X': 'stats/ab_X',
#     'Прерывистость вдоль Y': 'stats/ab_Y',
#     'Прерывистость вдоль Z': 'stats/ab_Z'
# }
# df = df.rename(columns=columns)
# df.to_csv(os.path.join(path, 'stats.csv'))
# df = pd.read_csv(os.path.join(path, 'variables_full.csv'), index_col=0)
# columns = {f'variables/{key}': key.lower() for key in df.keys()}
# df = df.rename(columns=columns)
# columns = [key for key in df.keys() if 'seed' not in key]
# df.drop(columns=columns)
# df.to_csv(os.path.join(path, 'variables_full_.csv'))
# df = pd.read_csv(os.path.join(path, 'variograms_full_log.csv'), index_col=0)
# df = np.exp(df)
# df.to_csv(os.path.join(path, 'variograms_full.csv'))
# df = pd.read_csv(os.path.join(path, 'derivatives.csv'), index_col=0)
# df = np.log(df)
# df.to_csv(os.path.join(path, 'derivatives_log.csv'))
# df = pd.read_csv(path, index_col=0)
# keys = {key: f'lorenz/{key}' for key in df.keys()}
# df = df.rename(columns=keys)
# df.to_csv(path)
# perm = np.load(os.path.join(path, 'perm.npy'))
# poro = np.load(os.path.join(path, 'poro.npy'))
# mask = np.load(os.path.join(path, 'mask.npy'))
# mask_x = mask[1:] * mask[:-1]
# mask_y = mask[:, 1:] * mask[:, :-1]
# mask_z = mask[:, :, 1:] * mask[:, :, :-1]
# dx_perm = np.abs(perm[:, 1:] - perm[:, :-1])[:, mask_x].mean(axis=1)
# dy_perm = np.abs(perm[:, :, 1:] - perm[:, :, :-1])[:, mask_y].mean(axis=1)
# dz_perm = np.abs(perm[:, :, :, 1:] - perm[:, :, :, :-1])[:, mask_z].mean(axis=1)

# dx_poro = np.abs(poro[:, 1:] - poro[:, :-1])[:, mask_x].mean(axis=1)
# dy_poro = np.abs(poro[:, :, 1:] - poro[:, :, :-1])[:, mask_y].mean(axis=1)
# dz_poro = np.abs(poro[:, :, :, 1:] - poro[:, :, :, :-1])[:, mask_z].mean(axis=1)

# df = pd.DataFrame()
# df['der/poro_X'] = dx_poro
# df['der/poro_Y'] = dy_poro
# df['der/poro_Z'] = dz_poro
# df['der/perm_X'] = dx_perm
# df['der/perm_Y'] = dy_perm
# df['der/perm_Z'] = dz_perm
# df.to_csv(os.path.join(path, 'derivatives.csv'))

# p50 = np.percentile(perm, 50, axis=-1)
# p16 = np.percentile(perm, 16, axis=-1)
# dp = (p50 - p16) / p50
# np.save(os.path.join(path, 'dykstra-parsons.npy'), dp)

# df = pd.DataFrame()
# df['dp/mean'] = np.nanmean(dp[:, mask[:, :, 40]], axis=1)
# df['dp/std'] = np.nanstd(dp[:, mask[:, :, 40]], axis=1)
# df['dp/median'] = np.nanmedian(dp[:, mask[:, :, 40]], axis=1)
# df['dp/min'] = np.nanmin(dp[:, mask[:, :, 40]], axis=1)
# df['dp/max'] = np.nanmax(dp[:, mask[:, :, 40]], axis=1)
# df.to_csv(os.path.join(path, 'dykstra-parsons.csv'))

# print(f'Nan values in df: {np.isnan(df.values).sum()}')
# print(f'Inf values in df: {np.isinf(df.values).sum()}')
# print(f'\nMin:\n{df.min()}')
# print(f'\nMax:\n{df.max()}')

# mean_perm = np.mean(perm, axis=-1)
# var_perm = np.var(perm, axis=-1)
# exp_mu = np.square(mean_perm) / np.sqrt(var_perm + np.square(mean_perm))
# sigma = np.sqrt(np.log(var_perm / np.square(mean_perm) + 1))

# p16 = stats.lognorm.ppf(0.16, s=sigma, scale=exp_mu)
# p50 = stats.lognorm.ppf(0.50, s=sigma, scale=exp_mu)
# dpi = (p50 - p16) / p50
# np.save(os.path.join(path, 'dykstra-parsons-improved.npy'), dpi)

# df = pd.DataFrame()
# df['dpi/mean'] = np.nanmean(dpi[:, mask[:, :, 40]], axis=1)
# df['dpi/std'] = np.nanstd(dpi[:, mask[:, :, 40]], axis=1)
# df['dpi/median'] = np.nanmedian(dpi[:, mask[:, :, 40]], axis=1)
# df['dpi/min'] = np.nanmin(dpi[:, mask[:, :, 40]], axis=1)
# df['dpi/max'] = np.nanmax(dpi[:, mask[:, :, 40]], axis=1)
# df.to_csv(os.path.join(path, 'dykstra-parsons-improved.csv'))

# print(f'Nan values in df: {np.isnan(df.values).sum()}')
# print(f'Inf values in df: {np.isinf(df.values).sum()}')
# print(f'\nMin:\n{df.min()}')
# print(f'\nMax:\n{df.max()}')
