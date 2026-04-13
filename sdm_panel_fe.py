
# -*- coding: utf-8 -*-
"""
两维固定效应空间杜宾模型（SDM）
适配当前项目数据：
- 数据文件：相关系数矩阵(1).xlsx
- 权重文件：权重矩阵.xlsx

模型：
y_it = rho * W y_it + X_it beta + W X_it theta + mu_i + tau_t + eps_it

实现思路：
1. 读取 11 省面板数据
2. 构造行标准化空间权重矩阵
3. 做双向去均值（省份固定效应 + 年份固定效应）
4. 用极大似然估计 rho、beta、theta
5. 计算 impacts（直接效应 / 间接效应 / 总效应）
6. 输出主回归表和 impacts 表

依赖：
pip install pandas numpy scipy statsmodels openpyxl
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from statsmodels.tools.numdiff import approx_hess


REGIONS = ['天津','河北','辽宁','上海','江苏','浙江','福建','山东','广东','广西','海南']


def spatial_lag_long(vec: np.ndarray, W: np.ndarray, n_regions: int) -> np.ndarray:
    """
    对长面板向量做空间滞后。
    要求 vec 排序为：每个年份内，地区顺序固定。
    """
    arr = np.asarray(vec).reshape(-1, n_regions)   # (T, N)
    lag = arr @ W.T
    return lag.reshape(-1, 1)


def two_way_demean(arr: np.ndarray, ids: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    双向去均值：
    x_it - x_i. - x_.t + x_..
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    out = np.zeros_like(arr, dtype=float)
    df_tmp = pd.DataFrame({'id': ids, 'time': times})
    for j in range(arr.shape[1]):
        s = pd.Series(arr[:, j])
        mean_i = s.groupby(df_tmp['id']).transform('mean').to_numpy()
        mean_t = s.groupby(df_tmp['time']).transform('mean').to_numpy()
        mean_all = float(s.mean())
        out[:, j] = s.to_numpy() - mean_i - mean_t + mean_all

    return out


def load_panel_data(
    data_path: str,
    years=(2014, 2023),
    y_col='LN(COD) - LN(GDP总）',
    x_cols=None,
) -> pd.DataFrame:
    """
    读取变量表2，并按论文主方案处理：
    y = ln(COD/GDP)
    core x = ln(ENV_INV_L1)
    controls = ln(GDP_PC), URB, IND2, RD
    """
    if x_cols is None:
        x_cols = {
            'inv_l1': '滞后一期：\n工业污染治理完成投资额（亿元）',
            'gdppc': '人均地区生产总值（元）',
            'urb': '城镇化率',
            'ind2': '第二产业增加值占GDP比重',
            'rd': 'R&D占GDP比重',
        }

    df = pd.read_excel(data_path, sheet_name='变量表2')
    df = df.rename(columns={'地区': 'region', '时间': 'year', y_col: 'y'})

    rename_map = {v: k for k, v in x_cols.items()}
    df = df.rename(columns=rename_map)

    need_cols = ['region', 'year', 'y'] + list(x_cols.keys())
    df = df[need_cols].copy()

    # 样本期
    start_year, end_year = years
    df = df[(df['year'] >= start_year) & (df['year'] <= end_year)].copy()

    # 变量处理：严格按你的主方案
    df['ln_inv_l1'] = np.log(df['inv_l1'])
    df['ln_gdppc'] = np.log(df['gdppc'])

    # 排序：年份优先，地区顺序固定
    df['region'] = pd.Categorical(df['region'], categories=REGIONS, ordered=True)
    df = df.sort_values(['year', 'region']).reset_index(drop=True)

    # 平衡面板检查
    n_regions = len(REGIONS)
    n_years = df['year'].nunique()
    assert len(df) == n_regions * n_years, '当前数据不是平衡面板，请先补齐缺失值或删去缺口年份。'

    return df


def load_weight_matrix(weight_path: str, kind: str = 'adjacency') -> np.ndarray:
    """
    kind:
    - adjacency : 邻接矩阵
    - distance_inverse : 距离倒数矩阵
    """
    if kind == 'adjacency':
        raw = pd.read_excel(weight_path, sheet_name='邻接矩阵').iloc[:11, :12].copy()
        raw.columns = ['region'] + REGIONS
        W = raw.set_index('region').loc[REGIONS, REGIONS].astype(float).to_numpy()

    elif kind == 'distance_inverse':
        raw = pd.read_excel(weight_path, sheet_name='距离矩阵（省会距离）').iloc[:11, :12].copy()
        raw.columns = ['region'] + REGIONS
        D = raw.set_index('region').loc[REGIONS, REGIONS].astype(float).to_numpy()
        with np.errstate(divide='ignore'):
            W = 1.0 / D
        np.fill_diagonal(W, 0.0)

    else:
        raise ValueError("kind 只能是 'adjacency' 或 'distance_inverse'")

    # 行标准化
    row_sum = W.sum(axis=1, keepdims=True)
    if np.any(row_sum == 0):
        raise ValueError('存在全零行，无法做行标准化，请检查权重矩阵。')
    W = W / row_sum
    return W


class TwoWayFESDM:
    """
    两维固定效应 SDM 的一个轻量实现。
    """

    def __init__(self, df: pd.DataFrame, W: np.ndarray, x_names=None, include_wx: bool = True):
        self.df = df.copy()
        self.W = W
        self.N = len(REGIONS)
        self.T = self.df['year'].nunique()
        self.n = len(self.df)
        self.include_wx = include_wx

        if x_names is None:
            x_names = ['ln_inv_l1', 'ln_gdppc', 'urb', 'ind2', 'rd']
        self.x_names = x_names

        self.ids = self.df['region'].astype(str).to_numpy()
        self.times = self.df['year'].to_numpy()

        self.y = self.df['y'].to_numpy().reshape(-1, 1)
        self.X = self.df[self.x_names].to_numpy()

        # 双向去均值
        self.y_dm = two_way_demean(self.y, self.ids, self.times)
        self.X_dm = two_way_demean(self.X, self.ids, self.times)
        self.Wy = spatial_lag_long(self.y.ravel(), self.W, self.N)
        self.Wy_dm = two_way_demean(self.Wy, self.ids, self.times)
        if self.include_wx:
            self.WX = np.hstack([spatial_lag_long(self.X[:, j], self.W, self.N) for j in range(self.X.shape[1])])
            self.WX_dm = two_way_demean(self.WX, self.ids, self.times)
            self.Z = np.hstack([self.X_dm, self.WX_dm])
            self.coef_names = self.x_names + [f'W_{x}' for x in self.x_names]
        else:
            self.WX = None
            self.WX_dm = None
            self.Z = self.X_dm.copy()
            self.coef_names = self.x_names.copy()

        self.evals = np.linalg.eigvals(self.W)

        self.params_ = None
        self.result_table_ = None
        self.impact_table_ = None
        self.diagnostic_tables_ = None
        self.llf_ = None
        self.aic_ = None
        self.bic_ = None

    def logdet_A(self, rho: float) -> float:
        vals = 1.0 - rho * self.evals
        if np.any(np.real(vals) <= 0):
            return -np.inf
        return float(self.T * np.sum(np.log(np.real(vals))))

    def neg_ll(self, params: np.ndarray) -> float:
        k = self.Z.shape[1]
        rho = float(params[0])
        beta = np.asarray(params[1:1 + k]).reshape(-1, 1)
        log_sigma2 = float(params[-1])
        sigma2 = np.exp(log_sigma2)

        if not (-0.98 < rho < 0.98):
            return 1e12

        logdet = self.logdet_A(rho)
        if not np.isfinite(logdet):
            return 1e12

        Ay = self.y_dm - rho * self.Wy_dm
        e = Ay - self.Z @ beta
        rss = float((e.T @ e).item())

        ll = logdet - (self.n / 2.0) * np.log(2.0 * np.pi * sigma2) - rss / (2.0 * sigma2)
        return -ll

    def fit(self):
        # 初值：先做 profile 思路给 rho 初始值
        def profile_obj(rho: float) -> float:
            Ay = self.y_dm - rho * self.Wy_dm
            b = np.linalg.lstsq(self.Z, Ay, rcond=None)[0]
            e = Ay - self.Z @ b
            sigma2 = float((e.T @ e).item() / self.n)
            ll = self.logdet_A(rho) - (self.n / 2.0) * np.log(sigma2)
            return -ll

        from scipy.optimize import minimize_scalar
        rho0 = minimize_scalar(profile_obj, bounds=(-0.95, 0.95), method='bounded').x

        Ay0 = self.y_dm - rho0 * self.Wy_dm
        b0 = np.linalg.lstsq(self.Z, Ay0, rcond=None)[0]
        e0 = Ay0 - self.Z @ b0
        sigma20 = float((e0.T @ e0).item() / self.n)

        x0 = np.r_[rho0, b0.ravel(), np.log(sigma20)]
        bounds = [(-0.98, 0.98)] + [(None, None)] * self.Z.shape[1] + [(-20, 20)]

        opt = minimize(self.neg_ll, x0, method='L-BFGS-B', bounds=bounds)
        if not opt.success:
            raise RuntimeError(f'优化失败：{opt.message}')

        self.params_ = opt.x
        self.llf_ = -float(opt.fun)
        k_all = len(self.params_)
        self.aic_ = 2.0 * k_all - 2.0 * self.llf_
        self.bic_ = np.log(self.n) * k_all - 2.0 * self.llf_

        # 近似标准误：数值 Hessian
        H = approx_hess(opt.x, self.neg_ll)
        cov = np.linalg.pinv(H)
        se = np.sqrt(np.diag(cov))
        zval = self.params_ / se
        pval = 2.0 * (1.0 - norm.cdf(np.abs(zval)))

        # 线性部分的异方差稳健近似标准误（rho 固定条件下）
        k = self.Z.shape[1]
        beta = np.asarray(self.params_[1:1 + k]).reshape(-1, 1)
        Ay = self.y_dm - float(self.params_[0]) * self.Wy_dm
        e = Ay - self.Z @ beta
        XtX_inv = np.linalg.pinv(self.Z.T @ self.Z)
        meat = self.Z.T @ ((e ** 2) * self.Z)
        vcov_beta_robust = XtX_inv @ meat @ XtX_inv
        se_beta_robust = np.sqrt(np.clip(np.diag(vcov_beta_robust), a_min=0, a_max=None))

        param_names = ['rho'] + self.coef_names + ['log_sigma2']
        se_robust = np.r_[np.nan, se_beta_robust, np.nan]
        z_robust = self.params_ / se_robust
        p_robust = 2.0 * (1.0 - norm.cdf(np.abs(z_robust)))
        self.result_table_ = pd.DataFrame({
            '变量': param_names,
            '系数': self.params_,
            '标准误': se,
            'z值': zval,
            'p值': pval,
            '稳健标准误(线性部分)': se_robust,
            '稳健z值(线性部分)': z_robust,
            '稳健p值(线性部分)': p_robust,
        })

        self._calc_impacts(cov)

        return self

    def _calc_impacts(self, cov: np.ndarray | None = None, sim_draws: int = 1000):
        """
        impacts 点估计 + 参数模拟近似标准误
        """
        assert self.params_ is not None, '请先 fit()'

        rho = float(self.params_[0])
        coeffs = dict(zip(self.coef_names, self.params_[1:1 + len(self.coef_names)]))

        A_inv = np.linalg.inv(np.eye(self.N) - rho * self.W)

        rows = []
        for x in self.x_names:
            beta = float(coeffs[x])
            theta = float(coeffs.get(f'W_{x}', 0.0))

            S = A_inv @ (beta * np.eye(self.N) + theta * self.W)
            direct = float(np.trace(S) / self.N)
            total = float(S.sum() / self.N)
            indirect = total - direct
            rows.append([x, direct, indirect, total])

        impacts = pd.DataFrame(rows, columns=['变量', '直接效应', '间接效应', '总效应'])

        # 若提供协方差矩阵，做参数模拟求 impacts 的标准误与 p 值
        if cov is not None:
            mean = self.params_[:1 + len(self.coef_names)]
            cov_sub = cov[:1 + len(self.coef_names), :1 + len(self.coef_names)]

            rng = np.random.default_rng(2026)
            draws = rng.multivariate_normal(mean, cov_sub, size=sim_draws)

            sim_store = {x: {'direct': [], 'indirect': [], 'total': []} for x in self.x_names}

            for draw in draws:
                rho_d = draw[0]
                if not (-0.98 < rho_d < 0.98):
                    continue
                vals = 1.0 - rho_d * self.evals
                if np.any(np.real(vals) <= 0):
                    continue

                try:
                    A_inv_d = np.linalg.inv(np.eye(self.N) - rho_d * self.W)
                except np.linalg.LinAlgError:
                    continue

                coeffs_d = dict(zip(self.coef_names, draw[1:]))

                for x in self.x_names:
                    beta_d = float(coeffs_d[x])
                    theta_d = float(coeffs_d[f'W_{x}'])
                    S_d = A_inv_d @ (beta_d * np.eye(self.N) + theta_d * self.W)

                    direct_d = float(np.trace(S_d) / self.N)
                    total_d = float(S_d.sum() / self.N)
                    indirect_d = total_d - direct_d

                    sim_store[x]['direct'].append(direct_d)
                    sim_store[x]['indirect'].append(indirect_d)
                    sim_store[x]['total'].append(total_d)

            for effect in ['direct', 'indirect', 'total']:
                std_list = []
                z_list = []
                p_list = []
                col_name = {'direct': '直接效应', 'indirect': '间接效应', 'total': '总效应'}[effect]

                for _, row in impacts.iterrows():
                    x = row['变量']
                    sims = np.asarray(sim_store[x][effect], dtype=float)
                    se = np.std(sims, ddof=1) if len(sims) > 5 else np.nan
                    z = row[col_name] / se if (se is not None and np.isfinite(se) and se > 0) else np.nan
                    p = 2.0 * (1.0 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
                    std_list.append(se)
                    z_list.append(z)
                    p_list.append(p)

                impacts[f'{col_name}_标准误'] = std_list
                impacts[f'{col_name}_z值'] = z_list
                impacts[f'{col_name}_p值'] = p_list

        self.impact_table_ = impacts

    @staticmethod
    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float).ravel()
        b = np.asarray(b, dtype=float).ravel()
        if a.size != b.size or a.size < 3:
            return np.nan
        sa = np.std(a, ddof=1)
        sb = np.std(b, ddof=1)
        if sa <= 0 or sb <= 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    def run_diagnostics(self, out_path: str | None = None, verbose: bool = True):
        """
        诊断模式：
        1) 去均值后变量标准差
        2) ln_inv_l1 与其余变量/空间滞后项相关性
        3) 条件数
        4) VIF
        """
        # 1) 去均值后标准差
        std_rows = []
        std_rows.append(['y_dm', float(np.std(self.y_dm, ddof=1))])
        for i, x in enumerate(self.x_names):
            std_rows.append([x, float(np.std(self.X_dm[:, i], ddof=1))])
        for i, x in enumerate(self.x_names):
            std_rows.append([f'W_{x}', float(np.std(self.WX_dm[:, i], ddof=1))])
        std_table = pd.DataFrame(std_rows, columns=['变量', '去均值后标准差'])

        # 2) 与 ln_inv_l1 的相关性（含空间滞后项）
        corr_rows = []
        inv = self.X_dm[:, self.x_names.index('ln_inv_l1')]

        for i, x in enumerate(self.x_names):
            if x == 'ln_inv_l1':
                continue
            corr_rows.append([x, self._corr(inv, self.X_dm[:, i])])
        if self.include_wx and self.WX_dm is not None:
            for i, x in enumerate(self.x_names):
                corr_rows.append([f'W_{x}', self._corr(inv, self.WX_dm[:, i])])
        corr_table = pd.DataFrame(corr_rows, columns=['变量', 'corr(ln_inv_l1, ·)'])

        # 3) 条件数（基于 Z）
        cond_number = float(np.linalg.cond(self.Z))
        cond_table = pd.DataFrame([['Z', cond_number]], columns=['矩阵', '条件数'])

        # 4) VIF（基于 Z）
        vif_rows = []
        names = self.coef_names
        Z = self.Z
        ncol = Z.shape[1]

        for j in range(ncol):
            yj = Z[:, j]
            Xo = np.delete(Z, j, axis=1)
            b = np.linalg.lstsq(Xo, yj, rcond=None)[0]
            yhat = Xo @ b
            sst = float(np.sum((yj - yj.mean()) ** 2))
            ssr = float(np.sum((yj - yhat) ** 2))
            if sst <= 1e-12:
                r2 = np.nan
                vif = np.nan
            else:
                r2 = max(0.0, min(1.0, 1.0 - ssr / sst))
                vif = np.inf if (1.0 - r2) <= 1e-10 else 1.0 / (1.0 - r2)
            vif_rows.append([names[j], r2, vif])
        vif_table = pd.DataFrame(vif_rows, columns=['变量', 'R2(对其余解释变量回归)', 'VIF'])

        # 5) 自动判读：投资项不显著的主要风险来源
        inv_std = float(std_table.loc[std_table['变量'] == 'ln_inv_l1', '去均值后标准差'].iloc[0])
        median_std = float(std_table['去均值后标准差'].median())
        weak_id_flag = inv_std < max(1e-8, 0.25 * median_std)

        corr_abs_max = float(np.nanmax(np.abs(corr_table['corr(ln_inv_l1, ·)'].to_numpy(dtype=float))))
        high_corr_flag = corr_abs_max >= 0.7

        vif_inv = float(vif_table.loc[vif_table['变量'] == 'ln_inv_l1', 'VIF'].iloc[0])
        vif_flag = vif_inv >= 5.0

        cond_flag = cond_number >= 1000.0
        if vif_flag or high_corr_flag:
            primary_reason = '共线性主导'
        elif weak_id_flag:
            primary_reason = '识别偏弱主导'
        elif cond_flag:
            primary_reason = '整体矩阵病态（尺度/近共线）主导'
        else:
            primary_reason = '未见强共线性，可能是统计功效不足'

        assess_rows = [
            ['ln_inv_l1去均值后标准差', inv_std, '低于中位数25%则提示识别偏弱', weak_id_flag],
            ['ln_inv_l1相关性最大绝对值', corr_abs_max, '>=0.7 视为高相关', high_corr_flag],
            ['ln_inv_l1的VIF', vif_inv, '>=5 视为明显共线性', vif_flag],
            ['设计矩阵条件数', cond_number, '>=1000 视为病态风险较高', cond_flag],
            ['自动判读', primary_reason, '用于解释投资项p值偏大', np.nan],
        ]
        assess_table = pd.DataFrame(assess_rows, columns=['指标', '数值', '判据', '是否触发'])

        # 6) 投资变量极端值与口径突变检测
        s_inv_raw = self.df['ln_inv_l1'].astype(float)
        z = (s_inv_raw - s_inv_raw.mean()) / (s_inv_raw.std(ddof=1) + 1e-12)
        extreme_cnt = int((np.abs(z) > 3).sum())

        ychg = self.df.sort_values(['region', 'year']).groupby('region')['ln_inv_l1'].diff()
        jump_thr = float(3.0 * ychg.std(ddof=1)) if np.isfinite(ychg.std(ddof=1)) else np.nan
        jump_cnt = int((np.abs(ychg) > jump_thr).sum()) if np.isfinite(jump_thr) else 0
        outlier_table = pd.DataFrame([
            ['|z(ln_inv_l1)| > 3 的观测数', extreme_cnt],
            ['ln_inv_l1 年度跳变阈值(3σ)', jump_thr],
            ['|Δln_inv_l1| > 3σ 的观测数', jump_cnt],
        ], columns=['指标', '数值'])

        self.diagnostic_tables_ = {
            'std': std_table,
            'corr_inv': corr_table,
            'condition_number': cond_table,
            'vif': vif_table,
            'assessment': assess_table,
            'outlier_check': outlier_table,
        }

        if verbose:
            print('\n========== 诊断模式：去均值后标准差 ==========')
            print(std_table.round(6).to_string(index=False))

            print('\n========== 诊断模式：ln_inv_l1 相关性 ==========')
            print(corr_table.round(6).to_string(index=False))

            print('\n========== 诊断模式：条件数 ==========')
            print(cond_table.round(6).to_string(index=False))

            print('\n========== 诊断模式：VIF ==========')
            print(vif_table.round(6).to_string(index=False))

            print('\n========== 诊断模式：自动判读 ==========')
            print(assess_table.to_string(index=False))

            print('\n========== 诊断模式：投资变量极端值/口径突变检查 ==========')
            print(outlier_table.to_string(index=False))

        if out_path:
            with pd.ExcelWriter(out_path) as writer:
                std_table.to_excel(writer, sheet_name='std_demeaned', index=False)
                corr_table.to_excel(writer, sheet_name='corr_with_ln_inv_l1', index=False)
                cond_table.to_excel(writer, sheet_name='condition_number', index=False)
                vif_table.to_excel(writer, sheet_name='vif', index=False)
                assess_table.to_excel(writer, sheet_name='assessment', index=False)
                outlier_table.to_excel(writer, sheet_name='outlier_check', index=False)

    def save_results(self, out_prefix='sdm_main'):
        assert self.result_table_ is not None and self.impact_table_ is not None, '请先 fit()'
        self.result_table_.to_excel(f'{out_prefix}_coef.xlsx', index=False)
        self.impact_table_.to_excel(f'{out_prefix}_impacts.xlsx', index=False)


def main():
    def standardize_columns(df_in: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        df_out = df_in.copy()
        for c in cols:
            s = df_out[c].astype(float)
            sd = float(s.std(ddof=1))
            if sd > 0:
                df_out[c] = (s - float(s.mean())) / sd
        return df_out

    def build_lag_df(df_in: pd.DataFrame, lag_order: int) -> tuple[pd.DataFrame, str]:
        """
        lag_order=1: 使用 ln_inv_l1
        lag_order=2/3: 在地区内继续向后平移 1/2 期
        """
        if lag_order == 1:
            return df_in.copy(), 'ln_inv_l1'
        col = f'ln_inv_l{lag_order}'
        tmp = df_in.sort_values(['region', 'year']).copy()
        tmp[col] = tmp.groupby('region')['ln_inv_l1'].shift(lag_order - 1)
        tmp = tmp.dropna(subset=[col]).copy()
        return tmp, col

    def run_stepwise_and_lag_checks(df_base: pd.DataFrame, W: np.ndarray):
        rows = []
        # 投资滞后 1/2/3 期 + 全模型（含 WX）
        for lag in [1, 2, 3]:
            dlag, inv_col = build_lag_df(df_base, lag)
            x_full = [inv_col, 'ln_gdppc', 'urb', 'ind2', 'rd']
            m = TwoWayFESDM(dlag, W, x_names=x_full, include_wx=True).fit()
            r = m.result_table_.set_index('变量')
            rows.append({
                '实验': f'lag{lag}_full_sdm',
                '样本量': len(dlag),
                'AIC': m.aic_,
                'BIC': m.bic_,
                '投资项': inv_col,
                '投资系数': float(r.loc[inv_col, '系数']),
                '投资标准误': float(r.loc[inv_col, '标准误']),
                '投资p值': float(r.loc[inv_col, 'p值']),
                '投资稳健p值': float(r.loc[inv_col, '稳健p值(线性部分)']),
                '条件数': float(np.linalg.cond(m.Z)),
            })

        # 分步回归（lag1）
        specs = [
            ('step1_core_no_wx', ['ln_inv_l1'], False),
            ('step2_core_with_wx', ['ln_inv_l1'], True),
            ('step3_add_controls_no_wx', ['ln_inv_l1', 'ln_gdppc', 'urb', 'ind2', 'rd'], False),
            ('step4_full_sdm', ['ln_inv_l1', 'ln_gdppc', 'urb', 'ind2', 'rd'], True),
        ]
        for name, xs, use_wx in specs:
            m = TwoWayFESDM(df_base, W, x_names=xs, include_wx=use_wx).fit()
            r = m.result_table_.set_index('变量')
            rows.append({
                '实验': name,
                '样本量': len(df_base),
                'AIC': m.aic_,
                'BIC': m.bic_,
                '投资项': 'ln_inv_l1',
                '投资系数': float(r.loc['ln_inv_l1', '系数']),
                '投资标准误': float(r.loc['ln_inv_l1', '标准误']),
                '投资p值': float(r.loc['ln_inv_l1', 'p值']),
                '投资稳健p值': float(r.loc['ln_inv_l1', '稳健p值(线性部分)']),
                '条件数': float(np.linalg.cond(m.Z)),
            })

        out = pd.DataFrame(rows)
        out.to_excel('W1_sensitivity_checks.xlsx', index=False)
        print('\n========== 敏感性分析（滞后/分步/稳健SE） ==========')
        print(out.round(4).to_string(index=False))

    # 1. 读取主样本
    df = load_panel_data(
        data_path='data/相关系数矩阵(1).xlsx',
        years=(2011, 2023)
    )
    # print('数据预览：')
    # print(df.head().to_string(index=False))

    # 2. 主模型：邻接矩阵
    W1 = load_weight_matrix('data/权重矩阵.xlsx', kind='adjacency')
    # print('空间权重矩阵 W1 预览：')
    # print(pd.DataFrame(W1, index=REGIONS, columns=REGIONS).round(4).to_string())
    model1 = TwoWayFESDM(df, W1)
    model1.fit()
    model1.run_diagnostics(out_path='W1_sdm_diagnostics.xlsx', verbose=True)

    print('\n========== SDM 主回归结果（W1 邻接矩阵） ==========')
    print(model1.result_table_.round(4).to_string(index=False))

    print('\n========== impacts 效应分解（W1 邻接矩阵） ==========')
    print(model1.impact_table_.round(4).to_string(index=False))

    model1.save_results('W1_sdm')

    # 3. 稳健性：距离倒数矩阵
    W2 = load_weight_matrix('data/权重矩阵.xlsx', kind='distance_inverse')
    # print('空间权重矩阵 W2 预览：')
    # print(pd.DataFrame(W2, index=REGIONS, columns=REGIONS).round(4).to_string())

    model2 = TwoWayFESDM(df, W2)
    model2.fit()
    model2.run_diagnostics(out_path='W2_sdm_diagnostics.xlsx', verbose=True)
    model2.save_results('W2_sdm')

    # 4. 用户关注项专项检查（以 W1 为主）
    # 4.1 变量标准化后重估（查看条件数是否下降）
    x_base = ['ln_inv_l1', 'ln_gdppc', 'urb', 'ind2', 'rd']
    df_std = standardize_columns(df, x_base)
    model1_std = TwoWayFESDM(df_std, W1, x_names=x_base, include_wx=True).fit()
    cond_raw = float(np.linalg.cond(model1.Z))
    cond_std = float(np.linalg.cond(model1_std.Z))
    pd.DataFrame(
        [{'模型': 'W1_raw', '条件数': cond_raw}, {'模型': 'W1_standardized_X', '条件数': cond_std}]
    ).to_excel('W1_condition_number_compare.xlsx', index=False)
    print('\n========== 标准化前后条件数对比（W1） ==========')
    print(pd.DataFrame(
        [{'模型': 'W1_raw', '条件数': cond_raw}, {'模型': 'W1_standardized_X', '条件数': cond_std}]
    ).round(4).to_string(index=False))

    # 4.2 投资滞后2/3期 + 分步回归 + 稳健SE对比
    run_stepwise_and_lag_checks(df, W1)

    print('\n结果已保存：')
    print('- W1_sdm_coef.xlsx')
    print('- W1_sdm_impacts.xlsx')
    print('- W2_sdm_coef.xlsx')
    print('- W2_sdm_impacts.xlsx')
    print('- W1_sdm_diagnostics.xlsx')
    print('- W2_sdm_diagnostics.xlsx')
    print('- W1_condition_number_compare.xlsx')
    print('- W1_sensitivity_checks.xlsx')


if __name__ == '__main__':
    main()
