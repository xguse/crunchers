#!/usr/bin/env python
"""Functions for streamlining analysis."""

# Imports
from collections import defaultdict, Sequence, namedtuple
import functools
import itertools
import concurrent.futures as futures

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels



import munch
import patsy

from crunchers import parallel as putils
from crunchers import ipython_info

if ipython_info():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

# Metadata
__author__ = "Gus Dunn"
__email__ = "w.gus.dunn@gmail.com"

def tree():
    return defaultdict(tree)

def report_logitreg(formula, data, verbose=True, disp=1):
    """Fit logistic regression, print a report, and return the fit object."""
    y, X = patsy.dmatrices(formula, data=data, return_type='dataframe')

    results = smf.Logit(endog=y, exog=X).fit(disp=disp)
    # results = smf.logit(formula, data=data).fit()
    summary = results.summary()
    margeff = results.get_margeff().summary()

    if verbose:
        report = """\n{summary}\n\n\n{margeff}\n""".format(summary=summary,margeff=margeff)
        print(report)

    return results


def report_glm(formula, data, verbose=True, **kwargs):
    """Fit GLM, print a report, and return the fit object."""
    results = smf.glm(formula, data=data, **kwargs).fit(disp=False, **kwargs)
    summary = results.summary()

    if verbose:
        report = """\n{summary}\n""".format(summary=summary)
        print(report)

    return results


def report_ols(formula, data, fit_regularized=False, L1_wt=1, refit=False, **kwargs):
    """Fit OLS regression, print a report, and return the fit object."""

    RegressionResultsWrapper = statsmodels.regression.linear_model.RegressionResultsWrapper

    # parse formula string into design matrix
    y, X = patsy.dmatrices(formula, data=data, return_type='dataframe')

    if fit_regularized:
        # Does not wrap in RegressionResultsWrapper automatically when using elastic_net
        results = RegressionResultsWrapper(smf.OLS(endog=y, exog=X,).fit_regularized(start_params=None, L1_wt=L1_wt, refit=refit, **kwargs))
    else:
        results = smf.OLS(endog=y, exog=X,).fit(**kwargs)

    return results


def report_rlm(formula, data, verbose=True, **kwargs):
    """Fit RLM, print a report, and return the fit object."""
    results = smf.rlm(formula, data=data, **kwargs).fit(**kwargs)
    summary = results.summary()

    if verbose:
        report = """\n{summary}\n""".format(summary=summary)
        print(report)

    return results


def do_regression(data, y_var, X_ctrls=None, X_hyp=None, kind='OLS', **kwargs):
    """Provide a further abstracted way to build and run multiple types of regressions.

    data (pd.DataFrame): data table to use when retrieving the column headers
    y_var (str): column header of the outcome variable
    X_ctrls (str): formula specification of the "boring" variables "column_header_1 + column_header_2"...
    X_hyp (str): formula specification of the "interesting" variables "column_header_1 + column_header_2"...
    kind (str): the type of regression to run `kind in ['GLM','OLS','RLM'] == True`
    """
    assert (X_ctrls is not None) or (X_hyp is not None)

    if X_hyp is not None:
        X_hyp = ' + {X_hyp}'.format(X_hyp=X_hyp)
    else:
        X_hyp = ''

    if X_ctrls is None:
        X_ctrls = ''

    formula = '{y_var} ~ {X_ctrls}{X_hyp}'.format(y_var=y_var,
                                                     X_ctrls=X_ctrls,
                                                     X_hyp=X_hyp)

    regs = munch.Munch()
    regs.GLM = report_glm
    regs.OLS = report_ols
    regs.RLM = report_rlm
    regs.LOGIT = report_logitreg

    return regs[kind](formula=formula,
                      data=data,
                      **kwargs)



def build_regression_models_grid(X_hyps_dicts, ctrl_coefs_dicts, outcomes_dicts):
    grid = munch.Munch()

    combos = itertools.product(X_hyps_dicts.keys(),ctrl_coefs_dicts.keys(),outcomes_dicts.keys())

    for model_vars in tqdm(combos, desc="building regression model grid"):
        grid_key = putils.GridKey(*model_vars)
        grid[grid_key] = putils.GridValues(X_hyps=X_hyps_dicts[grid_key.X_hyps],
                                    ctrl_coefs=ctrl_coefs_dicts[grid_key.ctrl_coefs],
                                    outcomes=outcomes_dicts[grid_key.outcomes])

    return grid

def regression_grid_single(grid_item, data, kind, **kwargs):

    grid_key, grid_value = grid_item

    y_var = grid_value.outcomes
    X_ctrls = ' + '.join(grid_value.ctrl_coefs)
    X_hyp = ' + '.join(grid_value.X_hyps)

    try:
        result = do_regression(data=data,
                               y_var=y_var,
                               X_ctrls=X_ctrls,
                               X_hyp=X_hyp,
                               kind=kind,
                               **kwargs
                              )

        return grid_key, result
    except np.linalg.linalg.LinAlgError:
        print('error with: {x}'.format(x=X_hyp))
        return grid_key, 'LinAlgError'


def run_regressions_grid(grid, data, kind, max_workers=None, **kwargs):

    regressions = tree()

    partial_regression_grid_single = functools.partial(regression_grid_single,
                                                       data=data,
                                                       kind=kind,
                                                       **kwargs
                                                      )

    with futures.ProcessPoolExecutor(max_workers=max_workers) as worker_pool:
        # results = tqdm(worker_pool.map(partial_regression_grid_single, (grid_item for grid_item in grid.items())))
        results = worker_pool.map(partial_regression_grid_single, (grid_item for grid_item in grid.items()))

        for job in tqdm(results, total=len(grid)):
            grid_key, reg_result = job
            X_hyps = grid_key.X_hyps
            ctrl_coefs = grid_key.ctrl_coefs
            outcomes = grid_key.outcomes

            regressions[outcomes][ctrl_coefs][X_hyps] = reg_result

    return munch.munchify(regressions)



def summarize_multi_LOGIT(results):
    """Return dataframe aggregating over-all stats from a dictionary-like object containing LOGIT result objects."""
    s = defaultdict(defaultdict)

    for name,reg in results.items():
        s['converged'][name] = reg.mle_retvals['converged']
        s['iterations'][name] = reg.mle_retvals['iterations']
        s['warnflag'][name] = reg.mle_retvals['warnflag']
        s['pseudo_rsqrd'][name] = reg.prsquared
        s['aic'][name] = reg.aic

    return pd.DataFrame(s)


def summarize_single_OLS(regression, col_dict, name, is_regularized=False):
    """Return dataframe aggregating over-all stats from a dictionary-like object containing OLS result objects."""
    reg = regression

    try:
        col_dict['rsquared'][name] = reg.rsquared
    except AttributeError:
        col_dict['rsquared'][name] = 'NA'

    try:
        col_dict['rsquared_adj'][name] = reg.rsquared_adj
    except AttributeError:
        col_dict['rsquared_adj'][name] = 'NA'

    col_dict['f_pvalue'][name] = reg.f_pvalue
    col_dict['condition_number'][name] = reg.condition_number
    col_dict['regularized'][name] = is_regularized

    if not is_regularized:
        outliers = reg.outlier_test(method='fdr_bh')['fdr_bh(p)'] <= 0.05
        col_dict['n_outliers'][name] = (outliers).sum()
        col_dict['outliers'][name] = ','.join(outliers.index[outliers].values)
    else:
        col_dict['n_outliers'][name] = "NA"
        col_dict['outliers'][name] = "NA"

    col_dict['aic'][name] = reg.aic

    return col_dict


def summarize_multi_OLS(results):
    """Return dataframe aggregating over-all stats from a dictionary-like object containing OLS result objects."""
    col_dict = defaultdict(defaultdict)

    test_reg = list(results.values())[0]
    try:
        is_regularized = test_reg.regularized
    except AttributeError:
        is_regularized = False

    for name, reg in results.items():
        # TODO: Adpat summarize_single_OLS to be used here
        col_dict = summarize_single_OLS(regression=reg, col_dict=col_dict, name=name, is_regularized=is_regularized)

    df = pd.DataFrame(col_dict)
    df.index.name = 'outcome'
    return df


def summarize_grid_OLS(regs, reg_grid):
    summaries_overall = []

    grid_keys = list(reg_grid.keys())

    for gk in tqdm(grid_keys, desc="initial summary"):
        # col_dict = defaultdict(defaultdict)
        r = regs[gk.outcomes][gk.ctrl_coefs][gk.X_hyps]
        df_part = summarize_multi_OLS({gk.X_hyps: r})
        df_part.index.name = 'X_hyps'
        df_part = df_part.reset_index()
        df_part['ctrl_coefs'] = gk.ctrl_coefs
        df_part['outcomes'] = gk.outcomes

        summaries_overall.append(df_part)

    summaries_overall_df = pd.concat(summaries_overall)

    # calculate q-values and add to regression objects
    outcome_ctrl_grps = summaries_overall_df.groupby(["outcomes","ctrl_coefs"])
    qvals = []

    def assign_qvalue_reg(row, regs):
        qval = row['qvalue_reg']
        r = regs[row.outcomes][row.ctrl_coefs][row.X_hyps]
        r.qvalue_reg = qval

    for name,df in tqdm(outcome_ctrl_grps, desc="adding qvalues"):
        df = df.copy()
        df.index.name = 'dropme'
        df['qvalue_reg'] = statsmodels.stats.multitest.multipletests(pvals=df['f_pvalue'], alpha=0.05, method='fdr_bh')[1]
        df["reg_obj"] = df.apply(lambda row: assign_qvalue_reg(row, regs), axis=1)
        df = df.set_index(["outcomes", "ctrl_coefs","X_hyps"])
        qvals.append(df)

    summaries_overall_df = pd.concat(qvals)

    columns = ["aic","condition_number","f_pvalue","qvalue_reg","n_outliers","outliers","regularized","rsquared","rsquared_adj"]

    return summaries_overall_df[columns]



def get_diff(a,b):
    return abs(a) - abs(b)

def get_log2_fold(a,b):

    return np.log2(abs(a) / abs(b))


def compare_coefs(row, value, results):
    reg = results[row['regression']]
    try:
        X_var = reg.params[row['X_hyp']]
        value = reg.params[value]
    except KeyError:
        return "NA"

    comparisons =  "{value} | {diff} | {log2_fold}".format(value=round(value,4),
                                                   diff=round(get_diff(value,X_var),4),
                                                   log2_fold=round(get_log2_fold(value,X_var),4)
                                                  )
    return comparisons


def identify_full_ctrl_names(X_vars, orig_ctrl_names):
    """Return set of variable names actually used in regression, tolerating mangling of categoricals."""
    X_vars = set(X_vars)

    ctrls = []
    for X_var in X_vars:
        for orig_ctrl in orig_ctrl_names:
            if X_var == orig_ctrl:
                ctrls.append(X_var)

            elif X_var.startswith(orig_ctrl) and (X_var.startswith('C(') and X_var.endswith(']')):
                ctrls.append(X_var)

            else:
                pass

    return set(ctrls)




def summarize_X_vars(results, sig_thresh=0.05, X_ctrls=None, X_ignore=None):
    if sig_thresh is None:
        sig_thresh = 1

    if X_ctrls is None:
        X_ctrls = []

    if X_ignore is None:
        X_ignore = []

    regs_dfs = []
    for name,reg in results.items():

        regs_dfs.append(pd.DataFrame(data={'pvalue_coef':reg.pvalues,
                                           'coef':reg.params,
                                           'regression': name,
                                           'pvalue_reg': reg.f_pvalue,
                                           'qvalue_reg': reg.qvalue_reg
                                          },
                                     columns=['regression','pvalue_reg','qvalue_reg','pvalue_coef','coef'],
                                    )
                       )

    pvals = pd.concat(regs_dfs)

    pvals['X_hyp'] = pvals.index.values
    pvals = pvals.reset_index(drop=True)

    X_ctrls = identify_full_ctrl_names(X_vars=pvals['X_hyp'].values, orig_ctrl_names=X_ctrls)


    X_ignore.extend(X_ctrls)
    ignore_these = pvals.X_hyp.isin(X_ignore)

    sig = pvals[~ignore_these].query(''' pvalue_coef <= {thresh} '''.format(thresh=sig_thresh))


    return sig


def summarize_grid_X_vars_OLS(regs, reg_grid, sig_thresh=0.05):
    rename_map = {"pvalue_coef": "pvalue_param_coef",
                  "coef": "param_coef",
                  "X_hyp": "param_name"}



    summaries_X_vars = []

    grid_keys = list(reg_grid.keys())
    for gk in tqdm(grid_keys, desc="initial summaries"):
        # col_dict = defaultdict(defaultdict)
        r = regs[gk.outcomes][gk.ctrl_coefs][gk.X_hyps]
    #     df_part = stats.summarize_multi_OLS({gk.X_hyps:r})
        df_part = summarize_X_vars(results={gk.X_hyps:r}, sig_thresh=sig_thresh, X_ctrls=None, X_ignore=["Intercept"])
    #     df_part.index.name = 'X_hyps'
    #     df_part = df_part.reset_index(inplace=False)
        df_part['ctrl_coefs'] = gk.ctrl_coefs
        df_part['qvalue_reg'] = r.qvalue_reg
        df_part['outcomes'] = gk.outcomes

        summaries_X_vars.append(df_part)


    columns = ["pvalue_reg","qvalue_reg","pvalue_param_coef","param_coef"]
    summaries_X_vars_df = pd.concat(summaries_X_vars).rename(columns=rename_map).set_index(["outcomes", "ctrl_coefs","regression","param_name"], append=False, inplace=False).sort_index()[columns]
    return summaries_X_vars_df


def format_all_regression_models(regs, total):
    """Return tuple of string formated versions of all regression tables in the `regs` object.

    Args:
        regs (reg-tree: dict-like): tree-like dict containing the regression
                                    results objects as leaves and descriptors as nodes.
        total (int):    total number of results tables to format.

    Returns:
        tuple
    """
    # GridKey(X_hyps='ITGA4', ctrl_coefs='ctrl_coefs_no_ster', outcomes='L_Thalamus_Proper')
    # args = Munch()
    # args.X_hyps = []
    # args.ctrl_coefs = []
    # args.outcomes = []
    # args.table = []
    #
    # # with futures.ProcessPoolExecutor(max_workers=max_workers) as worker_pool:
    # #     all_tables = worker_pool.map(partial_regression_grid_single, )

    divider = "#"*66
    tmpl = "-- {outcome} {ctrl} {X_hyp} --\n\n" \
    "outcomes:  {outcome}\n" \
    "ctrl:      {ctrl}\n" \
    "X_hyp:     {X_hyp}\n\n" \
    "{table}\n\n" \
    "{divider}\n" \
    "{divider}\n\n"

    prog_bar = tqdm(total=total)

    all_tables = []
    for outcome, ctrl_coefs_dicts in regs.items():
        for ctrl, X_hyps_dicts in ctrl_coefs_dicts.items():
            for X_hyp, result in X_hyps_dicts.items():
                table = result.summary2()
                all_tables.append(tmpl.format(**locals()))
                prog_bar.update(n=1)

    return all_tables
