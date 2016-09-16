"""Provide wrapper funcs that do some extra stuff all in a go."""

import numpy as np
import statsmodels.formula.api as smf

def report_logitreg(formula, data, verbose=True):
    """Fit logistic regression, print a report, and return the fit object."""
    results = smf.logit(formula, data=data).fit()
    summary = results.summary()
    margeff = results.get_margeff().summary()


    if verbose:
        report = """
{summary}\n\n
{margeff}\n""".format(summary=summary,margeff=margeff)

        print(report)

    return results


def report_glm(formula, data, verbose=True, **kwargs):
    """Fit GLM, print a report, and return the fit object."""
    results = smf.glm(formula, data=data, **kwargs).fit(disp=False, **kwargs)
    summary = results.summary()


    if verbose:
        report = """
{summary}\n""".format(summary=summary)

        print(report)

    return results

def report_ols(formula, data, verbose=True, fit_regularized=False, maxiter=50, L1_wt=1, **kwargs):
    """Fit OLS regression, print a report, and return the fit object."""
    # parse formula string into design matrix
    y, X = patsy.dmatrices(formula, data=data, return_type='dataframe')

    if fit_regularized:
        results = smf.OLS(endog=y, exog=X,).fit_regularized(start_params=None, L1_wt=L1_wt, maxiter=maxiter, **kwargs)
    else:
        results = smf.OLS(endog=y, exog=X,).fit(maxiter=maxiter, **kwargs)

    return results


def report_rlm(formula, data, verbose=True, **kwargs):
    """Fit RLM, print a report, and return the fit object."""
    results = smf.rlm(formula, data=data, **kwargs).fit(disp=False, **kwargs)
    summary = results.summary()


    if verbose:
        report = """
{summary}\n""".format(summary=summary)

        print(report)

    return results


def do_regression(data, y_var, X_ctrls=None, X_hyp=None, kind='OLS', **kwargs):
    """Provide a further abstracted way to build and run multiple types of regressions."""
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


    return regs[kind](formula=formula,
                      data=data,
                      verbose=True,
                      **kwargs)




def summarize_multi_OLS(results):
    """Return dataframe aggregating over-all stats from a dictionary-like object containing OLS result objects."""
    s = defaultdict(defaultdict)

    for name,reg in results.items():
        s['rsquared'][name] = reg.rsquared
        s['rsquared_adj'][name] = reg.rsquared_adj
        s['f_pvalue'][name] = reg.f_pvalue
        s['condition_number'][name] = reg.condition_number

        outliers = reg.outlier_test(method='fdr_bh')['fdr_bh(p)'] <= 0.05
        s['n_outliers'][name] = (outliers).sum()
        s['outliers'][name] = ','.join(outliers.index[outliers].values)
        s['aic'][name] = reg.aic

    return pd.DataFrame(s)




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

    return "{value} | {diff} | {log2_fold}".format(value=round(value,4),
                                                   diff=round(get_diff(value,X_var),4),
                                                   log2_fold=round(get_log2_fold(value,X_var),4)
                                                  )


def summarize_X_vars(results, sig_thresh=0.05, X_ctrls=None, X_ignore=None):
    if X_ctrls is None:
        X_ctrls = []

    if X_ignore is None:
        X_ignore = []

    regs_dfs = []
    for name,reg in results.items():

        regs_dfs.append(pd.DataFrame(data={'pvalues':reg.pvalues,
                                           'coef':reg.params,
                                           'regression': name,
                                          },
                                     columns=['regression','pvalues','coef'],
                                    )
                       )

    s = pd.concat(regs_dfs)
    s['X_hyp'] = s.index.values
    s = s.reset_index(drop=True)

    sig = s[~s.X_hyp.isin(X_ignore)].query(''' pvalues <= {thresh} '''.format(thresh=sig_thresh))

    for cvar in X_ctrls:
        col_name = cvar
        sig.loc[:,col_name] = sig.apply(lambda x: compare_coefs(row=x,value=cvar,results=results), axis=1)

    # set up multi-index
    col_num = sig.shape[1]
    lvl_t = ['']*(col_num-len(X_ctrls))  + [r'Control Coefs Compared to X_hyp ( X_ctrl | X_ctrl - X_hyp | log2(X_ctrl / X_hyp) )']*len(X_ctrls)
    lvl_b = sig.columns.values

    tups = list(zip(*[lvl_t,lvl_b]))

    sig.columns = pd.MultiIndex.from_tuples(tups)

    return sig
