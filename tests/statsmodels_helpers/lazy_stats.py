"""Provide wrapper funcs that do some extra stuff all in a go."""

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
