#!/usr/bin/env python3
"""report.pdf from results/summary.md and results/figures/*.png (fpdf2).   python scripts/report.py [results] [report.pdf]"""
import os
import sys

from fpdf import FPDF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "report.pdf")

INTRO = """Question. A research team hands a trading desk an equity signal with a gross backtest. What does the desk hand back? Three things, each measured here: which broker algorithm to route to (a wheel run as a randomised experiment, with the sample size the answer needs), what an order of a given size will cost (a pre-trade model fitted on the wheel's fills and tested out of sample), and how much of the signal's gross return survives once the cost model sits inside the portfolio optimiser and the trades go through the brokers.

Method. Free daily data (Yahoo, 156 US large caps, 2006-2026); an ordinary ML signal (eight price and volume features, ridge and gradient boosting, refit each January walk-forward, Grinold alpha); a Ledoit-Wolf risk model; a convex optimiser with dollar neutrality, position and gross limits, a 10 % of ADV desk cap and a transaction-cost penalty from the fitted model. Orders execute in a reduced-form intraday market: a Brownian bridge from the real open to the real close, the real volume on a U-shaped curve, a quoted spread from the liquidity rule, temporary impact concave in participation (exponent 0.6) and linear permanent impact, both at Almgren et al. (2005) scale, and three simulated brokers whose quality differs with liquidity. Because the simulator is ours, every order carries an oracle (its impact-free fill) and every order was also executed with every other broker, so the estimators can be checked against the truth. Phase A (2010-2016) is the desk's calibration: wheel, pre-trade model, cost penalty. Phase B (2017-2026) is the handoff, run once, out of sample.

Caveats. Brokers, fills and impact are simulated at published scale; the prices, volumes and the signal are real. The universe is today's large caps, so the signal's gross return carries survivorship bias; the cost results are about the difference between paper and realised books, which the bias affects little. No financing or borrow costs; weekly rebalancing on a five-day signal is deliberately turnover-heavy so that costs matter."""

FIGS = [("signal.png", "The signal: out-of-sample IC by year, return by prediction decile, in-sample against realised IC."),
        ("wheel.png", "The wheel: broker effects with cluster-bootstrap intervals for three cost benchmarks, convergence of the estimate as weeks accrue, and the orders needed to resolve a given difference."),
        ("adaptive.png", "The cost of learning: cumulative cost above oracle-best routing under fixed randomisation and Thompson sampling, for two reward definitions."),
        ("pretrade.png", "Pre-trade cost model: calibration by predicted decile, cost against order size by broker with the optimiser's fitted curve, and observed against true cost by stratum."),
        ("handoff.png", "The handoff: paper against realised P&L out of sample, annual return per variant, turnover against return, and the implementation-shortfall decomposition.")]


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9); self.set_text_color(120); self.cell(0, 6, "ProjectF - broker algo wheel, pre-trade cost model and the research handoff", align="R"); self.ln(8); self.set_text_color(0)

    def footer(self):
        self.set_y(-12); self.set_font("Helvetica", "", 8); self.set_text_color(120); self.cell(0, 6, f"{self.page_no()}", align="C")


def clean(s):
    return (s.replace("–", "-").replace("—", "-").replace("−", "-").replace("×", "x").replace("≥", ">=").replace("≤", "<=").replace("…", "...").replace("²", "^2").replace("±", "+/-").replace("**", "").replace("`", "")
             .replace("→", "->").replace("≈", "~").replace("é", "e").replace("ö", "o").replace("’", "'").replace("γ", "gamma").replace("σ", "sigma").replace("β", "beta").replace("Σ", "Sigma").replace("λ", "lambda").replace("α", "alpha"))


def md_table(pdf, rows):
    cols = [c.strip() for c in rows[0].strip("|").split("|")]
    data = [[clean(c.strip()) for c in r.strip("|").split("|")] for r in rows[2:]]
    n = len(cols); w = (pdf.w - 20) / n; fs = 6.5 if n <= 7 else 5.0; cut = 42 if n <= 7 else 22
    pdf.set_font("Helvetica", "B", fs)
    for c in cols:
        pdf.cell(w, 5, clean(c)[:cut], border=1)
    pdf.ln(5); pdf.set_font("Helvetica", "", fs)
    for r in data[:80]:
        if pdf.get_y() > pdf.h - 20:
            pdf.add_page()
        for c in r:
            pdf.cell(w, 4.5, c[:cut], border=1)
        pdf.ln(4.5)
    pdf.ln(2)


def main():
    pdf = PDF(); pdf.set_auto_page_break(auto=True, margin=15); pdf.add_page()
    pdf.set_font("Helvetica", "B", 16); pdf.cell(0, 10, "Broker Algo Wheel, Pre-Trade Cost Model and the Research Handoff", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for para in INTRO.split("\n\n"):
        pdf.multi_cell(0, 4.5, clean(para)); pdf.ln(2)
    for fn, cap in FIGS:
        p = os.path.join(R, "figures", fn)
        if not os.path.exists(p):
            continue
        if pdf.get_y() > pdf.h - 90:
            pdf.add_page()
        pdf.image(p, w=pdf.w - 20); pdf.set_font("Helvetica", "I", 8); pdf.multi_cell(0, 4, clean(cap)); pdf.ln(3); pdf.set_font("Helvetica", "", 9)
    sm = os.path.join(R, "summary.md")
    if os.path.exists(sm):
        pdf.add_page(); lines = open(sm, encoding="utf-8").read().splitlines(); i = 0
        while i < len(lines):
            l = lines[i]
            if l.startswith("## "):
                pdf.set_font("Helvetica", "B", 11); pdf.ln(2); pdf.cell(0, 7, clean(l[3:]), new_x="LMARGIN", new_y="NEXT"); pdf.set_font("Helvetica", "", 9); i += 1
            elif l.startswith("|"):
                j = i
                while j < len(lines) and lines[j].startswith("|"):
                    j += 1
                if j - i >= 2:
                    md_table(pdf, lines[i:j])
                i = j
            elif l.startswith("# "):
                i += 1
            elif l.strip():
                pdf.set_x(pdf.l_margin); pdf.multi_cell(0, 4.5, clean(l.strip())); i += 1
            else:
                i += 1
    pdf.output(OUT); print("wrote", OUT)


if __name__ == "__main__":
    main()
