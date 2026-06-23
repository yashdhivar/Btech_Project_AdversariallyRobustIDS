import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import json
import os
import torch

# ── Class label map (CICIDS) ──────────────────────────────────────────────────
LABEL_MAP = {0: 'BENIGN', 1: 'DoS/DDoS', 2: 'Probe/PortScan', 3: 'R2L/Bot', 4: 'U2R/Web'}
CICIDS_ATTACK_LABELS = {
    'BENIGN': 0,
    'DoS Hulk': 1, 'DoS GoldenEye': 1, 'DoS slowloris': 1,
    'DoS Slowhttptest': 1, 'Heartbleed': 1, 'DDoS': 1,
    'PortScan': 2,
    'FTP-Patator': 3, 'SSH-Patator': 3, 'Bot': 3, 'Infiltration': 3,
    'Web Attack \u2013 Brute Force': 4, 'Web Attack \u2013 XSS': 4,
    'Web Attack \u2013 Sql Injection': 4,
}

ATTACK_COLORS = {
    'Normal': '#2ecc71',
    'DoS': '#e74c3c',
    'Probe': '#e67e22',
    'R2L': '#9b59b6',
    'U2R': '#3498db',
}


# ─────────────────────────────────────────────────────────────────────────────
# Top metric row
# ─────────────────────────────────────────────────────────────────────────────

def render_metric_row():
    json_path = _find_results_json()
    if json_path is None:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Tier 2 Clean Acc", "N/A")
        col2.metric("Tier 3 Clean Acc", "N/A")
        col3.metric("Tier 2 Robust (PGD)", "N/A")
        col4.metric("Tier 3 Robust (PGD)", "N/A")
        col5.metric("Tier 3 AUC-ROC", "N/A")
        return

    with open(json_path) as f:
        res = json.load(f)

    t2c = res.get('Tier2-DNN_clean', {})
    t3c = res.get('Tier3-RobustDNN_clean', {})
    t2p = res.get('Tier2-DNN_robust_pgd', {})
    t3p = res.get('Tier3-RobustDNN_robust_pgd', {})

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Tier 2 Clean Acc",
                f"{t2c.get('accuracy',0)*100:.2f}%", delta="Full test set")
    col2.metric("Tier 3 Clean Acc",
                f"{t3c.get('accuracy',0)*100:.2f}%", delta="Robust model")
    col3.metric("Tier 2 Robust (PGD)",
                f"{t2p.get('robust_accuracy',0)*100:.1f}%",
                delta=f"drop {t2p.get('accuracy_drop',0)*100:.1f}pp", delta_color="inverse")
    col4.metric("Tier 3 Robust (PGD)",
                f"{t3p.get('robust_accuracy',0)*100:.1f}%",
                delta=f"drop {t3p.get('accuracy_drop',0)*100:.1f}pp")
    col5.metric("Tier 3 AUC-ROC",
                f"{t3c.get('auc_roc',0):.4f}", delta="Near perfect")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — Model Performance (existing, enhanced)
# ─────────────────────────────────────────────────────────────────────────────

def render_model_performance():
    st.subheader("📈 Evaluation Results — Tier 2 vs Tier 3")
    st.markdown(
        "Results computed on the test set (15% holdout). "
        "Adversarial evaluation used a stratified subset with "
        "FGSM (ε=0.1) and PGD (ε=0.1, 20 iterations). "
        "**Tier 3 is only evaluated on samples Tier 2 flags as attacks.**"
    )

    json_path = _find_results_json()
    if json_path is None:
        st.error("❌ `evaluation_results.json` not found. Run `python main.py --mode evaluate` first.")
        return

    with open(json_path) as f:
        res = json.load(f)

    t2c = res.get('Tier2-DNN_clean', {})
    t3c = res.get('Tier3-RobustDNN_clean', {})
    t2f = res.get('Tier2-DNN_robust_fgsm', {})
    t2p = res.get('Tier2-DNN_robust_pgd',  {})
    t3f = res.get('Tier3-RobustDNN_robust_fgsm', {})
    t3p = res.get('Tier3-RobustDNN_robust_pgd',  {})

    # ── KPI cards ─────────────────────────────────────────────────────────────
    st.markdown("### 🔢 Key Metrics at a Glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tier 2 Clean Accuracy",   f"{t2c.get('accuracy',0)*100:.2f}%")
    c2.metric("Tier 3 Clean Accuracy",   f"{t3c.get('accuracy',0)*100:.2f}%")
    c3.metric("Tier 2 Robust Acc (PGD)", f"{t2p.get('robust_accuracy',0)*100:.2f}%",
              delta=f"drop {t2p.get('accuracy_drop',0)*100:.1f}pp", delta_color="inverse")
    c4.metric("Tier 3 Robust Acc (PGD)", f"{t3p.get('robust_accuracy',0)*100:.2f}%",
              delta=f"drop {t3p.get('accuracy_drop',0)*100:.1f}pp")

    st.divider()

    # ── Full metrics table ─────────────────────────────────────────────────────
    st.markdown("### 📊 Full Metrics Table (Clean Evaluation)")
    table = {
        'Model':         ['Tier 2 — DNN', 'Tier 3 — Robust DNN'],
        'Accuracy (%)':  [round(t2c.get('accuracy',0)*100,2),  round(t3c.get('accuracy',0)*100,2)],
        'Precision (%)': [round(t2c.get('precision',0)*100,2), round(t3c.get('precision',0)*100,2)],
        'Recall (%)':    [round(t2c.get('recall',0)*100,2),    round(t3c.get('recall',0)*100,2)],
        'F1 (%)':        [round(t2c.get('f1_score',0)*100,2),  round(t3c.get('f1_score',0)*100,2)],
        'FPR (%)':       [round(t2c.get('fpr',0)*100,4),       round(t3c.get('fpr',0)*100,4)],
        'AUC-ROC':       [round(t2c.get('auc_roc',0),4),       round(t3c.get('auc_roc',0),4)],
    }
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

    st.divider()

    # ── Chart 1: Clean vs Robust ───────────────────────────────────────────────
    st.markdown("### 🔵 Clean vs Robust Accuracy Under Adversarial Attack")
    systems   = ['Tier 2 (DNN)', 'Tier 3 (Robust DNN)']
    clean_acc = [t2c.get('accuracy',0)*100, t3c.get('accuracy',0)*100]
    fgsm_acc  = [t2f.get('robust_accuracy',0)*100, t3f.get('robust_accuracy',0)*100]
    pgd_acc   = [t2p.get('robust_accuracy',0)*100, t3p.get('robust_accuracy',0)*100]

    fig1 = go.Figure(data=[
        go.Bar(name='Clean', x=systems, y=clean_acc, marker_color='#3498db',
               text=[f"{v:.2f}%" for v in clean_acc], textposition='outside'),
        go.Bar(name='FGSM Robust', x=systems, y=fgsm_acc, marker_color='#e67e22',
               text=[f"{v:.2f}%" for v in fgsm_acc], textposition='outside'),
        go.Bar(name='PGD Robust',  x=systems, y=pgd_acc,  marker_color='#e74c3c',
               text=[f"{v:.2f}%" for v in pgd_acc],  textposition='outside'),
    ])
    fig1.update_layout(
        barmode='group',
        yaxis=dict(title='Accuracy (%)', range=[0, 115]),
        height=440,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    # ── Chart 2: Accuracy drop ─────────────────────────────────────────────────
    st.markdown("### 📉 Accuracy Drop Under Attack (lower = more robust)")
    attacks   = ['FGSM', 'PGD']
    t2_drops  = [t2f.get('accuracy_drop',0)*100, t2p.get('accuracy_drop',0)*100]
    t3_drops  = [t3f.get('accuracy_drop',0)*100, t3p.get('accuracy_drop',0)*100]

    fig2 = go.Figure(data=[
        go.Bar(name='Tier 2 (DNN)',        x=attacks, y=t2_drops, marker_color='#e74c3c',
               text=[f"{v:.1f}pp" for v in t2_drops], textposition='outside'),
        go.Bar(name='Tier 3 (Robust DNN)', x=attacks, y=t3_drops, marker_color='#2ecc71',
               text=[f"{v:.1f}pp" for v in t3_drops], textposition='outside'),
    ])
    fig2.update_layout(
        barmode='group',
        yaxis=dict(title='Accuracy Drop (pp)', range=[0, 80]),
        height=380,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Robustness ratio ────────────────────────────────────────────────────────
    st.markdown("### 🛡️ Robustness Ratio  (robust acc / clean acc,  1.0 = perfectly robust)")
    rr = {
        'Model':           ['Tier 2 (DNN)', 'Tier 3 (Robust DNN)'],
        'FGSM Robustness': [round(t2f.get('robustness_ratio',0),4), round(t3f.get('robustness_ratio',0),4)],
        'PGD Robustness':  [round(t2p.get('robustness_ratio',0),4), round(t3p.get('robustness_ratio',0),4)],
    }
    st.dataframe(pd.DataFrame(rr), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — Per-Tier Results
# ─────────────────────────────────────────────────────────────────────────────

def render_per_tier_results():
    st.subheader("📋 Per-Tier Evaluation Results")

    json_path = _find_results_json()
    if json_path is None:
        st.error("❌ `evaluation_results.json` not found. Run `python main.py --mode evaluate` first.")
        return

    with open(json_path) as f:
        res = json.load(f)

    tab1, tab2, tab3 = st.tabs(["Tier 1 — Signatures", "Tier 2 — ML DNN", "Tier 3 — Robust DNN"])

    # ── Tier 1 ────────────────────────────────────────────────────────────────
    with tab1:
        t1 = res.get('tier1', {})
        if not t1 or 'total_signatures' not in t1:
            st.warning("Tier 1 evaluation data not available.")
        else:
            st.markdown("### Signature Database Coverage")
            c1, c2 = st.columns(2)
            c1.metric("Total Signatures", t1['total_signatures'])
            c2.metric("Categories", len(t1.get('signatures_by_category', {})))

            sig_cats = t1.get('signatures_by_category', {})
            if sig_cats:
                st.markdown("### Signatures by Category")
                df_sigs = pd.DataFrame([
                    {'Category': cat, 'Rules': count}
                    for cat, count in sig_cats.items()
                ])
                st.dataframe(df_sigs, use_container_width=True, hide_index=True)

                fig = go.Figure(go.Bar(
                    x=list(sig_cats.keys()), y=list(sig_cats.values()),
                    marker_color=['#e74c3c', '#e67e22', '#9b59b6', '#3498db',
                                  '#2ecc71', '#f39c12'][:len(sig_cats)],
                    text=list(sig_cats.values()), textposition='outside',
                ))
                fig.update_layout(title='Signature Rules per Category',
                                  yaxis_title='Number of Rules', height=350)
                st.plotly_chart(fig, use_container_width=True)

            st.info(
                f"📝 Test set contains {t1.get('test_set_attacks', 'N/A')} attack samples "
                f"and {t1.get('test_set_normal', 'N/A')} normal samples. "
                "Tier 1 operates on raw traffic features before preprocessing."
            )

    # ── Tier 2 ────────────────────────────────────────────────────────────────
    with tab2:
        t2c = res.get('Tier2-DNN_clean', {})
        t2f = res.get('Tier2-DNN_robust_fgsm', {})
        t2p = res.get('Tier2-DNN_robust_pgd', {})

        if not t2c:
            st.warning("Tier 2 evaluation data not available.")
        else:
            st.markdown("### Tier 2 — Clean Evaluation")
            _render_clean_metrics_cards(t2c, "Tier 2")

            if t2f or t2p:
                st.markdown("### Tier 2 — Adversarial Robustness")
                _render_robust_table("Tier 2", t2c, t2f, t2p)

            # Per-class
            t2_pc = res.get('tier2_per_class_clean', {})
            if t2_pc:
                st.markdown("### Tier 2 — Per-Attack Class Metrics (Clean)")
                _render_per_class_table(t2_pc)

    # ── Tier 3 ────────────────────────────────────────────────────────────────
    with tab3:
        t3c = res.get('Tier3-RobustDNN_clean', {})
        t3f = res.get('Tier3-RobustDNN_robust_fgsm', {})
        t3p = res.get('Tier3-RobustDNN_robust_pgd', {})

        if not t3c:
            st.warning("Tier 3 evaluation data not available.")
        else:
            st.info("⚠️ Tier 3 is evaluated only on samples that Tier 2 classified as attacks (pipeline gating).")
            st.markdown("### Tier 3 — Clean Evaluation (attack-gated)")
            _render_clean_metrics_cards(t3c, "Tier 3")

            if t3f or t3p:
                st.markdown("### Tier 3 — Adversarial Robustness")
                _render_robust_table("Tier 3", t3c, t3f, t3p)

            t3_pc = res.get('tier3_per_class_clean', {})
            if t3_pc:
                st.markdown("### Tier 3 — Per-Attack Class Metrics (Clean)")
                _render_per_class_table(t3_pc)


# ─────────────────────────────────────────────────────────────────────────────
# MODE 3 — Tier Comparison
# ─────────────────────────────────────────────────────────────────────────────

def render_tier_comparison():
    st.subheader("🔀 Tier Comparison")

    json_path = _find_results_json()
    if json_path is None:
        st.error("❌ `evaluation_results.json` not found. Run `python main.py --mode evaluate` first.")
        return

    with open(json_path) as f:
        res = json.load(f)

    comp12 = res.get('comparison_tier1_tier2', {})
    comp_all = res.get('comparison_all_tiers', {})

    # ── Tier 1 + Tier 2 comparison ────────────────────────────────────────────
    st.markdown("### Tier 1 + Tier 2 Comparison")
    if not comp12:
        st.warning("Comparison data not available.")
    else:
        rows = []
        for tier_name, info in comp12.items():
            row = {'Tier': tier_name.upper(), 'Type': info.get('type', 'N/A')}
            if 'clean_accuracy' in info:
                row['Clean Accuracy (%)'] = round(info['clean_accuracy'] * 100, 2)
                row['Clean F1 (%)'] = round(info['clean_f1'] * 100, 2)
                row['FGSM Robust Acc (%)'] = round(info.get('fgsm_robust_accuracy', 0) * 100, 2)
                row['PGD Robust Acc (%)'] = round(info.get('pgd_robust_accuracy', 0) * 100, 2)
            elif 'total_signatures' in info:
                row['Total Signatures'] = info['total_signatures']
                row['Categories'] = ', '.join(info.get('categories_covered', []))
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── All Tiers comparison ──────────────────────────────────────────────────
    st.markdown("### Tier 1 + Tier 2 + Tier 3 Full Comparison")
    if not comp_all:
        st.warning("Full comparison data not available.")
    else:
        rows = []
        for tier_name, info in comp_all.items():
            row = {'Tier': tier_name.upper(), 'Type': info.get('type', 'N/A')}
            if 'clean_accuracy' in info:
                row['Clean Accuracy (%)'] = round(info['clean_accuracy'] * 100, 2)
                row['Clean F1 (%)'] = round(info['clean_f1'] * 100, 2)
                row['Clean Precision (%)'] = round(info.get('clean_precision', 0) * 100, 2)
                row['Clean Recall (%)'] = round(info.get('clean_recall', 0) * 100, 2)
                row['FGSM Robust Acc (%)'] = round(info.get('fgsm_robust_accuracy', 0) * 100, 2)
                row['PGD Robust Acc (%)'] = round(info.get('pgd_robust_accuracy', 0) * 100, 2)
                row['PGD Drop (pp)'] = round(info.get('pgd_accuracy_drop', 0) * 100, 2)
            elif 'total_signatures' in info:
                row['Total Signatures'] = info['total_signatures']
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Chart: accuracy comparison across tiers
        ml_tiers = {k: v for k, v in comp_all.items() if 'clean_accuracy' in v}
        if ml_tiers:
            tier_names = [k.upper() for k in ml_tiers.keys()]
            clean_vals = [v['clean_accuracy'] * 100 for v in ml_tiers.values()]
            fgsm_vals = [v.get('fgsm_robust_accuracy', 0) * 100 for v in ml_tiers.values()]
            pgd_vals = [v.get('pgd_robust_accuracy', 0) * 100 for v in ml_tiers.values()]

            fig = go.Figure(data=[
                go.Bar(name='Clean', x=tier_names, y=clean_vals, marker_color='#3498db',
                       text=[f"{v:.1f}%" for v in clean_vals], textposition='outside'),
                go.Bar(name='FGSM Robust', x=tier_names, y=fgsm_vals, marker_color='#e67e22',
                       text=[f"{v:.1f}%" for v in fgsm_vals], textposition='outside'),
                go.Bar(name='PGD Robust', x=tier_names, y=pgd_vals, marker_color='#e74c3c',
                       text=[f"{v:.1f}%" for v in pgd_vals], textposition='outside'),
            ])
            fig.update_layout(
                barmode='group',
                title='All Tiers — Clean vs Robust Accuracy',
                yaxis=dict(title='Accuracy (%)', range=[0, 115]),
                height=440,
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Accuracy drop comparison
            drop_fgsm = [v.get('fgsm_accuracy_drop', 0) * 100 for v in ml_tiers.values()]
            drop_pgd = [v.get('pgd_accuracy_drop', 0) * 100 for v in ml_tiers.values()]

            fig2 = go.Figure(data=[
                go.Bar(name='FGSM Drop', x=tier_names, y=drop_fgsm, marker_color='#e67e22',
                       text=[f"{v:.1f}pp" for v in drop_fgsm], textposition='outside'),
                go.Bar(name='PGD Drop', x=tier_names, y=drop_pgd, marker_color='#e74c3c',
                       text=[f"{v:.1f}pp" for v in drop_pgd], textposition='outside'),
            ])
            fig2.update_layout(
                barmode='group',
                title='Accuracy Drop Under Attack (lower = better)',
                yaxis=dict(title='Accuracy Drop (pp)'),
                height=380,
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODE 4 — Per-Attack Analysis
# ─────────────────────────────────────────────────────────────────────────────

def render_per_attack_analysis():
    st.subheader("🎯 Per-Attack Type Evaluation")
    st.markdown(
        "Breakdown of detection performance for each attack category: "
        "**Normal**, **DoS**, **Probe**, **R2L**, **U2R**."
    )

    json_path = _find_results_json()
    if json_path is None:
        st.error("❌ `evaluation_results.json` not found. Run `python main.py --mode evaluate` first.")
        return

    with open(json_path) as f:
        res = json.load(f)

    tab_clean, tab_robust = st.tabs(["Clean Data", "Adversarial Robustness"])

    # ── Clean per-attack ──────────────────────────────────────────────────────
    with tab_clean:
        t2_pc = res.get('tier2_per_class_clean', {})
        t3_pc = res.get('tier3_per_class_clean', {})

        if not t2_pc and not t3_pc:
            st.warning("Per-attack evaluation data not available. Run evaluation first.")
            return

        for tier_label, pc_data in [("Tier 2", t2_pc), ("Tier 3", t3_pc)]:
            if not pc_data:
                continue

            st.markdown(f"### {tier_label} — Per-Attack Metrics (Clean)")

            # Table
            rows = []
            for cls_name, m in pc_data.items():
                rows.append({
                    'Attack Type': cls_name,
                    'Precision (%)': round(m['precision'] * 100, 2),
                    'Recall (%)': round(m['recall'] * 100, 2),
                    'F1-Score (%)': round(m['f1_score'] * 100, 2),
                    'FPR (%)': round(m.get('fpr', 0) * 100, 4),
                    'Support': m['support'],
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Grouped bar chart: precision, recall, F1 per attack type
            attack_names = [r['Attack Type'] for r in rows]
            colors = [ATTACK_COLORS.get(n, '#95a5a6') for n in attack_names]

            fig = go.Figure(data=[
                go.Bar(name='Precision', x=attack_names,
                       y=[r['Precision (%)'] for r in rows],
                       marker_color='#3498db',
                       text=[f"{r['Precision (%)']:.1f}%" for r in rows],
                       textposition='outside'),
                go.Bar(name='Recall', x=attack_names,
                       y=[r['Recall (%)'] for r in rows],
                       marker_color='#2ecc71',
                       text=[f"{r['Recall (%)']:.1f}%" for r in rows],
                       textposition='outside'),
                go.Bar(name='F1-Score', x=attack_names,
                       y=[r['F1-Score (%)'] for r in rows],
                       marker_color='#e67e22',
                       text=[f"{r['F1-Score (%)']:.1f}%" for r in rows],
                       textposition='outside'),
            ])
            fig.update_layout(
                barmode='group',
                title=f'{tier_label} — Precision / Recall / F1 per Attack Type',
                yaxis=dict(title='Score (%)', range=[0, 115]),
                height=440,
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Support distribution (pie chart)
            fig_pie = go.Figure(go.Pie(
                labels=attack_names,
                values=[r['Support'] for r in rows],
                marker=dict(colors=colors),
                textinfo='label+percent+value',
            ))
            fig_pie.update_layout(title=f'{tier_label} — Sample Distribution by Attack Type',
                                  height=380)
            st.plotly_chart(fig_pie, use_container_width=True)

            st.divider()

    # ── Robust per-attack ─────────────────────────────────────────────────────
    with tab_robust:
        for tier_label, fgsm_key, pgd_key in [
            ("Tier 2", 'tier2_per_class_robust_fgsm', 'tier2_per_class_robust_pgd'),
            ("Tier 3", 'tier3_per_class_robust_fgsm', 'tier3_per_class_robust_pgd'),
        ]:
            fgsm_data = res.get(fgsm_key, {})
            pgd_data = res.get(pgd_key, {})

            if not fgsm_data and not pgd_data:
                continue

            st.markdown(f"### {tier_label} — Per-Attack Robustness")

            # Combined table
            rows = []
            all_classes = set(list(fgsm_data.keys()) + list(pgd_data.keys()))
            for cls_name in sorted(all_classes):
                fm = fgsm_data.get(cls_name, {})
                pm = pgd_data.get(cls_name, {})
                rows.append({
                    'Attack Type': cls_name,
                    'Clean Acc (%)': round(fm.get('clean_accuracy', pm.get('clean_accuracy', 0)) * 100, 2),
                    'FGSM Robust (%)': round(fm.get('robust_accuracy', 0) * 100, 2),
                    'FGSM Drop (pp)': round(fm.get('accuracy_drop', 0) * 100, 2),
                    'PGD Robust (%)': round(pm.get('robust_accuracy', 0) * 100, 2),
                    'PGD Drop (pp)': round(pm.get('accuracy_drop', 0) * 100, 2),
                    'Support': fm.get('support', pm.get('support', 0)),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Chart: clean vs FGSM vs PGD per attack type
            attack_names = [r['Attack Type'] for r in rows]
            fig = go.Figure(data=[
                go.Bar(name='Clean', x=attack_names,
                       y=[r['Clean Acc (%)'] for r in rows],
                       marker_color='#3498db',
                       text=[f"{r['Clean Acc (%)']:.1f}%" for r in rows],
                       textposition='outside'),
                go.Bar(name='FGSM', x=attack_names,
                       y=[r['FGSM Robust (%)'] for r in rows],
                       marker_color='#e67e22',
                       text=[f"{r['FGSM Robust (%)']:.1f}%" for r in rows],
                       textposition='outside'),
                go.Bar(name='PGD', x=attack_names,
                       y=[r['PGD Robust (%)'] for r in rows],
                       marker_color='#e74c3c',
                       text=[f"{r['PGD Robust (%)']:.1f}%" for r in rows],
                       textposition='outside'),
            ])
            fig.update_layout(
                barmode='group',
                title=f'{tier_label} — Clean vs Adversarial Accuracy per Attack Type',
                yaxis=dict(title='Accuracy (%)', range=[0, 115]),
                height=440,
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Accuracy drop chart
            fig_drop = go.Figure(data=[
                go.Bar(name='FGSM Drop', x=attack_names,
                       y=[r['FGSM Drop (pp)'] for r in rows],
                       marker_color='#e67e22',
                       text=[f"{r['FGSM Drop (pp)']:.1f}pp" for r in rows],
                       textposition='outside'),
                go.Bar(name='PGD Drop', x=attack_names,
                       y=[r['PGD Drop (pp)'] for r in rows],
                       marker_color='#e74c3c',
                       text=[f"{r['PGD Drop (pp)']:.1f}pp" for r in rows],
                       textposition='outside'),
            ])
            fig_drop.update_layout(
                barmode='group',
                title=f'{tier_label} — Accuracy Drop per Attack Type (lower = more robust)',
                yaxis=dict(title='Accuracy Drop (pp)'),
                height=380,
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig_drop, use_container_width=True)

            st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# MODE 5 — Live CSV Inference (existing)
# ─────────────────────────────────────────────────────────────────────────────

def render_batch_analysis():
    st.subheader("📁 Live Batch Inference")
    st.markdown(
        "Upload a **CICIDS-format CSV** file. The pipeline will preprocess it with the "
        "trained scaler + feature selector, run both Tier 2 (standard DNN) and "
        "Tier 3 (robust DNN) on clean data, then generate FGSM + PGD adversarial examples "
        "to demonstrate the robustness gap live."
    )
    st.info(
        "ℹ️ Clean inference runs on the **full** uploaded file. "
        "Adversarial evaluation is capped at **2,000 samples** for speed."
    )

    uploaded_file = st.file_uploader("Choose a CICIDS CSV file", type="csv")
    if uploaded_file is None:
        return

    df_raw = pd.read_csv(uploaded_file)
    st.write(f"**Loaded:** {len(df_raw):,} rows × {df_raw.shape[1]} columns")
    st.dataframe(df_raw.head(8), use_container_width=True)

    if st.button("🚀 Run Full Pipeline", type="primary"):
        _run_live_inference(df_raw)


def _run_live_inference(df_raw):
    # ── Preprocess ────────────────────────────────────────────────────────────
    with st.spinner("Preprocessing…"):
        try:
            X, y, has_labels = _preprocess_cicids(df_raw)
        except Exception as e:
            st.error(f"❌ Preprocessing failed: {e}")
            st.stop()

    st.success(f"✅ Preprocessed: {X.shape[0]:,} samples × {X.shape[1]} features")

    if has_labels:
        label_counts = pd.Series(y).map(LABEL_MAP).value_counts()
        st.markdown("**Label distribution:**")
        st.dataframe(label_counts.rename("Count"), use_container_width=True)

    # ── Load models ───────────────────────────────────────────────────────────
    with st.spinner("Loading models…"):
        num_classes = int(len(np.unique(y))) if has_labels else 5
        tier2, tier3 = _load_models(X.shape[1], num_classes)

    if tier2 is None and tier3 is None:
        st.error("❌ No models found. Run training first.")
        st.stop()

    # ── Clean inference ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔵 Clean Inference  (full dataset)")
    with st.spinner("Running Tier 2 clean inference…"):
        t2_preds = _predict_keras(tier2, X) if tier2 else None
    with st.spinner("Running Tier 3 clean inference…"):
        t3_preds = _predict_torch(tier3, X) if tier3 else None

    _render_clean_results(t2_preds, t3_preds, y, has_labels)

    # ── Adversarial inference ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔴 Adversarial Evaluation  (2,000-sample subset)")

    if tier3 is None:
        st.warning("⚠️ Tier 3 not loaded — cannot generate adversarial examples.")
        return

    ADV_CAP = 2000
    rng = np.random.default_rng(42)
    idx = rng.choice(X.shape[0], size=min(ADV_CAP, X.shape[0]), replace=False)
    X_sub = X[idx]
    y_sub = y[idx] if has_labels else None

    with st.spinner(f"Generating FGSM adversarial examples ({len(idx):,} samples)…"):
        X_fgsm = _generate_fgsm(tier3, X_sub)
    with st.spinner(f"Generating PGD adversarial examples ({len(idx):,} samples, 10 iters)…"):
        X_pgd = _generate_pgd(tier3, X_sub)

    with st.spinner("Running adversarial inference on both models…"):
        t2_sub  = _predict_keras(tier2, X_sub)  if tier2 else None
        t3_sub  = _predict_torch(tier3, X_sub)
        t2_fgsm = _predict_keras(tier2, X_fgsm) if tier2 else None
        t2_pgd  = _predict_keras(tier2, X_pgd)  if tier2 else None
        t3_fgsm = _predict_torch(tier3, X_fgsm)
        t3_pgd  = _predict_torch(tier3, X_pgd)

    _render_adversarial_results(y_sub, has_labels,
                                 t2_sub,  t3_sub,
                                 t2_fgsm, t2_pgd,
                                 t3_fgsm, t3_pgd)


# ─────────────────────────────────────────────────────────────────────────────
# Shared render helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_clean_metrics_cards(metrics, tier_label):
    """Render KPI cards for a single tier's clean metrics."""
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"{tier_label} Accuracy", f"{metrics.get('accuracy',0)*100:.2f}%")
    c2.metric("Precision", f"{metrics.get('precision',0)*100:.2f}%")
    c3.metric("Recall", f"{metrics.get('recall',0)*100:.2f}%")
    c4.metric("F1-Score", f"{metrics.get('f1_score',0)*100:.2f}%")
    c5.metric("AUC-ROC", f"{metrics.get('auc_roc',0):.4f}" if 'auc_roc' in metrics else "N/A")


def _render_robust_table(tier_label, clean, fgsm, pgd):
    """Render a robustness summary table for a tier."""
    rows = []
    if fgsm:
        rows.append({
            'Attack': 'FGSM',
            'Clean Accuracy (%)': round(fgsm.get('clean_accuracy', 0) * 100, 2),
            'Robust Accuracy (%)': round(fgsm.get('robust_accuracy', 0) * 100, 2),
            'Accuracy Drop (pp)': round(fgsm.get('accuracy_drop', 0) * 100, 2),
            'Robustness Ratio': round(fgsm.get('robustness_ratio', 0), 4),
        })
    if pgd:
        rows.append({
            'Attack': 'PGD',
            'Clean Accuracy (%)': round(pgd.get('clean_accuracy', 0) * 100, 2),
            'Robust Accuracy (%)': round(pgd.get('robust_accuracy', 0) * 100, 2),
            'Accuracy Drop (pp)': round(pgd.get('accuracy_drop', 0) * 100, 2),
            'Robustness Ratio': round(pgd.get('robustness_ratio', 0), 4),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_per_class_table(per_class_data):
    """Render a per-class metrics table."""
    rows = []
    for cls_name, m in per_class_data.items():
        rows.append({
            'Attack Type': cls_name,
            'Precision (%)': round(m['precision'] * 100, 2),
            'Recall (%)': round(m['recall'] * 100, 2),
            'F1-Score (%)': round(m['f1_score'] * 100, 2),
            'FPR (%)': round(m.get('fpr', 0) * 100, 4),
            'Support': m['support'],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_clean_results(t2_preds, t3_preds, y, has_labels):
    cols = st.columns(2)
    for col, preds, name, color in zip(
        cols,
        [t2_preds, t3_preds],
        ['Tier 2 — DNN', 'Tier 3 — Robust DNN'],
        ['#3498db', '#2ecc71']
    ):
        if preds is None:
            col.warning(f"{name}: model not loaded")
            continue
        with col:
            st.markdown(f"**{name}**")
            if has_labels:
                acc = np.mean(preds == y)
                st.metric("Clean Accuracy", f"{acc*100:.2f}%")
            counts = pd.Series([LABEL_MAP.get(p, str(p)) for p in preds]).value_counts()
            fig = go.Figure(go.Bar(
                x=counts.index.tolist(), y=counts.values.tolist(),
                marker_color=color,
                text=counts.values.tolist(), textposition='outside'
            ))
            fig.update_layout(title=f'{name} — Prediction Distribution',
                              yaxis_title='Count', height=320,
                              margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)


def _render_adversarial_results(y_sub, has_labels,
                                 t2_sub, t3_sub,
                                 t2_fgsm, t2_pgd,
                                 t3_fgsm, t3_pgd):
    rows = []
    for label, clean_p, fgsm_p, pgd_p in [
        ('Tier 2 (DNN)',        t2_sub,  t2_fgsm, t2_pgd),
        ('Tier 3 (Robust DNN)', t3_sub,  t3_fgsm, t3_pgd),
    ]:
        if clean_p is None:
            continue
        row = {'Model': label}
        if has_labels and y_sub is not None:
            ca = np.mean(clean_p == y_sub) * 100
            row['Clean Acc (%)']   = round(ca, 2)
            if fgsm_p is not None:
                fa = np.mean(fgsm_p == y_sub) * 100
                row['FGSM Acc (%)']  = round(fa, 2)
                row['FGSM Drop (pp)'] = round(ca - fa, 2)
            if pgd_p is not None:
                pa = np.mean(pgd_p == y_sub) * 100
                row['PGD Acc (%)']   = round(pa, 2)
                row['PGD Drop (pp)'] = round(ca - pa, 2)
        rows.append(row)

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if has_labels and y_sub is not None and rows:
        systems   = [r['Model'] for r in rows]
        clean_v   = [r.get('Clean Acc (%)', 0) for r in rows]
        fgsm_v    = [r.get('FGSM Acc (%)',  0) for r in rows]
        pgd_v     = [r.get('PGD Acc (%)',   0) for r in rows]

        fig = go.Figure(data=[
            go.Bar(name='Clean',      x=systems, y=clean_v, marker_color='#3498db',
                   text=[f"{v:.1f}%" for v in clean_v], textposition='outside'),
            go.Bar(name='FGSM Robust', x=systems, y=fgsm_v, marker_color='#e67e22',
                   text=[f"{v:.1f}%" for v in fgsm_v], textposition='outside'),
            go.Bar(name='PGD Robust',  x=systems, y=pgd_v,  marker_color='#e74c3c',
                   text=[f"{v:.1f}%" for v in pgd_v],  textposition='outside'),
        ])
        fig.update_layout(
            barmode='group', title='Live Inference — Clean vs Adversarial Accuracy',
            yaxis=dict(title='Accuracy (%)', range=[0, 115]),
            height=420,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        for preds, attack in [(t3_fgsm, 'FGSM'), (t3_pgd, 'PGD')]:
            if preds is None:
                continue
            counts = pd.Series([LABEL_MAP.get(p, str(p)) for p in preds]).value_counts()
            fig = go.Figure(go.Bar(
                x=counts.index.tolist(), y=counts.values.tolist(),
                marker_color='#e74c3c',
                text=counts.values.tolist(), textposition='outside'
            ))
            fig.update_layout(title=f'Tier 3 Predictions on {attack} Data', height=300)
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_cicids(df_raw):
    import pickle

    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    # Labels
    has_labels = 'Label' in df.columns
    if has_labels:
        y = df['Label'].str.strip().map(CICIDS_ATTACK_LABELS).fillna(0).astype(int).values
        df = df.drop(columns=['Label'])
    else:
        y = np.zeros(len(df), dtype=int)

    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))
    X = df.values.astype(np.float32)

    # Scaler
    sp = _find_artifact('scaler.pkl')
    if sp:
        with open(sp, 'rb') as f:
            scaler = pickle.load(f)
        n = scaler.n_features_in_
        if X.shape[1] >= n:
            X = scaler.transform(X[:, :n])
        else:
            st.warning(f"⚠️ Scaler expects {n} features, got {X.shape[1]}. Skipping scaling.")

    # Feature selector
    fp = _find_artifact('feature_selector.pkl')
    if fp:
        with open(fp, 'rb') as f:
            selector = pickle.load(f)
        try:
            X = selector.transform(X)
        except Exception:
            X = X[:, :35]

    return X.astype(np.float32), y, has_labels


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_models(input_dim, num_classes):
    from src.adversarial_attacks.attack_utils import PyTorchDNN
    tier2, tier3 = None, None

    t2p = _find_model('models/tier2/best_model.h5')
    if t2p:
        try:
            import tensorflow as tf
            tier2 = tf.keras.models.load_model(t2p)
        except Exception as e:
            st.warning(f"⚠️ Tier 2 load error: {e}")

    t3p = _find_model('models/tier3/robust_model.pth')
    if t3p:
        try:
            tier3 = PyTorchDNN(input_dim=input_dim, num_classes=num_classes)
            tier3.load_state_dict(torch.load(t3p, map_location='cpu'))
            tier3.eval()
        except Exception as e:
            st.warning(f"⚠️ Tier 3 load error: {e}")

    return tier2, tier3


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def _predict_keras(model, X):
    return np.argmax(model.predict(X, verbose=0), axis=1)


def _predict_torch(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X)).argmax(dim=1).numpy()


def _generate_fgsm(model, X, epsilon=0.1):
    from src.adversarial_attacks.fgsm import fgsm_attack
    model.eval()
    x_t = torch.FloatTensor(X)
    with torch.no_grad():
        y_t = model(x_t).argmax(dim=1)
    return fgsm_attack(model, x_t, y_t, epsilon=epsilon).detach().numpy()


def _generate_pgd(model, X, epsilon=0.1, alpha=0.01, iters=10):
    from src.adversarial_attacks.pgd import pgd_attack
    model.eval()
    x_t = torch.FloatTensor(X)
    with torch.no_grad():
        y_t = model(x_t).argmax(dim=1)
    return pgd_attack(model, x_t, y_t,
                      epsilon=epsilon, alpha=alpha,
                      num_iterations=iters).detach().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def _project_root():
    here = os.path.abspath(__file__)
    for _ in range(6):
        here = os.path.dirname(here)
        if os.path.exists(os.path.join(here, 'main.py')):
            return here
    return os.getcwd()


def _find_model(rel):
    p = os.path.join(_project_root(), rel)
    return p if os.path.exists(p) else None


def _find_artifact(filename):
    p = os.path.join(_project_root(), 'models', 'preprocessing', filename)
    return p if os.path.exists(p) else None


def _find_results_json():
    p = os.path.join(_project_root(), 'evaluation_results', 'evaluation_results.json')
    return p if os.path.exists(p) else None
