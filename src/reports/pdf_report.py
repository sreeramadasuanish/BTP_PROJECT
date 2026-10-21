"""ReportLab PDF report generator for FED4FED statistical fairness evaluations."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def generate_pdf_report(
    output_pdf_path: Path | str,
    manifest: Dict[str, Any],
    partition_metadata: Dict[str, Any],
    round_logs: pd.DataFrame,
    bootstrap_summary: Dict[str, Any],
    fairness_results: Dict[str, Any],
    client_statistics_df: pd.DataFrame
) -> None:
    """Generate professional multi-page research PDF evaluation report."""
    pdf_path = Path(output_pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    primary_color = colors.HexColor("#1E3A8A")   # Deep navy
    slate_dark = colors.HexColor("#0F172A")
    slate_muted = colors.HexColor("#475569")
    accent_fair = colors.HexColor("#059669")     # Emerald green
    accent_unfair = colors.HexColor("#DC2626")   # Rose red
    light_bg = colors.HexColor("#F8FAFC")
    border_color = colors.HexColor("#CBD5E1")

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=primary_color,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=slate_muted,
    )
    h1_style = ParagraphStyle(
        "Heading1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=slate_dark,
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=slate_dark,
    )
    bold_style = ParagraphStyle(
        "BodyBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=slate_dark,
    )

    elements = []

    # 1. Header Banner
    elements.append(Paragraph("FED4FED: Statistical Fairness Evaluation Report", title_style))
    elements.append(Paragraph("Privacy-Preserving Verification for Federated Learning Models", subtitle_style))
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceBefore=2, spaceAfter=10))

    # 2. Executive Verdict Banner
    is_unfair = fairness_results.get("reject_h0", False)
    verdict_text = fairness_results.get("overall_verdict", "INCONCLUSIVE")
    verdict_color = accent_unfair if is_unfair else accent_fair

    verdict_p = Paragraph(
        f"<b>FAIRNESS VERDICT: {verdict_text}</b><br/>"
        f"<font size=8>{fairness_results.get('plain_english', '')}</font>",
        ParagraphStyle(
            "VerdictStyle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=verdict_color,
        )
    )

    verdict_table = Table([[verdict_p]], colWidths=[530])
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), light_bg),
        ("BOX", (0, 0), (-1, -1), 1.5, verdict_color),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    elements.append(verdict_table)
    elements.append(Spacer(1, 12))

    # 3. Experiment Configuration Summary Table
    elements.append(Paragraph("1. Experiment & Architecture Configuration", h1_style))
    exp_info = [
        [Paragraph("<b>Experiment ID:</b>", body_style), Paragraph(manifest.get("experiment_id", "N/A"), body_style),
         Paragraph("<b>Date / Time:</b>", body_style), Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), body_style)],
        [Paragraph("<b>Dataset:</b>", body_style), Paragraph(manifest.get("dataset", "N/A"), body_style),
         Paragraph("<b>Scenario:</b>", body_style), Paragraph(manifest.get("scenario", "N/A"), body_style)],
        [Paragraph("<b>Clients (K):</b>", body_style), Paragraph(str(manifest.get("num_clients", 100)), body_style),
         Paragraph("<b>Seed:</b>", body_style), Paragraph(str(manifest.get("seed", 42)), body_style)],
        [Paragraph("<b>FL Rounds:</b>", body_style), Paragraph(str(manifest.get("num_rounds", 10)), body_style),
         Paragraph("<b>Final Val Accuracy:</b>", body_style), Paragraph(f"{manifest.get('final_accuracy', 0.0):.4f}", body_style)],
        [Paragraph("<b>Partition Method:</b>", body_style), Paragraph(partition_metadata.get("method", "N/A"), body_style),
         Paragraph("<b>Bootstrap Replications (B):</b>", body_style), Paragraph(str(bootstrap_summary.get("num_resamples", 200)), body_style)],
    ]
    cfg_table = Table(exp_info, colWidths=[120, 145, 120, 145])
    cfg_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), light_bg),
        ("GRID", (0, 0), (-1, -1), 0.5, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(cfg_table)
    elements.append(Spacer(1, 12))

    # 4. Statistical Testing Outcomes
    elements.append(Paragraph("2. Statistical Fairness Hypothesis Tests", h1_style))
    test_rows = [
        [Paragraph("<b>Test Algorithm</b>", bold_style),
         Paragraph("<b>Statistic</b>", bold_style),
         Paragraph("<b>df</b>", bold_style),
         Paragraph("<b>p-value</b>", bold_style),
         Paragraph("<b>Significance (α)</b>", bold_style),
         Paragraph("<b>Decision</b>", bold_style)]
    ]

    if fairness_results.get("alg1_result"):
        a1 = fairness_results["alg1_result"]
        dec1 = "Reject H0 (Disparity)" if a1["reject_h0"] else "Fail to Reject (Fair)"
        p_str = f"{a1['p_value']:.4e}" if a1['p_value'] < 1e-3 else f"{a1['p_value']:.4f}"
        test_rows.append([
            Paragraph("Algorithm 1 (Graybill-Deal / T1)", body_style),
            Paragraph(f"{a1['test_statistic']:.4f}", body_style),
            Paragraph(str(a1['degrees_of_freedom']), body_style),
            Paragraph(p_str, body_style),
            Paragraph(f"{a1['significance_level']:.3f}", body_style),
            Paragraph(f"<b>{dec1}</b>", body_style),
        ])

    if fairness_results.get("alg2_result"):
        a2 = fairness_results["alg2_result"]
        dec2 = "Reject H0 (Disparity)" if a2["reject_h0"] else "Fail to Reject (Fair)"
        p_str = f"{a2['p_value']:.4e}" if a2['p_value'] < 1e-3 else f"{a2['p_value']:.4f}"
        test_rows.append([
            Paragraph("Algorithm 2 (Welch-Type F)", body_style),
            Paragraph(f"{a2['test_statistic']:.4f}", body_style),
            Paragraph(f"({a2['df_numerator']:.0f}, {a2['df_denominator']:.1f})", body_style),
            Paragraph(p_str, body_style),
            Paragraph(f"{a2['significance_level']:.3f}", body_style),
            Paragraph(f"<b>{dec2}</b>", body_style),
        ])

    tests_table = Table(test_rows, colWidths=[170, 70, 70, 70, 75, 75])
    tests_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ("GRID", (0, 0), (-1, -1), 0.5, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(tests_table)
    elements.append(Spacer(1, 12))

    # 5. Algorithm 3 Performance Confidence Interval
    elements.append(Paragraph("3. Algorithm 3: Global Performance Confidence Interval", h1_style))
    ci_res = fairness_results.get("confidence_interval", {})
    if ci_res and ci_res.get("is_permitted", False):
        ci_text = (
            f"<b>Status:</b> Computed under homogeneous fairness condition.<br/>"
            f"<b>Confidence Level:</b> {ci_res['confidence_level']*100:.1f}%<br/>"
            f"<b>Pooled Performance Estimate (Graybill-Deal):</b> {ci_res['pooled_estimate']:.4f}<br/>"
            f"<b>Confidence Interval:</b> [{ci_res['pooled_ci_lower']:.4f}, {ci_res['pooled_ci_upper']:.4f}]<br/>"
            f"<b>Standard Error:</b> {ci_res['pooled_se']:.4f} (Effective df = {ci_res['effective_df']:.1f})"
        )
    else:
        reason = ci_res.get("reason", "Condition not met.") if ci_res else "Fairness condition violated."
        ci_text = (
            f"<b>Status:</b> UNAVAILABLE (Condition Not Met)<br/>"
            f"<b>Decision Rule Rationale:</b> {reason}"
        )

    ci_table = Table([[Paragraph(ci_text, body_style)]], colWidths=[530])
    ci_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), light_bg),
        ("BOX", (0, 0), (-1, -1), 0.5, border_color),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(ci_table)
    elements.append(Spacer(1, 12))

    # 6. Sample Client Bootstrap Statistics Table
    elements.append(Paragraph("4. Sample Decentralized Client Performance Statistics", h1_style))
    sample_df = client_statistics_df.head(8)
    client_rows = [
        [Paragraph("<b>Client ID</b>", bold_style),
         Paragraph("<b>Test Samples (n_i)</b>", bold_style),
         Paragraph("<b>Observed Metric</b>", bold_style),
         Paragraph("<b>Bootstrap Mean</b>", bold_style),
         Paragraph("<b>Variance (S_i^2)</b>", bold_style),
         Paragraph("<b>Std Error (SE)</b>", bold_style)]
    ]
    for _, row in sample_df.iterrows():
        client_rows.append([
            Paragraph(f"Client {int(row['client_id']):02d}", body_style),
            Paragraph(str(int(row["sample_count"])), body_style),
            Paragraph(f"{row['observed_metric']:.4f}", body_style),
            Paragraph(f"{row['bootstrap_mean']:.4f}", body_style),
            Paragraph(f"{row['bootstrap_variance']:.6f}", body_style),
            Paragraph(f"{row['bootstrap_se']:.4f}", body_style),
        ])

    c_table = Table(client_rows, colWidths=[80, 90, 90, 90, 90, 90])
    c_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ("GRID", (0, 0), (-1, -1), 0.5, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(c_table)
    elements.append(Spacer(1, 14))

    # 7. Methodology & Software Footer
    elements.append(HRFlowable(width="100%", thickness=0.5, color=border_color, spaceBefore=4, spaceAfter=8))
    elements.append(Paragraph(
        "<b>Scientific Methodology:</b> Fed4Fed privacy-preserving statistical homogeneity testing using non-parametric "
        "bootstrapping, Graybill-Deal common mean estimation, and Welch-type heteroscedastic F-testing. "
        "Generated by FED4FED Research Suite v1.0.0.",
        subtitle_style
    ))

    doc.build(elements)
