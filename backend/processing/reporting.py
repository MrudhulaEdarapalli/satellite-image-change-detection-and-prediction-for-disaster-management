from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
import os
from datetime import datetime

def generate_disaster_report(output_path, before_img_path, after_img_path, overlay_img_path, stats):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor("#0f172a"),
        alignment=1,
        spaceAfter=20
    )
    story.append(Paragraph("DISASTER IMPACT ANALYSIS REPORT", title_style))
    
    # Metadata
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"<b>Date generated:</b> {date_str}", styles['Normal']))
    story.append(Paragraph(f"<b>Analysis Engine:</b> Neural Siamese U-Net v2.0", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Stats Table
    data = [
        ["Metric", "Value"],
        ["Overall Impact Score", f"{stats['damage_percentage']:.2f}%"],
        ["Total Pixels Affected", f"{stats['total_changed_pixels']:,}"],
        ["Predicted Severity", stats['severity']],
        ["Vegetation Loss Est.", f"{stats['sectors']['vegetation_loss']:.1f}%"],
        ["Infrastructure Impact Est.", f"{stats['sectors']['infrastructure_damage']:.1f}%"]
    ]
    
    table = Table(data, colWidths=[2.5*inch, 2.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1"))
    ]))
    story.append(table)
    story.append(Spacer(1, 0.4 * inch))

    # Images Section
    story.append(Paragraph("<b>Imagery Assessment:</b>", styles['Heading2']))
    
    # Add Before and After
    img_width = 3 * inch
    img_height = 3 * inch
    
    img_row = [
        Image(before_img_path, width=img_width, height=img_height),
        Image(after_img_path, width=img_width, height=img_height)
    ]
    
    t_imgs = Table([img_row])
    story.append(t_imgs)
    
    story.append(Paragraph("<center>Left: T1 Baseline | Right: T2 Impact Image</center>", styles['Italic']))
    story.append(Spacer(1, 0.3 * inch))

    # Analytical Overlay
    story.append(Paragraph("<b>Neural Analytics Overlay:</b>", styles['Heading2']))
    story.append(Image(overlay_img_path, width=5*inch, height=5*inch))
    story.append(Paragraph("<center>Change detection mask overlaid on impacted areas</center>", styles['Italic']))

    # Conclusion
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("<b>Confidentiality & Disclaimer:</b>", styles['Heading3']))
    story.append(Paragraph("This report is generated for emergency response planning. Values are estimates based on neural inference and should be ground-verified where possible.", styles['Normal']))

    doc.build(story)
    return output_path
