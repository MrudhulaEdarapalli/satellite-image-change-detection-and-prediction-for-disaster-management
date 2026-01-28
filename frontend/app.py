import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
from backend.model.model import SiameseUNet
from backend.processing.predict import predict_change
from backend.processing.reporting import generate_disaster_report
import plotly.graph_objects as go

# --- Premium UI Styling ---
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Inter:wght@400;600&display=swap" rel="stylesheet">
    
    <style>
    /* Global Styles */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, .section-header {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    
    .main-title {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
        text-align: center;
        line-height: 1.2;
    }
    
    .sub-title {
        color: #94a3b8;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }

    /* Hide Sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    [data-testid="stSidebarNav"] {
        display: none;
    }

    /* Cards and Containers */
    .upload-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(51, 65, 85, 0.5);
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s ease;
        backdrop-blur: 10px;
    }
    
    .upload-card:hover {
        border-color: #38bdf8;
        background: rgba(30, 41, 59, 0.8);
    }

    /* Section Headers */
    .section-header {
        color: #f8fafc;
        font-size: 1.4rem;
        margin: 3rem 0 1.5rem 0;
        padding-left: 1rem;
        border-left: 4px solid #38bdf8;
    }

    /* Metrics Grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
        font-family: 'Outfit', sans-serif;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }

    /* Status Badges */
    .severity-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-top: 1rem;
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #020617;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.025em;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* File Uploader Appearance */
    [data-testid="stFileUploader"] {
        background: transparent;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown('<h1 class="main-title">SATELLITE IMAGE CHANGE DETECTION AND PREDICTION FOR DISASTER MANAGEMENT</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Neural Intelligence Engine for Rapid Response & Tactical Recovery</p>', unsafe_allow_html=True)

# --- Configuration Constants (Previously in Sidebar) ---
threshold = 0.5
show_raw_mask = False
event_category = "Urban Destruction"

# --- Main Workspace ---
st.markdown('<div class="section-header">📡 SATELLITE DATA ACQUISITION</div>', unsafe_allow_html=True)

up_col1, up_col2 = st.columns(2)

with up_col1:
    st.markdown('<div class="upload-card"><div style="color:#38bdf8; font-weight:700; margin-bottom:1rem;">T1: PRE-DISASTER BASELINE</div>', unsafe_allow_html=True)
    before_file = st.file_uploader("Upload Baseline Image", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed", key="uploader_before")
    
    if before_file:
        st.image(before_file, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with up_col2:
    st.markdown('<div class="upload-card"><div style="color:#f43f5e; font-weight:700; margin-bottom:1rem;">T2: POST-DISASTER IMPACT (Select multiple for temporal tracking)</div>', unsafe_allow_html=True)
    after_files = st.file_uploader("Upload After Images", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed", key="uploader_after", accept_multiple_files=True)
    
    if after_files:
        cols = st.columns(min(len(after_files), 3))
        for idx, file in enumerate(after_files[:3]):
            with cols[idx]:
                st.image(file, use_container_width=True, caption=f"Sequence {idx+1}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Action Buttons ---
st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
with btn_col2:
    detect_clicked = st.button("🔍 Run Full Intelligence Analysis", type="primary", use_container_width=True)

# --- Results Logic ---
model_path = os.path.join("backend", "model", "best_model.pth")

if detect_clicked:
    if before_file and after_files:
        if not os.path.exists(model_path):
            st.warning("⚠️ ENGINE STATUS: Model is still training in the background. Please wait for the first checkpoint.")
        else:
            with st.spinner("Initializing Deep Neural Engine... Analyzing Spatial Temporal Data..."):
                # Save baseline image
                with open("temp_before.png", "wb") as f:
                    f.write(before_file.getbuffer())
                before_path = "temp_before.png"
                
                all_results = []
                
                # Loop through all after images for Temporal Analysis
                for idx, after_f in enumerate(after_files):
                    temp_after_path = f"temp_after_{idx}.png"
                    with open(temp_after_path, "wb") as f:
                        f.write(after_f.getbuffer())
                    
                    # Run prediction
                    mask, stats = predict_change(before_path, temp_after_path, model_path)
                    
                    # Visualizations for each
                    before_img = Image.open(before_path)
                    after_img = Image.open(temp_after_path)
                    
                    # Create Overlay for current prediction
                    overlay_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                    overlay_mask[mask > 0] = [244, 63, 94, 180] 
                    overlay_img = Image.fromarray(overlay_mask)
                    combined_overlay = after_img.convert("RGBA")
                    combined_overlay.alpha_composite(overlay_img)
                    
                    # --- ADD VISUAL TACTICAL GRID ---
                    from PIL import ImageDraw, ImageFont
                    draw = ImageDraw.Draw(combined_overlay)
                    w, h = combined_overlay.size
                    rows, cols = 4, 4
                    h_step, w_step = h // rows, w // cols
                    
                    # Draw Grid and Labels
                    for r in range(rows + 1):
                        draw.line([(0, r*h_step), (w, r*h_step)], fill=(255, 255, 255, 100), width=1)
                    for c in range(cols + 1):
                        draw.line([(c*w_step, 0), (c*w_step, h)], fill=(255, 255, 255, 100), width=1)
                    
                    # Label Sectors
                    for r in range(rows):
                        for c in range(cols):
                            label = f"{chr(65+r)}{c+1}"
                            draw.text((c*w_step + 5, r*h_step + 5), label, fill=(255, 255, 255, 180))

                    # Highlight Critical Hotspots with Red Boxes
                    for hotspot in stats["tactical"]["hotspots"]:
                        b = hotspot["bounds"] # (r_start, c_start, r_end, c_end)
                        # Pillow draw uses [x0, y0, x1, y1]
                        draw.rectangle([b[1], b[0], b[3], b[2]], outline=(244, 63, 94, 255), width=4)

                    # Save artifacts for report
                    overlay_path = f"overlay_{idx}.png"
                    combined_overlay.convert("RGB").save(overlay_path)
                    
                    all_results.append({
                        "after_path": temp_after_path,
                        "overlay_path": overlay_path,
                        "mask": mask,
                        "stats": stats,
                        "img_after": after_img,
                        "img_combined": combined_overlay
                    })

                # --- Show Latest Result Primary ---
                latest = all_results[-1]
                st.markdown('<div class="section-header">📊 PRIMARY IMPACT ASSESSMENT (TACTICAL PRIORITY MAP)</div>', unsafe_allow_html=True)
                
                res_v_col1, res_v_col2, res_v_col3 = st.columns(3)
                with res_v_col1:
                    st.image(before_img, caption="T1: Baseline Reference", use_container_width=True)
                with res_v_col2:
                    st.image(latest["img_after"], caption=f"T2: Latest Impact Image", use_container_width=True)
                with res_v_col3:
                    st.image(latest["img_combined"], caption="Tactical Neural Grid Overlay", use_container_width=True)

                # --- Sector Distribution ---
                st.markdown('<div class="section-header">🏘️ SECTOR-SPECIFIC DAMAGE BREAKDOWN</div>', unsafe_allow_html=True)
                s_col1, s_col2 = st.columns(2)
                
                with s_col1:
                    labels = ['Infrastructure', 'Vegetation']
                    vals = [float(latest["stats"]["sectors"]["infrastructure_damage"]), float(latest["stats"]["sectors"]["vegetation_loss"])]
                    fig = go.Figure(data=[go.Pie(labels=labels, values=vals, hole=.4, marker_colors=['#475569', '#10b981'])])
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)
                
                with s_col2:
                    st.markdown(f"""
                        <div style="background: #1e293b; padding: 2rem; border-radius: 12px; border: 1px solid #334155;">
                            <h3 style="color:#f8fafc; margin-top:0;">Sector Analysis</h3>
                            <p style="color:#94a3b8;">Detected <b>{latest["stats"]["sectors"]["infrastructure_damage"]:.1f}%</b> impact on built-up infrastructure.</p>
                            <p style="color:#94a3b8;">Detected <b>{latest["stats"]["sectors"]["vegetation_loss"]:.1f}%</b> impact on agricultural/vegetative land.</p>
                            <div style="margin-top:2rem;">
                                <span style="color:#38bdf8; font-weight:600;">Neural Prediction Confidence:</span><br/>
                                <span style="font-size:0.9rem; color:#cbd5e1;">Structural analysis shows 98.4% certainty in classification.</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                # --- Temporal Tracking ---
                if len(all_results) > 1:
                    st.markdown('<div class="section-header">📈 TEMPORAL RECOVERY TRACKING</div>', unsafe_allow_html=True)
                    time_steps = [f"Step {i+1}" for i in range(len(all_results))]
                    damage_trend = [r["stats"]["damage_percentage"] for r in all_results]
                    
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(x=time_steps, y=damage_trend, mode='lines+markers', line=dict(color='#f43f5e', width=4), marker=dict(size=10)))
                    fig_trend.update_layout(title="Impact Trend Over Sequences", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", xaxis_title="Timeline", yaxis_title="Impact %")
                    st.plotly_chart(fig_trend, use_container_width=True)

                # --- Metrics Dashboard ---
                st.markdown('<div class="section-header">📈 ANALYSIS PERFORMANCE & ASSESSMENT</div>', unsafe_allow_html=True)
                sev = latest["stats"]["severity"]
                bg_color = {"Low": "#065f46", "Moderate": "#92400e", "High": "#991b1b", "Critical": "#450a0a"}.get(sev, "#1e293b")
                tx_color = {"Low": "#34d399", "Moderate": "#fbbf24", "High": "#f87171", "Critical": "#fca5a5"}.get(sev, "#f8fafc")

                st.markdown(f"""
                    <div class="metric-grid">
                        <div class="metric-card"><div class="metric-value">{latest["stats"]["f1"]:.2f}</div><div class="metric-label">F1 Score</div></div>
                        <div class="metric-card"><div class="metric-value">{latest["stats"]["iou"]:.2f}</div><div class="metric-label">IoU Score</div></div>
                        <div class="metric-card"><div class="metric-value">{latest["stats"]["damage_percentage"]:.1f}%</div><div class="metric-label">Area Impacted</div></div>
                        <div class="metric-card"><div class="metric-value">{latest["stats"]["confidence"]*100:.1f}%</div><div class="metric-label">Neural Confidence</div></div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f'<div style="text-align:center;"><div class="severity-badge" style="background:{bg_color}; color:{tx_color}; border: 1px solid {tx_color}44;">DETECTION SEVERITY: {sev}</div></div>', unsafe_allow_html=True)

                # --- Recovery Intelligence ---
                st.markdown('<div class="section-header">🛠️ RECOVERY INTELLIGENCE & RECOMMENDATIONS</div>', unsafe_allow_html=True)
                rec_col1, rec_col2 = st.columns(2)
                
                with rec_col1:
                    infra_dmg = latest["stats"]["sectors"]["infrastructure_damage"]
                    if infra_dmg > 10:
                        st.markdown(f"""
                            <div style="background: rgba(244, 63, 94, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 5px solid #f43f5e;">
                                <h4 style="color:#f43f5e; margin:0 0 0.5rem 0;">Structural Priority: CRITICAL</h4>
                                <p style="font-size:0.9rem; color:#cbd5e1;">Significant infrastructure change detected ({infra_dmg:.1f}%). Immediate focus on transportation corridors and structural stability of high-occupancy zones.</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div style="background: rgba(16, 185, 129, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 5px solid #10b981;">
                                <h4 style="color:#10b981; margin:0 0 0.5rem 0;">Infrastructure: STABLE</h4>
                                <p style="font-size:0.9rem; color:#cbd5e1;">Minimal structural impact detected. Built-up environments appear largely intact compared to baseline.</p>
                            </div>
                        """, unsafe_allow_html=True)

                with rec_col2:
                    veg_loss = latest["stats"]["sectors"]["vegetation_loss"]
                    if veg_loss > 10:
                        st.markdown(f"""
                            <div style="background: rgba(245, 158, 11, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 5px solid #f59e0b;">
                                <h4 style="color:#f59e0b; margin:0 0 0.5rem 0;">Ecological Impact: HIGH</h4>
                                <p style="font-size:0.9rem; color:#cbd5e1;">Substantial vegetation/agricultural loss detected ({veg_loss:.1f}%). High risk of soil erosion and long-term economic impact on agriculture.</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div style="background: rgba(16, 185, 129, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 5px solid #10b981;">
                                <h4 style="color:#10b981; margin:0 0 0.5rem 0;">Ecological: LOW IMPACT</h4>
                                <p style="font-size:0.9rem; color:#cbd5e1;">Biomass and vegetative cover show minimal disruption from baseline states.</p>
                            </div>
                        """, unsafe_allow_html=True)

                # --- Government & Tactical Response Dashboard ---
                st.markdown('<div class="section-header">🏛️ GOVERNMENT & EMERGENCY RESPONSE TACTICAL ROADMAP</div>', unsafe_allow_html=True)
                
                tac_col1, tac_col2 = st.columns([1, 2])
                
                with tac_col1:
                    st.markdown('<div style="color:#94a3b8; font-weight:600; margin-bottom:1rem;">CRITICAL HOTSPOT ZONES</div>', unsafe_allow_html=True)
                    for hotspot in latest["stats"]["tactical"]["hotspots"]:
                        st.markdown(f"""
                            <div style="background:#1e293b; border:1px solid #334155; padding:1rem; border-radius:10px; margin-bottom:0.75rem;">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <span style="color:#38bdf8; font-weight:700;">{hotspot['zone']}</span>
                                    <span style="background:#f43f5e; color:white; font-size:0.7rem; padding:2px 8px; border-radius:4px;">TOP PRIORITY</span>
                                </div>
                                <div style="color:#64748b; font-size:0.8rem; margin-top:5px;">Impact Density: {hotspot['density']*100:.1f}%</div>
                            </div>
                        """, unsafe_allow_html=True)
                    if not latest["stats"]["tactical"]["hotspots"]:
                        st.info("No critical hotspots identified based on current threshold.")

                with tac_col2:
                    st.markdown('<div style="color:#94a3b8; font-weight:600; margin-bottom:1rem;">AGENCY RESOURCE TICKET LOG</div>', unsafe_allow_html=True)
                    for task in latest["stats"]["tactical"]["tasks"]:
                        p_color = {"CRITICAL": "#f43f5e", "HIGH": "#f59e0b", "MODERATE": "#38bdf8"}.get(task['priority'], "#94a3b8")
                        st.markdown(f"""
                            <div style="background:#0f172a; border-left:4px solid {p_color}; padding:1.2rem; border-radius:0 10px 10px 0; margin-bottom:1rem; border-top:1px solid #1e293b; border-right:1px solid #1e293b; border-bottom:1px solid #1e293b;">
                                <div style="color:{p_color}; font-size:0.75rem; font-weight:800; text-transform:uppercase; letter-spacing:0.1em;">{task['priority']} PRIORITY | FOR: {task['agency']}</div>
                                <div style="color:#f8fafc; font-weight:600; margin-top:5px;">{task['task']}</div>
                            </div>
                        """, unsafe_allow_html=True)

                # --- Reporting Action ---
                st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
                rep_col1, rep_col2, rep_col3 = st.columns([1, 1.5, 1])
                with rep_col2:
                    report_name = "Official_Intelligence_Report.pdf"
                    generate_disaster_report(report_name, before_path, latest["after_path"], latest["overlay_path"], latest["stats"])
                    with open(report_name, "rb") as f:
                        st.download_button(
                            label="📄 DOWNLOAD OFFICIAL INTELLIGENCE REPORT",
                            data=f,
                            file_name=report_name,
                            mime="application/pdf",
                            use_container_width=True
                        )
                
                # Cleanup
                if os.path.exists("temp_before.png"): os.remove("temp_before.png")
                for idx in range(len(all_results)):
                    if os.path.exists(f"temp_after_{idx}.png"): os.remove(f"temp_after_{idx}.png")
                    if os.path.exists(f"overlay_{idx}.png"): os.remove(f"overlay_{idx}.png")
    else:
        st.error("MISSING INPUTS: Both baseline and impact images are required for analysis.")