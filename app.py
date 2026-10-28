import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
import json
import random
import io
import time
from datetime import datetime
from fpdf import FPDF

# -----------------------------------------------------------------------------
# Configuration de la page Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Matrice de Risque Collaborative & IA",
    page_icon="🛡️",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .ai-box {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        border: 1px solid #7DD3FC;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 15px;
    }
    .action-card {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 16px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .export-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .badge-ai {
        background-color: #0284C7;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    div[data-testid="stRadio"] > div {
        flex-direction: row;
        gap: 8px;
        background-color: #F1F5F9;
        padding: 6px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    div[data-testid="stRadio"] label {
        background: white;
        padding: 8px 16px;
        border-radius: 8px;
        border: 1px solid #CBD5E1;
        font-weight: 600;
        cursor: pointer;
    }
    div[data-testid="stDataFrame"] td, 
    div[data-testid="stDataFrame"] th, 
    div[data-testid="stDataFrame"] [role="gridcell"] {
        white-space: nowrap !important;
        text-overflow: ellipsis;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Fonction utilitaire : Arrondi mathématique au plus près
# -----------------------------------------------------------------------------
def round_nearest(val):
    try:
        return int(np.floor(float(val) + 0.5))
    except:
        return 1

# -----------------------------------------------------------------------------
# Initialisation de la Session State
# -----------------------------------------------------------------------------
if 'project_created' not in st.session_state:
    st.session_state.project_created = False

if 'api_validated' not in st.session_state:
    st.session_state.api_validated = False

if 'saved_api_key' not in st.session_state:
    st.session_state.saved_api_key = ""

if 'saved_model' not in st.session_state:
    st.session_state.saved_model = "gemini-3.8-flash"

if 'current_main_tab' not in st.session_state:
    st.session_state.current_main_tab = "1️⃣ Saisies Individuelles (Répondants)"

if 'project_config' not in st.session_state:
    st.session_state.project_config = {
        "project_name": "Projet Modernisation SI",
        "project_code": "PRJ-2026-001",
        "project_period": "Q1 2026 - Q4 2026",
        "business_analyst": "Alexandre Martin",
        "max_probability": 5,
        "max_impact": 5,
        "participants": [],
        "suggested_risks": []
    }

if 'raw_submissions' not in st.session_state:
    st.session_state.raw_submissions = pd.DataFrame(columns=[
        "N°", "Participant", "User_Slot", "Risque identifié", "Probabilité", "Impact", "Mitigation"
    ])

if 'consensus_data' not in st.session_state:
    st.session_state.consensus_data = pd.DataFrame()

if not st.session_state.raw_submissions.empty and "N°" not in st.session_state.raw_submissions.columns:
    st.session_state.raw_submissions.insert(0, "N°", range(1, len(st.session_state.raw_submissions) + 1))

# -----------------------------------------------------------------------------
# Fonction de consolidation locale (sans Gemini)
# -----------------------------------------------------------------------------
def consolidate_locally(df_submissions):
    if df_submissions.empty:
        return pd.DataFrame()
        
    grouped = df_submissions.groupby("Risque identifié")
    rows = []
    for idx, (risk_title, group) in enumerate(grouped):
        contributors = [str(c) for c in group["Participant"].dropna().unique() if str(c).strip()]
        contributors_count = len(contributors)
        contributors_str = ", ".join(contributors)
        
        p_min = int(group["Probabilité"].min())
        p_max = int(group["Probabilité"].max())
        p_avg = round(float(group["Probabilité"].mean()), 1)
        
        i_min = int(group["Impact"].min())
        i_max = int(group["Impact"].max())
        i_avg = round(float(group["Impact"].mean()), 1)
        
        prio_series = group["Probabilité"] * group["Impact"]
        prio_min = int(prio_series.min())
        prio_max = int(prio_series.max())
        prio_avg = round(float(prio_series.mean()), 1)
        
        cons_prob = round_nearest(p_avg)
        cons_imp = round_nearest(i_avg)
        final_priority = cons_prob * cons_imp
        
        prob_range = p_max - p_min
        impact_range = i_max - i_min
        
        if prob_range >= 2 or impact_range >= 2:
            status = "À discuter"
            dispersion = f"Écart important (Prob: {p_min}-{p_max}, Impact: {i_min}-{i_max})"
        else:
            status = "Validé"
            dispersion = "Consensus général (écarts faibles)"
            
        raw_mitigations = [str(r.get("Mitigation", "")).strip() for _, r in group.iterrows() if str(r.get("Mitigation", "")).strip()]
        unique_mitigs = list(dict.fromkeys(raw_mitigations))
        mitigation_notes = " ; ".join(unique_mitigs) if unique_mitigs else "Aucune mesure documentée"
        
        rows.append({
            "CODE": f"R{idx+1}",
            "CANONICAL RISK": risk_title,
            "CONTRIBUTORS COUNT": contributors_count,
            "CONTRIBUTORS": contributors_str,
            "PROB_MIN": p_min,
            "PROB_MAX": p_max,
            "PROB_AVG": p_avg,
            "IMPACT_MIN": i_min,
            "IMPACT_MAX": i_max,
            "IMPACT_AVG": i_avg,
            "PRIO_MIN": prio_min,
            "PRIO_MAX": prio_max,
            "PRIO_AVG": prio_avg,
            "CONSENSUS PROBABILITY": cons_prob,
            "CONSENSUS IMPACT": cons_imp,
            "FINAL PRIORITY": final_priority,
            "DISPERSION NOTE": dispersion,
            "MITIGATION NOTES": mitigation_notes,
            "STATUS": status
        })
    df_res = pd.DataFrame(rows)
    df_res = df_res.sort_values(by=["FINAL PRIORITY", "CONTRIBUTORS COUNT"], ascending=[False, False]).reset_index(drop=True)
    df_res["CODE"] = [f"R{i+1}" for i in range(len(df_res))]
    return df_res

# -----------------------------------------------------------------------------
# Fonctions d'Exportation (Excel & PDF avec Matrice Visuelle)
# -----------------------------------------------------------------------------
def generate_excel_export(cfg, df_final, df_raw):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        rename_mapping = {
            "CODE": "Code",
            "CANONICAL RISK": "Risque synthétisé",
            "CONSENSUS PROBABILITY": "Probabilité Validée",
            "CONSENSUS IMPACT": "Impact Validé",
            "FINAL PRIORITY": "Priorité Finale",
            "STATUS": "Statut",
            "CONTRIBUTORS": "Contributeurs",
            "CONTRIBUTORS COUNT": "Nb contributeurs",
            "PROB_MIN": "Prob Min",
            "PROB_MAX": "Prob Max",
            "PROB_AVG": "Prob Moy",
            "IMPACT_MIN": "Imp Min",
            "IMPACT_MAX": "Imp Max",
            "IMPACT_AVG": "Imp Moy",
            "PRIO_MIN": "Prio Min",
            "PRIO_MAX": "Prio Max",
            "PRIO_AVG": "Prio Moy",
            "MITIGATION NOTES": "Mitigation retenue",
            "DISPERSION NOTE": "Analyse des écarts"
        }
        
        df_export = df_final.rename(columns=rename_mapping)
        cols_register_fr = [
            "Code", "Risque synthétisé", "Probabilité Validée", "Impact Validé", 
            "Priorité Finale", "Statut", "Contributeurs", "Nb contributeurs", 
            "Prob Min", "Prob Max", "Prob Moy", "Imp Min", "Imp Max", "Imp Moy",
            "Prio Min", "Prio Max", "Prio Moy", "Mitigation retenue", "Analyse des écarts"
        ]
        cols_present = [c for c in cols_register_fr if c in df_export.columns]
        df_export[cols_present].to_excel(writer, sheet_name="Registre_Consensus", index=False)
        
        max_p = cfg["max_probability"]
        max_i = cfg["max_impact"]
        matrix_grid = [["" for _ in range(max_i)] for _ in range(max_p)]
        for idx, row in df_final.iterrows():
            try:
                p = int(row["CONSENSUS PROBABILITY"])
                i = int(row["CONSENSUS IMPACT"])
                code_val = str(row.get("CODE", f"R{idx+1}"))
                if 1 <= p <= max_p and 1 <= i <= max_i:
                    if matrix_grid[p-1][i-1] == "":
                        matrix_grid[p-1][i-1] = code_val
                    else:
                        matrix_grid[p-1][i-1] += f", {code_val}"
            except:
                pass
        df_matrix = pd.DataFrame(matrix_grid, index=[f"Probabilité {p}" for p in range(1, max_p+1)], columns=[f"Impact {i}" for i in range(1, max_i+1)])
        df_matrix.to_excel(writer, sheet_name="Grille_Heatmap")
        
        params_data = {
            "Paramètre": ["Nom du Projet", "Code du Projet", "Période", "Analyste d'Affaires", "Nombre de Participants", "Échelle Max Probabilité", "Échelle Max Impact", "Date d'Exportation"],
            "Valeur": [cfg["project_name"], cfg["project_code"], cfg["project_period"], cfg["business_analyst"], len(cfg["participants"]), cfg["max_probability"], cfg["max_impact"], datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        }
        pd.DataFrame(params_data).to_excel(writer, sheet_name="Paramètres_Projet", index=False)
        
        if not df_raw.empty:
            df_raw.to_excel(writer, sheet_name="Soumissions_Brutes", index=False)
            
    return output.getvalue()

def clean_txt(text):
    if not text:
        return ""
    text_str = str(text)
    replacements = {
        '’': "'", '‘': "'", '“': '"', '”': '"', '–': '-', '—': '-',
        '…': '...', '€': 'EUR', '•': '*'
    }
    for k, v in replacements.items():
        text_str = text_str.replace(k, v)
    return text_str.encode('latin-1', 'replace').decode('latin-1')

def generate_pdf_export(cfg, df_final):
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 9, clean_txt("Rapport de Gestion des Risques"), ln=1)

    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(2, 132, 199)
    pdf.cell(0, 6, clean_txt(f"Projet : {cfg['project_name']} ({cfg['project_code']})"), ln=1)

    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, clean_txt(f"Analyste : {cfg['business_analyst']} | Période : {cfg['project_period']} | Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}"), ln=1)

    pdf.ln(3)
    pdf.set_draw_color(226, 232, 240)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    total_r = len(df_final)
    max_p = cfg["max_probability"]
    max_i = cfg["max_impact"]
    max_crit = max_p * max_i
    top_score = int(df_final["FINAL PRIORITY"].max()) if not df_final.empty else 0
    nb_disc = len(df_final[df_final["STATUS"] == "À discuter"])
    nb_val = len(df_final[df_final["STATUS"] == "Validé"])

    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(241, 245, 249)
    pdf.set_text_color(30, 41, 59)
    kpi_txt = f"Total risques : {total_r}  |  Criticité Max : {top_score}/{max_crit}  |  Validés : {nb_val}  |  À arbitrer : {nb_disc}  |  Échelle : {max_p}x{max_i}"
    pdf.cell(0, 7, clean_txt(kpi_txt), border=1, fill=True, align='C', ln=1)
    pdf.ln(5)

    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 7, clean_txt(f"1. Matrice Visuelle de Criticité ({max_p} x {max_i})"), ln=1)
    pdf.ln(2)

    grid_risks = {(p, i): [] for p in range(1, max_p + 1) for i in range(1, max_i + 1)}
    for idx, row in df_final.iterrows():
        try:
            p_pos = int(row.get("CONSENSUS PROBABILITY", 1))
            i_pos = int(row.get("CONSENSUS IMPACT", 1))
            c_code = str(row.get("CODE", f"R{idx+1}"))
            if (p_pos, i_pos) in grid_risks:
                grid_risks[(p_pos, i_pos)].append(c_code)
        except:
            pass

    start_x = 35
    cell_w = min(28, int(150 / max_i))
    cell_h = min(13, int(65 / max_p))

    for p in range(max_p, 0, -1):
        cur_y = pdf.get_y()
        pdf.set_x(10)
        pdf.set_font('Helvetica', 'B', 8)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(22, cell_h, clean_txt(f"Prob. {p}"), align='R')
        
        for i in range(1, max_i + 1):
            ratio = (p * i) / max_crit
            if ratio >= 0.65:
                pdf.set_fill_color(254, 202, 202)
                pdf.set_text_color(153, 27, 27)
            elif ratio >= 0.40:
                pdf.set_fill_color(254, 215, 170)
                pdf.set_text_color(154, 52, 18)
            elif ratio >= 0.20:
                pdf.set_fill_color(254, 240, 138)
                pdf.set_text_color(133, 77, 14)
            else:
                pdf.set_fill_color(187, 247, 208)
                pdf.set_text_color(22, 101, 52)

            cell_x = start_x + (i - 1) * cell_w
            pdf.rect(cell_x, cur_y, cell_w, cell_h, 'DF')
            
            pdf.set_xy(cell_x, cur_y + 1)
            pdf.set_font('Helvetica', 'B', 7)
            pdf.cell(cell_w, 4, str(p * i), align='C')
            
            codes_in_cell = ", ".join(grid_risks.get((p, i), []))
            if codes_in_cell:
                pdf.set_xy(cell_x, cur_y + 5)
                pdf.set_font('Helvetica', 'B', 7)
                pdf.cell(cell_w, 4, clean_txt(codes_in_cell[:16]), align='C')

        pdf.set_xy(start_x, cur_y + cell_h)

    pdf.set_x(start_x)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(71, 85, 105)
    for i in range(1, max_i + 1):
        pdf.cell(cell_w, 5, clean_txt(f"Impact {i}"), align='C')
    pdf.ln(8)

    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 7, clean_txt("2. Registre des Risques Consolidé"), ln=1)
    pdf.ln(2)

    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_fill_color(241, 245, 249)
    pdf.set_text_color(30, 41, 59)

    col_widths = [13, 80, 13, 13, 15, 20, 36]
    headers = ["Code", "Risque Synthétisé", "Prob.", "Imp.", "Prio.", "Statut", "Contributeurs"]

    for w, h in zip(col_widths, headers):
        pdf.cell(w, 7, clean_txt(h), border=1, fill=True, align='C')
    pdf.ln()

    pdf.set_font('Helvetica', '', 8)
    for idx, row in df_final.iterrows():
        code_val = str(row.get("CODE", f"R{idx + 1}"))
        risk_val = str(row.get("CANONICAL RISK", ""))[:45]
        prob_val = str(row.get("CONSENSUS PROBABILITY", "1"))
        imp_val = str(row.get("CONSENSUS IMPACT", "1"))
        prio_val = str(row.get("FINAL PRIORITY", "1"))
        status_val = str(row.get("STATUS", ""))[:10]
        contrib_val = str(row.get("CONTRIBUTORS", ""))[:22]

        pdf.cell(col_widths[0], 6, clean_txt(code_val), border=1, align='C')
        pdf.cell(col_widths[1], 6, clean_txt(risk_val), border=1)
        pdf.cell(col_widths[2], 6, clean_txt(prob_val), border=1, align='C')
        pdf.cell(col_widths[3], 6, clean_txt(imp_val), border=1, align='C')
        pdf.cell(col_widths[4], 6, clean_txt(prio_val), border=1, align='C')
        pdf.cell(col_widths[5], 6, clean_txt(status_val), border=1, align='C')
        pdf.cell(col_widths[6], 6, clean_txt(contrib_val), border=1)
        pdf.ln()

    pdf.ln(6)
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 7, clean_txt("3. Fiches Détaillées & Plans de Mitigation"), ln=1)
    pdf.ln(2)

    for idx, row in df_final.iterrows():
        code_val = str(row.get("CODE", f"R{idx + 1}"))
        risk_title = str(row.get("CANONICAL RISK", ""))[:140]
        mitigation = str(row.get("MITIGATION NOTES", "")) if pd.notna(row.get("MITIGATION NOTES", "")) else "Aucune mesure documentée"
        dispersion = str(row.get("DISPERSION NOTE", "")) if pd.notna(row.get("DISPERSION NOTE", "")) else "N/A"
        contributors = str(row.get("CONTRIBUTORS", "")) if pd.notna(row.get("CONTRIBUTORS", "")) else "Non spécifié"

        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_text_color(2, 132, 199)
        pdf.multi_cell(190, 5, clean_txt(f"[{code_val}] {risk_title} (Criticité : {row.get('FINAL PRIORITY', '1')})"))

        pdf.set_font('Helvetica', '', 8)
        pdf.set_text_color(51, 65, 85)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(190, 4.5, clean_txt(f"Plan de mitigation : {mitigation}"))
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(190, 4.5, clean_txt(f"Analyse des écarts : {dispersion}"))
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(190, 4.5, clean_txt(f"Contributeurs : {contributors}"))
        pdf.ln(2)

    try:
        out = pdf.output(dest='S')
        if isinstance(out, str):
            return out.encode('latin-1', 'replace')
        return bytes(out)
    except Exception as e:
        print(f"Erreur génération PDF : {e}")
        raise

# -----------------------------------------------------------------------------
# Barre latérale : Paramètres, Clé API & Modèle Gemini
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Paramètres & IA")
    usable_models = ["gemini-3.8-flash", "gemini-3.5-flash"]
    
    if not st.session_state.api_validated:
        input_api_key = st.text_input("Clé API Gemini (Optionnel) :", type="password", help="Optionnel : Entrez votre clé API si vous souhaitez utiliser le regroupement intelligent Gemini.")
        selected_model = st.selectbox(
            "Modèle Gemini :",
            options=usable_models,
            index=0,
            help="Modèle Gemini utilisé pour le regroupement sémantique."
        )
        
        if st.button("Valider clé API", type="primary"):
            if not input_api_key.strip():
                st.warning("⚠️ Veuillez entrer une clé API avant de valider.")
            else:
                try:
                    with st.spinner("Validation de la clé..."):
                        genai.configure(api_key=input_api_key.strip())
                        models = [m.name for m in genai.list_models()]
                        if models:
                            st.session_state.api_validated = True
                            st.session_state.saved_api_key = input_api_key.strip()
                            st.session_state.saved_model = selected_model
                            st.success("✅ Clé API validée avec succès !")
                            st.rerun()
                except Exception as e:
                    st.error(f"❌ Échec de validation : {e}\nVérifiez votre clé API.")
        gemini_api_key = input_api_key
    else:
        st.text_input("Clé API Gemini (Optionnel) :", value="************************", type="password", disabled=True)
        st.selectbox("Modèle Gemini :", options=[st.session_state.saved_model], disabled=True)
        st.success(f"🔒 Connexion active ({st.session_state.saved_model})")
        
        if st.button("Modifier connexion API"):
            st.session_state.api_validated = False
            st.rerun()
        
        gemini_api_key = st.session_state.saved_api_key
        selected_model = st.session_state.saved_model

    st.markdown("---")
    if st.session_state.project_created:
        cfg_curr = st.session_state.project_config
        proj_name_curr = cfg_curr['project_name']
        proj_code_curr = cfg_curr['project_code']
        proj_period_curr = cfg_curr['project_period']
        ba_curr = cfg_curr['business_analyst']
        nb_parts_curr = len(cfg_curr['participants'])
        max_p_curr = cfg_curr['max_probability']
        max_i_curr = cfg_curr['max_impact']
        max_crit_curr = max_p_curr * max_i_curr

        st.success(f"📌 **{proj_name_curr}** ({proj_code_curr})")
        st.caption(f"📅 Période : {proj_period_curr}")
        st.caption(f"👤 Analyste : {ba_curr}")
        st.caption(f"👥 Participants : {nb_parts_curr}")
        st.info(f"Échelle : **{max_p_curr} x {max_i_curr}** (Max: {max_crit_curr})")
        
        if st.button("🔄 Réinitialiser / Modifier Config"):
            st.session_state.project_created = False
            st.rerun()

# =============================================================================
# ÉCRAN DE CONFIGURATION (ANALYSTE D'AFFAIRES)
# =============================================================================
if not st.session_state.project_created:
    st.markdown("""<div class="main-header">🛡️ Configuration de l'Atelier de Gestion des Risques</div>""", unsafe_allow_html=True)
    st.markdown("""<div class="sub-header">Espace Analyste d'Affaires : Définissez les paramètres globaux du projet et préparez l'atelier.</div>""", unsafe_allow_html=True)

    st.subheader("1. Paramètres Généraux du Projet")
    col_pname, col_pcode = st.columns([3, 2])
    with col_pname:
        project_name = st.text_input("Nom du Projet :", value=st.session_state.project_config["project_name"])
    with col_pcode:
        project_code = st.text_input("Code du Projet :", value=st.session_state.project_config["project_code"], placeholder="Ex: PRJ-2026-001")

    col_pperiod, col_analyst = st.columns([2, 3])
    with col_pperiod:
        project_period = st.text_input("Période du Projet :", value=st.session_state.project_config["project_period"], placeholder="Ex: Janvier 2026 - Décembre 2026")
    with col_analyst:
        business_analyst = st.text_input("Nom de l'Analyste d'Affaires :", value=st.session_state.project_config["business_analyst"])

    st.markdown("---")
    st.subheader("2. Échelle de Criticité")
    col_p_max, col_i_max = st.columns([2, 2])
    with col_p_max:
        max_prob = st.number_input("Échelle Max Probabilité (1 à 10) :", min_value=1, max_value=10, value=st.session_state.project_config["max_probability"])
    with col_i_max:
        max_impact = st.number_input("Échelle Max Impact (1 à 10) :", min_value=1, max_value=10, value=st.session_state.project_config["max_impact"])

    st.markdown("---")
    col_title_p, col_btn_p = st.columns([3, 1])
    with col_title_p:
        st.subheader("3. Liste des Répondants / Participants")
        st.markdown("Personnalisez le nom et courriel des participants.")
    with col_btn_p:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("👥 Générer une liste de répondants"):
            default_participants = [
                {"name": "Marc Villeneuve", "email": "marc.v@entreprise.com"},
                {"name": "Sophie Martin", "email": "sophie.m@entreprise.com"},
                {"name": "Thomas Bernard", "email": "thomas.b@entreprise.com"},
                {"name": "Julie Tremblay", "email": "julie.t@entreprise.com"}
            ]
            st.session_state.project_config["participants"] = default_participants
            st.rerun()

    current_parts = st.session_state.project_config["participants"]
    participants_df = pd.DataFrame(current_parts if current_parts else [{"name": "", "email": ""}])
    
    edited_participants = st.data_editor(
        participants_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "name": st.column_config.TextColumn("Nom du répondant", required=True),
            "email": st.column_config.TextColumn("Courriel", required=True)
        },
        key="editor_participants"
    )

    st.markdown("---")
    col_title_r, col_btn_r = st.columns([3, 1])
    with col_title_r:
        st.subheader("4. Bibliothèque de Risques Suggérés")
        st.markdown("Saisissez ou générez une liste de risques suggérés pour guider les répondants.")
    with col_btn_r:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📋 Générer une liste de risques"):
            st.session_state.project_config["suggested_risks"] = [
                "Retard de migration vers le cloud",
                "Résistance au changement des équipes opérationnelles",
                "Dépassement budgétaire initial",
                "Vulnérabilités de cybersécurité non détectées",
                "Pénurie de ressources expertes en interne"
            ]
            st.rerun()

    suggested_text = st.text_area(
        "Risques pré-suggérés (Saisir un risque par ligne) :",
        value="\n".join(st.session_state.project_config["suggested_risks"]),
        height=130,
        placeholder="Ex: Retard de livraison des équipements..."
    )

    st.markdown("---")
    if st.button("🚀 Valider la configuration et démarrer l'atelier", type="primary"):
        if not project_name:
            st.error("⚠️ Veuillez indiquer un nom de projet.")
        else:
            st.session_state.project_config["project_name"] = project_name
            st.session_state.project_config["project_code"] = project_code
            st.session_state.project_config["project_period"] = project_period
            st.session_state.project_config["business_analyst"] = business_analyst
            st.session_state.project_config["max_probability"] = int(max_prob)
            st.session_state.project_config["max_impact"] = int(max_impact)
            
            valid_participants = edited_participants.dropna(subset=["name"])[["name", "email"]].to_dict("records")
            if len(valid_participants) == 0:
                valid_participants = [{"name": "Participant 1", "email": "p1@exemple.com"}]
            st.session_state.project_config["participants"] = valid_participants

            risks_list = [r.strip() for r in suggested_text.split("\n") if r.strip()]
            st.session_state.project_config["suggested_risks"] = risks_list

            st.session_state.project_created = True
            st.rerun()

# =============================================================================
# ÉCRAN PRINCIPAL DE L'ATELIER
# =============================================================================
else:
    cfg = st.session_state.project_config
    p_name = cfg["project_name"]
    p_code = cfg["project_code"]
    b_analyst = cfg["business_analyst"]
    p_period = cfg["project_period"]
    max_p = cfg["max_probability"]
    max_i = cfg["max_impact"]
    max_crit = max_p * max_i

    st.markdown(f"""<div class="main-header">🛡️ Atelier : {p_name} ({p_code})</div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="sub-header">Analyste : <b>{b_analyst}</b> | Période : <b>{p_period}</b> | Échelle Max : <b>{max_p} x {max_i}</b> (Criticité Max: {max_crit})</div>""", unsafe_allow_html=True)

    nav_tabs = [
        "1️⃣ Saisies Individuelles (Répondants)", 
        "2️⃣ Consolidation & Analyse des Écarts", 
        "3️⃣ Analyse des risques consolidés",
        "4️⃣ Matrice Finale & Exportation"
    ]
    
    current_index = nav_tabs.index(st.session_state.current_main_tab) if st.session_state.current_main_tab in nav_tabs else 0
    selected_tab = st.radio(
        "Navigation principale :",
        options=nav_tabs,
        index=current_index,
        horizontal=True,
        label_visibility="collapsed"
    )
    if selected_tab != st.session_state.current_main_tab:
        st.session_state.current_main_tab = selected_tab
        st.rerun()

    # -------------------------------------------------------------------------
    # ONGLET 1 : SAISIES INDIVIDUELLES
    # -------------------------------------------------------------------------
    if st.session_state.current_main_tab == nav_tabs[0]:
        st.subheader("Espace de saisie individuelle")
        
        col_demo_r1, col_demo_r2 = st.columns([3, 1])
        with col_demo_r1:
            st.markdown("Chaque participant évalue ses risques dans son onglet dédié. Utilisez le bouton démo pour pré-remplir des essais.")
        with col_demo_r2:
            if st.button("⚡ Générer risques de démo", type="secondary"):
                pool_risks = [
                    ("Retard de migration vers le cloud", "Prévoir un plan de rattrapage et des ressources externes.", 4, 4, False),
                    ("Glissement d'échéancier sur le projet cloud", "Revoir le chemin critique avec l'équipe de pilotage.", 4, 4, False),
                    ("Résistance au changement des équipes", "Mettre en place un plan de conduite du changement.", 3, 3, True),
                    ("Dépassement budgétaire initial", "Établir un suivi rigoureux des coûts par sprint.", 2, 4, False),
                    ("Vulnérabilités de cybersécurité", "Réaliser un audit de sécurité et des tests d'intrusion.", 2, 5, True),
                    ("Risque de fuite de données sensibles", "Renforcer le chiffrement et la gestion des accès IAM.", 2, 5, True),
                    ("Pénurie de ressources expertes en interne", "Sous-traiter certains lots critiques.", 4, 3, False),
                ]
                
                new_rows = []
                p_cap = cfg['max_probability']
                i_cap = cfg['max_impact']
                
                for idx, p in enumerate(cfg["participants"]):
                    p_name_loop = p["name"]
                    slot_id = f"slot_{idx}"
                    num_risks = random.randint(4, min(6, len(pool_risks)))
                    selected_pool = random.sample(pool_risks, num_risks)
                    
                    for risk_title, mitigation, bp, bi, diverge in selected_pool:
                        base_p = min(bp, p_cap)
                        base_i = min(bi, i_cap)
                        
                        if diverge:
                            prob = max(1, min(p_cap, base_p + random.choice([-2, 2])))
                            impact = max(1, min(i_cap, base_i + random.choice([-2, 2])))
                        else:
                            prob = max(1, min(p_cap, base_p + random.choice([0, 0, 1])))
                            impact = max(1, min(i_cap, base_i + random.choice([0, 0, 1])))
                            
                        new_rows.append({
                            "Participant": p_name_loop,
                            "User_Slot": slot_id,
                            "Risque identifié": risk_title,
                            "Probabilité": prob,
                            "Impact": impact,
                            "Mitigation": mitigation
                        })
                
                if new_rows:
                    df_demo = pd.DataFrame(new_rows)
                    start_id = len(st.session_state.raw_submissions) + 1
                    df_demo.insert(0, "N°", range(start_id, start_id + len(df_demo)))
                    st.session_state.raw_submissions = pd.concat([st.session_state.raw_submissions, df_demo], ignore_index=True)
                    st.success("Risques de démo générés (avec consensus et divergences) !")
                    st.rerun()

        st.markdown("---")
        participants = cfg["participants"]
        tab_names = [f"👤 {p['name']}" for p in participants]
        user_tabs = st.tabs(tab_names)

        def render_user_slot(participant_info, slot_id):
            p_name_part = participant_info["name"]
            p_email = participant_info["email"]
            st.caption(f"Répondant : **{p_name_part}** ({p_email})")

            version_key = f"ver_{slot_id}"
            if version_key not in st.session_state:
                st.session_state[version_key] = 0
            v = st.session_state[version_key]

            st.markdown("##### Soumettre un risque :")
            
            risk_text = ""
            if cfg["suggested_risks"]:
                risk_choice_type = st.radio("Origine du risque :", ["Choisir dans la bibliothèque", "Rédiger un risque personnalisé"], key=f"choice_type_{slot_id}_{v}", horizontal=True)
                if risk_choice_type == "Choisir dans la bibliothèque":
                    risk_text = st.selectbox("Sélectionnez un risque :", cfg["suggested_risks"], key=f"select_{slot_id}_{v}")
                else:
                    risk_text = st.text_input("Description personnalisée du risque :", placeholder="Ex: Risque de...", key=f"custom_{slot_id}_{v}")
            else:
                risk_text = st.text_input("Description du risque :", placeholder="Ex: Risque de...", key=f"risk_input_{slot_id}_{v}")

            st.markdown("---")
            prob_options = [None] + list(range(1, cfg['max_probability'] + 1))
            prob_val = st.selectbox(
                f"Probabilité (1 = plus faible, {cfg['max_probability']} = plus élevé)",
                options=prob_options,
                format_func=lambda x: "Veuillez sélectionner..." if x is None else f"{x} / {cfg['max_probability']}",
                key=f"p_{slot_id}_{v}"
            )

            impact_options = [None] + list(range(1, cfg['max_impact'] + 1))
            impact_val = st.selectbox(
                f"Impact (1 = plus faible, {cfg['max_impact']} = plus élevé)",
                options=impact_options,
                format_func=lambda x: "Veuillez sélectionner..." if x is None else f"{x} / {cfg['max_impact']}",
                key=f"i_{slot_id}_{v}"
            )
                
            mitig_text = st.text_area("Pistes de mitigation suggérées :", key=f"m_{slot_id}_{v}")
            
            if st.button("📤 Ajouter ce risque à ma liste", key=f"btn_{slot_id}_{v}"):
                cleaned_risk = risk_text.strip() if risk_text else ""
                if not cleaned_risk:
                    st.warning("⚠️ Veuillez saisir ou sélectionner une description.")
                elif prob_val is None:
                    st.warning("⚠️ Veuillez sélectionner une probabilité.")
                elif impact_val is None:
                    st.warning("⚠️ Veuillez sélectionner un impact.")
                else:
                    next_id = len(st.session_state.raw_submissions) + 1
                    new_sub = pd.DataFrame([{
                        "N°": next_id,
                        "Participant": p_name_part,
                        "User_Slot": slot_id,
                        "Risque identifié": cleaned_risk,
                        "Probabilité": int(prob_val),
                        "Impact": int(impact_val),
                        "Mitigation": mitig_text
                    }])
                    st.session_state.raw_submissions = pd.concat([st.session_state.raw_submissions, new_sub], ignore_index=True)
                    st.session_state[version_key] += 1
                    st.success("Risque ajouté avec succès !")
                    st.rerun()

            st.markdown("---")
            st.markdown(f"##### 📋 Risques saisis par {p_name_part}")
            df_all = st.session_state.raw_submissions
            if not df_all.empty and "User_Slot" in df_all.columns:
                df_user = df_all[df_all["User_Slot"] == slot_id]
                if not df_user.empty:
                    st.dataframe(df_user[["N°", "Participant", "Risque identifié", "Probabilité", "Impact", "Mitigation"]], use_container_width=True, hide_index=True)
                else:
                    st.info("Aucun risque saisi pour cet onglet.")

        for idx, p in enumerate(participants):
            with user_tabs[idx]:
                render_user_slot(p, f"slot_{idx}")

    # -------------------------------------------------------------------------
    # ONGLET 2 : CONSOLIDATION & ANALYSE DES ÉCARTS (HYBRIDE OPTIMISÉ)
    # -------------------------------------------------------------------------
    elif st.session_state.current_main_tab == nav_tabs[1]:
        st.subheader("Consolidation & Analyse des Écarts")
        df_submissions = st.session_state.raw_submissions

        if df_submissions.empty:
            st.warning("⚠️ Aucun risque n'a encore été soumis par les participants.")
        else:
            st.markdown("##### 📥 Risques bruts soumis par tous les participants")
            st.dataframe(df_submissions[["N°", "Participant", "Risque identifié", "Probabilité", "Impact", "Mitigation"]], use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("##### Choisir le Mode de Consolidation")
            
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                st.markdown("""
                <div class="ai-box">
                    <h4>📊 Consolidation Statistique (Locale / Sans IA)</h4>
                    <p>Calcule les statistiques exactes (min, max, moyennes) et regroupe par libellé textuel identique sans nécessiter de clé API.</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("📊 Lancer la consolidation locale", type="secondary", use_container_width=True):
                    df_loc = consolidate_locally(df_submissions)
                    st.session_state.consensus_data = df_loc
                    st.success("Consolidation locale effectuée avec succès !")

            with col_c2:
                st.markdown("""
                <div class="ai-box">
                    <h4>🤖 Consolidation Intelligente (Gemini Hybride) <span class="badge-ai">IA Rapide</span></h4>
                    <p>Regroupe sémantiquement les formulations, analyse la dispersion et délègue les calculs à Python pour une vitesse maximale.</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("🤖 Lancer la consolidation Gemini", type="primary", use_container_width=True):
                    if not gemini_api_key:
                        st.error("⚠️ Veuillez valider votre clé API Gemini dans la barre latérale pour activer ce mode.")
                    else:
                        try:
                            genai.configure(api_key=gemini_api_key)
                            
                            submissions_clean = [
                                {
                                    "id": int(r["N°"]),
                                    "participant": str(r["Participant"]),
                                    "risk": str(r["Risque identifié"]),
                                    "mitigation": str(r.get("Mitigation", ""))
                                }
                                for _, r in df_submissions.iterrows()
                            ]
                            
                            prompt = f"""
                            Tu es un analyste d'affaires expert en gestion des risques projet.
                            Projet : {cfg["project_name"]} ({cfg["project_code"]})

                            Voici la liste des soumissions de risques individuels faites par différents participants d'un atelier :
                            {json.dumps(submissions_clean, ensure_ascii=False)}

                            Effectue les tâches suivantes :
                            1. **Regroupement sémantique** : Regroupe les risques similaires sous un titre "canonical_risk" unique, clair et professionnel. Assure-toi que chaque 'id' de la liste est inclus dans exactement un groupe.
                            2. **Association des identifiants** : Liste dans "submission_ids" les entiers exacts du champ 'id' correspondant au groupe.
                            3. **Amalgame des mitigations** : Rédige dans "mitigation_notes" une synthèse claire, cohérente et opérationnelle fusionnant les propositions des participants.
                            4. **Analyse qualitative de la dispersion** : Rédige dans "dispersion_insight" une brève note explicative (1 à 2 phrases) analysant le consensus ou la nature du désaccord potentiel entre les points de vue des participants sur ce risque.

                            Retourne STRICTEMENT un tableau JSON d'objets avec ce format exact :
                            [
                              {{
                                "canonical_risk": "Titre synthétisé",
                                "submission_ids": [1, 4],
                                "mitigation_notes": "Synthèse claire et actionnable du plan d'action",
                                "dispersion_insight": "Analyse qualitative du consensus ou des divergences observées"
                              }}
                            ]
                            """

                            start_time = time.time()
                            with st.spinner(f"Analyse sémantique et regroupement via {selected_model}..."):
                                model = genai.GenerativeModel(selected_model)
                                response = model.generate_content(
                                    prompt,
                                    generation_config={
                                        "response_mime_type": "application/json",
                                        "temperature": 0.2
                                    }
                                )
                                elapsed_time = time.time() - start_time

                            clusters = json.loads(response.text.strip())
                            
                            rows = []
                            for idx, c in enumerate(clusters):
                                sub_df = df_submissions[df_submissions["N°"].isin(c.get("submission_ids", []))]
                                if sub_df.empty:
                                    continue
                                
                                contributors = [str(x) for x in sub_df["Participant"].dropna().unique() if str(x).strip()]
                                contributors_count = len(contributors)
                                contributors_str = ", ".join(contributors)
                                
                                p_min = int(sub_df["Probabilité"].min())
                                p_max = int(sub_df["Probabilité"].max())
                                p_avg = round(float(sub_df["Probabilité"].mean()), 1)
                                
                                i_min = int(sub_df["Impact"].min())
                                i_max = int(sub_df["Impact"].max())
                                i_avg = round(float(sub_df["Impact"].mean()), 1)
                                
                                prio_series = sub_df["Probabilité"] * sub_df["Impact"]
                                prio_min = int(prio_series.min())
                                prio_max = int(prio_series.max())
                                prio_avg = round(float(prio_series.mean()), 1)
                                
                                cons_prob = round_nearest(p_avg)
                                cons_imp = round_nearest(i_avg)
                                final_priority = cons_prob * cons_imp
                                
                                prob_range = p_max - p_min
                                impact_range = i_max - i_min
                                
                                insight = str(c.get("dispersion_insight", "")).strip()
                                if prob_range >= 2 or impact_range >= 2:
                                    status = "À discuter"
                                    dispersion = f"Écart de notation (Prob: {p_min}-{p_max}, Impact: {i_min}-{i_max})"
                                    if insight:
                                        dispersion += f" - {insight}"
                                else:
                                    status = "Validé"
                                    dispersion = f"Consensus général (écarts faibles)"
                                    if insight:
                                        dispersion += f" - {insight}"

                                rows.append({
                                    "CODE": f"R{idx+1}",
                                    "CANONICAL RISK": c.get("canonical_risk", f"Risque {idx+1}"),
                                    "CONTRIBUTORS COUNT": contributors_count,
                                    "CONTRIBUTORS": contributors_str,
                                    "PROB_MIN": p_min,
                                    "PROB_MAX": p_max,
                                    "PROB_AVG": p_avg,
                                    "IMPACT_MIN": i_min,
                                    "IMPACT_MAX": i_max,
                                    "IMPACT_AVG": i_avg,
                                    "PRIO_MIN": prio_min,
                                    "PRIO_MAX": prio_max,
                                    "PRIO_AVG": prio_avg,
                                    "CONSENSUS PROBABILITY": cons_prob,
                                    "CONSENSUS IMPACT": cons_imp,
                                    "FINAL PRIORITY": final_priority,
                                    "DISPERSION NOTE": dispersion,
                                    "MITIGATION NOTES": c.get("mitigation_notes", "Aucune mesure documentée"),
                                    "STATUS": status
                                })

                            df_consensus = pd.DataFrame(rows).sort_values(
                                by=["FINAL PRIORITY", "CONTRIBUTORS COUNT"], 
                                ascending=[False, False]
                            ).reset_index(drop=True)
                            
                            df_consensus["CODE"] = [f"R{i+1}" for i in range(len(df_consensus))]
                            st.session_state.consensus_data = df_consensus
                            st.success(f"✅ Consolidation effectuée avec succès via **{selected_model}** en **{elapsed_time:.2f} s** !")

                        except Exception as e:
                            st.error(f"❌ Erreur lors du traitement Gemini : {e}")

            if not st.session_state.consensus_data.empty:
                st.markdown("---")
                nb_total = len(st.session_state.consensus_data)
                nb_disc = len(st.session_state.consensus_data[st.session_state.consensus_data["STATUS"] == "À discuter"])
                nb_val = len(st.session_state.consensus_data[st.session_state.consensus_data["STATUS"] == "Validé"])

                st.success(f"✨ **Consolidation terminée !** Total : **{nb_total}** risques identifiés (**{nb_disc}** nécessitent un arbitrage, **{nb_val}** sont déjà au consensus).")
                
                if st.button("➡️ Passer à l'analyse des risques consolidés", type="primary", use_container_width=True):
                    st.session_state.current_main_tab = nav_tabs[2]
                    st.rerun()

    # -------------------------------------------------------------------------
    # ONGLET 3 : ANALYSE DES RISQUES CONSOLIDÉS (ARBITRAGE ENRICHI)
    # -------------------------------------------------------------------------
    elif st.session_state.current_main_tab == nav_tabs[2]:
        st.subheader("📋 Analyse des risques consolidés")
        
        if st.session_state.consensus_data.empty:
            st.warning("⚠️ Aucune donnée consolidée. Veuillez d'abord lancer une consolidation dans l'onglet 2.")
        else:
            df_cons = st.session_state.consensus_data.copy()
            
            for col in ["PROB_MIN", "PROB_MAX", "IMPACT_MIN", "IMPACT_MAX", "CONSENSUS PROBABILITY", "CONSENSUS IMPACT"]:
                if col in df_cons.columns:
                    df_cons[col] = pd.to_numeric(df_cons[col], errors='coerce').fillna(1).astype(int)
            for col in ["PROB_AVG", "IMPACT_AVG"]:
                if col in df_cons.columns:
                    df_cons[col] = pd.to_numeric(df_cons[col], errors='coerce').fillna(1.0).round(1)

            if "CONTRIBUTORS COUNT" in df_cons.columns:
                df_cons["CONTRIBUTORS COUNT"] = pd.to_numeric(df_cons["CONTRIBUTORS COUNT"], errors='coerce').fillna(1).astype(int)
            else:
                df_cons["CONTRIBUTORS COUNT"] = 1

            if "PRIO_MIN" not in df_cons.columns:
                df_cons["PRIO_MIN"] = df_cons["PROB_MIN"] * df_cons["IMPACT_MIN"]
            if "PRIO_MAX" not in df_cons.columns:
                df_cons["PRIO_MAX"] = df_cons["PROB_MAX"] * df_cons["IMPACT_MAX"]
            if "PRIO_AVG" not in df_cons.columns:
                df_cons["PRIO_AVG"] = (df_cons["PROB_AVG"] * df_cons["IMPACT_AVG"]).round(1)

            df_cons["FINAL PRIORITY"] = df_cons["CONSENSUS PROBABILITY"] * df_cons["CONSENSUS IMPACT"]
            st.session_state.consensus_data = df_cons

            # Section 1 : Éléments à discuter
            st.markdown("### 🔍 Synthèse des risques - Éléments à discuter")
            st.caption("Risques présentant une dispersion de notation (écart ≥ 2) ou nécessitant un arbitrage collégial.")

            mask_disc = df_cons["STATUS"] == "À discuter"
            df_discuter = df_cons[mask_disc].copy()

            if df_discuter.empty:
                st.info("🎉 Aucun risque actuellement en statut 'À discuter'. Tous les risques sont validés !")
            else:
                display_disc = df_discuter[[
                    "CODE", "CANONICAL RISK", "CONTRIBUTORS COUNT",
                    "PROB_MIN", "PROB_MAX", "PROB_AVG",
                    "IMPACT_MIN", "IMPACT_MAX", "IMPACT_AVG",
                    "PRIO_MIN", "PRIO_MAX", "PRIO_AVG",
                    "CONTRIBUTORS", "DISPERSION NOTE", "MITIGATION NOTES"
                ]].copy()

                display_disc = display_disc.rename(columns={
                    "CODE": "Code",
                    "CANONICAL RISK": "Risque synthétisé",
                    "CONTRIBUTORS COUNT": "Nb contributeurs",
                    "PROB_MIN": "Prob Min",
                    "PROB_MAX": "Prob Max",
                    "PROB_AVG": "Prob Moy",
                    "IMPACT_MIN": "Imp Min",
                    "IMPACT_MAX": "Imp Max",
                    "IMPACT_AVG": "Imp Moy",
                    "PRIO_MIN": "Prio Min",
                    "PRIO_MAX": "Prio Max",
                    "PRIO_AVG": "Prio Moy",
                    "CONTRIBUTORS": "Contributeurs",
                    "DISPERSION NOTE": "Analyse des écarts",
                    "MITIGATION NOTES": "Mitigation proposée"
                })

                st.dataframe(display_disc, use_container_width=True, hide_index=True)

                st.markdown("""<div class="action-card"><b>⚖️ Arbitrer et valider un risque :</b></div>""", unsafe_allow_html=True)
                
                risk_options = {f"[{r['CODE']}] {r['CANONICAL RISK']}": r['CODE'] for _, r in df_discuter.iterrows()}
                chosen_label = st.selectbox("Sélectionner le risque à arbitrer :", options=list(risk_options.keys()), key="select_disc_risk")
                chosen_code = risk_options[chosen_label]
                curr_risk = df_discuter[df_discuter["CODE"] == chosen_code].iloc[0]

                col_ctx1, col_ctx2 = st.columns([1, 2])
                with col_ctx1:
                    st.text_input(
                        "👥 Contributeurs ayant mentionné ce risque :", 
                        value=str(curr_risk.get("CONTRIBUTORS", "Non spécifié")), 
                        disabled=True,
                        key=f"contrib_disp_{chosen_code}"
                    )
                with col_ctx2:
                    st.text_area(
                        "🔍 Analyse des écarts (Réflexion IA & Données) :", 
                        value=str(curr_risk.get("DISPERSION NOTE", "N/A")), 
                        height=68, 
                        disabled=True,
                        key=f"disp_disp_{chosen_code}"
                    )

                avg_p = float(curr_risk.get("PROB_AVG", curr_risk.get("CONSENSUS PROBABILITY", 1)))
                avg_i = float(curr_risk.get("IMPACT_AVG", curr_risk.get("CONSENSUS IMPACT", 1)))
                def_p = max(1, min(max_p, round_nearest(avg_p)))
                def_i = max(1, min(max_i, round_nearest(avg_i)))

                col_arb_p, col_arb_i = st.columns(2)
                with col_arb_p:
                    selected_p = st.selectbox(f"Probabilité validée (Moyenne : {avg_p}) :", options=list(range(1, max_p + 1)), index=def_p - 1, key=f"prob_val_{chosen_code}")
                with col_arb_i:
                    selected_i = st.selectbox(f"Impact validé (Moyenne : {avg_i}) :", options=list(range(1, max_i + 1)), index=def_i - 1, key=f"imp_val_{chosen_code}")

                current_mitig_val = str(curr_risk.get("MITIGATION NOTES", ""))
                val_mitig = st.text_area("Plan de mitigation consolidé (Ajuster ou enrichir au besoin) :", value=current_mitig_val, height=70, key=f"mitig_val_{chosen_code}")

                if st.button("💾 Enregistrer la validation", type="primary", use_container_width=True):
                    mask_target = st.session_state.consensus_data["CODE"] == chosen_code
                    st.session_state.consensus_data.loc[mask_target, "CONSENSUS PROBABILITY"] = int(selected_p)
                    st.session_state.consensus_data.loc[mask_target, "CONSENSUS IMPACT"] = int(selected_i)
                    st.session_state.consensus_data.loc[mask_target, "FINAL PRIORITY"] = int(selected_p) * int(selected_i)
                    st.session_state.consensus_data.loc[mask_target, "MITIGATION NOTES"] = val_mitig.strip()
                    st.session_state.consensus_data.loc[mask_target, "STATUS"] = "Validé"
                    st.success(f"✅ Risque {chosen_code} validé (Priorité : {int(selected_p) * int(selected_i)}) !")
                    st.rerun()

            st.markdown("---")

            # Section 2 : Consensus
            st.markdown("### ✅ Synthèse des risques - Consensus")
            st.caption("Risques validés et intégrés au registre officiel du projet.")

            mask_val = df_cons["STATUS"] == "Validé"
            df_valides = df_cons[mask_val].copy()

            if df_valides.empty:
                st.info("ℹ️ Aucun risque actuellement validé.")
            else:
                display_val = df_valides[[
                    "CODE", "CANONICAL RISK", "CONTRIBUTORS COUNT",
                    "CONSENSUS PROBABILITY", "CONSENSUS IMPACT", "FINAL PRIORITY",
                    "CONTRIBUTORS", "MITIGATION NOTES"
                ]].copy()

                display_val = display_val.rename(columns={
                    "CODE": "Code",
                    "CANONICAL RISK": "Risque synthétisé",
                    "CONTRIBUTORS COUNT": "Nb contributeurs",
                    "CONSENSUS PROBABILITY": "Prob Validée",
                    "CONSENSUS IMPACT": "Imp Validé",
                    "FINAL PRIORITY": "Priorité Finale",
                    "CONTRIBUTORS": "Contributeurs",
                    "MITIGATION NOTES": "Mitigation retenue"
                })

                st.dataframe(display_val, use_container_width=True, hide_index=True)

                st.markdown("""<div class="action-card"><b>↩️ Rouvrir la discussion sur un risque validé :</b></div>""", unsafe_allow_html=True)
                col_rev_sel, col_rev_btn = st.columns([4, 2])
                rev_options = {f"[{r['CODE']}] {r['CANONICAL RISK']} (Prio: {r['FINAL PRIORITY']})": r['CODE'] for _, r in df_valides.iterrows()}
                with col_rev_sel:
                    chosen_rev_label = st.selectbox("Sélectionner un risque validé à réexaminer :", options=list(rev_options.keys()), key="select_rev_risk")
                    chosen_rev_code = rev_options[chosen_rev_label]
                    
                with col_rev_btn:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("↩️ Remonter aux Éléments à discuter", type="secondary", use_container_width=True):
                        mask_rev = st.session_state.consensus_data["CODE"] == chosen_rev_code
                        st.session_state.consensus_data.loc[mask_rev, "STATUS"] = "À discuter"
                        st.warning(f"Le risque {chosen_rev_code} a été replacé dans 'Éléments à discuter'.")
                        st.rerun()

    # -------------------------------------------------------------------------
    # ONGLET 4 : MATRICE FINALE & EXPORTATION
    # -------------------------------------------------------------------------
    elif st.session_state.current_main_tab == nav_tabs[3]:
        st.subheader("🎯 Matrice Finale & Exportation")
        if st.session_state.consensus_data.empty:
            st.warning("⚠️ Veuillez d'abord compléter la consolidation.")
        else:
            df_final = st.session_state.consensus_data.copy()
            df_final["CONSENSUS PROBABILITY"] = pd.to_numeric(df_final["CONSENSUS PROBABILITY"], errors='coerce').fillna(1).astype(int)
            df_final["CONSENSUS IMPACT"] = pd.to_numeric(df_final["CONSENSUS IMPACT"], errors='coerce').fillna(1).astype(int)
            df_final["FINAL PRIORITY"] = df_final["CONSENSUS PROBABILITY"] * df_final["CONSENSUS IMPACT"]
            
            if "CONTRIBUTORS COUNT" in df_final.columns:
                df_final = df_final.sort_values(by=["FINAL PRIORITY", "CONTRIBUTORS COUNT"], ascending=[False, False]).reset_index(drop=True)

            col_mat_table, col_mat_plot = st.columns([3, 2])
            with col_mat_table:
                st.markdown("##### 📌 Registre Consolidé Officiel")
                df_final_display = df_final[["CODE", "CANONICAL RISK", "CONTRIBUTORS COUNT", "CONSENSUS PROBABILITY", "CONSENSUS IMPACT", "FINAL PRIORITY", "CONTRIBUTORS", "STATUS"]].copy()
                df_final_display = df_final_display.rename(columns={
                    "CODE": "Code",
                    "CANONICAL RISK": "Risque synthétisé",
                    "CONTRIBUTORS COUNT": "Nb contributeurs",
                    "CONSENSUS PROBABILITY": "Probabilité",
                    "CONSENSUS IMPACT": "Impact",
                    "FINAL PRIORITY": "Priorité",
                    "CONTRIBUTORS": "Contributeurs",
                    "STATUS": "Statut"
                })
                df_final_display.insert(0, "N°", range(1, len(df_final_display) + 1))
                st.dataframe(df_final_display, use_container_width=True, hide_index=True)

            with col_mat_plot:
                st.markdown(f"##### 🎯 Heatmap de Criticité ({max_p}x{max_i})")
                matrix_scores = np.zeros((max_p, max_i), dtype=int)
                for p in range(1, max_p + 1):
                    for i in range(1, max_i + 1):
                        matrix_scores[p-1, i-1] = p * i

                text_matrix = [["" for _ in range(max_i)] for _ in range(max_p)]
                for idx, row in df_final.iterrows():
                    try:
                        p = int(row["CONSENSUS PROBABILITY"])
                        i = int(row["CONSENSUS IMPACT"])
                        code_val = str(row.get("CODE", f"R{idx+1}"))
                        if 1 <= p <= max_p and 1 <= i <= max_i:
                            risk_label = f"<b>{code_val}</b>"
                            if text_matrix[p-1][i-1] == "":
                                text_matrix[p-1][i-1] = risk_label
                            else:
                                text_matrix[p-1][i-1] += f", {risk_label}"
                    except:
                        pass

                fig = go.Figure(data=go.Heatmap(
                    z=matrix_scores,
                    x=list(range(1, max_i + 1)),
                    y=list(range(1, max_p + 1)),
                    colorscale="YlOrRd",
                    showscale=False,
                    text=[[f"<b>Score {matrix_scores[r, c]}</b><br>{text_matrix[r][c]}" for c in range(max_i)] for r in range(max_p)],
                    texttemplate="%{text}",
                    textfont={"size": 11, "color": "black"}
                ))
                fig.update_layout(
                    xaxis_title="IMPACT", yaxis_title="PROBABILITÉ",
                    height=450, margin=dict(l=30, r=30, t=20, b=30),
                    xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                    yaxis=dict(tickmode='linear', tick0=1, dtick=1)
                )
                st.plotly_chart(fig, use_container_width=True)

            # -----------------------------------------------------------------
            # SECTION EXPORTATION (EXCEL & PDF)
            # -----------------------------------------------------------------
            st.markdown("---")
            st.subheader("📥 Exportation du Registre & de la Matrice")
            st.markdown("Téléchargez le registre consolidé et son analyse sous format Excel ou PDF.")
            
            col_exp_xl, col_exp_pdf = st.columns(2)
            
            with col_exp_xl:
                st.markdown("""
                <div class="export-box">
                    <h4>📊 Exporter au format Excel (.xlsx)</h4>
                    <p>Contient le registre des risques, les valeurs statistiques min/max/moyenne, la grille matricielle textuelle et les soumissions brutes.</p>
                </div>
                """, unsafe_allow_html=True)
                
                excel_bytes = generate_excel_export(cfg, df_final, st.session_state.raw_submissions)
                st.download_button(
                    label="📥 Télécharger le fichier Excel (.xlsx)",
                    data=excel_bytes,
                    file_name=f"Registre_Risques_{cfg['project_code']}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
            with col_exp_pdf:
                st.markdown("""
                <div class="export-box">
                    <h4>📄 Exporter au format PDF (.pdf)</h4>
                    <p>Génère un rapport formel destiné au comité de pilotage avec la matrice visuelle positionnée, le registre et les fiches détaillées.</p>
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    pdf_bytes = generate_pdf_export(cfg, df_final)
                    st.download_button(
                        label="📄 Télécharger le Rapport PDF (.pdf)",
                        data=pdf_bytes,
                        file_name=f"Rapport_Risques_{cfg['project_code']}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e_pdf:
                    st.error(f"Erreur lors de la génération du PDF : {e_pdf}")