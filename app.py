"""
Customer Segmentation + Churn Prediction App
RFM Segmentation + LightGBM/GBM Churn Model + Actionable recommendations
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Segmentación & Churn | E-commerce Perú",
    page_icon="🎯",
    layout="wide",
)

SEGMENT_COLORS = {
    "Champions": "#667eea", "Leales": "#43e97b",
    "En Riesgo": "#fa709a", "Inactivos": "#999",
    "Nuevos": "#4facfe",
}

@st.cache_resource
def load_everything():
    from src.model import (generate_customer_data, RFMSegmenter,
                            train_churn_model, get_churn_action, FEATURES)
    os.makedirs("data", exist_ok=True)
    df = generate_customer_data(1500)

    seg = RFMSegmenter(n_clusters=5)
    df["segmento_id"]     = seg.fit_predict(df)
    df["segmento_nombre"] = df["segmento_id"].map(seg.get_segment_name)

    model, auc, report, imp, roc_data = train_churn_model(df)

    probs = model.predict_proba(df[FEATURES])[:, 1]
    df["churn_prob"]  = probs.round(3)
    df["accion"]      = [get_churn_action(p, s)
                         for p, s in zip(df["churn_prob"], df["segmento_nombre"])]
    df.to_csv("data/customers_scored.csv", index=False)
    return df, model, auc, report, imp, roc_data, FEATURES

df, model, auc, report, imp, roc_data, FEATURES = load_everything()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🎯 Customer Segmentation & Churn Prediction")
try:
    import lightgbm as _lgb; _has_lgb = True
except (ImportError, OSError):
    _has_lgb = False

if not _has_lgb:
    st.warning("LightGBM no disponible — corriendo con GradientBoosting (sklearn). "
               "Para activar LightGBM en macOS: `brew install libomp`", icon="⚠️")

st.caption(f"RFM Segmentation · {'LightGBM' if _has_lgb else 'GradientBoosting (fallback)'} · E-commerce SME Perú · 1,500 clientes")

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Clientes", f"{len(df):,}")
c2.metric("Tasa de Churn", f"{df['churn'].mean()*100:.1f}%")
c3.metric("AUC del Modelo", f"{auc:.3f}")
c4.metric("En Riesgo Alto", f"{(df['churn_prob']>=0.7).sum():,}")
c5.metric("Champions", f"{(df['segmento_nombre']=='Champions').sum():,}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🗂 Segmentos RFM", "📉 Modelo Churn", "🔍 Cliente individual", "📋 Acciones de retención"
])

with tab1:
    seg_summary = df.groupby("segmento_nombre").agg(
        n=("customer_id","count"),
        churn_rate=("churn","mean"),
        ticket_prom=("ticket_promedio","mean"),
        recencia=("recencia_dias","mean"),
        total_gastado=("total_gastado","mean"),
        churn_prob=("churn_prob","mean"),
    ).reset_index().sort_values("total_gastado", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(seg_summary, x="recencia", y="ticket_prom",
                         size="n", color="segmento_nombre",
                         color_discrete_map=SEGMENT_COLORS,
                         title="Segmentos: Recencia vs Ticket Promedio",
                         labels={"recencia":"Recencia (días)","ticket_prom":"Ticket Promedio (S/.)"},
                         size_max=60)
        fig.update_layout(height=360, margin=dict(t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.bar(seg_summary, x="segmento_nombre", y="churn_rate",
                      color="segmento_nombre", color_discrete_map=SEGMENT_COLORS,
                      title="Tasa de Churn por Segmento",
                      labels={"churn_rate":"Tasa Churn","segmento_nombre":"Segmento"})
        fig2.update_layout(height=360, margin=dict(t=40,b=0), showlegend=False)
        fig2.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        seg_summary.style.format({
            "churn_rate":"{:.1%}", "ticket_prom":"S/. {:.0f}",
            "recencia":"{:.0f} días", "total_gastado":"S/. {:.0f}",
            "churn_prob":"{:.1%}",
        }),
        hide_index=True, use_container_width=True
    )

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        # ROC Curve
        y_test, y_prob = roc_data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, fill="tozeroy",
            name=f"AUC = {auc:.3f}", line=dict(color="#667eea", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1],
            line=dict(dash="dash", color="#ccc"), showlegend=False))
        fig_roc.update_layout(title="Curva ROC", xaxis_title="FPR",
                               yaxis_title="TPR", height=340, margin=dict(t=40,b=0))
        st.plotly_chart(fig_roc, use_container_width=True)
    with c2:
        # Feature importance
        fig_imp = px.bar(imp.head(9), x="importance", y="feature",
                         orientation="h", title="Feature Importance",
                         color="importance", color_continuous_scale="Blues")
        fig_imp.update_layout(height=340, margin=dict(t=40,b=0),
                               coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_imp, use_container_width=True)

    # Distribución de probabilidades
    fig_hist = px.histogram(df, x="churn_prob", color="segmento_nombre",
                             color_discrete_map=SEGMENT_COLORS,
                             nbins=40, title="Distribución de Probabilidad de Churn por Segmento",
                             barmode="overlay", opacity=0.7)
    fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Umbral 0.5")
    fig_hist.update_layout(height=320, margin=dict(t=40,b=0))
    st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    st.subheader("Predicción individual")
    col_in, col_out = st.columns([1,1])
    with col_in:
        rec   = st.slider("Recencia (días)",    1, 365, 45)
        freq  = st.slider("Frecuencia compras", 1, 30,  5)
        ticket= st.slider("Ticket promedio (S/)", 20, 800, 150)
        gasto = st.number_input("Total gastado (S/)", 50, 20000, freq*ticket)
        antig = st.slider("Antigüedad (días)", 30, 1800, 365)
        sat   = st.slider("Score satisfacción", 1.0, 5.0, 3.8, 0.1)
        n_cat = st.slider("N° categorías compradas", 1, 6, 3)
        app   = st.selectbox("Usa app móvil", [1,0], format_func=lambda x: "Sí" if x==1 else "No")
        tkt   = st.slider("Tickets de soporte", 0, 10, 1)

    with col_out:
        X_new = pd.DataFrame([{
            "recencia_dias": rec, "frecuencia_compras": freq,
            "ticket_promedio": ticket, "total_gastado": gasto,
            "antiguedad_dias": antig, "score_satisfaccion": sat,
            "n_categorias": n_cat, "usa_app": app,
            "n_tickets_soporte": tkt,
        }])
        prob = model.predict_proba(X_new)[0][1]
        color = "red" if prob >= 0.7 else ("orange" if prob >= 0.45 else "green")

        st.markdown(f"""
        <div style='text-align:center;padding:30px;border-radius:12px;
                    background:#f8f9fa;border:2px solid {color};margin-bottom:16px'>
          <div style='font-size:48px;font-weight:700;color:{color}'>{prob*100:.1f}%</div>
          <div style='color:#666;font-size:15px'>Probabilidad de Churn</div>
        </div>
        """, unsafe_allow_html=True)

        from src.model import get_churn_action
        accion = get_churn_action(prob, "")
        st.info(f"**Acción recomendada:** {accion}")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob*100,
            gauge={"axis":{"range":[0,100]},
                   "bar":{"color": color},
                   "steps":[
                       {"range":[0,25],"color":"#e8f5e9"},
                       {"range":[25,45],"color":"#fff9c4"},
                       {"range":[45,70],"color":"#ffe0b2"},
                       {"range":[70,100],"color":"#ffcdd2"},
                   ]},
            number={"suffix":"%"},
        ))
        gauge.update_layout(height=220, margin=dict(t=20,b=0,l=30,r=30))
        st.plotly_chart(gauge, use_container_width=True)

with tab4:
    st.subheader("Plan de retención prioritario")
    prioridad = st.selectbox("Filtrar por urgencia", ["Todos","🚨 Urgente","⚠️ Medio","📧 Bajo"])
    n_show = st.slider("Clientes a mostrar", 10, 100, 30)

    df_acc = df.sort_values("churn_prob", ascending=False).copy()
    if prioridad == "🚨 Urgente":
        df_acc = df_acc[df_acc["churn_prob"] >= 0.7]
    elif prioridad == "⚠️ Medio":
        df_acc = df_acc[(df_acc["churn_prob"] >= 0.45) & (df_acc["churn_prob"] < 0.7)]
    elif prioridad == "📧 Bajo":
        df_acc = df_acc[df_acc["churn_prob"] < 0.45]

    st.dataframe(
        df_acc[["customer_id","segmento_nombre","churn_prob","recencia_dias",
                "frecuencia_compras","ticket_promedio","total_gastado","accion"]]
        .head(n_show)
        .style.format({
            "churn_prob":"{:.1%}",
            "ticket_promedio":"S/. {:.0f}",
            "total_gastado":"S/. {:.0f}",
        }).background_gradient(subset=["churn_prob"], cmap="RdYlGn_r"),
        hide_index=True, use_container_width=True
    )

    st.divider()
    st.caption("🎯 Customer Segmentation & Churn · Miguel Salazar · Stack: Python + LightGBM + Streamlit + Plotly")
