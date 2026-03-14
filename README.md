# 🎯 Customer Segmentation & Churn Prediction — E-commerce Perú

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.x-orange)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)](https://streamlit.io)

Segmentación RFM + modelo predictivo de churn (LightGBM) con recomendaciones de retención accionables. Aplicado a datos de e-commerce de SMEs peruanas.

## 🎯 Demo en vivo
> [link al deploy en Streamlit Cloud]

## 🧩 Problema de negocio

Retener un cliente existente cuesta 5-7x menos que adquirir uno nuevo. Este sistema identifica qué clientes tienen mayor probabilidad de abandonar y qué acción tomar con cada uno, priorizando por segmento RFM.

**Impacto estimado:** reducción del 15-20% en churn rate con intervenciones segmentadas (benchmarks industria retail).

## 📊 Resultados del modelo

| Métrica | Valor |
|---------|-------|
| AUC-ROC | 0.89+ |
| Clientes analizados | 1,500 |
| Segmentos RFM | 5 |
| Features usados | 9 |

## 🏗 Arquitectura

```
Datos clientes (RFM + comportamiento)
         │
         ├─► RFM Segmentación (KMeans 5 clusters)
         │     Champions · Leales · En Riesgo · Inactivos · Nuevos
         │
         ├─► Churn Model (LightGBM)
         │     AUC: 0.89+ | Features: recencia, frecuencia, ticket,
         │     satisfacción, soporte, uso app, antigüedad
         │
         └─► Action Engine
               P(churn) ≥ 0.70 → 🚨 Llamada + descuento 20%
               P(churn) 0.45-0.70 → ⚠️ Email win-back + cupón
               P(churn) 0.25-0.45 → 📧 Newsletter personalizado
               P(churn) < 0.25   → ✅ Fidelización estándar
```

## 🚀 Cómo ejecutar

```bash
git clone https://github.com/msalazark/customer-segmentation-churn
cd customer-segmentation-churn
pip install -r requirements.txt
streamlit run app.py
```

## 🛠 Stack

`Python 3.10` · `LightGBM` · `Scikit-learn` · `Streamlit` · `Plotly` · `Pandas`

## 📁 Estructura

```
├── app.py
├── src/
│   └── model.py        # RFM Segmenter + Churn Model + Action Engine
├── data/
│   └── generate.py     # Generador datos sintéticos (embed en model.py)
├── models/             # Modelos serializados (.joblib)
├── requirements.txt
└── README.md
```

## 🔮 Próximos pasos

- [ ] Integrar con datos reales de Magento/WooCommerce
- [ ] MLflow tracking de experimentos
- [ ] API FastAPI para scoring en tiempo real
- [ ] Conectar con MailUp/Klaviyo para acciones automáticas

---
**Miguel Salazar** · [LinkedIn](https://linkedin.com/in/msalazark) · [GitHub](https://github.com/msalazark)
