"""
Customer Segmentation + Churn Prediction
RFM segmentation → LightGBM churn model → Actionable recommendations
Contexto: e-commerce SME peruano
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, classification_report,
                              confusion_matrix, roc_curve)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGB = True
except (ImportError, OSError):
    # OSError cubre el caso de macOS sin libomp instalado
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_LGB = False


# ── Generador de datos ────────────────────────────────────────────────────────
def generate_customer_data(n=1500, seed=42):
    np.random.seed(seed)
    regiones = np.random.choice(
        ["Lima","Arequipa","Trujillo","Cusco","Piura"],
        size=n, p=[0.55,0.15,0.12,0.10,0.08]
    )
    canales  = np.random.choice(
        ["Orgánico","Paid Search","Email","Social","Directo"],
        size=n, p=[0.25,0.30,0.20,0.15,0.10]
    )
    # Perfiles base
    segmento_latente = np.random.choice(
        ["champion","loyal","at_risk","lost","new"],
        size=n, p=[0.12,0.23,0.28,0.22,0.15]
    )

    def noise(mu, sigma, low=0):
        return np.maximum(low, np.random.normal(mu, sigma, n))

    recencia      = np.where(segmento_latente=="champion", noise(15,8,1),
                    np.where(segmento_latente=="loyal",   noise(30,12,1),
                    np.where(segmento_latente=="at_risk", noise(75,20,1),
                    np.where(segmento_latente=="lost",    noise(180,40,1),
                                                           noise(5,3,1)))))
    frecuencia    = np.where(segmento_latente=="champion", noise(18,4,3),
                    np.where(segmento_latente=="loyal",   noise(10,3,2),
                    np.where(segmento_latente=="at_risk", noise(4,2,1),
                    np.where(segmento_latente=="lost",    noise(2,1,1),
                                                           noise(1,0.5,1)))))
    ticket_prom   = np.where(segmento_latente=="champion", noise(320,60,50),
                    np.where(segmento_latente=="loyal",   noise(185,40,30),
                    np.where(segmento_latente=="at_risk", noise(120,35,20),
                    np.where(segmento_latente=="lost",    noise(80,25,20),
                                                           noise(95,30,20)))))

    total_gastado = frecuencia * ticket_prom * np.random.uniform(0.8, 1.2, n)
    antigüedad    = np.random.randint(30, 1800, n)
    satisfaccion  = np.where(segmento_latente=="champion", noise(4.5,0.3,3),
                    np.where(segmento_latente=="lost",     noise(2.8,0.6,1),
                                                            noise(3.8,0.5,1)))
    satisfaccion  = np.clip(satisfaccion, 1, 5)
    n_categorias  = np.random.randint(1, 7, n)
    usa_app       = np.random.binomial(1, 0.45, n)
    n_tickets_soporte = np.where(segmento_latente=="lost", np.random.poisson(3, n),
                                  np.random.poisson(0.5, n))

    # Churn label (variable objetivo)
    churn_prob = (
        0.008 * recencia +
        -0.04 * frecuencia +
        -0.001 * ticket_prom +
        -0.08 * satisfaccion +
        0.12  * n_tickets_soporte +
        -0.03 * usa_app +
        np.random.normal(0, 0.08, n)
    )
    churn_prob = 1 / (1 + np.exp(-churn_prob + 0.5))  # sigmoid
    churn      = np.random.binomial(1, np.clip(churn_prob, 0, 1), n)

    return pd.DataFrame({
        "customer_id":        [f"C{i:05d}" for i in range(n)],
        "recencia_dias":      recencia.round(0).astype(int),
        "frecuencia_compras": frecuencia.round(0).astype(int),
        "ticket_promedio":    ticket_prom.round(2),
        "total_gastado":      total_gastado.round(2),
        "antiguedad_dias":    antigüedad,
        "score_satisfaccion": satisfaccion.round(1),
        "n_categorias":       n_categorias,
        "usa_app":            usa_app,
        "n_tickets_soporte":  n_tickets_soporte,
        "region":             regiones,
        "canal_adquisicion":  canales,
        "churn":              churn,
        "segmento_latente":   segmento_latente,
    })


# ── RFM + Segmentación KMeans ─────────────────────────────────────────────────
class RFMSegmenter:
    SEGMENT_NAMES = {
        0: "Champions",
        1: "Leales",
        2: "En Riesgo",
        3: "Inactivos",
        4: "Nuevos",
    }

    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.fitted = False

    def fit_predict(self, df):
        rfm = df[["recencia_dias","frecuencia_compras","ticket_promedio","total_gastado"]].copy()
        # Invertir recencia (menor = mejor)
        rfm["recencia_inv"] = 1 / (rfm["recencia_dias"] + 1)
        features = rfm[["recencia_inv","frecuencia_compras","ticket_promedio","total_gastado"]]
        scaled   = self.scaler.fit_transform(features)
        labels   = self.kmeans.fit_predict(scaled)

        # Ordenar clusters por total_gastado promedio → asignar nombres consistentes
        cluster_means = pd.DataFrame({"cluster": labels, "gasto": rfm["total_gastado"]})
        order = cluster_means.groupby("cluster")["gasto"].mean().sort_values(ascending=False).index
        remap = {old: new for new, old in enumerate(order)}
        labels_sorted = np.array([remap[l] for l in labels])

        self.fitted = True
        return labels_sorted

    def get_segment_name(self, cluster_id):
        return self.SEGMENT_NAMES.get(cluster_id, f"Segmento {cluster_id}")


# ── Modelo de Churn ───────────────────────────────────────────────────────────
FEATURES = [
    "recencia_dias","frecuencia_compras","ticket_promedio",
    "total_gastado","antiguedad_dias","score_satisfaccion",
    "n_categorias","usa_app","n_tickets_soporte",
]

def train_churn_model(df):
    X = df[FEATURES]
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    if HAS_LGB:
        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            class_weight="balanced", verbose=-1,
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
        )

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    auc    = roc_auc_score(y_test, y_prob)

    # Feature importance
    if HAS_LGB:
        imp = pd.DataFrame({
            "feature": FEATURES,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
    else:
        imp = pd.DataFrame({
            "feature": FEATURES,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

    return model, auc, classification_report(y_test, y_pred, output_dict=True), imp, (y_test, y_prob)


def get_churn_action(churn_prob, segment_name):
    """Retorna acción de retención según probabilidad y segmento."""
    if churn_prob >= 0.7:
        return "🚨 Intervención urgente: llamada + descuento 20%"
    elif churn_prob >= 0.45:
        if "Champion" in segment_name or "Leal" in segment_name:
            return "⚠️ Email win-back + oferta exclusiva VIP"
        return "⚠️ Email win-back + cupón 15%"
    elif churn_prob >= 0.25:
        return "📧 Newsletter personalizado + recomendaciones"
    else:
        return "✅ Programa de fidelización estándar"


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    df = generate_customer_data(1500)
    df.to_csv("data/customers.csv", index=False)

    # Segmentación RFM
    seg = RFMSegmenter(n_clusters=5)
    df["segmento_id"]   = seg.fit_predict(df)
    df["segmento_nombre"] = df["segmento_id"].map(seg.get_segment_name)

    print("Segmentos:")
    print(df.groupby("segmento_nombre").agg(
        n=("customer_id","count"),
        churn_rate=("churn","mean"),
        ticket=("ticket_promedio","mean"),
        recencia=("recencia_dias","mean"),
    ).round(2))

    # Modelo churn
    model, auc, report, imp, _ = train_churn_model(df)
    print(f"\nChurn Model AUC: {auc:.4f}")
    print("\nTop features:")
    print(imp.head(5).to_string(index=False))
