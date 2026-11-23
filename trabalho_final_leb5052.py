# LEB5052 - Previsao de Chuva Diaria na ESALQ/USP (1917-2024)
# Autor: Arthur Cruvinel Marques
# Departamento de Engenharia de Biossistemas - ESALQ/USP - Piracicaba, SP

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import FuncFormatter

from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 10, "figure.titlesize": 11})

OUTPUT_DIR = "figures_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_PATH = "Dados_chuva_esalq.csv"


def fmt_virgula(x, _pos=None, casas=1):
    return f"{x:.{casas}f}".replace(".", ",")


def carregar_dados(filepath=DATA_PATH):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo nao encontrado: {filepath}")

    df = pd.read_csv(filepath, encoding="utf-8-sig")
    df.columns = ["dia_num", "ano", "precip_raw", "temp_raw"]

    df["precip"] = pd.to_numeric(df["precip_raw"].astype(str).str.replace(",", "."), errors="coerce")
    df["temp"] = pd.to_numeric(df["temp_raw"].astype(str).str.replace(",", "."), errors="coerce")

    # Remove picos isolados de leitura de 1 dia (salto >= 10 C em relacao ao dia anterior e ao seguinte, ex.: 0 C em 1981)
    picos_temp = (
        ((df["temp"] - df["temp"].shift(1)).abs() >= 10)
        & ((df["temp"] - df["temp"].shift(-1)).abs() >= 10)
    )
    df.loc[picos_temp, "temp"] = np.nan

    # Imputacao estritamente causal (forward-fill) para evitar uso de dados futuros (t + 1)
    df["precip_imputed"] = df["precip"].ffill()
    df["temp_imputed"] = df["temp"].ffill()

    return df


def preparar_features(df):
    df = df.copy()

    # Alvo: remocao explicita da ultima observacao (30/04/2024), que nao possui dia seguinte (t + 1)
    df["precip_amanha"] = df["precip_imputed"].shift(-1)
    df = df.dropna(subset=["precip_amanha"]).copy()
    df["chuva_amanha"] = (df["precip_amanha"] >= 0.1).astype(int)

    # Defasagens (lags) apenas do passado
    for lag in [1, 2, 3, 7]:
        df[f"precip_lag_{lag}"] = df["precip_imputed"].shift(lag)

    for lag in [1, 2]:
        df[f"temp_lag_{lag}"] = df["temp_imputed"].shift(lag)

    # Acumulados e medias moveis
    df["precip_acum_7d"] = df["precip_imputed"].rolling(window=7, min_periods=1).sum()
    df["temp_media_3d"] = df["temp_imputed"].rolling(window=3, min_periods=1).mean()

    # Sazonalidade
    df["dia_do_ano"] = df["dia_num"]
    df["mes"] = (((df["dia_num"] - 1) // 30) + 1).clip(1, 12)

    # Descartar colunas brutas e os 7 primeiros dias sem historico completo de lags
    df = df.drop(columns=["precip_raw", "temp_raw", "precip", "temp", "precip_amanha"])

    return df.dropna().copy()


def resumo_exploratorio(df_model):
    blocos = [
        ("Sub-treino (1917-2014)", df_model["ano"] <= 2014),
        ("Validacao (2015-2019)", (df_model["ano"] >= 2015) & (df_model["ano"] <= 2019)),
        ("Treino completo (1917-2019)", df_model["ano"] <= 2019),
        ("Teste independente (2020-2024)", df_model["ano"] > 2019),
    ]
    linhas = []
    for nome, mask in blocos:
        sub = df_model[mask]
        linhas.append({
            "Periodo": nome,
            "Dias": len(sub),
            "Dias chuvosos": int(sub["chuva_amanha"].sum()),
            "Taxa chuva": f"{sub['chuva_amanha'].mean()*100:.2f}%",
            "Precip. media (mm)": f"{sub['precip_imputed'].mean():.2f}",
            "Temp. media (C)": f"{sub['temp_imputed'].mean():.2f} +/- {sub['temp_imputed'].std():.2f}",
        })
    df_eda = pd.DataFrame(linhas)
    print("\nResumo exploratorio por particao temporal:")
    print(df_eda.to_string(index=False))
    return df_eda


def split_temporal(df_model, ano_val_ini=2015, ano_corte=2019):
    features = [
        "precip_imputed", "temp_imputed", "mes", "dia_do_ano",
        "precip_lag_1", "precip_lag_2", "precip_lag_3", "precip_lag_7",
        "temp_lag_1", "temp_lag_2", "precip_acum_7d", "temp_media_3d"
    ]

    X = df_model[features]
    y = df_model["chuva_amanha"]

    mask_subtrain = df_model["ano"] < ano_val_ini
    mask_val = (df_model["ano"] >= ano_val_ini) & (df_model["ano"] <= ano_corte)
    mask_train = df_model["ano"] <= ano_corte
    mask_test = df_model["ano"] > ano_corte

    splits = {
        "X_subtrain": X[mask_subtrain], "y_subtrain": y[mask_subtrain],
        "X_val": X[mask_val], "y_val": y[mask_val],
        "X_train": X[mask_train], "y_train": y[mask_train],
        "X_test": X[mask_test], "y_test": y[mask_test],
    }
    return splits, features


def rodar_modelos(splits):
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    modelos = {}
    metricas = []

    # 1. Majoritario (Naive - classe mais frequente)
    t0 = time.time()
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    tempo_dum = time.time() - t0
    prob_tr_dum = dummy.predict_proba(X_train)[:, 1]
    prob_te_dum = dummy.predict_proba(X_test)[:, 1]
    pred_tr_dum = dummy.predict(X_train)
    pred_te_dum = dummy.predict(X_test)
    modelos["Majoritário (Naive)"] = (pred_tr_dum, prob_tr_dum, pred_te_dum, prob_te_dum, dummy, tempo_dum)

    # 2. Persistencia (Naive temporal - repete ocorrencia de chuva do dia atual)
    t0 = time.time()
    pred_tr_pers = (X_train["precip_imputed"] >= 0.1).astype(int).values
    pred_te_pers = (X_test["precip_imputed"] >= 0.1).astype(int).values
    tempo_pers = time.time() - t0
    modelos["Persistência (Naive)"] = (
        pred_tr_pers, pred_tr_pers.astype(float),
        pred_te_pers, pred_te_pers.astype(float),
        None, tempo_pers
    )

    # 3. Regressao Logistica (Baseline linear com padronizacao ajustada apenas no treino)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    tempo_lr = time.time() - t0

    prob_tr_lr = lr.predict_proba(X_train_scaled)[:, 1]
    prob_te_lr = lr.predict_proba(X_test_scaled)[:, 1]
    pred_tr_lr = (prob_tr_lr >= 0.50).astype(int)
    pred_te_lr = (prob_te_lr >= 0.50).astype(int)
    modelos["Regressão Logística"] = (pred_tr_lr, prob_tr_lr, pred_te_lr, prob_te_lr, lr, tempo_lr)

    # 4. Random Forest
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    tempo_rf = time.time() - t0

    prob_tr_rf = rf.predict_proba(X_train)[:, 1]
    prob_te_rf = rf.predict_proba(X_test)[:, 1]
    pred_tr_rf = (prob_tr_rf >= 0.50).astype(int)
    pred_te_rf = (prob_te_rf >= 0.50).astype(int)
    modelos["Random Forest"] = (pred_tr_rf, prob_tr_rf, pred_te_rf, prob_te_rf, rf, tempo_rf)

    # 5. HistGradientBoosting
    t0 = time.time()
    hgb = HistGradientBoostingClassifier(max_iter=150, max_depth=8, random_state=42)
    hgb.fit(X_train, y_train)
    tempo_hgb = time.time() - t0

    prob_tr_hgb = hgb.predict_proba(X_train)[:, 1]
    prob_te_hgb = hgb.predict_proba(X_test)[:, 1]
    pred_tr_hgb = (prob_tr_hgb >= 0.50).astype(int)
    pred_te_hgb = (prob_te_hgb >= 0.50).astype(int)
    modelos["HistGradientBoosting"] = (pred_tr_hgb, prob_tr_hgb, pred_te_hgb, prob_te_hgb, hgb, tempo_hgb)

    for nome, (pred_tr, prob_tr, pred_te, prob_te, _, tempo) in modelos.items():
        metricas.append({
            "Modelo": nome,
            "ROC-AUC Treino": f"{roc_auc_score(y_train, prob_tr):.4f}",
            "ROC-AUC Teste": f"{roc_auc_score(y_test, prob_te):.4f}",
            "Acuracia Teste": f"{accuracy_score(y_test, pred_te)*100:.2f}%",
            "Sensibilidade": f"{recall_score(y_test, pred_te)*100:.2f}%",
            "Precisao": f"{precision_score(y_test, pred_te, zero_division=0)*100:.2f}%",
            "F1-Score": f"{f1_score(y_test, pred_te, zero_division=0):.4f}",
            "Tempo Treino": f"{tempo:.2f} s" if tempo >= 0.01 else "< 0.01 s",
        })

    df_res = pd.DataFrame(metricas)
    print("\nResultados no treino (1917-2019) e teste independente (2020-2024) - limiar 0,50:")
    print(df_res.to_string(index=False))

    return modelos, df_res


def calibrar_limiar_validacao(splits, prob_te_hgb):
    # Calibracao feita exclusivamente na janela de validacao (2015-2019), com modelo treinado em 1917-2014
    hgb_val = HistGradientBoostingClassifier(max_iter=150, max_depth=8, random_state=42)
    hgb_val.fit(splits["X_subtrain"], splits["y_subtrain"])
    prob_val = hgb_val.predict_proba(splits["X_val"])[:, 1]
    y_val = splits["y_val"]

    limiares = [0.50, 0.40, 0.35, 0.30]
    linhas = []
    for th in limiares:
        pred_val = (prob_val >= th).astype(int)
        linhas.append({
            "Conjunto": "Validação (2015–2019)",
            "Limiar": f"{th:.2f}",
            "Acuracia": f"{accuracy_score(y_val, pred_val)*100:.2f}%",
            "Sensibilidade": f"{recall_score(y_val, pred_val)*100:.2f}%",
            "Precisao": f"{precision_score(y_val, pred_val)*100:.2f}%",
            "F1-Score": f"{f1_score(y_val, pred_val):.4f}",
        })

    # Aplicacao do limiar escolhido (0,35) no teste independente (2020-2024)
    y_test = splits["y_test"]
    pred_te_035 = (prob_te_hgb >= 0.35).astype(int)
    linhas.append({
        "Conjunto": "Teste (2020–2024)",
        "Limiar": "0.35 (aplicado)",
        "Acuracia": f"{accuracy_score(y_test, pred_te_035)*100:.2f}%",
        "Sensibilidade": f"{recall_score(y_test, pred_te_035)*100:.2f}%",
        "Precisao": f"{precision_score(y_test, pred_te_035)*100:.2f}%",
        "F1-Score": f"{f1_score(y_test, pred_te_035):.4f}",
    })

    df_calib = pd.DataFrame(linhas)
    print("\nCalibracao de limiar na validacao (2015-2019) e aplicacao no teste (2020-2024):")
    print(df_calib.to_string(index=False))
    return df_calib


def gerar_fluxograma():
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0.35, 4.85)
    ax.axis("off")

    caixas = [
        (0.3, 2.85, 3.8, 1.7, "#eef4fa", "#1b365d", "1. Definição e Coleta",
         "Posto Meteorológico ESALQ/USP (1917–2024)\n39.174 registros diários; alvo: chuva ≥ 0,1 mm"),
        (4.6, 2.85, 3.8, 1.7, "#eef4fa", "#1b365d", "2. Limpeza dos Dados",
         "4 erros isolados de leitura (ex.: 0 °C em 1981)\nFalhas (< 2%) preenchidas com forward-fill"),
        (8.9, 2.85, 3.8, 1.7, "#eef4fa", "#1b365d", "3. Criação de Variáveis",
         "12 variáveis de chuva, temperatura,\nacumulados e época do ano (39.166 dias)"),
        (0.3, 0.65, 3.8, 1.7, "#eef7f1", "#1b5e3a", "4. Divisão Cronológica",
         "Sem sorteio aleatório (evita vazamento)\nTreino: 1917–2019 (Val: 2015–2019) | Teste: 2020–2024"),
        (4.6, 0.65, 3.8, 1.7, "#eef7f1", "#1b5e3a", "5. Comparação de Modelos",
         "Majoritário, Persistência, Regressão Logística,\nRandom Forest e HistGradientBoosting"),
        (8.9, 0.65, 3.8, 1.7, "#fdf3e7", "#9a5b13", "6. Interpretação e Decisão",
         "Importância por permutação (HistGradientBoosting)\nAnálise prática para manejo da irrigação"),
    ]

    for x, y, w, h, fc, ec, titulo, texto in caixas:
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.18",
            facecolor=fc, edgecolor=ec, linewidth=1.6
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 0.42, titulo, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=ec)
        ax.plot([x + 0.3, x + w - 0.3], [y + h - 0.72, y + h - 0.72], color="#b8c4d0", lw=0.9)
        ax.text(x + w / 2, y + 0.48, texto, ha="center", va="center",
                fontsize=8.8, color="#222222", linespacing=1.45)

    arrow_kw = dict(arrowstyle="-|>,head_length=0.6,head_width=0.35", color="#4a5568", lw=1.4)
    # Setas superiores
    ax.add_patch(FancyArrowPatch((4.15, 3.70), (4.55, 3.70), **arrow_kw))
    ax.add_patch(FancyArrowPatch((8.45, 3.70), (8.85, 3.70), **arrow_kw))
    # Seta em degrau da caixa 3 para caixa 4
    ax.plot([10.8, 10.8, 2.2], [2.80, 2.58, 2.58], color="#4a5568", lw=1.4)
    ax.add_patch(FancyArrowPatch((2.2, 2.58), (2.2, 2.40), **arrow_kw))
    # Setas inferiores
    ax.add_patch(FancyArrowPatch((4.15, 1.50), (4.55, 1.50), **arrow_kw))
    ax.add_patch(FancyArrowPatch((8.45, 1.50), (8.85, 1.50), **arrow_kw))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig0_fluxograma_metodologico.png"), dpi=300, bbox_inches="tight")
    plt.close()


def salvar_graficos(splits, modelos, features):
    X_test, y_test = splits["X_test"], splits["y_test"]
    modelos_prob = {
        k: v for k, v in modelos.items()
        if k not in ("Majoritário (Naive)", "Persistência (Naive)")
    }

    # 1. Curva ROC (com acentos e virgula decimal)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    cores = {
        "Regressão Logística": "darkorange",
        "Random Forest": "royalblue",
        "HistGradientBoosting": "forestgreen",
    }
    for nome, (_, _, _, prob_te, _, _) in modelos_prob.items():
        fpr, tpr, _ = roc_curve(y_test, prob_te)
        auc = roc_auc_score(y_test, prob_te)
        auc_str = f"{auc:.3f}".replace(".", ",")
        ax.plot(fpr, tpr, label=f"{nome} (AUC = {auc_str})", color=cores[nome], lw=1.9)

    # Ponto operacional da Persistencia
    pred_pers = modelos["Persistência (Naive)"][2]
    tn, fp, fn, tp = confusion_matrix(y_test, pred_pers).ravel()
    fpr_pers = fp / (fp + tn)
    tpr_pers = tp / (tp + fn)
    auc_pers = roc_auc_score(y_test, pred_pers)
    ax.scatter(
        [fpr_pers], [tpr_pers], color="#8e44ad", s=65, zorder=5,
        label=f"Persistência (AUC = {auc_pers:.3f})".replace(".", ",")
    )

    ax.plot([0, 1], [0, 1], "k--", lw=1.1, label="Acaso (AUC = 0,500)")
    ax.set_title("Curvas ROC no Período de Teste Independente (2020–2024)")
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: fmt_virgula(x, pos, 1)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: fmt_virgula(x, pos, 1)))
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_curvas_roc_finais.png"), dpi=300)
    plt.close()

    # 2. Matrizes de confusao (limiar 0,50) com acentos e eixos padronizados
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    cmaps = ["Oranges", "Blues", "Greens"]
    for idx, (nome, (_, _, pred_te, _, _, _)) in enumerate(modelos_prob.items()):
        cm = confusion_matrix(y_test, pred_te)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Seco", "Chuva"])
        disp.plot(ax=axes[idx], cmap=cmaps[idx], values_format="d", colorbar=False)
        axes[idx].set_title(nome, fontweight="bold")
        axes[idx].set_xlabel("Classe Prevista")
        axes[idx].set_ylabel("Classe Real" if idx == 0 else "")
        axes[idx].grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_matrizes_confusao.png"), dpi=300)
    plt.close()

    # 3. Importancia por permutacao no modelo selecionado (HistGradientBoosting)
    hgb_model = modelos["HistGradientBoosting"][4]
    perm = permutation_importance(
        hgb_model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1
    )
    ordem = perm.importances_mean.argsort()

    print("\nImportancia por permutacao (HistGradientBoosting no teste - ROC-AUC):")
    for i in ordem[::-1]:
        print(f"  {features[i]:16s}: {perm.importances_mean[i]:.4f} +/- {perm.importances_std[i]:.4f}")

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.barh(range(len(ordem)), perm.importances_mean[ordem], align="center", color="#2c3e50")
    ax.set_yticks(range(len(ordem)))
    ax.set_yticklabels(np.array(features)[ordem])
    ax.set_title("Importância dos Atributos – HistGradientBoosting (Permutação no Teste)")
    ax.set_xlabel("Queda média na ROC-AUC")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: fmt_virgula(x, pos, 2)))
    ax.axvline(0, color="#7f8c8d", lw=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_permutation_importance.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    df = carregar_dados()
    df_model = preparar_features(df)
    print(f"Registros brutos: {len(df)} | Registros validos apos bordas temporais: {len(df_model)}")
    resumo_exploratorio(df_model)
    splits, features = split_temporal(df_model, ano_val_ini=2015, ano_corte=2019)
    modelos, _ = rodar_modelos(splits)
    calibrar_limiar_validacao(splits, modelos["HistGradientBoosting"][3])
    gerar_fluxograma()
    salvar_graficos(splits, modelos, features)
    print("\nExecucao finalizada com sucesso.")
