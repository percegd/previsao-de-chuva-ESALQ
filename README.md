# Previsão de Chuva Diária na ESALQ/USP (1917–2024)

Trabalho final da disciplina **LEB5052 – Inteligência Artificial em Biossistemas: Princípios e Aplicações**  
Departamento de Engenharia de Biossistemas, Escola Superior de Agricultura "Luiz de Queiroz" (ESALQ/USP) – Piracicaba, SP

**Autor:** Arthur Cruvinel Marques  
**Docente:** Prof. Dr. Roberto Fray da Silva

---

## Sobre o projeto

Modelagem preditiva supervisionada (classificação binária) para prever a ocorrência de chuva no dia seguinte (`≥ 0,1 mm`) a partir da série histórica diária do Posto Meteorológico do Departamento de Engenharia de Biossistemas da ESALQ/USP (01/01/1917 a 30/04/2024; 39.174 registros brutos e 39.166 dias válidos após descarte das bordas temporais).

O pipeline adota imputação estritamente causal (*forward-fill*), partição cronológica sem embaralhamento (sub-treino 1917–2014, validação temporal 2015–2019 para calibração do limiar operacional e teste independente 2020–2024), comparação de cinco modelos (Majoritário, Persistência, Regressão Logística, Random Forest e HistGradientBoosting) e interpretação via importância por permutação no modelo selecionado (`HistGradientBoosting`).

---

## Estrutura dos arquivos

```
├── Dados_chuva_esalq.csv                 # Base climatológica diária original (1917–2024)
├── trabalho_final_leb5052.py             # Script principal (EDA, preparo, treino, validação e gráficos)
├── Trabalho_Final_LEB5052_Colab.ipynb    # Notebook interativo para execução no Google Colab
├── figures_final/                        # Gráficos gerados em 300 dpi (padrão PT-BR)
│   ├── fig0_fluxograma_metodologico.png
│   ├── fig1_curvas_roc_finais.png
│   ├── fig2_matrizes_confusao.png
│   └── fig3_permutation_importance.png
├── Artigo_Final_LEB5052_ESALQ.docx       # Artigo final formatado em ABNT (Word)
├── Artigo_Final_LEB5052_ESALQ.pdf        # Artigo final formatado em ABNT (PDF)
├── Apresentacao_Final_LEB5052_ESALQ.pdf  # Apresentação final em slides (10 slides, PDF)
└── README.md
```

---

## Dicionário de variáveis

| Variável | Papel | Tipo / Unidade | Descrição |
| :--- | :---: | :---: | :--- |
| `precip_imputed` | X1 | Contínua (mm) | Chuva do dia atual após imputação causal (*forward-fill*) |
| `temp_imputed` | X2 | Contínua (°C) | Temperatura média diária do ar no dia atual |
| `precip_lag_1, 2, 3, 7` | X3–X6 | Contínua (mm) | Chuva registrada 1, 2, 3 e 7 dias antes |
| `temp_lag_1, 2` | X7–X8 | Contínua (°C) | Temperatura média registrada 1 e 2 dias antes |
| `precip_acum_7d` | X9 | Contínua (mm) | Chuva acumulada nos últimos 7 dias (soma móvel) |
| `temp_media_3d` | X10 | Contínua (°C) | Média móvel de temperatura dos últimos 3 dias |
| `dia_do_ano` | X11 | Inteiro (1 a 365) | Dia sequencial no ano (ciclo sazonal) |
| `mes` | X12 | Inteiro (1 a 12) | Mês aproximado do ano (estações seca e chuvosa) |
| **`chuva_amanha`** | **Alvo (y)** | **Binário (0 ou 1)** | **1 se chuva no dia seguinte ≥ 0,1 mm; 0 caso contrário** |

---

## Como executar

### Localmente
Instale os pacotes necessários:
```bash
pip install numpy pandas scikit-learn matplotlib
```

Execute o script:
```bash
python trabalho_final_leb5052.py
```

### No Google Colab
Abra o arquivo `Trabalho_Final_LEB5052_Colab.ipynb` no Google Colab e execute as células em sequência.
