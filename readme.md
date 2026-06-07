# Detecção de Spam em E-mails

> Trabalho da disciplina de **Machine Learning - Teoria e Aplicado**  
> Professora: Ma. Nathália A. Lima | Grupo 15

---

## Integrantes

| Nome | RA |
|---|---|
| João Pedro Torres | 2007848 |
| João Pedro Duarte | 1993686 |
| Mateus Felix | 1998770 |

---

## Descrição do Problema

O spam é um problema persistente na comunicação digital. E-mails não solicitados podem carregar golpes financeiros, phishing, malware e conteúdo enganoso, representando risco real aos usuários. Filtrar spam manualmente é inviável, pois o volume de mensagens é enorme e as técnicas dos remetentes evoluem constantemente.

---

## Objetivo do Projeto

Desenvolver um classificador automático de spam capaz de analisar o conteúdo de um e-mail e determinar, com alta precisão, se ele é **spam** (1) ou **ham** — mensagem legítima (0). O classificador treinado é exposto via uma aplicação web interativa no Streamlit, permitindo que qualquer usuário teste o modelo em tempo real.

---

## Dataset Utilizado

- **Nome:** Spambase
- **Fonte:** [Kaggle](https://www.kaggle.com/)
- **Arquivo:** `spambase_updated.csv`
- **Descrição:** Cada linha representa um e-mail. As colunas descrevem a frequência percentual de palavras comuns (ex.: `free`, `money`, `credit`) e caracteres especiais (ex.: `!`, `$`, `#`) no corpo do e-mail. A coluna alvo `spam` indica se o e-mail é spam (`1`) ou legítimo (`0`).
- **Total de features:** 57 atributos numéricos

---

## Tipo de Problema de Machine Learning

**Classificação Binária Supervisionada** — o modelo aprende a partir de exemplos rotulados para distinguir entre duas classes: spam e ham.

---

## Metodologia

O projeto foi desenvolvido seguindo as seguintes etapas:

1. **Análise Exploratória de Dados (EDA):** verificação da distribuição das classes (spam vs. ham), estatísticas descritivas, identificação de valores nulos/duplicados, análise de correlação e importância de variáveis via Random Forest.

2. **Pré-processamento:** remoção de duplicatas e nulos, separação de features (`X`) e target (`y`), divisão treino/teste com estratificação (80/20, `random_state=42`), e normalização dos dados com `StandardScaler` (para modelos lineares e SVM) e `MinMaxScaler` (para Naive Bayes).

3. **Treinamento e Validação:** os modelos foram avaliados com **Stratified K-Fold Cross-Validation** (5 folds), usando `Pipeline` do scikit-learn para evitar vazamento de dados entre os folds.

4. **Análise Comparativa:** comparação das métricas entre os modelos para seleção do mais adequado à produção.

5. **Análise de Erros:** quantificação de falsos positivos (ham classificado como spam) e falsos negativos (spam não detectado).

6. **Deploy:** o modelo final e o scaler foram serializados com `joblib` e servidos via API Flask, integrada a uma interface Streamlit.

---

## Modelos Treinados

| Modelo | Normalização Aplicada |
|---|---|
| Regressão Logística | StandardScaler |
| SVM (SVC) | StandardScaler |
| Naive Bayes Multinomial | MinMaxScaler |

---

## Modelo Final Escolhido

**SVC** — Support Vector Classifier (SVC, probability=True)

O SVC foi o modelo com melhor desempenho geral na validação cruzada, apresentando o maior F1-Score harmônico entre os três candidatos. O modelo treinado foi exportado como `melhor_modelo_spam.pkl`.

---

## Métricas de Avaliação

As métricas utilizadas na validação cruzada foram:

- **Acurácia** — proporção de predições corretas sobre o total.
- **Precisão** — dos e-mails classificados como spam, quantos realmente são spam.
- **Recall** — dos e-mails que são spam, quantos foram corretamente identificados.
- **F1-Score** — média harmônica entre precisão e recall; métrica principal de seleção do modelo.
- **AUC-ROC** — área sob a curva ROC; mede a capacidade discriminatória do modelo.

---

## Principais Resultados

- O **SVC** obteve o maior F1-Score e AUC-ROC na validação cruzada.
- A **Regressão Logística** apresentou resultados muito próximos ao SVC.
- O **Naive Bayes Multinomial** foi o modelo com desempenho mais modesto, especialmente em precisão.
- Na análise de erros do modelo final sobre o conjunto de teste:
  - **Falsos negativos** (spam não detectado): expõem o usuário a riscos de fraude.
  - **Falsos positivos** (e-mail legítimo filtrado): prejudicam a experiência do usuário.
- As palavras/caracteres mais indicativos de **spam** incluem: `free`, `money`, `credit`, `$`, `!`, entre outros relacionados a promoções e urgência.
- As palavras mais associadas a **ham** são termos neutros do cotidiano corporativo, como `george`, `hpl`, `re`, `edu`.

---

## Estrutura dos Arquivos

```
.
SpamIdentifier/
│
├── app.py # Aplicação Streamlit
├── requirements.txt # Dependências do projeto
├── README.md # Documentação do projeto
│
├── notebooks/
│ └── notebook_atualizado.ipynb # Notebook revisado da P1
│
├── model/
│ └── modelo_final.pkl # Modelo final salvo
│ └── modelo_final.joblib # Alternativa ao .pkl, caso o grupo use .joblib
│
├── reports/
│ └── relatorio_atualizado.pdf # Relatório final atualizado
│
└── data/
└── dataset.csv # Dataset utilizado, se puder ser versionado
```

---

## Tecnologias Utilizadas

| Categoria | Tecnologias |
|---|---|
| Linguagem | Python 3 |
| Machine Learning | scikit-learn (`LogisticRegression`, `SVC`, `MultinomialNB`, `RandomForestClassifier`) |
| Análise de Dados | pandas, numpy |
| Visualização | matplotlib, seaborn |
| Serialização | joblib |
| API Backend | Flask, Flask-CORS |
| Interface Web | Streamlit |
| Ambiente | Jupyter Notebook |

---

## Instruções para Executar o Notebook

### Pré-requisitos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

### Execução

1. Clone ou baixe este repositório.
2. Certifique-se de que o arquivo `spambase_updated.csv` está na mesma pasta do notebook.
3. Faça o upload do notebook no Google Colab
4. Execute todas as células em ordem.
5. Ao final, os arquivos `melhor_modelo_spam.pkl` e `scaler_spam.pkl` serão gerados automaticamente.

---

## Instruções para Executar o App Streamlit

### Pré-requisitos

```bash
pip install streamlit flask flask-cors joblib numpy scikit-learn
```

### Passo 1 — Iniciar a API Flask

Em um terminal, execute:

```bash
python app.py
```

O servidor estará disponível em `http://localhost:5000`.

### Passo 2 — Iniciar o Streamlit

Em outro terminal, execute:

```bash
streamlit run streamlit_app.py
```

Acesse `http://localhost:8501` no navegador, cole o texto de um e-mail e clique em **Verificar** para obter a predição.

---

## Link do App Publicado
 
> `https://emailspam-kvtcuuwsg5cuqochfkm7c6.streamlit.app/`

---

## Limitações

- **Idioma:** o modelo foi treinado em um dataset predominantemente em inglês. E-mails em português podem apresentar desempenho reduzido.
- **Features fixas:** a extração de características é baseada em uma lista estática de 57 palavras/caracteres. Novos padrões de spam que não utilizem essas palavras podem não ser detectados.
- **Texto puro:** o modelo analisa apenas o conteúdo textual do e-mail. Spam baseado em imagens, links ou código HTML não é detectado diretamente.
- **Deriva temporal:** os padrões de spam evoluem; o modelo pode perder desempenho ao longo do tempo sem retreinamento.
- **Thresholds:** o modelo usa o limiar padrão de 0.5 para classificação; em cenários específicos pode ser necessário ajustá-lo para priorizar recall (evitar falsos negativos) ou precisão.

---

## Conclusão

O projeto demonstrou que é possível construir um classificador de spam eficaz utilizando técnicas clássicas de Machine Learning sobre features estatísticas de texto. A comparação sistemática entre Regressão Logística, SVM e Naive Bayes, com validação cruzada estratificada, garantiu uma seleção de modelo robusta e sem viés. O uso de pipelines do scikit-learn assegurou a integridade do processo de avaliação. O modelo final foi integrado a uma aplicação web funcional, evidenciando o ciclo completo de um projeto de Machine Learning: da exploração de dados ao deploy em produção.
