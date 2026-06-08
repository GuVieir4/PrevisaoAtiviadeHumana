# Preditor de Atividade Humana

Projeto desenvolvido para aplicação de técnicas de Machine Learning na classificação automática de atividades humanas utilizando dados coletados por sensores de smartphones.

---

## 👨‍💻 Integrantes

| Nome | RA |
|--------|--------|
| Daniel Costa | 1989218 |
| Gustavo Henrique | 1992080 |
| Renan Caixeta | 2011940 |

---

## 📋 Descrição do Problema

O reconhecimento automático de atividades humanas é um problema relevante em áreas como saúde, monitoramento de pacientes, dispositivos vestíveis (wearables), esportes e Internet das Coisas (IoT).

A partir de dados capturados por sensores presentes em smartphones, é possível identificar qual atividade uma pessoa está realizando. Entretanto, devido à semelhança entre alguns movimentos, essa tarefa exige o uso de técnicas de Machine Learning capazes de encontrar padrões nos dados.

---

## 🎯 Objetivo do Projeto

Desenvolver um modelo de Machine Learning capaz de classificar automaticamente atividades humanas utilizando dados de sensores de smartphones.

As atividades previstas pelo sistema são:

- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

Além do treinamento dos modelos, foi desenvolvido um aplicativo interativo utilizando Streamlit para realizar previsões em tempo real.

---

## 📊 Dataset Utilizado

O projeto utiliza o dataset **UCI Human Activity Recognition Using Smartphones (UCI HAR Dataset)**.

### Características do dataset:

- Dados coletados por acelerômetros e giroscópios de smartphones.
- Participação de 30 voluntários.
- Mais de 500 atributos derivados dos sinais capturados.
- 6 classes de atividades humanas.

### Fonte

https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

---

## 🤖 Tipo de Problema de Machine Learning

**Classificação Multiclasse Supervisionada**

O modelo recebe como entrada informações dos sensores e prevê uma entre seis atividades possíveis.

---

## 🔬 Metodologia

O desenvolvimento do projeto foi dividido nas seguintes etapas:

1. Carregamento dos dados.
2. Análise exploratória dos dados (EDA).
3. Tratamento e preparação dos dados.
4. Remoção de atributos com baixa variância.
5. Normalização utilizando StandardScaler.
6. Divisão dos dados em treinamento, validação e teste.
7. Treinamento dos modelos.
8. Ajuste de hiperparâmetros.
9. Avaliação dos modelos.
10. Seleção do melhor modelo.
11. Implantação utilizando Streamlit.

---

## 🧠 Modelos Treinados

### K-Nearest Neighbors (KNN)

Modelo baseado em distância que classifica uma amostra utilizando seus vizinhos mais próximos.

### Support Vector Machine (SVC - RBF)

Modelo capaz de encontrar fronteiras complexas entre as classes utilizando Kernel RBF.

### Random Forest

Modelo baseado em múltiplas árvores de decisão combinadas.

---

## 🏆 Modelo Final Escolhido

Após a comparação dos resultados, o modelo selecionado foi:

**Support Vector Machine com Kernel RBF Tunado (SVC-RBF Tunado)**

O modelo passou por um processo de otimização de hiperparâmetros (Hyperparameter Tuning), buscando a melhor combinação de configurações para maximizar o desempenho.

Após os testes e avaliações, o SVC-RBF Tunado apresentou os melhores resultados em termos de capacidade de generalização e desempenho nas métricas utilizadas, tornando-se o modelo final adotado na aplicação.

---

## 📈 Métricas de Avaliação

Para avaliar os modelos foram utilizadas:

- Accuracy (Acurácia)
- Precision (Precisão)
- Recall
- F1-Score
- Matriz de Confusão
- Curva ROC
- Área sob a Curva (AUC)

---

## 📌 Principais Resultados

Os modelos apresentaram elevado desempenho na classificação das atividades humanas.

Principais observações:

- Alta capacidade de distinguir atividades estáticas e dinâmicas.
- Melhor desempenho obtido pelo SVC-RBF Tunado.
- Baixa taxa de erro entre as classes.
- As maiores confusões ocorreram entre atividades de caminhada devido à semelhança dos movimentos.

---

## 📁 Estrutura do Projeto

```text
PrevisaoAtividadeHumana/
│
├── app.py
├── requirements.txt
├── README.md
├── train_model.py
│
├── notebooks/
│   └── notebook_atualizado.ipynb
│
├── model/
│   └── modelo_final.joblib
│
├── reports/
│
└── data/
    ├── train.csv
    └── test.csv
```

---

## 🛠 Tecnologias Utilizadas

- Python 3
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

---

## ▶️ Como Executar o Notebook

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Abra o Jupyter Notebook

```bash
jupyter notebook
```

### 4. Execute o notebook

```text
notebooks/notebook_atualizado.ipynb
```

---

## 🚀 Como Executar o Aplicativo Streamlit

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute a aplicação:

```bash
streamlit run app.py
```

---

## 🌐 Aplicação Publicada

Acesse o sistema online:

https://previsao-atividade-humana.streamlit.app/

---

## ⚠️ Limitações

- O modelo foi treinado apenas com os dados presentes no dataset UCI HAR.
- O desempenho pode variar em dispositivos diferentes dos utilizados na coleta original.
- Algumas atividades possuem características semelhantes, aumentando a chance de confusão entre classes.
- O sistema não realiza aprendizado contínuo após a implantação.

---

## ✅ Conclusão

O projeto demonstrou que técnicas de Machine Learning são capazes de identificar atividades humanas com elevada precisão utilizando dados de sensores de smartphones.

Entre os modelos avaliados, o SVC-RBF Tunado apresentou o melhor desempenho, tornando-se o modelo final da aplicação.

O processo de ajuste de hiperparâmetros contribuiu para melhorar a capacidade de generalização do modelo, resultando em previsões mais precisas e robustas.

O sistema desenvolvido possui potencial para aplicações em monitoramento de saúde, dispositivos inteligentes, esportes e Internet das Coisas, demonstrando na prática o uso de aprendizado de máquina em problemas reais de classificação.
