# Preditor de Atividade Humana

## Deploy

Aplicacao publicada no Streamlit: https://previsao-atividade-humana.streamlit.app/

## Estrutura

```text
PrevisaoAtividadeHumana/
|-- app.py
|-- requirements.txt
|-- README.md
|-- train_model.py
|-- notebooks/
|   `-- desafio11.ipynb
|-- model/
|   `-- modelo_final.joblib
|-- reports/
`-- data/
    |-- train.csv
    `-- test.csv
```

## Arquivos principais

- `app.py`: aplicacao Streamlit que carrega o modelo, recebe dados e executa predicoes.
- `model/modelo_final.joblib`: modelo final salvo.
- `notebooks/notebook_atualizado.ipynb`: notebook atualizado.
- `data/train.csv` e `data/test.csv`: bases usadas pelo notebook.
- `train_model.py`: script usado para gerar novamente o modelo, se necessario.

## Como rodar localmente

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
