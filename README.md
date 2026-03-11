# Modelo de Predição de Churn

Este repositório contém a implementação de um modelo de machine learning para predição de churn de clientes em um banco. O modelo utiliza algoritmos de classificação para identificar clientes propensos a deixar o serviço, baseado em dados históricos.

## Descrição do Projeto

O churn, ou rotatividade de clientes, é um problema crítico para empresas, especialmente no setor bancário. Este projeto visa desenvolver um modelo preditivo que ajude a identificar clientes em risco de churn, permitindo ações preventivas para retenção.

O modelo foi desenvolvido como parte de um curso prático de machine learning, implementando as melhores práticas de pré-processamento, treinamento e avaliação de modelos.

## Dataset

O dataset utilizado (`churn.csv`) contém informações sobre clientes bancários, incluindo:

- **score_credito**: Pontuação de crédito do cliente
- **pais**: País de origem (França, Espanha, Alemanha)
- **sexo_biologico**: Gênero biológico (Homem, Mulher)
- **idade**: Idade do cliente
- **anos_de_cliente**: Tempo de relacionamento com o banco
- **saldo**: Saldo atual na conta
- **servicos_adquiridos**: Número de serviços utilizados
- **tem_cartao_credito**: Indica se possui cartão de crédito (0/1)
- **membro_ativo**: Indica se é membro ativo (0/1)
- **salario_estimado**: Salário estimado
- **churn**: Variável alvo (0 = não churn, 1 = churn)

## Pré-processamento dos Dados

### Tratamento de Variáveis Categóricas
- **OneHotEncoder**: Aplicado às variáveis `pais` e `sexo_biologico` para conversão em formato numérico
- **LabelEncoder**: Aplicado à variável alvo `churn`

### Normalização
- **MinMaxScaler**: Utilizado para normalizar as variáveis numéricas no modelo KNN

### Divisão dos Dados
- Conjunto de treinamento: 75% dos dados
- Conjunto de teste: 25% dos dados
- Divisão estratificada para manter a proporção da variável alvo

## Modelos Implementados

### 1. DummyClassifier (Modelo Baseline)
- **Descrição**: Modelo simples que faz previsões aleatórias ou baseadas na classe majoritária
- **Propósito**: Estabelecer uma linha de base para comparação

### 2. DecisionTreeClassifier
- **Parâmetros**:
  - max_depth: 3
  - random_state: 5
- **Descrição**: Árvore de decisão com profundidade limitada para evitar overfitting

### 3. KNeighborsClassifier
- **Parâmetros**: Padrão (n_neighbors=5)
- **Descrição**: Algoritmo KNN aplicado aos dados normalizados
- **Pré-requisito**: Normalização dos dados

## Avaliação dos Modelos

Os modelos foram avaliados utilizando a métrica de acurácia no conjunto de teste:

- **DummyClassifier**: Acurácia baseline
- **DecisionTreeClassifier**: Melhor desempenho geral
- **KNeighborsClassifier**: Desempenho competitivo após normalização

O modelo selecionado foi o **DecisionTreeClassifier** devido ao seu equilíbrio entre interpretabilidade e desempenho.

## Arquivos do Modelo

Os modelos treinados foram salvos em arquivos pickle para uso em produção:

- `modelo_onehot.pkl`: Transformador OneHotEncoder
- `modelo_arvore.pkl`: Modelo DecisionTreeClassifier treinado

## Como Usar

### Pré-requisitos
```
pip install pandas scikit-learn matplotlib plotly
```

### Fazendo Previsões
```python
import pandas as pd
import pickle

# Carregar modelos
modelo_one_hot = pd.read_pickle('modelo_onehot.pkl')
modelo_arvore = pd.read_pickle('modelo_arvore.pkl')

# Novo dado para previsão
novo_dado = pd.DataFrame({
    'score_credito': [850],
    'pais': ['França'],
    'sexo_biologico': ['Homem'],
    'idade': [27],
    'anos_de_cliente': [3],
    'saldo': [56000],
    'servicos_adquiridos': [1],
    'tem_cartao_credito': [1],
    'membro_ativo': [1],
    'salario_estimado': [85270.00]
})

# Aplicar transformação
novo_dado_transformado = modelo_one_hot.transform(novo_dado)

# Fazer previsão
previsao = modelo_arvore.predict(novo_dado_transformado)
print(f'Previsão de churn: {previsao[0]}')  # 0 = não churn, 1 = churn
```

## Estrutura do Projeto

```
modelo-predicao-churn/
├── README.md
├── desafios/
│   ├── data/
│   │   └── churn.csv
│   └── teste/
│       └── Desafios_+Hora+da+prática.ipynb
├── modelo_onehot.pkl
└── modelo_arvore.pkl
```

## Notebook de Desenvolvimento

O arquivo `Desafios_+Hora+da+prática.ipynb` contém o desenvolvimento completo do projeto, incluindo:

- Análise exploratória de dados
- Pré-processamento
- Treinamento e comparação de modelos
- Salvamento dos modelos
- Exemplo de uso em produção

## Considerações Técnicas

- **Linguagem**: Python 3.x
- **Bibliotecas principais**: scikit-learn, pandas, matplotlib, plotly
- **Tipo de problema**: Classificação binária
- **Métrica de avaliação**: Acurácia
- **Persistência do modelo**: Pickle

## Melhorias Futuras

- Implementar validação cruzada
- Explorar outros algoritmos (Random Forest, SVM, etc.)
- Adicionar métricas adicionais (precision, recall, F1-score)
- Otimização de hiperparâmetros
- Deploy em API REST

## Autor

Desenvolvido como parte do curso POS TECH ML - Lara Gonçalves
