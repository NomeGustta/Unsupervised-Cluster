# 📚 Análise de Clustering com K-Means e PCA no Conjunto de Dados Iris

## 🔧 Dependências

Para executar este código, você precisa instalar as bibliotecas Python listadas abaixo. Utilize o `pip` para instalá-las:

```bash
pip install matplotlib pandas scikit-learn
```

### Bibliotecas Utilizadas

- **`matplotlib`**: Para a criação de gráficos e visualizações.
- **`pandas`**: Para manipulação e análise de dados em estruturas de dados, como DataFrames.
- **`scikit-learn`**: Para a implementação de algoritmos de aprendizado de máquina, como K-Means e PCA, além de métricas de avaliação.

## 🔍 Explicação do Código

### 1. Importação de Bibliotecas

```python
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
```

- **`matplotlib.pyplot`**: Usado para criar gráficos e visualizações.
- **`pandas`**: Facilita a manipulação e análise de dados em formato de tabelas.
- **`sklearn.datasets.load_iris`**: Carrega o conjunto de dados Iris, um conjunto de dados clássico de aprendizado de máquina.
- **`sklearn.cluster.KMeans`**: Implementa o algoritmo K-Means para clustering.
- **`sklearn.decomposition.PCA`**: Realiza a redução de dimensionalidade usando PCA (Principal Component Analysis).
- **`sklearn.metrics.silhouette_score`**: Calcula o índice de silhueta para medir a qualidade da clusterização.

### 2. Carregamento dos Dados

```python
# Carregando o conjunto de dados Iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Exibindo as primeiras linhas do conjunto de dados
print("Dados Iris:\n", df.head())
```

- O conjunto de dados Iris é carregado e armazenado em `data`.
- Um DataFrame `df` é criado a partir dos dados com os nomes das colunas correspondentes às características do conjunto de dados.
- `df.head()` exibe as primeiras linhas para inspecionar os dados.

### 3. Aplicação do Modelo K-Means

```python
# Configuração do modelo de K-Means
kmeans = KMeans(n_clusters=3, random_state=42)  # Três clusters porque há três espécies reais

# Ajustando o modelo aos dados (sem os rótulos)
kmeans.fit(df)

# Adicionando os rótulos previstos pelo K-Means ao DataFrame
df['Cluster'] = kmeans.labels_
```

- Um modelo de K-Means é configurado para identificar 3 clusters.
- O modelo é ajustado aos dados usando `kmeans.fit(df)`.
- Os rótulos dos clusters são adicionados ao DataFrame como uma nova coluna chamada `Cluster`.

### 4. Avaliação da Qualidade dos Clusters

```python
# Calculando o índice de silhueta para avaliar a qualidade dos clusters
silhouette_avg = silhouette_score(df.iloc[:, :-1], kmeans.labels_)
print(f"Índice de Silhueta: {silhouette_avg:.2f}")
```

- O índice de silhueta é calculado para avaliar a qualidade da clusterização.
- O resultado é impresso para análise.


## 📈 Visualização dos Clusters

Abaixo, uma visualização 2D dos clusters identificados pelo K-Means após a redução de dimensionalidade com PCA:

![Captura de tela 2024-12-05 114748](https://github.com/user-attachments/assets/9e3694c2-3bd9-4793-a1bb-47bda005c514)

Esta imagem mostra a distribuição dos dados em dois componentes principais, com cores diferentes representando cada cluster identificado.

### 5. Visualização com PCA

```python
# Reduzindo a dimensionalidade para visualização (2D) com PCA
pca = PCA(n_components=2)
df_pca = pd.DataFrame(pca.fit_transform(df.iloc[:, :-1]), columns=['Componente 1', 'Componente 2'])
df_pca['Cluster'] = kmeans.labels_

# Plotando os clusters
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green']
for cluster in range(3):
    cluster_data = df_pca[df_pca['Cluster'] == cluster]
    plt.scatter(cluster_data['Componente 1'], cluster_data['Componente 2'],
                label=f'Cluster {cluster}', color=colors[cluster])

plt.title('Clusters Identificados pelo K-Means')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.legend()
plt.show()
```

- O PCA é usado para reduzir a dimensionalidade dos dados para 2 componentes principais.
- Os clusters são visualizados em um gráfico de dispersão, onde cada cluster é representado por uma cor diferente.

### 6. Comparação com as Espécies Reais

```python
# Comparação com as espécies reais
real_labels = pd.Series(data.target, name='Espécie Real')
cluster_labels = pd.Series(kmeans.labels_, name='Cluster')
comparison = pd.crosstab(real_labels, cluster_labels)
print("\nComparação entre Clusters e Espécies Reais:\n")
print(comparison)
```

- Uma tabela de contagem é criada para comparar as espécies reais com os clusters identificados.
- O resultado é exibido para verificar como o modelo de clustering se alinha com as espécies reais.

## 🚀 Execução do Código

1. Certifique-se de ter o Python instalado.
2. Instale as bibliotecas necessárias com `pip install matplotlib pandas scikit-learn`.
3. Copie e cole o código em um arquivo Python (`clustering_iris.py`) e execute-o com `python clustering_iris.py`.

## 🔗 Referências

- [Documentação do scikit-learn](https://scikit-learn.org/stable/)
- [Documentação do pandas](https://pandas.pydata.org/pandas-docs/stable/)
- [Documentação do matplotlib](https://matplotlib.org/stable/contents.html)

---

Espero que este README tenha tornado o código mais compreensível e útil! 😊

