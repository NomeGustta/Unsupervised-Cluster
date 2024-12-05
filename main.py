# Importação de bibliotecas necessárias
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 1. Escolha dos Dados
# Carregando o conjunto de dados Iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Exibindo as primeiras linhas do conjunto de dados
print("Dados Iris:\n", df.head())

# 2. Aplicação do Modelo
# Configuração do modelo de K-Means
kmeans = KMeans(n_clusters=3, random_state=42)  # Três clusters porque há três espécies reais

# Ajustando o modelo aos dados (sem os rótulos)
kmeans.fit(df)

# Adicionando os rótulos previstos pelo K-Means ao DataFrame
df['Cluster'] = kmeans.labels_

# 3. Interpretação e Análise dos Resultados
# Calculando o índice de silhueta para avaliar a qualidade dos clusters
silhouette_avg = silhouette_score(df.iloc[:, :-1], kmeans.labels_)
print(f"Índice de Silhueta: {silhouette_avg:.2f}")

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

# Comparação com as espécies reais
real_labels = pd.Series(data.target, name='Espécie Real')
cluster_labels = pd.Series(kmeans.labels_, name='Cluster')
comparison = pd.crosstab(real_labels, cluster_labels)
print("\nComparação entre Clusters e Espécies Reais:\n")
print(comparison)
