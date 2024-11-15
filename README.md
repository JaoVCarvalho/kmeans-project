# Algoritmo K-Means


## Índice

- [Descrição](#descricao)
- [Equipe](#equipe)
- [Relatório](#relatorio)
  
## Descrição

Este repositório contém o trabalho 02 da disciplina de Inteligência Artificial do curso de Sistemas de Informação da Universidade Federal de Lavras. O trabalho tem como objetivo a implementação manual do algoritmo de clusterização K-means, utilizando a base de dados Iris.

Neste projeto, será desenvolvido um algoritmo em Python, sem o uso de bibliotecas que já implementam o K-means. O experimento será conduzido com 3 e 5 clusters, e o desempenho da clusterização será avaliado pelo silhouette score, calculado com o auxílio da biblioteca Sklearn. Em seguida, será feita uma nova implementação utilizando uma biblioteca que já contenha o K-means, permitindo a comparação dos resultados obtidos com a versão manual.

Como extensão, será realizada uma redução de dimensionalidade com PCA para 1 e 2 componentes principais, visualizando os clusters e centróides. Além disso, será feita uma análise de desempenho entre a implementação manual e a da biblioteca, comparando métricas de avaliação e o tempo de execução.

  ## Equipe

- [Diogo Carrer de Macedo](https://github.com/diogocarrer)
- [João Victor Carvalho dos Santos](https://github.com/JaoVCarvalho) 

## Relatório

A comparação entre a implementação manual e a biblioteca Sklearn para o algoritmo K-means na base de dados Iris revelou diferenças nos aspectos de uso de memória e tempo de execução.

Qualidade de Clusterização: Ambas as abordagens apresentaram a mesma qualidade de clusterização, com resultados equivalentes em termos de separação dos clusters, sendo o k = 3, o melhor.

Memória e Tempo de Execução: A implementação manual demonstrou menor uso de memória, enquanto a biblioteca Sklearn foi mais eficiente em termos de tempo de execução, destacando-se pela rapidez nos cálculos devido às suas otimizações internas.
