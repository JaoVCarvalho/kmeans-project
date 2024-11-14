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

A comparação entre a implementação manual (hardcore) e a biblioteca Sklearn para o algoritmo K-means na base de dados Iris revelou diferenças em qualidade de clusterização, memória e tempo de execução.

Silhouette Score:
A Sklearn apresentou um Silhouette Score mais alto, indicando uma separação superior dos clusters. Com k=3, obteve o melhor desempenho, com um Silhouette Score de 0.5528, enquanto a implementação manual alcançou 0.5296 para o mesmo valor de k. Esse resultado demonstra uma vantagem da biblioteca em termos de qualidade de clusterização.

Memória e Tempo de Execução:
A Sklearn teve maior consumo de memória (0.1457 MB vs. 0.0123 MB), reflexo das otimizações internas e recursos adicionais. No entanto, foi mais rápida, com um tempo de execução de 0.0486 segundos, em comparação com 0.0702 segundos na implementação manual.
