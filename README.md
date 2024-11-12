# Algoritmo KNN


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

A análise das métricas de avaliação dos classificadores KNN, tanto na implementação manual quanto na biblioteca Sklearn, revelou resultados idênticos em diferentes configurações de K, conforme a randomização realizada com: 
```python
train_test_split(attributes, species, test_size=0.20, random_state=13)
```

Ambos os classificadores apresentaram desempenhos coincidentes, validando a eficácia da implementação manual e confirmando a precisão da biblioteca Sklearn. A precisão e revocação foram iguais entre os dois modelos para todas as configurações de K.

Em relação ao Tempo de Execução, o classificador manual apresentou um tempo total de 0.12 segundos, enquanto o classificador Sklearn foi significativamente mais rápido, com 0.01 segundos. Essa diferença indica que a implementação da biblioteca é mais eficiente, devido à otimização interna e ao uso de estruturas de dados mais adequadas. 

Em relação ao Uso de Memória, em termos de consumo de memória, o classificador manual utilizou 37.27 MB, enquanto o Sklearn consumiu 25.35 MB. Embora ambos os métodos consumam uma quantidade razoável de memória, a implementação da biblioteca se mostrou mais econômica, refletindo uma melhor gestão de recursos. Foi utilizada a biblioteca psutil para fazer a medição.

É relevante destacar que, ao executar o código novamente, os valores de tempo e memória irão variar, mas é esperado que permaneçam em uma faixa semelhante. Isso se deve a fatores como a carga do sistema e a aleatoriedade na divisão dos dados, que podem influenciar o desempenho dos classificadores.
