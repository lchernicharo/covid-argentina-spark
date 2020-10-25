# Algoritmos de classificação sobre base de dados do COVID-19
Projeto para ser executado no Apache Spark e que aplica diversos modelos de classificação na base de dados compartilhada pelo governo da Argentina sobre os casos de COVID-19.

## Funcionamento

Este programa foi desenvolvido utilizando o pacote **MLlib** do **Apache Spark 3.0.1** para o processamento dos algoritmos e o sistema de arquivos **HDFS** do **Apache Hadoop** para armazenamento dos arquivos de dados e dos resultados gerados. Caso você não tenha o **Hadoop** instalado, recomendo que o faça, pois isso aumenta significativamente a qualidade de todo o processo em diversos aspectos. Entretanto, seu uso realmente não é obrigatório, mas algumas mudanças serão necessárias no código.