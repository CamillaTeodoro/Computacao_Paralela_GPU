# Projeto 02 - Computação Paralela**Grupo:**  
- Gabriel Vargas  
- Leticia Americano  
- Nilson Deon  
- Olga Camilla  

**Data:** Novembro/2024  

---

## Descrição
Este projeto é uma aplicação de redes neurais treinada com o dataset Iris. Utilizamos três versões para a execução do algoritmo de Perceptron com backpropagation: sequencial, paralela com OpenMP e distribuída com MPI, explorando diferentes abordagens de paralelização para otimizar o treinamento da rede neural.

O código original não foi desenvolvido pelo grupo e está disponível no repositório: alexandremstf/neural-network (Autor: Alexandre Magno).

## Estrutura do Projeto
A estrutura básica do código se organiza em módulos para tratar das diferentes configurações de execução (Sequencial, OpenMP, MPI) e nas funções de implementação da rede neural e do treinamento via backpropagation. Abaixo, uma visão geral dos principais diretórios e arquivos:

- `src/`: Código-fonte principal, com versões OpenMP_GPU, e CUDA do treinamento da rede neural.
- `Makefile`: Configurações de compilação para cada versão do código.

## Como Executar
### Compilação

Codigo OpenMP_GPU : 

Codigo MPI : 

### Executar o código:

Codigo Sequencial: time ./neuralnetwork

Codigo OpenMP_GPU : time ./neuralnetwork

Codigo CUDA

## Dependências
Certifique-se de ter configurado as dependências necessárias para a execução de cada versão. Para compilar e executar o projeto, certifique-se de ter as seguintes dependências instaladas:

1. **Compilador C/C++**: Necessário para compilar o código em qualquer versão. Recomendado `gcc` ou `g++`.
2. **Make**: Utilizado para gerenciar a compilação do projeto.
3. **OpenMP**: Necessário para a versão paralela com OpenMP. Geralmente incluído no `gcc` (versão 4.2 ou superior).
4. **CUDA**: Necessário para a versão para GPU. Pode se instalado com sudo apt install nvidia-cuda-toolkit (Linux) ou baixado no site https://developer.nvidia.com/cuda-toolkit

## Explicação da Aplicação
### Dataset: Iris
O dataset Iris é um conjunto de dados amplamente utilizado no aprendizado de máquina. Ele contém 150 amostras divididas igualmente entre três espécies de flores de íris: *Iris setosa*, *Iris virginica* e *Iris versicolor*. Cada amostra possui quatro características: comprimento e largura da sépala, e comprimento e largura da pétala. O objetivo da rede neural é classificar corretamente a espécie de uma flor com base nas suas características.

### Perceptron com Backpropagation
A rede neural utilizada neste projeto é baseada no modelo de Perceptron, uma das arquiteturas mais simples e eficazes para problemas de classificação linear. Neste caso, a rede neural é composta por uma camada de entrada, uma ou mais camadas ocultas, e uma camada de saída. O treinamento é realizado através do algoritmo de backpropagation, que ajusta os pesos da rede para minimizar o erro entre a saída prevista e a saída desejada.

### Versões de Execução
- **Sequencial**: Treinamento da rede neural executado em um único núcleo.
- **OpenMP**: Paralelização baseada em threads, utilizando OpenMP para acelerar o treinamento distribuindo o processamento entre múltiplos núcleos.
- **MPI**: Implementação distribuída utilizando MPI para execução em ambientes com múltiplas máquinas, dividindo o trabalho entre processos independentes.

## Alterações no Código

1. **Configuração da Camada de Saída**: Alteramos `output_layer_size` de 1 para 3 para permitir múltiplas saídas no problema de classificação.
2. **Aprimoramento de `autoTraining`**: Automatizamos a busca por configurações ótimas de camada oculta e taxa de aprendizado.
3. **Saídas Detalhadas em `trainingClassification`**: Adicionamos informações sobre o progresso do treinamento, como taxa de acerto e número de épocas.
4. **Cálculos Otimizados em `ForwardPropagation` e `BackPropagation`**: Ajustamos o cálculo de ativações e erros para lidar com múltiplas classes de saída.
5. **Inicialização Simplificada**: Consolidamos a configuração dos parâmetros principais em uma função (`setParameter`).
6. **Cálculo de Taxa de Acerto**: Otimizamos o cálculo para melhorar a precisão na avaliação de desempenho.

Essas mudanças melhoram a precisão, flexibilidade e eficiência do código, com suporte a paralelização e otimização de parâmetros.

## Conclusão 
Durante a execução do projeto, foram realizados testes para comparar o desempenho entre as versões. A paralelização permite um treinamento mais rápido nas versões OpenMP e MPI, especialmente em conjuntos de dados maiores ou redes neurais mais complexas.

O código foi executado em um computador com as seguintes especificações:

[ARRUMAR]:
Arquitetura do Sistema: x86_64 (64 bits)
Sistema Operacional: Ubuntu 22.04.1
Versão do Kernel do Linux: 6.8.0-48-generic
Processador: Intel(R) Core(TM) i7-10700F CPU @ 2.90GHz
Núcleos por Soquete: 8
Threads por Núcleo: 2
Soquetes: 1
Total de CPUs (Núcleos * Threads por Núcleo * Soquetes): 16
Memória RAM Total: 15 GiB
Swap Total: 2,0 GiB

### Versão Sequencial

- **Tempo**: 29.424 segundos

### VersãoOpenMP_GPU

- **Tempo 1 Thread**: 
- **Tempo 2 Threads**: 
- **Tempo 4 Threads**: 
- **Tempo 8 Threads**: 

### Versão CUDA

- **Tempo 1 Team e 4 Threads**: 
- **Tempo 2 Teams e 2 Threads**: 
- **Tempo 4 Teams e 0 Threads**: 

[ARRUMAR]: Esses resultados demonstram uma clara redução no tempo de execução com o aumento do nível de paralelismo. A versão OpenMP apresenta uma redução significativa do tempo à medida que o número de threads aumenta, com a execução em 8 threads sendo a mais rápida. A versão MPI também mostra uma diminuição do tempo de execução com o aumento do número de processos, destacando-se a execução com 4 processos como a mais eficiente entre as configurações testadas.
