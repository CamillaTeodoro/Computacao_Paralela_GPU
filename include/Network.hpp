#ifndef NETWORK_HPP_
#define NETWORK_HPP_

#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <cstring>

namespace Neural
{

    class Network
    {

        struct ForwardPropagation
        {
            double *sum_input_weight;            // Armazena a soma dos produtos entre as entradas e os pesos das entradas para cada neurônio na camada oculta
            double *sum_output_weight;           // Armazena a soma dos produtos entre as saídas da camada oculta e os pesos de saída para cada neurônio na camada de saída
            double *sum_input_weight_activation; // Armazena o resultado da função de ativação aplicada ao sum_input_weight
            double *output;                      // Armazena a saída final da rede neural

            ForwardPropagation(int size_input, int size_output)
            {
                sum_input_weight = new double[size_input]();
                sum_output_weight = new double[size_output]();
                sum_input_weight_activation = new double[size_input]();
                output = new double[size_output]();
            }

            ~ForwardPropagation()
            {
                delete[] sum_input_weight;
                delete[] sum_output_weight;
                delete[] sum_input_weight_activation;
                delete[] output;
            }
        };

        struct network
        {
            int epoch;
            int hidden_layer;
            double learning_rate;
            double *weight_input;
            double *weight_output;
        };

    private:
        int input_layer_size;
        int output_layer_size;
        int hidden_layer_size;

        double *input;         // Dados de entrada linearizados
        double *output;        // Dados de saída linearizados
        double *weight_input;  // Pesos de entrada linearizados
        double *weight_output; // Pesos de saída linearizados

        network best_network;

        int output_rows;

        int epoch;
        int max_epoch;

        int correct_output;
        int hit_percent;

        double desired_percent;
        double learning_rate;
        double error_tolerance;

        int input_weight_size;
        int output_weight_size;

    public:
#pragma omp declare target
        Network();

        void initializeWeight();
        void trainingClassification();
#pragma omp end declare target
        void autoTraining(int, double);
        void run();
        Network(double *, double *, int, int);

        ~Network();
        void hitRateCount(double *, unsigned int);
        void hitRateCalculate();

        ForwardPropagation forwardPropagation(double *);
        void backPropagation(ForwardPropagation &, double *, double *);

        double sigmoid(double);
        double sigmoidPrime(double);

        void setInput(double *, int, int);
        void setOutput(double *, int, int);
        void setMaxEpoch(int);
        void setDesiredPercent(int);
        void setHiddenLayerSize(int);
        void setLearningRate(double);
        void setErrorTolerance(double);
        void setParameter(int, int, double, double = 1, int = 1);
    };

}

#endif
