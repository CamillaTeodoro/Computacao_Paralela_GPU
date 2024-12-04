#include "../include/Network.hpp"

#ifdef NUM_THREADS
#define THREADS NUM_THREADS
#else
#define THREADS 1
#endif

namespace Neural
{
#pragma omp declare target
    Network::Network()
    {
        input = nullptr;
        output = nullptr;
        weight_input = nullptr;
        weight_output = nullptr;
        omp_set_num_threads(THREADS);
    }

#pragma omp end declare target
    Network::~Network()
    {
        std::cout << "Destrutor: Iniciando..." << std::endl;
        if (input != nullptr)
        {
            delete[] input;
            input = nullptr;
        }
        if (output != nullptr)
        {
            delete[] output;
            output = nullptr;
        }
        if (weight_input != nullptr)
        {
            delete[] weight_input;
            weight_input = nullptr;
        }
        if (weight_output != nullptr)
        {
            delete[] weight_output;
            weight_output = nullptr;
        }
        std::cout << "Destrutor: Finalizado" << std::endl;
    }

    Network::Network(double *user_input, double *user_output, int input_rows, int input_cols, int output_rows, int output_cols)
    {
        input = nullptr;
        output = nullptr;
        weight_input = nullptr;
        weight_output = nullptr;
        std::cout << "Construtor: Iniciando..." << std::endl;

        try
        {
            std::cout << "Construtor: Configurando input..." << std::endl;
            setInput(user_input, input_rows, input_cols);

            std::cout << "Construtor: Configurando output..." << std::endl;
            setOutput(user_output, output_rows, output_cols);

            std::cout << "Construtor: Configurando output_layer_size..." << std::endl;
            output_layer_size = output_cols;

            std::cout << "Construtor: Configurando threads..." << std::endl;
            omp_set_num_threads(THREADS);

            std::cout << "Construtor: Finalizado" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Erro no construtor: " << e.what() << std::endl;
            throw;
        }
        catch (...)
        {
            std::cerr << "Erro desconhecido no construtor" << std::endl;
            throw;
        }
    }

    void Network::setParameter(int user_max_epoch, int user_desired_percent, double user_error_tolerance, double user_learning_rate, int user_hidden_layer_size)
    {

        setMaxEpoch(user_max_epoch);
        setLearningRate(user_learning_rate);
        setErrorTolerance(user_error_tolerance);
        setDesiredPercent(user_desired_percent);
        setHiddenLayerSize(user_hidden_layer_size);
        best_network.epoch = max_epoch;

        initializeWeight();
    }

#pragma omp declare target
    void Network::run()
    {
        for (int data_row = 0; data_row < output_rows; data_row++)
        {
            int input_start = data_row * input_layer_size;
            ForwardPropagation forward;
            forwardPropagation(&input[input_start], forward);
            hitRateCount(forward.output, data_row);
        }
        hitRateCalculate();
    }
#pragma omp end declare target

    void Network::trainingClassification()
    {
        epoch = 0;
        while (epoch < max_epoch && hit_percent < desired_percent)
        {
            trainOneEpoch();
            epoch++;
        }

        printf("Hidden Layer Size: %d\tLearning Rate: %f\tHit Percent: %d%%\tEpoch: %d\n",
               hidden_layer_size, learning_rate, hit_percent, epoch);
    }

#pragma omp declare target
    void Network::trainOneEpoch()
    {
        for (int data_row = 0; data_row < output_rows; data_row++)
        {
            int input_start = data_row * input_layer_size;
            int output_start = data_row * output_layer_size;

            ForwardPropagation forward;
            forwardPropagation(&input[input_start], forward);
            backPropagation(forward, &input[input_start], &output[output_start]);
        }
        run();
    }
#pragma omp end declare target

    void Network::autoTraining(int hidden_layer_limit, double learning_rate_increase)
    {
        network global_best_network = best_network;
        Network *this_ptr = this;

        // Calcula o número total de iterações
        int num_hidden_layers = hidden_layer_limit - 2;
        int num_learning_rates = (int)(1.0 / learning_rate_increase);
        int total_iterations = num_hidden_layers * num_learning_rates;

        // Aloca e inicializa arrays
        int *epochs = new int[total_iterations]();
        double *learning_rates = new double[total_iterations]();
        int *hidden_layers = new int[total_iterations]();
        double *temp_weights_input = new double[input_weight_size * total_iterations]();
        double *temp_weights_output = new double[output_weight_size * total_iterations]();

        // Copia os dados necessários para arrays que serão mapeados para a GPU
        double *device_input = new double[input_layer_size * output_rows];
        double *device_output = new double[output_layer_size * output_rows];
        memcpy(device_input, input, input_layer_size * output_rows * sizeof(double));
        memcpy(device_output, output, output_layer_size * output_rows * sizeof(double));

#pragma omp target data map(to : this_ptr[0 : 1],                                   \
                                device_input[0 : input_layer_size * output_rows],   \
                                device_output[0 : output_layer_size * output_rows]) \
    map(tofrom : epochs[0 : total_iterations],                                      \
            learning_rates[0 : total_iterations],                                   \
            hidden_layers[0 : total_iterations],                                    \
            temp_weights_input[0 : input_weight_size * total_iterations],           \
            temp_weights_output[0 : output_weight_size * total_iterations])
        {
#pragma omp target teams distribute parallel for collapse(2)
            for (int i = 3; i <= hidden_layer_limit; i++)
            {
                for (int j = 0; j < num_learning_rates; j++)
                {
                    double current_learning_rate = (j + 1) * learning_rate_increase;
                    int idx = (i - 3) * num_learning_rates + j;

                    // Configura parâmetros locais
                    int local_hidden_size = i;
                    double local_learning_rate = current_learning_rate;
                    int local_epoch = 0;
                    int local_hit_percent = 0;

                    // Inicializa pesos locais
                    double *local_weight_input = new double[input_weight_size];
                    double *local_weight_output = new double[output_weight_size];

                    // Inicializa pesos aleatórios
                    for (int k = 0; k < input_weight_size; k++)
                    {
                        local_weight_input[k] = ((double)rand() / RAND_MAX);
                    }
                    for (int k = 0; k < output_weight_size; k++)
                    {
                        local_weight_output[k] = ((double)rand() / RAND_MAX);
                    }

                    // Treina a rede
                    while (local_epoch < max_epoch && local_hit_percent < desired_percent)
                    {
                        // Treina uma época
                        for (int data_row = 0; data_row < output_rows; data_row++)
                        {
                            int input_start = data_row * input_layer_size;
                            int output_start = data_row * output_layer_size;

                            ForwardPropagation forward;
                            forwardPropagation(&device_input[input_start], forward);
                            backPropagation(forward, &device_input[input_start], &device_output[output_start]);
                        }
                        local_epoch++;
                    }

                    // Salva resultados
                    epochs[idx] = local_epoch;
                    learning_rates[idx] = local_learning_rate;
                    hidden_layers[idx] = local_hidden_size;

                    // Copia pesos
                    int weight_input_offset = idx * input_weight_size;
                    int weight_output_offset = idx * output_weight_size;

#pragma omp parallel for
                    for (int k = 0; k < input_weight_size; k++)
                    {
                        temp_weights_input[weight_input_offset + k] = local_weight_input[k];
                    }

#pragma omp parallel for
                    for (int k = 0; k < output_weight_size; k++)
                    {
                        temp_weights_output[weight_output_offset + k] = local_weight_output[k];
                    }

                    // Libera memória local
                    delete[] local_weight_input;
                    delete[] local_weight_output;
                }
            }
        }

        // Encontra melhor resultado
        int best_idx = 0;
        for (int i = 1; i < total_iterations; i++)
        {
            if (epochs[i] < epochs[best_idx])
            {
                best_idx = i;
            }
        }

        // Atualiza rede com melhores parâmetros
        epoch = epochs[best_idx];
        learning_rate = learning_rates[best_idx];
        hidden_layer_size = hidden_layers[best_idx];

        // Copia melhores pesos
        int best_weight_input_offset = best_idx * input_weight_size;
        int best_weight_output_offset = best_idx * output_weight_size;
        memcpy(weight_input, &temp_weights_input[best_weight_input_offset],
               input_weight_size * sizeof(double));
        memcpy(weight_output, &temp_weights_output[best_weight_output_offset],
               output_weight_size * sizeof(double));

        printf("Best Network --> Hidden Layer Size: %d\tLearning Rate: %f\tEpoch: %d\n",
               hidden_layer_size, learning_rate, epoch);

        // Libera memória
        delete[] epochs;
        delete[] learning_rates;
        delete[] hidden_layers;
        delete[] temp_weights_input;
        delete[] temp_weights_output;
        delete[] device_input;
        delete[] device_output;
    }

#pragma omp declare target
    void Network::forwardPropagation(double *input_line, ForwardPropagation &forward)
    {
        forward.input_size = hidden_layer_size;
        forward.output_size = output_layer_size;

        // Calcula o somatório dos produtos entre entrada e pesos
        for (int i = 0; i < hidden_layer_size; i++)
        {
            forward.sum_input_weight[i] = 0;
            for (int j = 0; j < input_layer_size; j++)
            {
                forward.sum_input_weight[i] += input_line[j] * weight_input[j * hidden_layer_size + i];
            }
        }

        // Aplica função de ativação
        for (int i = 0; i < hidden_layer_size; i++)
        {
            forward.sum_input_weight_activation[i] = sigmoid(forward.sum_input_weight[i]);
        }

        // Calcula saídas
        for (int i = 0; i < output_layer_size; i++)
        {
            forward.sum_output_weight[i] = 0;
            for (int j = 0; j < hidden_layer_size; j++)
            {
                forward.sum_output_weight[i] += forward.sum_input_weight_activation[j] *
                                                weight_output[j * output_layer_size + i];
            }
            forward.output[i] = sigmoid(forward.sum_output_weight[i]);
        }
    }

    void Network::backPropagation(ForwardPropagation &forward, double *input_line, double *output_line)
    {
        double delta_output[ForwardPropagation::MAX_SIZE];
        double delta_input[ForwardPropagation::MAX_SIZE];

        // Calcula erro de saída
        for (int i = 0; i < output_layer_size; i++)
        {
            delta_output[i] = (output_line[i] - forward.output[i]) *
                              sigmoidPrime(forward.sum_output_weight[i]);
        }

        // Calcula erro da camada oculta
        for (int i = 0; i < hidden_layer_size; i++)
        {
            delta_input[i] = 0;
            for (int j = 0; j < output_layer_size; j++)
            {
                delta_input[i] += delta_output[j] * weight_output[i * output_layer_size + j];
            }
            delta_input[i] *= sigmoidPrime(forward.sum_input_weight[i]);
        }

        // Atualiza pesos
        for (int i = 0; i < hidden_layer_size; i++)
        {
            for (int j = 0; j < output_layer_size; j++)
            {
                weight_output[i * output_layer_size + j] += learning_rate *
                                                            delta_output[j] * forward.sum_input_weight_activation[i];
            }
        }

        for (int i = 0; i < input_layer_size; i++)
        {
            for (int j = 0; j < hidden_layer_size; j++)
            {
                weight_input[i * hidden_layer_size + j] += learning_rate *
                                                           delta_input[j] * input_line[i];
            }
        }
    }
    void Network::hitRateCount(double *neural_output, unsigned int data_row)
    {
        for (int i = 0; i < output_layer_size; i++)
        {
            // Acessa o output linearizado usando data_row * output_layer_size + i
            if (abs(neural_output[i] - output[data_row * output_layer_size + i]) < error_tolerance)
                correct_output++;
        }
    }

    void Network::hitRateCalculate()
    {

        hit_percent = (correct_output * 100) / (output_rows * output_layer_size);
        correct_output = 0;
    }

    void Network::initializeWeight()
    {
        input_weight_size = input_layer_size * hidden_layer_size;
        output_weight_size = hidden_layer_size * output_layer_size;

        // Liberar memória anterior se existir
        if (weight_input != nullptr)
        {
            delete[] weight_input;
            weight_input = nullptr;
        }
        if (weight_output != nullptr)
        {
            delete[] weight_output;
            weight_output = nullptr;
        }

        // Aloca memória para os arrays
        weight_input = new double[input_weight_size];
        weight_output = new double[output_weight_size];

        srand((unsigned int)time(0));

        // Inicializa os pesos
        for (int i = 0; i < input_weight_size; i++)
        {
            weight_input[i] = ((double)rand() / (RAND_MAX));
        }

        for (int i = 0; i < output_weight_size; i++)
        {
            weight_output[i] = ((double)rand() / (RAND_MAX));
        }

        hit_percent = 0;
        correct_output = 0;
    }

    double Network::sigmoid(double z)
    {
        return 1 / (1 + exp(-z));
    }

    double Network::sigmoidPrime(double z)
    {
        return exp(-z) / (pow(1 + exp(-z), 2));
    }
#pragma omp end declare target

    void Network::setMaxEpoch(int m)
    {
        max_epoch = m;
    }

    void Network::setDesiredPercent(int d)
    {
        desired_percent = d;
    }

    void Network::setHiddenLayerSize(int h)
    {
        hidden_layer_size = h;
    }

    void Network::setLearningRate(double l)
    {
        learning_rate = l;
    }

    void Network::setErrorTolerance(double e)
    {
        error_tolerance = e;
    }

    void Network::setInput(double *i, int rows, int cols)
    {
        std::cout << "setInput: Iniciando (rows=" << rows << ", cols=" << cols << ")" << std::endl;

        try
        {
            if (i == nullptr)
            {
                throw std::runtime_error("Input array is null");
            }

            input = new double[rows * cols];
            if (input == nullptr)
            {
                throw std::runtime_error("Failed to allocate input array");
            }

            std::cout << "setInput: Copiando dados..." << std::endl;
            for (int j = 0; j < rows * cols; j++)
            {
                input[j] = i[j];
            }

            input_layer_size = cols + 1; // +1 para bias
            std::cout << "setInput: Finalizado (input_layer_size=" << input_layer_size << ")" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Erro em setInput: " << e.what() << std::endl;
            throw;
        }
    }

    void Network::setOutput(double *o, int rows, int cols)
    {
        std::cout << "setOutput: Iniciando (rows=" << rows << ", cols=" << cols << ")" << std::endl;

        try
        {
            if (o == nullptr)
            {
                throw std::runtime_error("Output array is null");
            }

            output_rows = rows;
            output = new double[rows * cols];
            if (output == nullptr)
            {
                throw std::runtime_error("Failed to allocate output array");
            }

            std::cout << "setOutput: Copiando dados..." << std::endl;
            for (int j = 0; j < rows * cols; j++)
            {
                output[j] = o[j];
            }

            output_layer_size = cols;
            std::cout << "setOutput: Finalizado (output_layer_size=" << output_layer_size << ")" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Erro em setOutput: " << e.what() << std::endl;
            throw;
        }
    }

}