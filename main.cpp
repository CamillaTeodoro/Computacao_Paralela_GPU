#include "include/Network.hpp"
#include "include/Dataset.hpp"

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

int main(int argc, char *argv[])
{
	int num_atributos = 4;
	int num_classes = 3;

	Neural::Dataset data_learning;
	data_learning.loadInputOutputData(num_atributos, num_classes, "database/iris.txt");

	vector<vector<double>> input_vec = data_learning.getInput();
	vector<vector<double>> output_vec = data_learning.getOutput();

	// Converter vectors para arrays
	int input_rows = input_vec.size();
	int input_cols = input_vec[0].size();
	int output_rows = output_vec.size();
	int output_cols = output_vec[0].size();

	double *input_array = new double[input_rows * input_cols];
	double *output_array = new double[output_rows * output_cols];

	// Copiar dados dos vectors para os arrays
	for (int i = 0; i < input_rows; i++)
	{
		for (int j = 0; j < input_cols; j++)
		{
			input_array[i * input_cols + j] = input_vec[i][j];
		}
	}

	for (int i = 0; i < output_rows; i++)
	{
		for (int j = 0; j < output_cols; j++)
		{
			output_array[i * output_cols + j] = output_vec[i][j];
		}
	}

	int maximo_epocas_para_testar = 1000;
	int taxa_acerto_desejada = 95;
	double tolerancia_maxima_de_erro = 0.05;
	int maximo_camadas_escondidas = 15;
	double taxa_de_aprendizado = 0.25;

	// Criar rede neural com arrays
	Neural::Network neural_network(input_array, output_array, input_rows, output_rows);

	neural_network.setParameter(
		maximo_epocas_para_testar,
		taxa_acerto_desejada,
		tolerancia_maxima_de_erro,
		maximo_camadas_escondidas,
		taxa_de_aprendizado);

	neural_network.autoTraining(maximo_camadas_escondidas, taxa_de_aprendizado);

	// Limpar memória
	delete[] input_array;
	delete[] output_array;

	return 0;
}