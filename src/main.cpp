#include <iostream>
#include <stdio.h>
#include "threadsafe_list.h"
#include "thread_pool.h"
#include "classifier.h"
#include "experiment.h"
#include "file_writer.h"
#include <string>
#include <list>
#include <stdlib.h>

double print_speedup(std::chrono::duration<double> a, std::chrono::duration<double> b){
	double speed_up = ((double) a.count()) / b.count();
	std::cout << speed_up;
	return speed_up;
}

std::vector<int> gen_exp_vector(int start, int end, int inc){
	std::vector <int> v{};
	for(int i = start; i < end; i+=inc){
		v.push_back(i);
	}
	return v;
}

int main(int argc, char **argv)
{
	if(argc != 7){
		return -1;
	}
	int seed = atoi(argv[1]);
	//std::cout << "seed: " << seed << std::endl;
	int size = atoi(argv[2]);
	//std::cout << "size: " << size << std::endl;
	int dimension = atoi(argv[3]);
	//std::cout << "dimension: " << dimension << std::endl;
	int iterations = atoi(argv[4]);
	//std::cout << "iterations: " << iterations << std::endl;
	double match_percentage = atof(argv[5]);
	//std::cout << "match percentage: " << match_percentage << std::endl;
	int n_threads = atoi(argv[6]);
	//std::cout << "n_threads: " << n_threads << std::endl;
	//std::cout<<"init exp" << std::endl;
	experiment e{seed, size, dimension, iterations, match_percentage, n_threads};
	//std::cout<<"start exp" << std::endl;
	e.perform_exp();
	return 0;
}
