#ifndef EXPERIMENT_H
#define EXPERIMENT_H
#include <vector>
#include <chrono>
#include <string>
#include <memory>
#include <iostream>
#include "threadsafe_list.h"
#include "classifier.h"

class matching_chunk{
    public:
    matching_chunk():start(0), end(0), match_set(nullptr){
        //this->sigma = new std::vector<char>;
    };
    ~matching_chunk(){
        //delete this->sigma;
    }
    matching_chunk(int start, int end, threadsafe_list<classifier> * match_set, std::vector<char> * sigma): start(start), end(end), match_set(match_set), sigma(sigma){
        //if(this->sigma == nullptr){
        //    this->sigma = new std::vector<char>;
        //}
    };    
    int start;
    int end;
    threadsafe_list<classifier> * match_set;
    std::shared_ptr<std::vector<char>> sigma;
};

class experiment{

    public:
    double match_percentage;
    int seed;
    int size;
    int dim;
    int iterations;
    int n_threads;
    std::vector<classifier> population;
    std::shared_ptr<std::vector<char>> rand_instance();
    void rand_instance(std::vector<char> * instance);
    void create_rand_population();
    void create_rand_population(std::shared_ptr<std::vector<char>> instance);
    experiment();
    experiment(int seed, int size, int dim, int iterations, double match_percentage, int n_threads);
    experiment(int size, int dim, int iterations);
    ~experiment();
    void match_task_lock(matching_chunk * chunk);
    void match_task_tsx(matching_chunk * chunk);
    void match_task_merge_list(matching_chunk * chunk);   
    std::chrono::duration<double> matching_sequential(std::shared_ptr<std::vector<char>> instance);
    std::chrono::duration<double> matching_mutex(std::shared_ptr<std::vector<char>> instance);
    std::chrono::duration<double> matching_merge_list(std::shared_ptr<std::vector<char>> instance);
    std::chrono::duration<double> matching_tsx(std::shared_ptr<std::vector<char>> instance);
    void print(std::string arg);
    void perform_exp();
};

#endif