#include "experiment.h"
#include "classifier.h"
#include "threadsafe_list.h"
#include "thread_pool.h"
#include <stdlib.h> 
#include <time.h> 
#include <thread>
#include <math.h>
#include <functional>
#include <iostream>
#include <memory>

experiment::experiment(){}

experiment::experiment(int size, int dim, int iterations){
    this->seed = 42;
    this->size = size;
    this->dim = dim;
    this->iterations = iterations;
    this->match_percentage = 0.15;
    srand(this->seed);
}

experiment::experiment(int seed, int size, int dim, int iterations, double match_percentage, int n_threads){
    this->seed = seed;
    this->size = size;
    this->dim = dim;
    this->iterations = iterations;
    this->match_percentage = match_percentage;
    this->n_threads = n_threads;
    srand(this->seed);
}

experiment::~experiment(){
}

void experiment::perform_exp(){
    std::chrono::duration<double> seq{0};
    std::chrono::duration<double> mut{0};
    std::chrono::duration<double> tsx{0};
    for(int i = 0; i < iterations; i++){
        this->population.clear();
        //std::cout<<"init instance" << std::endl;
        auto instance = this->rand_instance();
        //std::cout<<"init population" << std::endl;
        this->create_rand_population(instance);
        //std::cout<<"start matching" << std::endl;
        seq = this->matching_sequential(instance);
        mut = this->matching_mutex(instance);
        tsx = this->matching_merge_list(instance);
        std::cout << seq.count() << " " << mut.count() << " " << tsx.count() << std::endl;
    }
}

void experiment::create_rand_population(std::shared_ptr<std::vector<char>> instance){
    std::vector<classifier> matches{};
    std::vector<classifier> no_matches{};
    int matches_size = ceil(this->size * this->match_percentage);
    int no_matches_size = this->size - matches_size;
    // create matching classifier
    for(int i = 0; i < matches_size; i++){
        std::vector<char> conditions{};
        for(int j = 0; j < instance->size(); j++){
            if(rand()%2){
                conditions.push_back(2);
            }else{
                conditions.push_back(instance->at(j));
            }
        }
        classifier clazz{std::move(conditions)};
        matches.push_back(clazz);        
    }
    // create non matching classifier
    for(int i = 0; i < no_matches_size; i++){
        std::vector<char> conditions{};
        for(int j = 0; j < instance->size(); j++){
            if(rand()%2){
                conditions.push_back(2);
            }else{
                conditions.push_back(instance->at(j));
            }
            int index = (int) rand() % instance->size();
            int new_val = instance->at(index) == 0 ? 1 : 0;
            conditions[index] = new_val;
        }
        classifier clazz{std::move(conditions)};
        no_matches.push_back(clazz);        
    }
    int border = this->match_percentage * 1000;
    // shuffle
    for(int i = 0; i < this->size; i++){
        bool draw_matching = (rand() % 1000) * this->match_percentage <= border;
        if(matches.size() == 0){
            this->population.push_back(no_matches.back());
            no_matches.pop_back();
        }else if(no_matches.size() == 0){
            this->population.push_back(matches.back());
            matches.pop_back();
        }else if(draw_matching){
            this->population.push_back(matches.back());
            matches.pop_back();
        }else{
            this->population.push_back(no_matches.back());
            no_matches.pop_back();
        }
    }
}

std::chrono::duration<double> experiment::matching_sequential(std::shared_ptr<std::vector<char>> instance){
    auto start = std::chrono::high_resolution_clock::now();
    threadsafe_list<classifier> match_set{};
    for(auto cl : this->population){
        if(cl.matches(instance)){
            match_set.insert_to_list_sequential(cl);
	    }
    }
    auto finish = std::chrono::high_resolution_clock::now();
    return finish - start;    
}

std::chrono::duration<double> experiment::matching_merge_list(std::shared_ptr<std::vector<char>> instance){
    auto start = std::chrono::high_resolution_clock::now();
    threadsafe_list<classifier> match_set{};
    int n_threads = this->n_threads;
    int chunk_size = ceil(this->size / n_threads);
    std::vector<matching_chunk> chunks;
    for(int i = 0; i < n_threads; i++){
        chunks.push_back(matching_chunk(i * chunk_size, (i + 1) * chunk_size, &match_set, nullptr));
    }
    chunks[n_threads - 1].end = this->population.size();
    std::vector<std::thread> threads;
    for(int j = 0; j < n_threads; j++){
        chunks[j].sigma = instance;
        threads.push_back(std::thread(&experiment::match_task_merge_list, this, &chunks[j]));
    }
    for(std::thread &t : threads){
        t.join();
    }
    auto finish = std::chrono::high_resolution_clock::now();
    return finish - start;
}

std::chrono::duration<double> experiment::matching_mutex(std::shared_ptr<std::vector<char>> instance){
    auto start = std::chrono::high_resolution_clock::now();
    threadsafe_list<classifier> match_set{};
    int n_threads = this->n_threads;
    int chunk_size = ceil(this->size / n_threads);
    std::vector<matching_chunk> chunks;
    for(int i = 0; i < n_threads; i++){
        chunks.push_back(matching_chunk(i * chunk_size, (i + 1) * chunk_size, &match_set, nullptr));
    }
    chunks[n_threads - 1].end = this->population.size();
    std::vector<std::thread> threads;
    for(int j = 0; j < n_threads; j++){
        chunks[j].sigma = instance;
        threads.push_back(std::thread(&experiment::match_task_lock, this, &chunks[j]));
    }
    for(std::thread &t : threads){
        t.join();
    }
    auto finish = std::chrono::high_resolution_clock::now();
    return finish - start;
}

std::chrono::duration<double> experiment::matching_tsx(std::shared_ptr<std::vector<char>> instance){
    auto start = std::chrono::high_resolution_clock::now();
    threadsafe_list<classifier> match_set{};
    int n_threads = this->n_threads;
    int chunk_size = ceil(this->size / n_threads);
    std::vector<matching_chunk> chunks;
    for(int i = 0; i < n_threads; i++){
        chunks.push_back(matching_chunk(i * chunk_size, (i + 1) * chunk_size, &match_set, nullptr));
    }
    chunks[n_threads - 1].end = this->population.size();
    std::vector<std::thread> threads;
    for(int j = 0; j < n_threads; j++){
        chunks[j].sigma = instance;
        threads.push_back(std::thread(&experiment::match_task_tsx, this, &chunks[j]));
    }
    for(std::thread &t : threads){
        t.join();
    }
    match_set.clear();
    auto finish = std::chrono::high_resolution_clock::now();
    return finish - start;
}


std::shared_ptr<std::vector<char>> experiment::rand_instance(){
    std::shared_ptr<std::vector<char>> instance = std::make_shared<std::vector<char>>();
    for(int i = 0; i < this->dim; i++){
        instance->push_back(((char) rand() % 2));
    }
    return instance;
}

void experiment::match_task_lock(matching_chunk * chunk){
    for(int i = chunk->start; i < chunk->end; i++){
        if(this->population[i].matches(chunk->sigma)){
            (chunk->match_set)->insert_to_list(population[i]);
        }
    }
}

void experiment::match_task_tsx(matching_chunk * chunk){
    for(int i = chunk->start; i < chunk->end; i++){
        if(this->population[i].matches(chunk->sigma)){
            (chunk->match_set)->insert_to_list_tsx(population[i]);
        }
    }
}

void experiment::match_task_merge_list(matching_chunk * chunk){
    std::list<classifier> hits{};
    for(int i = chunk->start; i < chunk->end; i++){
        if(this->population[i].matches(chunk->sigma)){
            hits.push_back(population[i]);
        }
    }
    (chunk->match_set)->merge_to_list_lock(hits);
}
