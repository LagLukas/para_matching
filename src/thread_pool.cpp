#include <iostream>
#include "thread_pool.h"
#include "classifier.h"
#include "experiment.h"
template<class T, class M> 
using task_pointer = void (M::*) (T);
///** 
template <class T, class M> thread_pool<T, M>::thread_pool()
{
	this->done = false;
	this->working = 0;
	int n_threads = std::thread::hardware_concurrency();
	for(int i = 0; i < n_threads; i++){
		this->threads.push_back(std::move(std::thread(&thread_pool::work, this)));
	}
}

template <class T, class M> thread_pool<T, M>::~thread_pool()
{
	this->close();
}

template <class T, class M> void thread_pool<T, M>::insert_task(task_pointer f, T arg, M * object_pointer){
	std::lock_guard<std::mutex> lock(this->lock);
	this->task_functions.push_back(f);
	this->task_args.push_back(arg);
	this->object_references.push_back(object_pointer);
}

template <class T, class M>  task_pointer<T, M> thread_pool<T, M>::get_task(T & arg, M * object_pointer){
	std::lock_guard<std::mutex> lock(this->lock);
	if(this->task_functions.size()>0){
		auto f = this->task_functions.back();
		this->task_functions.pop_back();	
		arg = this->task_args.back();
		this->task_args.pop_back();
		object_pointer = this->object_references.back();
		//std::cout<<object_pointer->size;
		this->object_references.pop_back();
		return f;
	}
	return nullptr;
}

template <class T, class M>  void thread_pool<T, M>::exec_task(){
	//std::lock_guard<std::mutex> lock(this->lock);
	this->lock.lock();
	if(this->task_functions.size()>0){
		auto task = this->task_functions.back();
		this->task_functions.pop_back();	
		auto arg = this->task_args.back();
		this->task_args.pop_back();
		auto object_pointer = this->object_references.back();
		this->object_references.pop_back();
		this->lock.unlock();
		if(arg->sigma == nullptr){
			std::cout<<"was";
		}
		working++;
		(object_pointer->*task)(arg);
		working--;		
	}else{
		this->lock.unlock();
		std::this_thread::yield();
	}
}


template <class T, class M> void thread_pool<T, M>::work(){
	while(!this->done){
		this->exec_task();
		/**
		T a;
		M obj;
		auto task = this->get_task(a, &obj);
		M * obj_pointer = &obj;
		std::cout<<obj_pointer->size<<std::endl;
		if((task != nullptr) && (obj_pointer != nullptr)){

		}else{
			std::this_thread::yield();
		}
		*/
	}
}

template <class T, class M> void thread_pool<T, M>::close(){
	this->done = true;
	this->join();
}

template <class T, class M> void thread_pool<T, M>::join(){
	for(std::thread &t : this->threads){
		t.join();
	}
}

template <class T, class M> bool thread_pool<T, M>::is_working(){
	if(this->working == 0){
		return false;
	}
	return true;
}

//template class thread_pool<int>; 
//template class thread_pool<classifier>;
//template class thread_pool<matching_chunk, experiment>;
template class thread_pool<matching_chunk *, experiment>;
//*/
