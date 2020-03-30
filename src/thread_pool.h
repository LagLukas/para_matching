#ifndef THREAD_POOL_H
#define THREAD_POOL_H
#include <thread>
#include <mutex>
#include <vector>
#include <atomic>
#include <functional>

///** 
template <class T, class M> class thread_pool
{
public:
	typedef void (M::*task_pointer) (T);
	thread_pool();
	~thread_pool();
	void insert_task(task_pointer f, T arg, M * object_pointer);
	void exec_task();
	task_pointer get_task(T & arg, M * object_pointer);
	bool is_working();
	void close();
	void work();
	void join();	
private:
	std::vector<task_pointer> task_functions;
	std::vector<T> task_args;
	std::vector<M *> object_references;
	std::mutex lock;
	std::atomic_bool done;
	std::atomic_int working;
	std::vector<std::thread> threads;

};
//*/
#endif // THREAD_POOL_H
