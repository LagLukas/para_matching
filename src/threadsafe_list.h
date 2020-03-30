#ifndef THREADSAFE_LIST_H
#define THREADSAFE_LIST_H
#include <mutex>
#include <list>

template <class T> class threadsafe_list
{
private:
	bool is_locked;
	std::list<T> v;
	std::mutex insert_lock;
public:
	threadsafe_list();
	~threadsafe_list();
	void merge_to_list_lock(std::list<T> other);
	void insert_to_list_sequential(T element);
	void insert_to_list(T element);
	void insert_to_list_tsx(T element);
	void clear();
};

#endif // THREADSAFE_LIST_H
