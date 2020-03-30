#include <immintrin.h>
#include <iostream>
#include "threadsafe_list.h"
#include "classifier.h"

constexpr int MAX_RETRIES = 5;

template <class T> threadsafe_list<T>::threadsafe_list()
{
	//this->insert_lock = new std::mutex();
	is_locked = false;
	//this->v = {};
}

template <class T> threadsafe_list<T>::~threadsafe_list()
{
	//delete this->insert_lock;
	//delete this->v;
}

template <class T> void threadsafe_list<T>::insert_to_list(T element){
	std::lock_guard<std::mutex> lock(this->insert_lock);
	this->v.push_back(element);
}

template <class T> void threadsafe_list<T>::insert_to_list_tsx(T element){
	unsigned status;
	for(int i = 0; i < MAX_RETRIES; i++){
		if((status = _xbegin()) == _XBEGIN_STARTED) {
			if(this->is_locked){ 
			/*Lockvariable muss im Read Set sein */
			_xabort(0xFF);
			}
			this->v.push_back(element);
			/* kritischer Abschnitt ohne Lock*/
			_xend();
			break;
		}
		if((status & _XABORT_RETRY) && i < MAX_RETRIES){
			while(this->is_locked){
				_mm_pause();
			}
		}else{
			std::lock_guard<std::mutex> lock(this->insert_lock);
			this->is_locked = true;
			this->v.push_back(element);
			this->is_locked = false;
		}
	}
}

template <class T> void threadsafe_list<T>::merge_to_list_lock(std::list<T> other){
	std::lock_guard<std::mutex> lock(this->insert_lock);
	this->v.splice(this->v.end(), other);
	//this->v.merge(other);
}

template <class T> void threadsafe_list<T>::insert_to_list_sequential(T element){
	this->v.push_back(element);
}

template <class T> void threadsafe_list<T>::clear(){
	this->v.clear();
}

template class threadsafe_list<int>; 
template class threadsafe_list<classifier>;
//template class threadsafe_list<classifier&>;
