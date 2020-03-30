#include "classifier.h"
#include <iostream>

classifier::classifier(){

}

classifier::classifier(std::vector<char>&& conditions){
    this->conditions = conditions;
}

classifier::~classifier(){
}

bool classifier::matches(std::vector<char> situation){
    for(int i = 0; i<situation.size(); i++){
        if(situation.at(i) != this->conditions.at(i)){
            return false;
        }
    }
    return true;
}

bool classifier::matches(std::vector<char> * situation){
    for(int i = 0; i<situation->size(); i++){
        if(situation->at(i) != this->conditions.at(i)){
            return false;
        }
    }
    return true;
}

bool classifier::matches(std::shared_ptr<std::vector<char>> situation){
    for(int i = 0; i<situation->size(); i++){
        if((situation->at(i) != this->conditions.at(i)) && this->conditions.at(i) != 2){
            return false;
        }
    }
    return true;
}

bool classifier::operator==(const classifier& rhs){
    if(this->conditions.size() != rhs.conditions.size()){
        return false;
    }
    for(int i = 0; i < rhs.conditions.size(); i++){
        if(rhs.conditions.at(i) != this->conditions.at(i)){
            return false;
        }
    }
    return true;
}

bool classifier::operator<(const classifier& rhs){
    return true;
}
