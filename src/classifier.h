#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <memory>
#include <vector>

class classifier{

    public:
    std::vector<char> conditions;
    classifier();
    classifier(std::vector<char>&& conditions);
    ~classifier();
    bool matches(std::vector<char> situation);
    bool matches(std::vector<char>  * situation);
    bool matches(std::shared_ptr<std::vector<char>> situation);
    bool operator==( const classifier& rhs);
    bool operator<( const classifier& rhs);
};

#endif
