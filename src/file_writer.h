#ifndef FILE_WRITER
#define FILE_WRITER
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

class file_writer{
    public:
    int line;
    std::ofstream file_stream;
    file_writer(std::string path);
    void write_line(std::vector<double> values);
    ~file_writer();
};


#endif