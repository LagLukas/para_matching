#include "file_writer.h"

file_writer::file_writer(std::string path){
    this->file_stream.open(path);
    this->line = 0;
}

file_writer::~file_writer(){
    this->file_stream.close();
}

void file_writer::write_line(std::vector<double> values){
    for(int i = 0; i < values.size(); i++){
        this->file_stream << values[i];
        if(i != values.size() - 1){
            this->file_stream << ";";
        }else{
            this->file_stream << "\n";
        }
    }
    this->line = this->line + 1;
}
