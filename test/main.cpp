#include "net.h"
#include <fstream>
#include <iostream>
#include "utils.h"

vector<string> input_names = {"in","grid"};
vector<Dim> input_dims = {Dim(128,50,50),Dim(128,128,2)};
vector<string> output_names = {"ooo"};
vector<Dim> output_dims = {Dim(128, 128, 128)};
//vector<Dim> output_dims = {Dim(126, 126,2)};



int main(int argc, char **argv)
{
    
    string trt_name = "data/2.trt";
    string input_name = "data/input.bin";
    string grid_name = "data/grid.bin";
    

    cell<float,int>* grid_data= new cell<float,int>(1,2,128,128);
    grid_data->read_data(grid_name);

    cell<float,int>* in_data = new cell<float,int>(1,128,50,50);
    in_data->read_data(input_name);


    vector<float *> inputs = {in_data->host_ptr(),grid_data->host_ptr()};

    cout << "the   complete file is in a buffer ";
    
    // delete[] buffer;

    //vector<float *> inputs = {data,grid};

    BaseNet net = BaseNet();
    net.init(input_names, input_dims, output_names, output_dims);
    net.build(trt_name);
    net.infer(inputs);
    net.debug(128);

    return 0;
}
