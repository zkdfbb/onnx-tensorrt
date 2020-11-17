#include "net.h"
#include <fstream>
#include <iostream>
#include "utils.h"

vector<string> input_names = {"in","grid"};
vector<Dim> input_dims = {Dim(12,20,20),Dim(2,30,30)};
vector<string> output_names = {"ooo"};
vector<Dim> output_dims = {Dim(12, 30, 30)};



int main(int argc, char **argv)
{
    
    string trt_name = "data/2.trt";
    string input_name = "data/input.bin";
    string grid_name = "data/grid.bin";
    string pytorch_output_name = "data/pytorch_2.bin";
    

    
    
    // cell<float,int>* in_data = new cell<float,int>(1,12,20,20);
    cell<float,int>* in_data = new cell<float,int>(input_dims[0]);
    in_data->read_data(input_name);

    //cell<float,int>* grid_data= new cell<float,int>(1,2,30,30);
    cell<float,int>* grid_data= new cell<float,int>(input_dims[1]);
    grid_data->read_data(grid_name);

    vector<float *> inputs = {in_data->host_ptr(),grid_data->host_ptr()};

    cout << "the   complete file is in a buffer ";

    BaseNet net = BaseNet();
    net.init(input_names, input_dims, output_names, output_dims);
    net.build(trt_name);
    net.infer(inputs);
    net.debug(10);

    cout<<"out check"<<endl;
    cell<float,int>* out_data = new cell<float,int>(output_dims[0]);
    out_data->read_data(pytorch_output_name);
    cv::Mat out_mat(1,output_dims[0].size(),CV_32F,out_data->host_ptr());
    vector<cv::Mat> torch_outs = {out_mat};
    net.diff(torch_outs);

#ifdef WRITTEN
    vector<string> output_layers = {"data/ooo.txt"};
    vector<float*> out(output_dims.size());
    net.get_buffer(out);
    for(int i = 0;i<out.size();i++){
        int size = output_dims[i].size();
        grid_data->write_other_data(out[i],size,output_layers[i]);
    }
#endif
    return 0;
}
