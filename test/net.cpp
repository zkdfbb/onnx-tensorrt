#include "net.h"

BaseNet::BaseNet()
{
}

BaseNet::~BaseNet()
{
    cudaStreamDestroy(stream_);
    for (size_t i = 0; i < input_names_.size() + output_names_.size(); ++i)
    {
        CHECK(cudaFree(buffers_[i]));
    }
}

int BaseNet::init(vector<string> input_names, vector<Dim> input_dims,
                  vector<string> output_names, vector<Dim> output_dims)
{
    batch_size_ = 1;
    assert(input_names.size() == input_dims.size());
    assert(output_names.size() == output_dims.size());

    input_names_.resize(input_names.size());
    copy(input_names.begin(), input_names.end(), input_names_.begin());
    input_dims_.resize(input_dims.size());
    copy(input_dims.begin(), input_dims.end(), input_dims_.begin());
    output_names_.resize(output_names.size());
    copy(output_names.begin(), output_names.end(), output_names_.begin());
    output_dims_.resize(output_dims.size());
    copy(output_dims.begin(), output_dims.end(), output_dims_.begin());

    int size = input_names.size() + output_names.size();
    buffers_ = new void *[size];
    for (auto &input_dim : input_dims)
    {
        input_sizes_.push_back(input_dim.size());
    }
    for (auto &output_dim : output_dims)
    {
        output_sizes_.push_back(output_dim.size());
    }
}

int BaseNet::build(const string &trt_name)
{
    vector<char> trt_model;
    size_t size{0};
    ifstream file(trt_name, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trt_model.resize(size);
        file.read(trt_model.data(), size);
        file.close();
    }
    else
    {
        cout << "please check your rt file " << trt_name << " is valid" << endl;
        return 1;
    }

    runtime_ = nvinfer1::createInferRuntime(gLogger);
    nvonnxparser::IPluginFactory* pluginFactory = nvonnxparser::createPluginFactory(gLogger);
    engine_ = runtime_->deserializeCudaEngine(trt_model.data(), size, pluginFactory);
    context_ = engine_->createExecutionContext();
    cout << trt_name << " has been successfully loaded." << endl;

    for (int i = 0; i < input_names_.size(); ++i)
    {
        int index = engine_->getBindingIndex(input_names_[i].c_str());
        cout << "input " << input_names_[i] << " index " << index << endl;
        input_indices_.push_back(index);
        CHECK(cudaMalloc(&buffers_[index], input_sizes_[i] * sizeof(float)));
    }
    for (int i = 0; i < output_names_.size(); ++i)
    {
        int index = engine_->getBindingIndex(output_names_[i].c_str());
        cout << "output " << output_names_[i] << " index " << index << endl;
        output_indices_.push_back(index);
        CHECK(cudaMalloc(&buffers_[index], output_sizes_[i] * sizeof(float)));
    }
    CHECK(cudaStreamCreate(&stream_));
    return 0;
}

void BaseNet::infer(vector<float *> &inputs)
{
    cout << "begin to infer !!!" << endl;

    for (int i = 0; i < input_sizes_.size(); ++i)
    {
        cout << input_names_[i] << ": ";
        for (int j = 0; j < 10; ++j)
        {
            cout << inputs[i][j] << " ";
        }
        cout << endl;
    }

    assert(inputs.size() == input_sizes_.size());
    for (int i = 0; i < input_sizes_.size(); ++i)
    {
        int index = input_indices_[i];
        int size = input_sizes_[i] * sizeof(float);
        CHECK(cudaMemcpyAsync(buffers_[index], inputs[i], size, cudaMemcpyHostToDevice, stream_));
    }
    context_->enqueue(batch_size_, buffers_, stream_, nullptr);
    cudaStreamSynchronize(stream_);
}

void BaseNet::get_buffer(vector<float *> &outputs)
{
    cout << "get output buffer" << endl;
    for (int i = 0; i < output_sizes_.size(); ++i)
    {
        int index = output_indices_[i];
        int size = output_sizes_[i] * sizeof(float);
        outputs[i] = new float[size];
        CHECK(cudaMemcpyAsync(outputs[i], buffers_[index], size, cudaMemcpyDeviceToHost, stream_));
    }
    cudaStreamSynchronize(stream_);
}

void BaseNet::debug(int size)
{
    vector<float *> outputs(output_sizes_.size());
    this->get_buffer(outputs);
    for (int i = 0; i < output_sizes_.size(); ++i)
    {
        cout << output_names_[i] << ": ";
        for (int j = 0; j < size; ++j)
        {
            cout << outputs[i][j] << " ";
        }
        cout << endl;
    }
}
