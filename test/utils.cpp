#include "utils.h"
#include <opencv2/opencv.hpp>
template<typename T1,typename T2>
void cell<T1,T2>::init(){
    size_ = n_*c_*h_*w_;
    spatial_dim_ = h_*w_;
    host_ptr_ = (T1*) malloc(sizeof(T1)*size_);
    memset(host_ptr_,0,sizeof(T1)*size_);
    cudaMalloc(&device_ptr_, sizeof(T1)*size_);
    cudaMemset(device_ptr_, 0x0, sizeof(T1)*size_);
}


template<typename T1,typename T2>
void cell<T1,T2>::init_data(init_type type){
    switch(type){
        case ZERO:
        {
            cout<<"init data type zero"<<endl;
            //memset(host_ptr_, 0, size_*sizeof(T1));
            break;
        }

        case COMMON:
        {
            cout<<"init data type: common" <<endl;
            for(int n = 0;n<n_;n++){
                for(int c = 0;c<c_;c++){
                    for(int h = 0;h<h_;h++){
                        for(int w = 0;w<w_;w++){
                            host_ptr_[n*c_*spatial_dim_ + c * spatial_dim_ + h*w_ + w] = (h+1) * w;
                        }
                    }
                }
            }
            break;
        }
        case GRID:
        {
            cout<<"init data type: grid"<<endl;
            for(int n = 0;n<n_;n++){
                for(int i = 0;i<h_;i++){
                    for(int j = 0;j<w_;j++){
                        host_ptr_[n *h_*w_*2 + i*w_*2 + j*2] = -1 + j*2.0/(w_-1);
                        host_ptr_[n *h_*w_*2 + i*w_*2 + j*2 + 1] = -1 + i*2.0/(h_-1);
                    }
                }
            }
            break;
        }
        default:
            cout<<"fail "<<endl;
    } 
}

template<typename T1,typename T2>
void cell<T1,T2>::check_part_data(int size){
    for(int i = 0;i<size;i++){
        cout<<host_ptr_[i]<<" ";
    }
    cout<<endl;
}

template<typename T1,typename T2>
void cell<T1,T2>::check_whole_data(init_type type){
    switch(type){
        case ZERO:
        {
            cout<<"data filled with zero "<<endl;
            break;
        }
        case COMMON:
        {
            cout<<"COMMON: "<<COMMON<<endl;
            for(int n = 0;n<n_;n++){
                for(int c = 0;c<c_;c++){
                    for(int h = 0;h<h_;h++){
                        for(int w = 0;w<w_;w++){
                            cout<< host_ptr_[n*c_*spatial_dim_ + c * spatial_dim_ + h*w_ + w]<<" ";
                        }
                        cout<<endl;
                    }
                    cout<<endl<<endl;
                }
            }
            break;
        }
        case GRID:
        {
            cout<<"GRID: "<<GRID<<endl;
            for(int n = 0;n<n_;n++){
                for(int i = 0;i<h_;i++){
                    for(int j = 0;j<w_;j++){
                        cout<<host_ptr_[n*spatial_dim_*2 + i*w_*2 + j*2] <<" "<<host_ptr_[n*spatial_dim_*2 + i*w_*2 + j*2 + 1]<<" ";
                    }
                    cout<<endl;
                }
                cout<<endl<<endl;;
            }
            break;
        }
        default:
            cout<<"fail "<<endl;
    }
}


template<typename T1,typename T2>
void cell<T1,T2>::read_data(string filename){
    char * buffer;
    long size;
    ifstream file (filename,   ios::in|ios::binary|ios::ate);
    size = file.tellg();
    file.seekg (0, ios::beg);
    buffer = new char [size];
    file.read (buffer, size);
    file.close();
    float * data = (float*) buffer;
    memcpy(host_ptr_,data,sizeof(float) * size_);
}

template<typename T1,typename T2>
void cell<T1,T2>::write_data(int size,string filename){
    ofstream dout(filename);
    for(int i=0;i<size;i++)
		dout << host_ptr_[i]<< endl;
	dout.close();
	cout << "Finish write "<<filename << endl;
}

template<typename T1,typename T2>
void cell<T1,T2>::write_other_data(float*ptr,int size,string filename){
    ofstream dout(filename);
    for(int i=0;i<size;i++)
		dout << ptr[i]<< endl;
	dout.close();
	cout << "Finish write "<<filename << endl;
}



template<typename T1,typename T2>
void cell<T1,T2>::sync_H2D(){
    cudaMemcpy(device_ptr_, host_ptr_, size_*sizeof(T1), cudaMemcpyHostToDevice);
}
template<typename T1,typename T2>
void cell<T1,T2>::sync_D2H(){
    cudaMemcpy(host_ptr_, device_ptr_, size_*sizeof(T1), cudaMemcpyDeviceToHost);
}

template<typename T1,typename T2>
T1* cell<T1,T2>::host_ptr(){
    return host_ptr_;
}
template<typename T1,typename T2>
T1* cell<T1,T2>::device_ptr(){
    return (T1*)device_ptr_;
} 

INSTANTIATE_CLASS(cell);