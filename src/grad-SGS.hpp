#include "../../Dense-CSR/csr_matrix.h"
#include "../../Dense-CSR/vector_operations.h"
#include <cmath>

std::vector<double> grad_desc(const CSR_Matrix &A, std::vector<double> &x0, const std::vector<double> &b, const size_t &N, const double &tolerance){
    std::vector<double> x = x0;
    std::vector<double> r = std::vector<double>(b.size());
    double diff = 0.;
    double t;
    for(size_t i = 0; i < N; i++){
        //std::cout<<diff<<std::endl;
        r = (A*x - b);
        t = (r * r) / (r * (A * r));
        x = x - t*r;
        diff = 0;
        for(size_t j = 0; j < b.size(); j++){
            diff += (t * r[j]) * (t * r[j]);
        }
        if(std::sqrt(diff) < tolerance){
            //std::cout<<i<<" ";
            break;
        }
    }
    return x;
}

std::vector<double> sym_gauss_seidel(const CSR_Matrix &A, std::vector<double> &x0, const std::vector<double> &b, const size_t &N, const double &tolerance){
    std::vector<double> x = x0;
    std::vector<double> last = x;
    std::vector<double> tmp = std::vector<double>(b.size());
    double diff = 0.;
    for(size_t i = 0; i < N; i++){
        for(size_t m = 0; m < b.size(); m++){
            tmp[m] = 0;
            for(size_t k = 0; k < b.size(); k++){
                if (m != k){
                    tmp[m] += A(m, k) * x[k];
                }
            }
            x[m] = (b[m] - tmp[m]) / A(m, m);
        }
        for(int m = b.size() - 1; m >= 0; m--){
            tmp[m] = 0;
            for(size_t k = 0; k < b.size(); k++){
                if (m != k){
                    tmp[m] += A(m, k) * x[k];
                }
            }
            x[m] = (b[m] - tmp[m]) / A(m, m);
        }
        diff = 0;
        for(size_t j = 0; j < b.size(); j++){
            diff += (x[j] - last[j]) * (x[j] - last[j]);
        }
        if(std::sqrt(diff) < tolerance){
            break;
        }
    }
    return x;
}