#include "../src/grad-SGS.hpp"

int main(){
    std::vector<double> x0 = {1., 2., 3.};
    std::vector<double> v = {1., 2., 4., 4., 6.};
    std::vector<std::size_t> c = {0, 1, 1, 1, 2};
    std::vector<std::size_t> r = {0, 2, 3, 5};
    CSR_Matrix a{v, c, r};
    std::vector<double> b = {1., 1., 1.};
    std::vector<double> res = sym_gauss_seidel(a, x0, b, 1000, 0.001);
    //std::vector<double> res = jacobi(a, x0, b, 0.25, 100, 0.001);
    //std::vector<double> res = grad_desc(a, x0, b, 100, 0.00001);
    std::cout<<res[0]<<" "<<res[1]<<" "<<res[2];
}