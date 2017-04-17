#include "vec_expi.hh"
#include <Eigen/Dense>
#include <iostream>

int main(int argc, const char* argv[])
{
    const unsigned maxN = 256;

    Eigen::VectorXf  in(maxN);
    Eigen::VectorXcf out0(maxN);
    Eigen::VectorXcf out1(maxN);
    Eigen::VectorXcf out2(maxN);
    Eigen::VectorXcf out3(maxN);

    bool have_sse2 = false;
    bool have_avx = false;
    bool have_avx2 = false;
#ifdef __SSE2__
    have_sse2 = true;
#endif
#ifdef __AVX__
    have_avx = true;
#endif
#ifdef __AVX2__
    have_avx2 = true;
#endif

 std::cout << "sse2: " << (have_sse2 ? "" : "un") << "supported" << std::endl;
 std::cout << "avx: " << (have_avx ? "" : "un") << "supported" << std::endl;
 std::cout << "avx2: " << (have_avx2 ? "" : "un") << "supported" << std::endl;

    unsigned err_count = 0;

    for (unsigned size=1; size<=maxN; ++size) {
        in.head(size) = Eigen::ArrayXf::Random(size) * M_PI;
        vec_expi_libm(in.data(), out0.data(), size);
#ifdef __SSE2__
        vec_expi_sse2(in.data(), out1.data(), size);
        float err = (out0-out1).norm() / out0.norm();
        if (err>1e-7) {
            err_count++;
            std::cout << "sse2 " << size << " err: " << err << std::endl;
        }
#endif
#ifdef __AVX__
        vec_expi_avx(in.data(), out2.data(), size);
        err = (out0-out2).norm() / out0.norm();
        if (err>1e-7) {
            err_count++;
            std::cout << "avx " << size << " err: " << err << std::endl;
        }
#endif
        vec_expi(in.data(), out3.data(), size);
        err = (out0-out3).norm() / out0.norm();
        if (err>1e-7) {
            err_count++;
            std::cout << "vec_expi " << size << " err: " << err << std::endl;
        }
    }
    return err_count;
}
