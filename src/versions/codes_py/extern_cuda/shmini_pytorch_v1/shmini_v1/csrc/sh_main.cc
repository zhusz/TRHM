#include <torch/torch.h>

#include <vector>

#include "default_image.h"
#include "spherical_harmonics.h"
#include "default_image.h"

#define CHECK_NOT_CUDA(x) TORCH_CHECK(!x.type().is_cuda(), #x "must not be a CUDA tensor")
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_CPU(x) CHECK_NOT_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor projectEnvironment(int L, at::Tensor to_be_fit_input, at::Tensor result_output) {
    // to_be_fit: (H, W, 3(rgb))
    // result: ((L + 1) * (L + 1), 3(rgb))
    CHECK_INPUT_CPU(to_be_fit_input);
    CHECK_INPUT_CPU(result_output);

    assert((to_be_fit_input.dim() == 3) && (to_be_fit_input.size(2) == 3));
    const int H = to_be_fit_input.size(0);
    const int W = to_be_fit_input.size(1);
    const int N = (L + 1) * (L + 1);
    assert((result_output.dim() == 2) && (result_output.size(0) == N) && (result_output.size(1) == 3));

    sh::DefaultImage to_be_fit = sh::DefaultImage(W, H);
    const float *to_be_fit_data = to_be_fit_input.data_ptr<float>();
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            to_be_fit.SetPixel(
                w,
                h,
                Eigen::Array3f(
                    to_be_fit_data[h * W * 3 + w * 3 + 0],
                    to_be_fit_data[h * W * 3 + w * 3 + 1],
                    to_be_fit_data[h * W * 3 + w * 3 + 2]
                )
            );
        }
    }
    std::unique_ptr<std::vector<Eigen::Array3f> > result = sh::ProjectEnvironment(L, to_be_fit);
    float *result_data = result_output.data_ptr<float>();
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < 3; c++) {
            result_data[n * 3 + c] = (*result)[n][c];
        }
    }

    return result_output;
}

at::Tensor applyRotation(int L, at::Tensor q_input, at::Tensor sh_input, at::Tensor sh_output) {
    // input
    // L: int (so N == (L + 1) * (L + 1))
    // q_input: (4, )
    // sh_input: (m, N, 3)

    // output
    // sh_output: (m, N, 3)

    assert((q_input.dim() == 1) && (q_input.size(0) == 4));
    const int N = (L + 1) * (L + 1);
    assert((sh_input.dim() == 3) && (sh_input.size(1) == N) && (sh_input.size(2) == 3));
    const int m = sh_input.size(0);
    assert((sh_output.dim() == 3) && (sh_output.size(0) == m) 
        && (sh_output.size(1) == N) && (sh_output.size(2) == 3));
    CHECK_INPUT_CPU(q_input);
    CHECK_INPUT_CPU(sh_input);
    CHECK_INPUT_CPU(sh_output);

    const float *q_ptr = q_input.data_ptr<float>();
    Eigen::Quaterniond q(q_ptr[0], q_ptr[1], q_ptr[2], q_ptr[3]);
    std::unique_ptr<sh::Rotation> r = sh::Rotation::Create(L, q);

    const float *sh_input_ptr = sh_input.data_ptr<float>();
    float *sh_output_ptr = sh_output.data_ptr<float>();
    std::vector<Eigen::Array3f> sh(N);
    std::vector<Eigen::Array3f> result;
    for (int j = 0; j < m; j++) {
        for (int n = 0; n < N; n++) {
            sh[n] = Eigen::Array3f(
                sh_input_ptr[j * N * 3 + n * 3 + 0],
                sh_input_ptr[j * N * 3 + n * 3 + 1],
                sh_input_ptr[j * N * 3 + n * 3 + 2]
            );
        }
        r->Apply(sh, &result);
        for (int n = 0; n < N; n++) {
            sh_output_ptr[j * N * 3 + n * 3 + 0] = result[n][0];
            sh_output_ptr[j * N * 3 + n * 3 + 1] = result[n][1];
            sh_output_ptr[j * N * 3 + n * 3 + 2] = result[n][2];
        }
    }

    return sh_output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("projectEnvironment", &projectEnvironment, "projectEnvironment (CPU)");
    m.def("applyRotation", &applyRotation, "applyRotation (CPU)");
}
