#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cassert>

#define CHECK_NOT_CUDA(x) TORCH_CHECK(!x.type().is_cuda(), #x "must not be a CUDA tensor")
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_CPU(x) CHECK_NOT_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

struct float2
{
    float x, y;
};

struct float3
{
    float x, y, z;
};

class summed_area_table
{
protected:
    int width_, height_;
    // std::vector<float> sat_;
    const float* sat_;

    float I(int x, int y) const
    {
        if (x < 0 || y < 0) return 0;
        size_t i = y*width_ + x;
        return sat_[i];
    }

public:
    void create_lum(float* luminanceIntegralMap, int width, int height)
    {
        assert(nc > 2);

        width_ = width; height_ = height;

        sat_ = luminanceIntegralMap;
        /*
        sat_.clear();
        sat_.resize(width_ * height_);

        for (int y = 0; y < height_; ++y)
        for (int x = 0; x < width_;  ++x)
        {
            size_t i = y*width_ + x;

            T r = rgb[i*nc + 0];
            T g = rgb[i*nc + 1];
            T b = rgb[i*nc + 2];

            float ixy = luminance(r,g,b);

            sat_[i] = ixy + I(x-1, y) + I(x, y-1) - I(x-1, y-1);
        }
        */
    }

    int width() const  { return width_;  }
    int height() const { return height_; }

    /**
     * Returns the sum of a region defined by A,B,C,D.
     *
     * A----B
     * |    |  sum = C+A-B-D
     * D----C
     */
    int sum(int ax, int ay, int bx, int by, int cx, int cy, int dx, int dy) const
    {
        return I(cx, cy) + I(ax, ay) - I(bx, by) - I(dx, dy);
    }
};

struct sat_region
{
    int x_, y_, w_, h_;
    float sum_;
    const summed_area_table* sat_;

    void create(int x, int y, int w, int h, const summed_area_table* sat, float init_sum = -1)
    {
        x_ = x; y_ = y; w_ = w; h_ = h; sum_ = init_sum; sat_ = sat;

        if (sum_ < 0)
            sum_ = sat_->sum(x,       y,
                             x+(w-1), y,
                             x+(w-1), y+(h-1),
                             x,       y+(h-1));
    }

    void split_w_slow(sat_region& A) const
    {
        for (size_t w = 1; w <= w_; ++w)
        {
            A.create(x_, y_, w, h_, sat_);

            // if region left has approximately half the energy of the entire thing stahp
            if (A.sum_*2.f >= sum_)
                break;
        }
    }

    void split_w_fast(sat_region& A) const
    {
        int left = 0;
        int right = w_;
        int mid = (left + right) / 2;
        A.create(x_, y_, mid, h_, sat_);
        while (right - 1 > left) {
            if (A.sum_ * 2.f < sum_) {
                left = mid;
            } else {
                right = mid;
            }
            mid = (left + right + 1) / 2;  // the "+1" here makes it completely equivalent to the method split_w_slow
            A.create(x_, y_, mid, h_, sat_);
        }
    }

    /**
     * Split region horizontally into subregions A and B.
     */
    void split_w(sat_region& A, sat_region& B) const
    {
        split_w_fast(A);
        B.create(x_ + (A.w_-1), y_, w_ - A.w_, h_, sat_, sum_ - A.sum_);
    }

    void split_h_fast(sat_region& A) const 
    {
        int top = 0;
        int bottom = h_;
        int mid = (top + bottom) / 2;
        A.create(x_, y_, w_, mid, sat_);
        while (bottom - 1 > top) {
            if (A.sum_ * 2.f < sum_) {
                top = mid;
            } else {
                bottom = mid;
            }
            mid = (top + bottom + 1) / 2;  // the "+1" here makes it completely equivalent to the method split_h_slow
            A.create(x_, y_, w_, mid, sat_);
        }
    }

    void split_h_slow(sat_region& A) const
    {
        for (size_t h = 1; h <= h_; ++h)
        {
            A.create(x_, y_, w_, h, sat_);

            // if region top has approximately half the energy of the entire thing stahp
            if (A.sum_*2.f >= sum_)
                break;
        }
    }

    /**
     * Split region vertically into subregions A and B.
     */
    void split_h(sat_region& A, sat_region& B) const
    {
        split_h_fast(A);
        B.create(x_, y_ + (A.h_-1), w_, h_ - A.h_, sat_, sum_ - A.sum_);
    }

    float2 centroid() const
    {
        float2 c;

        sat_region A;

        split_w_fast(A);
        c.x = A.x_ + (A.w_-1);

        split_h_fast(A);
        c.y = A.y_ + (A.h_-1);

        return c;
    }
};

void split_recursive(const sat_region& r, size_t n, std::vector<sat_region>& regions)
{
    // check: can't split any further?
    if (r.w_ < 2 || r.h_ < 2 || n == 0)
    {
        regions.push_back(r);
        return;
    }

    sat_region A, B;

    if (r.w_ > r.h_)
        r.split_w(A, B);
    else
        r.split_h(A, B);

    split_recursive(A, n-1, regions);
    split_recursive(B, n-1, regions);
}

void median_cut(const summed_area_table& img, size_t n, std::vector<sat_region>& regions)
{
    regions.clear();

    // insert entire image as start region
    sat_region r;
    r.create(0, 0, img.width(), img.height(), &img);

    // recursively split into subregions
    split_recursive(r, n, regions);
}

std::vector<at::Tensor> doMedianCut(  // TODO: outputM
    at::Tensor outputM,  // scalar, denoting regions.size()
    at::Tensor outputRegion,  // (m, 4)  (XDXD)
    at::Tensor outputCentroid,  // (m, 2)
    at::Tensor outputAccumulatedLuminance,  // (m,)
    at::Tensor inputIntegralMap,  // (winHeight, winWidth)
    int n  // use 2^n cuts, and we define m to be 2^n.
) {
    CHECK_INPUT_CPU(outputM);
    CHECK_INPUT_CPU(outputRegion);
    CHECK_INPUT_CPU(outputCentroid);
    CHECK_INPUT_CPU(outputAccumulatedLuminance);
    CHECK_INPUT_CPU(inputIntegralMap);

    assert(inputIntegralMap.dim() == 2);
    const int H = inputIntegralMap.size(0);
    const int W = inputIntegralMap.size(1);

    assert ((outputM.dim() == 1) && (outputM.size(0) == 1));
    assert((outputRegion.dim() == 2) && (outputRegion.size(0) == m) && (outputRegion.size(1) == 4));
    assert((outputCentroid.dim() == 2) && (outputCentroid.size(0) == m) && (outputCentroid.size(1) == 2));
    assert((outputAccumulatedLuminance.dim() == 1) && (outputAccumulatedLuminance.size(0) == m));

    summed_area_table lum_sat;
    lum_sat.create_lum(inputIntegralMap.data_ptr<float>(), W, H);
    std::vector<sat_region> regions;
    median_cut(lum_sat, n, regions);

    int* outputM_ = outputM.data_ptr<int>();
    int* outputRegion_ = outputRegion.data_ptr<int>();
    float* outputCentroid_ = outputCentroid.data_ptr<float>();
    float* outputAccumulatedLuminance_ = outputAccumulatedLuminance.data_ptr<float>();
    float2 centroid;
    outputM_[0] = regions.size();
    for (int j = 0; j < regions.size(); j++) {
        outputRegion_[j * 4 + 0] = regions[j].x_;
        outputRegion_[j * 4 + 1] = regions[j].x_ + regions[j].w_;
        outputRegion_[j * 4 + 2] = regions[j].y_;
        outputRegion_[j * 4 + 3] = regions[j].y_ + regions[j].h_;
        centroid = regions[j].centroid();
        outputCentroid_[j * 2 + 0] = centroid.x;
        outputCentroid_[j * 2 + 1] = centroid.y;
        outputAccumulatedLuminance_[j] = regions[j].sum_;
    }

    return {
        outputM,
        outputRegion,
        outputCentroid,
        outputAccumulatedLuminance,
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("doMedianCut", &doMedianCut, "doMedianCut (CPU)");
}
