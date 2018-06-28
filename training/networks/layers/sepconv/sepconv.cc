#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Sepconv")
    .Input("input: float")      // A [batch, height, width, 3] tensor
    .Input("kv: float")         // The vertical kernel [height, width, kchannels] tensor
    .Input("kh: float")         // The horizontal kernel [height, width, kchannels] tensor
    .Output("output: float")    // The resulting [batch, height, width, 3] tensor
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
{    
    // Ensure the input rank is 4
    shape_inference::ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));

    // Ensure the input is an RGB image
    shape_inference::DimensionHandle channels = c->Dim(input_shape, 3);
    TF_RETURN_IF_ERROR(c->WithValue(channels, 3, &channels));

    // Ensure the kernels rank is 4
    shape_inference::ShapeHandle kv_shape;
    shape_inference::ShapeHandle kh_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &kv_shape));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 4, &kh_shape));

    // Ensure both kernels have the same shape, it should be [None, None, kchannels]    
    TF_RETURN_IF_ERROR(c->Merge(kh_shape, kv_shape, &kv_shape));

    // The output has the same size as the input
    c->set_output(0, c->input(0));

    return Status::OK();
});

// GPU launcher
void SepconvKernelLauncher(
    const float* inputs, 
    const float* kv,
    const float* kh,
    const int n, 
    const int h, 
    const int w,
    const int kchannels,
    float* output);

// GPU op
class SepconvOpGPU : public OpKernel
{
public:    
    explicit SepconvOpGPU(OpKernelConstruction* context) : OpKernel(context) { }

    void Compute(OpKernelContext* context) override 
    {
        // Check the number of tensors
        DCHECK_EQ(3, context->num_inputs());

        // Get the input tensor and the two kernels
        const Tensor& input = context->input(0);
        const Tensor& kv = context->input(1);
        const Tensor& kh = context->input(2);

        // Check the shapes of the input and kernel tensors
        const TensorShape& input_shape = input.shape();
        const TensorShape& kv_shape = kv.shape();
        const TensorShape& kh_shape = kh.shape();

        // Get the batch parameters
        const int n = input_shape.dim_size(0);
        const int h = input_shape.dim_size(1);
        const int w = input_shape.dim_size(2);
        const int kchannels = kv_shape.dim_size(3);

        // Ensure the batch number matches
        DCHECK_EQ(n, kv_shape.dim_size(0));
        DCHECK_EQ(n, kh_shape.dim_size(0));

        // Ensure the two kernels have the same depth
        DCHECK_EQ(kh_shape.dim_size(3), kchannels);

        // Ensure the depth of the kernels is an odd number
        DCHECK_EQ(kchannels % 2, 1);

        // Ensure all the input tensors have the same 2D resolution
        DCHECK_EQ(h, kv_shape.dim_size(1));
        DCHECK_EQ(h, kh_shape.dim_size(1));
        DCHECK_EQ(w, kv_shape.dim_size(2));        
        DCHECK_EQ(w, kh_shape.dim_size(2));
                
        // Create output tensor
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output));

        // Launch the GPU kernel
        auto f_input = input.flat<float>();
        auto f_kv = kv.flat<float>();
        auto f_kh = kh.flat<float>();
        auto f_output = output->template flat<float>();

        SepconvKernelLauncher(
            f_input.data(),
            f_kv.data(),
            f_kh.data(),
            n, 
            h, 
            w,
            kchannels,
            f_output.data()
        );
    }
};

REGISTER_KERNEL_BUILDER(Name("Sepconv").Device(DEVICE_GPU), SepconvOpGPU);
