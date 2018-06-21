#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

/* ==============
 * Forward op
 * =========== */
REGISTER_OP("SepConv")
    .Input("input: float")    // A [batch, height, width, channels] tensor
    .Input("kh: float")       // The horizontal kernel [height, width, kchannels] tensor
    .Input("kv: float")       // The vertical kernel [height, width, kchannels] tensor
    .Output("output: float")  // The resulting [batch, height, width, channels] tensor
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
{    
    // Ensure the input rank is 4
    shape_inference::ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));

    // Ensure the input is an RGB image
    shape_inference::DimensionHandle channels = c->Dim(input_shape, 3);
    TF_RETURN_IF_ERROR(c->WithValue(channels, 3, &channels));

    // Ensure the kernels rank is 3
    shape_inference::ShapeHandle kh_shape;
    shape_inference::ShapeHandle kv_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &kh_shape));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &kv_shape));

    // Ensure both kernels have the same shape, it should be [None, None, kchannels]    
    TF_RETURN_IF_ERROR(c->Merge(kh_shape, kv_shape, &kv_shape))

    // The output has the same size as the input
    c->set_output(0, c->input(0));

    return Status::OK();
});

// GPU launcher
void SepConvKernelLauncher(
    const float* inputs, 
    const float* kh,
    const float* kv,
    const int n, 
    const int h, 
    const int w,
    const int kchannels,
    float* output);

// GPU op
class SepConvOpGPU : public OpKernel
{
public:    
    explicit SepConvOpGPU(OpKernelConstruction* context) : OpKernel(context) { }

    void Compute(OpKernelContext* context) override 
    {
        // Check the number of tensors
        DCHECK_EQ(3, context->num_inputs());

        // Get the input tensor and the two kernels
        const Tensor& input = context->input(0);
        const Tensor& kh = context->input(1);
        const Tensor& kv = context->input(2);

        // Check the shapes of the input and kernel tensors
        const TensorShape& input_shape = input.shape();
        const TensorShape& kh_shape = kh.shape();
        const TensorShape& kv_shape = kv.shape();

        // Get the batch parameters
        const int n = input_shape.dim_size(0);
        const int h = input_shape.dim_size(1);
        const int w = input_shape.dim_size(2);
        const int kchannels = kh_shape.dim_size(2);

        // Ensure the batch number matches
        DCHECK_EQ(n, kh_shape.dim_size(0));
        DCHECK_EQ(n, kv_shape.dim_size(0));

        // Ensure the two kernels have the same depth
        DCHECK_EQ(kv_shape.dim_size(2), kchannels);

        // Ensure the depth of the kernels is an odd number
        DCHECK_EQ(kchannels % 2, 1)

        // Ensure all the input tensors have the same 2D resolution
        DCHECK_EQ(h, kh_shape.dim_size(0));
        DCHECK_EQ(h, kv_shape.dim_size(0));
        DCHECK_EQ(w, kh_shape.dim_size(1));
        DCHECK_EQ(w, kv_shape.dim_size(1));

        // Create the output tensor info
        TensorShape output_shape;
        output_shape.AddDim(batch_samples);
        output_shape.AddDim(h);
        output_shape.AddDim(w);
        output_shape.AddDim(3);
                
        // Create output tensor
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

        // Launch the GPU kernel
        auto f_input = input.flat<float>();
        auto f_kh = kh.flat<float>();
        auto f_kv = kv.flat<float>();
        auto f_output = output->template flat<float>();

        SepConvKernelLauncher(
            f_input.data(), 
            f_kh.data(),
            f_kv.data(),
            n, 
            h, 
            w,
            kchannels,
            f_output.data()
        );
    }
};

REGISTER_KERNEL_BUILDER(Name("SepConv").Device(DEVICE_GPU), SepConvOpGPU);

/* ==============
 * Backwards op
 * =========== */
REGISTER_OP("SepConvGrad")
    .Input("grad: float")
    .Input("kh: float")
    .Input("kv: float")
    .Input("source: float")
    .Output("backprop_kh: float")
    .Output("backprop_kv: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
{    
    // Ensure the input and source rank is 4
    shape_inference::ShapeHandle grad_shape;
    shape_inference::ShapeHandle source_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &grad_shape));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 4, &source_shape));

    // Ensure the source is an RGB image
    shape_inference::DimensionHandle channels = c->Dim(source_shape, 3);
    TF_RETURN_IF_ERROR(c->WithValue(channels, 3, &channels));

    // Ensure the kernels rank is 3
    shape_inference::ShapeHandle kh_shape;
    shape_inference::ShapeHandle kv_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &kh_shape));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &kv_shape));

    // Ensure both kernels have the same shape, it should be [None, None, kchannels]    
    TF_RETURN_IF_ERROR(c->Merge(kh_shape, kv_shape, &kv_shape))

    // The outputs have the same size as the kernels
    c->set_output(0, kh_shape);
    c->set_output(1, kv_shape);

    return Status::OK();
});

// GPU gradient launcher
void SepConvGradKernelLauncher(
    const float* grad, 
    const float* kh,
    const float* kv,
    const float* source,
    const int n, 
    const int h, 
    const int w,
    const int kchannels,
    float* backprop_kh,
    float* backprop_kv);
  
class SepConvGradOpGPU : public OpKernel
{
public:
    explicit SepConvGradOpGPU(OpKernelConstruction* context) : OpKernel(context) { }
  
    void Compute(OpKernelContext* context) override 
    {
        // Check the number of tensors
        DCHECK_EQ(4, context->num_inputs());

        // Get the op tensors
        const Tensor& grad = context->input(0);
        const Tensor& kh = context->input(1);
        const Tensor& kv = context->input(2);
        const Tensor& source = context->input(3);

        // Check the shapes of the input and kernel tensors
        const TensorShape& grad_shape = grad.shape();
        const TensorShape& kh_shape = kh.shape();
        const TensorShape& kv_shape = kv.shape();
        const TensorShape& source_shape = source.shape();

        // Get the batch parameters
        const int n = grad_shape.dim_size(0);
        const int h = grad_shape.dim_size(1);
        const int w = grad_shape.dim_size(2);
        const int channels = grad_shape.dim_size(3);
        const int kchannels = kh_shape.dim_size(2);

        // Ensure the batch number matches
        DCHECK_EQ(n, kh_shape.dim_size(0));
        DCHECK_EQ(n, kv_shape.dim_size(0));

        // Ensure the grad is an RGB image
        DCHECK_EQ(channels, 3);

        // Ensure the two kernels have the same depth
        DCHECK_EQ(kv_shape.dim_size(2), kchannels);

        // Ensure the depth of the kernels is an odd number
        DCHECK_EQ(kchannels % 2, 1)

        // Ensure all the input tensors have the same 2D resolution
        DCHECK_EQ(h, kh_shape.dim_size(0));
        DCHECK_EQ(h, kv_shape.dim_size(0));
        DCHECK_EQ(w, kh_shape.dim_size(1));
        DCHECK_EQ(w, kv_shape.dim_size(1));

        // Create output tensors
        Tensor* backprop_kh = NULL;
        Tensor* backprop_kv = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &kh_shape));
        OP_REQUIRES_OK(context, context->allocate_output(1, weights_shape, &kv_shape));

        // Launch the GPU kernel
        auto f_grad = grad.flat<float>();
        auto f_kh = kh.flat<float>();
        auto f_kv = kv.flat<float>();
        auto f_source = source.flat<float>();
        auto f_backprop_kh = backprop_kh->template flat<float>();
        auto f_backprop_kv = backprop_kv->template flat<float>();

        SepConvGradKernelLauncher(f_grad.data(), f_kh.data(), f_kv.data(), n, h, w, kchannels, f_backprop_kh.data(), f_backprop_kv.data());
    }
};

REGISTER_KERNEL_BUILDER(Name("SepConvGrad").Device(DEVICE_GPU), SepConvGradOpGPU);
