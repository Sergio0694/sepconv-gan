#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("SepconvGrad")
    .Input("grad: float")       // The output error
    .Input("input: float")      // A [batch, height, width, channels] tensor
    .Input("kv: float")         // The vertical kernel [height, width, kchannels] tensor
    .Input("kh: float")         // The horizontal kernel [height, width, kchannels] tensor
    .Output("kv_grad: float")   // The gradient with respect to kv
    .Output("kh_grad: float")   // The gradient with respect to kh
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
{    
    // Ensure the grad rank is 4
    shape_inference::ShapeHandle grad_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &grad_shape));

    // Ensure the grad shape matches the input
    shape_inference::ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &input_shape));
    TF_RETURN_IF_ERROR(c->Merge(grad_shape, input_shape, &input_shape));

    // Ensure the input is an RGB image (and the output grad too)
    shape_inference::DimensionHandle channels = c->Dim(input_shape, 3);
    TF_RETURN_IF_ERROR(c->WithValue(channels, 3, &channels));

    // Ensure the kernels rank is 4
    shape_inference::ShapeHandle kv_shape;
    shape_inference::ShapeHandle kh_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 4, &kv_shape));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 4, &kh_shape));

    // Set the shape of the kernel gradients
    c->set_output(0, kv_shape);
    c->set_output(1, kh_shape);

    return Status::OK();
});

// GPU launcher
void SepconvGradKernelLauncher(
    const float* grad,
    const float* input,
    const float* kv,
    const float* kh,
    const int n, 
    const int h, 
    const int w,
    const int kchannels,
    float* kv_grad,
    float* kh_grad);

// GPU op
class SepconvGradOpGPU : public OpKernel
{
public:    
    explicit SepconvGradOpGPU(OpKernelConstruction* context) : OpKernel(context) { }

    void Compute(OpKernelContext* context) override 
    {
        // Check the number of tensors
        DCHECK_EQ(4, context->num_inputs());

        // Get the input tensors
        const Tensor& grad = context->input(0);
        const Tensor& input = context->input(1);
        const Tensor& kv = context->input(2);
        const Tensor& kh = context->input(3);

        // Check the shapes of the input and kernel tensors
        const TensorShape& grad_shape = grad.shape();
        const TensorShape& input_shape = input.shape();
        const TensorShape& kv_shape = kv.shape();
        const TensorShape& kh_shape = kh.shape();

        // Get the batch parameters
        const int n = grad_shape.dim_size(0);
        const int h = grad_shape.dim_size(1);
        const int w = grad_shape.dim_size(2);
        const int kchannels = kv_shape.dim_size(3);

        // Same batch size for all inputs
        DCHECK_EQ(n, input_shape.dim_size(0));
        DCHECK_EQ(n, kv_shape.dim_size(0));
        DCHECK_EQ(n, kh_shape.dim_size(0));

        // Same 2D resolution for all inputs
        DCHECK_EQ(h, input_shape.dim_size(1));
        DCHECK_EQ(h, kv_shape.dim_size(1));
        DCHECK_EQ(h, kh_shape.dim_size(1));
        DCHECK_EQ(w, input_shape.dim_size(2));
        DCHECK_EQ(w, kv_shape.dim_size(2));
        DCHECK_EQ(w, kh_shape.dim_size(2));
        
        // Same depth for both kernels
        DCHECK_EQ(kchannels, kh_shape.dim_size(3));
                
        // Create output tensors
        Tensor* kv_grad = NULL;
        Tensor* kh_grad = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, kv_shape, &kv_grad));
        OP_REQUIRES_OK(context, context->allocate_output(1, kh_shape, &kh_grad));

        // Launch the GPU kernel
        auto f_grad = grad.flat<float>();
        auto f_input = input.flat<float>();
        auto f_kv = kv.flat<float>();
        auto f_kh = kh.flat<float>();
        auto f_kv_grad = kv_grad->template flat<float>();
        auto f_kh_grad = kh_grad->template flat<float>();

        SepconvGradKernelLauncher(
            f_grad.data(),
            f_input.data(),
            f_kv.data(),
            f_kh.data(),
            n, 
            h, 
            w,
            kchannels,
            f_kv_grad.data(),
            f_kh_grad.data()
        );
    }
};

REGISTER_KERNEL_BUILDER(Name("SepconvGrad").Device(DEVICE_GPU), SepconvGradOpGPU);
