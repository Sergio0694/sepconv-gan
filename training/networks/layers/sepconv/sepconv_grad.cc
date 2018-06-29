#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("SepconvGrad")
    .Input("grad: float")       // The output error
    .Input("input: float")      // A [batch, height, width, channels] tensor
    .Attr("kchannels: int")     // Depth of each separable kernel
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

    // The set the shape of the kernel gradients
    int kchannels_;
    c->GetAttr("kchannels", &kchannels_);
    shape_inference::DimensionHandle kchannels = c->MakeDim(kchannels_);
    shape_inference::ShapeHandle k_grad_shape;
    TF_RETURN_IF_ERROR(c->ReplaceDim(input_shape, 3, kchannels, &k_grad_shape));
    c->set_output(0, k_grad_shape);
    c->set_output(1, k_grad_shape);

    return Status::OK();
});

// GPU launcher
void SepconvGradKernelLauncher(
    const float* grad,
    const float* input,
    const int n, 
    const int h, 
    const int w,
    const int kchannels,
    float* kv_grad,
    float* kh_grad);

// GPU op
class SepconvGradOpGPU : public OpKernel
{
private:
    int kchannels_;

public:    
    explicit SepconvGradOpGPU(OpKernelConstruction* context) : OpKernel(context) 
    {
        OP_REQUIRES_OK(context, context->GetAttr("kchannels", &kchannels_));
        OP_REQUIRES(context, kchannels_ >= 3, errors::InvalidArgument("kchannels must be >= 3, got ", kchannels_));
        OP_REQUIRES(context, kchannels_ % 2 == 1, errors::InvalidArgument("kchannels must be an odd number, was ", kchannels_));
    }

    void Compute(OpKernelContext* context) override 
    {
        // Check the number of tensors
        DCHECK_EQ(2, context->num_inputs());

        // Get the input tensors
        const Tensor& grad = context->input(0);
        const Tensor& input = context->input(1);

        // Check the shapes of the input and kernel tensors
        const TensorShape& grad_shape = grad.shape();
        const TensorShape& input_shape = input.shape();

        // Get the batch parameters
        const int n = grad_shape.dim_size(0);
        const int h = grad_shape.dim_size(1);
        const int w = grad_shape.dim_size(2);

        // Ensure the grad and input tensors match
        DCHECK_EQ(n, input_shape.dim_size(0));
        DCHECK_EQ(h, input_shape.dim_size(1));
        DCHECK_EQ(w, input_shape.dim_size(2));
        DCHECK_EQ(3, input_shape.dim_size(3));
        DCHECK_EQ(3, grad_shape.dim_size(3));

        // Create the output tensor sinfo	
        TensorShape grads_shape;
        grads_shape.AddDim(n);
        grads_shape.AddDim(h);	
        grads_shape.AddDim(w);	
        grads_shape.AddDim(kchannels_);
                
        // Create output tensors
        Tensor* kv_grad = NULL;
        Tensor* kh_grad = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, grads_shape, &kv_grad));
        OP_REQUIRES_OK(context, context->allocate_output(1, grads_shape, &kh_grad));

        // Launch the GPU kernel
        auto f_grad = grad.flat<float>();
        auto f_input = input.flat<float>();
        auto f_kv_grad = kv_grad->template flat<float>();
        auto f_kh_grad = kh_grad->template flat<float>();

        SepconvGradKernelLauncher(
            f_grad.data(),
            f_input.data(),
            n, 
            h, 
            w,
            kchannels_,
            f_kv_grad.data(),
            f_kh_grad.data()
        );
    }
};

REGISTER_KERNEL_BUILDER(Name("SepconvGrad").Device(DEVICE_GPU), SepconvGradOpGPU);
