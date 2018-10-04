#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("NearestShader")
    .Input("source: float")      // A [batch, height, width, 3] tensor
    .Input("frame_0: float")
    .Input("frame_1: float")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
{    
    // Ensure the input rank is 4
    shape_inference::ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));

    // Ensure the input is an RGB image
    shape_inference::DimensionHandle channels = c->Dim(input_shape, 3);
    TF_RETURN_IF_ERROR(c->WithValue(channels, 3, &channels));

    // Ensure the input frames have the same shape
    shape_inference::ShapeHandle frame_0_shape;
    shape_inference::ShapeHandle frame_1_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &frame_0_shape));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 4, &frame_1_shape));

    // Ensure both frames have the same shape
    TF_RETURN_IF_ERROR(c->Merge(frame_0_shape, frame_1_shape, &frame_1_shape));

    // The output has the same size as the input
    c->set_output(0, c->input(0));

    return Status::OK();
});

// GPU launcher
void NearestShaderKernelLauncher(
    const float* input, 
    const float* frame_0,
    const float* frame_1,
    const int n, 
    const int h,
    const int w,
    float* output);

// GPU op
class NearestShaderOpGPU : public OpKernel
{
public:    
    explicit NearestShaderOpGPU(OpKernelConstruction* context) : OpKernel(context) { }

    void Compute(OpKernelContext* context) override 
    {
        // Check the number of tensors
        DCHECK_EQ(3, context->num_inputs());

        // Get the input tensor and the two kernels
        const Tensor& input = context->input(0);
        const Tensor& frame_0 = context->input(1);
        const Tensor& frame_1 = context->input(2);

        // Check the shapes of the input and kernel tensors
        const TensorShape& input_shape = input.shape();
        const TensorShape& frame_0_shape = frame_0.shape();
        const TensorShape& frame_1_shape = frame_1.shape();

        // Get the batch parameters
        const int n = input_shape.dim_size(0);
        const int h = input_shape.dim_size(1);
        const int w = input_shape.dim_size(2);

        // Ensure the batch number matches
        DCHECK_EQ(n, frame_0_shape.dim_size(0));
        DCHECK_EQ(n, frame_1_shape.dim_size(0));

        // Ensure all the input tensors have the same 2D resolution
        DCHECK_EQ(h, frame_0_shape.dim_size(1));
        DCHECK_EQ(h, frame_1_shape.dim_size(1));
        DCHECK_EQ(w, frame_0_shape.dim_size(2));        
        DCHECK_EQ(w, frame_1_shape.dim_size(2));

        // Create output tensor
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output));

        // Launch the GPU kernel
        auto f_input = input.flat<float>();
        auto f_frame_0 = frame_0.flat<float>();
        auto f_frame_1 = frame_1.flat<float>();
        auto f_output = output->template flat<float>();

        NearestShaderKernelLauncher(
            f_input.data(),
            f_frame_0.data(),
            f_frame_1.data(),
            n, 
            h, 
            w,
            f_output.data()
        );
    }
};

REGISTER_KERNEL_BUILDER(Name("NearestShader").Device(DEVICE_GPU), NearestShaderOpGPU);
