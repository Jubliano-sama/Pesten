#!/usr/bin/env python
"""
convert_to_onnx.py

This script converts a trained PyTorch student model (actor or critic) to the ONNX format.
ONNX (Open Neural Network Exchange) is an open format built to represent machine learning
models, allowing them to be used across different frameworks and runtimes for high-performance
inference.

This script can:
- Convert either the actor (StudentNet) or the critic (StudentCritic) model.
- Load weights from a specified .pth file.
- Set dynamic axes for the batch size, making the ONNX model flexible.
- Optionally, verify the converted model against the original PyTorch model using ONNX Runtime.
"""

import argparse
import sys
import torch
import torch.onnx
import numpy as np

# Import the model definitions.
# We import from ppo2.py as it contains the final definitions for the larger student models.
try:
    from ppo2 import StudentNet, StudentCritic
except ImportError:
    print("Error: Could not import model definitions from ppo2.py.")
    print("Please ensure this script is run from the root of the repository.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch student model to ONNX format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model_type',
        type=str,
        required=True,
        choices=['actor', 'critic'],
        help="The type of model to convert."
    )
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help="Path to the input PyTorch model weights file (.pth)."
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help="Path to save the output ONNX model. If not provided, it will be derived from the input path."
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help="Verify the converted ONNX model by comparing its output with the original PyTorch model."
    )

    args = parser.parse_args()

    # Determine output path if not specified
    if args.output_path is None:
        args.output_path = args.input_path.replace('.pth', '.onnx')

    print(f"Converting {args.model_type} model...")
    print(f"  Input PyTorch model: {args.input_path}")
    print(f"  Output ONNX model:   {args.output_path}")

    # 1. Initialize the correct model architecture
    if args.model_type == 'actor':
        model = StudentNet(input_dim=121, output_dim=55)
        input_names = ['observation']
        output_names = ['logits']
    elif args.model_type == 'critic':
        model = StudentCritic(input_dim=121, output_dim=1)
        input_names = ['observation']
        output_names = ['value']
    else:
        # This case is handled by argparse choices, but we keep it for safety
        raise ValueError(f"Unknown model type: {args.model_type}")

    # 2. Load the trained weights
    try:
        model.load_state_dict(torch.load(args.input_path, map_location='cpu'))
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)

    # 3. Set the model to evaluation mode
    # This is crucial for reproducibility as it disables layers like Dropout
    model.eval()

    # 4. Create a dummy input tensor with the correct shape
    # The shape is (batch_size, num_features) -> (1, 121)
    dummy_input = torch.randn(1, 121, requires_grad=False)

    # 5. Export the model to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            args.output_path,
            export_params=True,
            opset_version=12,  # A commonly used opset version
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                input_names[0]: {0: 'batch_size'},  # Make the batch dimension dynamic
                output_names[0]: {0: 'batch_size'}
            }
        )
        print(f"\n✅ Model successfully converted to {args.output_path}")
    except Exception as e:
        print(f"\n❌ Error during ONNX export: {e}")
        sys.exit(1)

    # 6. Optionally, verify the exported model
    if args.check:
        print("\nVerifying the ONNX model...")
        try:
            import onnx
            import onnxruntime

            # Check that the model is well-formed
            onnx_model = onnx.load(args.output_path)
            onnx.checker.check_model(onnx_model)
            print("  - ONNX model structure is valid.")

            # Compare the outputs of the PyTorch and ONNX models
            ort_session = onnxruntime.InferenceSession(args.output_path)
            
            # Get PyTorch model output
            with torch.no_grad():
                pytorch_output = model(dummy_input)

            # Get ONNX model output
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
            ort_outputs = ort_session.run(None, ort_inputs)

            # Compare outputs
            np.testing.assert_allclose(
                pytorch_output.numpy(),
                ort_outputs[0],
                rtol=1e-03,
                atol=1e-05
            )
            print("  - Verification successful: PyTorch and ONNX model outputs match.")
            print("✅ Verification complete.")

        except ImportError:
            print("\n⚠️ Verification skipped: `onnx` or `onnxruntime` not installed.")
            print("   Please install them with: pip install onnx onnxruntime")
        except Exception as e:
            print(f"\n❌ Verification failed: {e}")


if __name__ == '__main__':
    main()