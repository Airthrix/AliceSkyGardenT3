# AliceSkyGardenT3
AliceSkyGardenT3 is a Sparse Activation Architecture for Green Artificial Intelligence
The Energy Efficiency Optimization Language Model AliceSkyGardenT3 Framework Based on Ternary Parameters {-1,0,1}

The simple API allows easy adoption:

   # Compression
   model.compress_model_weights().save("compressed_model")



   # Deployment
   model = AliceSkyGardenT3ForCausalLM.load_compressed_model(
   
   "compressed_model", device="cuda"
   
   )

![training](https://github.com/user-attachments/assets/ae9a8e44-cc3d-4250-9d72-d55eea1deb86)
This picture is Training loss and accuracy curves for AliceSkyGardenT3
---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
![sparsity](https://github.com/user-attachments/assets/12571a0b-bd56-4bbe-9922-c0d9d006c166)
This picture is Weight sparsity by layer index in 24-layer model

The ternary architecture achieves 85.3% weight sparsity on average. The sparsity follows a logarithmic distribution across layers:
s(l) = smax − β log(1 + γl) (30)
where l is layer index, smax = 0.92, β = 0.15, γ = 0.2.

---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
Table: Model Compression Results (7B Parameters) 

Model Size (GB) Bits/Param Compression Ratio 

FP32 Baseline 26.8 32.00 1.00× 

GPTQ 4-bit 3.5 4.00 7.66× 

AliceSkyGardenT3 2.1 1.58 12.76× 

---------------------------------------------------------------------------------
---------------------------------------------------------------------------------
![accuracy](https://github.com/user-attachments/assets/084465f7-7fe2-408c-8c9b-0a5015211f28)
this picture is Accuracy comparison on GLUE benchmark tasks

Despite aggressive quantization, accuracy remains high across multiple benchmarks.

---------------------------------------------------------------------------------
---------------------------------------------------------------------------------

![latency](https://github.com/user-attachments/assets/e9857ef8-fece-4468-8d62-51405e965609)
this picture is Inference latency comparison for different sequence lengths

