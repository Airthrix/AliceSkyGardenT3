<h1 align="center">AliceSkyGardenT3</h1>

# This is a Sparse Activation Architecture for Green Artificial Intelligence: The Energy Efficiency Optimization Language Model AliceSkyGardenT3 Framework Based on Ternary Parameters {-1,0,1}

 
Due to the involvement of business secrets, it has not been fully open-sourced yet, so I encrypted the framework code to local API. You just need to make sure put ".API_KEY.key" "API_KEY.bin" "API_SAFE.bin" three files in the path. 

>After adding my ".API_KEY.key" and "API_KEY.bin" in the path, you can directly run python train_vocab.py for training ^^ Enjoy your training (and don't forget to create a data folder and put dataset into it)

      python train_vocab.py --resume_checkpoint checkpoint.pth  --use_amp
>If you have completed the training, you can interact with the command python interact_vocab.py

      python interact_vocab.py


![training](https://github.com/user-attachments/assets/7dcc44b7-f85a-4468-a673-cbe84a9ed07f)
<p align="center">
This picture is Training loss and accuracy curves for AliceSkyGardenT3
</p>

---------------------------------------------------------------------------------

>## When use this ".API_KEY.key" "API_KEY.bin" "API_SAFE.bin" you need:
      Python 3.9+ (Recommend 3.10,3.11,3.12)
      pip uninstall Crypto
      pip install pycryptodome
(Due to API data encryption and compression, volume of modeling_aliceskygarden_t3.py file has been temporarily reduced to 22KB, normal original file size is 89KB. If open sourced in the future, the volume will be the normal 89KB)
    
---------------------------------------------------------------------------------

>## Note:
The train_vocab.py here is just an example, but it can run 100% successfully. You can change the loading of the data set from pkl to h5 by yourself. Learning Rate lr is suggested to be increased to 2.6e-5. And the use of vocab.json can be replaced with a tokenizer (I have reserved tokenizer function in my train_vocab.py code).


---------------------------------------------------------------------------------
>## Compression (Already included in the framework)
      model.compress_model_weights().save("compressed_model")
(Before training, The framework will automatically quantify most of the parameters to {-1,0,1}. Finally, compressed_weights.safetensors generated after final training is the weight file with the smallest volume)


---------------------------------------------------------------------------------
>## Deployment (Already included in the framework)
      model = AliceSkyGardenT3ForCausalLM.load_compressed_model("compressed_model", device="cuda")
(If the GPU or CPU supports Ternary Operation in the future, there is no need to call for decompression, just run the original compressed weight file compressed_weights.safetensors directly)


---------------------------------------------------------------------------------
![sparsity](https://github.com/user-attachments/assets/12571a0b-bd56-4bbe-9922-c0d9d006c166)
<p align="center">
This picture is Weight sparsity by layer index in 24-layer model
</p>
The ternary architecture achieves 85.3% weight sparsity on average. The sparsity follows a logarithmic distribution across layers:
s(l) = smax − β log(1 + γl) (30)
where l is layer index, smax = 0.92, β = 0.15, γ = 0.2.


---------------------------------------------------------------------------------
## Model Compression Results (7B Parameters)

| Model              | Model Size (GB) | Bits/Param | Compression Ratio |
|--------------------|:---------------:|:----------:|:-----------------:|
| FP32 Baseline      | 26.8            | 32.00      | 1.00×             |
| GPTQ 4-bit         | 3.5             | 4.00       | 7.66×             |
| AliceSkyGardenT3   | 2.1             | 1.58       | 12.76×            |


---------------------------------------------------------------------------------
![accuracy](https://github.com/user-attachments/assets/084465f7-7fe2-408c-8c9b-0a5015211f28)
<p align="center">
this picture is Accuracy comparison on GLUE benchmark tasks
</p>
Despite aggressive quantization, accuracy remains high across multiple benchmarks.


---------------------------------------------------------------------------------

![latency](https://github.com/user-attachments/assets/e9857ef8-fece-4468-8d62-51405e965609)
<p align="center">
this picture is Inference latency comparison for different sequence lengths
</p>
