from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
fname="NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
model = AutoModelForCausalLM.from_pretrained(
    f"models/{fname}", trust_remote_code=True
) 
tokenizer = AutoTokenizer.from_pretrained( f"./models/{fname}/")
hybrid_override_pattern=model.config.hybrid_override_pattern




# # print("original length of model", len(model.model.layers))
layers_to_remove=[8,38,45]#[ 3,  4, 20, 21, 22, 23, 28, 29, 31, 32, 38, 39]
layertype_removed=[]
for idx in sorted(layers_to_remove, reverse=True):
    print("deleting layer", idx)
    del model.backbone.layers[idx] 
    layertype_removed.append(hybrid_override_pattern[idx])
    hybrid_override_pattern=hybrid_override_pattern[:idx] + hybrid_override_pattern[idx+1:]


# # print("new length of model", len(model.model.layers))
model.config.num_hidden_layers = len(model.backbone.layers)
model.config.hybrid_override_pattern=hybrid_override_pattern

result = "_".join(map(str, layers_to_remove))
removed_layers_type= "".join(layertype_removed[::-1])
end=f"L{layers_to_remove[0]}"
for i in range(1, len(layers_to_remove)):
    end=f"{end}-{layers_to_remove[i]}"
filename=f"output_path/{fname}_{end}"
print(f"saved at {filename}/")
model.save_pretrained(f"{filename}/")
tokenizer.save_pretrained(f"{filename}/")
