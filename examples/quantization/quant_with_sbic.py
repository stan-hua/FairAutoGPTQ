#  Non-standard libraries
import numpy as np
import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


def is_example_biased(example):
    """
    Function to filter for examples intentionally biased towards a certain group

    Parameters
    ----------
    example : dict
        Example row in SBIC dataset
    """
    return (example["whoTarget"] == "0.0") and (example["intentYN"] != "1.0")


def is_example_unbiased(example):
    """
    Function to filter for unbiased examples

    Parameters
    ----------
    example : dict
        Example row in SBIC dataset
    """
    return (example["whoTarget"] == "") and (example["intentYN"] != "0.0")


def get_data(nsamples, seed, seqlen, model,
             mark_biased=True,
             prop_unbiased=0.8):
    """
    Get Social Bias Inference Corpus (SBIC) dataset to use for quantization

    Parameters
    ----------
    nsamples : int
        Number of samples
    seed : int
        Random seed to set
    seqlen : int
        Maximum sequence length to pass through model
    model : str
        Name of model; whose  tokenizer will be imported
    mark_biased : bool, optional
        If True, specify the texts that are biased. Otherwise, let all texts
        be labeled as unbiased.
    prop_unbiased : float, optional
        Proportion of 

    Returns
    -------
    Tuple of (list, list)
        The first sub-list contains all sampled calibration text
        The second sub-list is a list of bools to denote which sample is
        unbiased text
    """
    from datasets import load_dataset

    # Load training/test data
    traindata = load_dataset("allenai/social_bias_frames",
                             split="train",
                             trust_remote_code=True)

    # Split into biased and unbiased text
    biased_train = traindata.filter(is_example_biased)
    unbiased_train = traindata.filter(is_example_unbiased)

    from transformers import AutoTokenizer

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    # Tokenizer text
    biased_train_tokens = tokenizer("\n\n".join(biased_train["post"]), return_tensors="pt")
    unbiased_train_tokens = tokenizer("\n\n".join(unbiased_train["post"]), return_tensors="pt")
    import random

    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)

    calib_train_tokens = []
    is_unbiased = []
    for idx in range(nsamples):
        # Alternate between unbiased and biased text, if specified
        sample_unbiased = random.choices(
            population=[True, False],
            weights=[prop_unbiased, 1 - prop_unbiased],
        )[0]

        if sample_unbiased:
            train_tokens = unbiased_train_tokens
            is_unbiased.append(True)
        else:
            train_tokens = biased_train_tokens
            is_unbiased.append(not mark_biased)

        # Sample input
        i = random.randint(0, train_tokens.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = train_tokens.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        calib_train_tokens.append({"input_ids": inp, "attention_mask": attention_mask})

    return calib_train_tokens, is_unbiased


def main():
    pretrained_model_dir = "facebook/opt-125m"
    quantized_model_dir = "opt-125m-4bit-128g-fairgpt"

    # Get data
    calib_train_tokens, is_unbiased = get_data(
        1024, 0, 2048, pretrained_model_dir,
        mark_biased=True,
        prop_unbiased=0.7,
    )

    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
        desc_act=False,  # desc_act and group size only works on triton
        damp_percent=0.05,
    )

    # load un-quantized model, the model will always be force loaded into cpu
    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    # with value under torch.LongTensor type.
    model.quantize(calib_train_tokens, is_unbiased, use_triton=False)

    # save quantized model
    model.save_quantized(quantized_model_dir)

    # save quantized model using safetensors
    model.save_quantized(quantized_model_dir, use_safetensors=True)

    # load quantized model, currently only support cpu or single gpu
    model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_triton=False)


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
