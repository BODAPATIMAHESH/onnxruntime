# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------


import torch
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer


def group_by_self_and_cross(present_key_values: tuple[torch.Tensor], concat: bool = False):
    """Categorize present_key_values into self and cross attention.

    Split present state from grouped by layer to grouped by self/cross attention.
    Before: (past_key_self_0, past_value_self_0, past_key_cross_0, past_value_cross_0),
            (past_key_self_1, past_value_self_1, past_key_cross_1, past_value_cross_1), ...
    After: (past_key_self_0, past_value_self_0, past_key_self_1, past_value_self_1, ...),
            (past_key_cross_0, past_value_cross_0, past_key_cross_1, past_value_cross_1, ...)

    Args:
        present_key_values: From past_key_values of a model (group by layer)
        concat: If concat self attention with cross attention key/value to return

    Returns:
        present_self (Tuple[torch.Tensor]): present key and values from self attention
        present_cross (Tuple[torch.Tensor]): present key and values from cross attention
    """
    present_self: list[torch.Tensor] = []
    present_cross: list[torch.Tensor] = []
    for _, present_layer_i in enumerate(present_key_values):
        assert len(present_layer_i) == 4, f"Expected to have four items. Got {len(present_layer_i)}"
        present_key_self, present_value_self, present_key_cross, present_value_cross = present_layer_i
        present_self.extend([present_key_self, present_value_self])
        present_cross.extend([present_key_cross, present_value_cross])
    if concat:
        return present_self + present_cross
    else:
        return present_self, present_cross


def back_group_by_layer(past_key_values: tuple[tuple[torch.Tensor]]):
    """Categorize present_key_values from self and cross attention to layer by layer.

    Reorder past state from grouped by self/cross attention to grouped by layer.
    Before: past_key_self_0, past_value_self_0, past_key_self_1, past_value_self_1, ...,
            past_key_cross_0, past_value_cross_0, past_key_cross_1, past_value_cross_1, ...
    After: (past_key_self_0, past_value_self_0, past_key_cross_0, past_value_cross_0),
            (past_key_self_1, past_value_self_1, past_key_cross_1, past_value_cross_1),

    Args:
        present_key_values: From past_key_values of a model (group by self and cross attention)

    Returns:
        past_tuples: present key and values grouped by layer.
    """
    past_tuples = ()
    half_idx = len(past_key_values) // 2
    for i in range(len(past_key_values) // 4):
        idx = 2 * i
        past_tuples += (
            (
                past_key_values[idx],
                past_key_values[idx + 1],
                past_key_values[half_idx + idx],
                past_key_values[half_idx + idx + 1],
            ),
        )
    return past_tuples


def get_input_names(past_key_values: tuple[tuple[torch.Tensor]], encoder=True):
    """Process input names of model wrapper.

    Args:
        past_key_values: Consider `self` and `cross` past_key_values

    Returns:
        names (List[string]): input names
    """
    names = []
    num_layers = len(past_key_values) // 4 if encoder else len(past_key_values)
    prefix = "past_" if not encoder else "present_"
    for i in range(num_layers):
        names.extend([prefix + s for s in [f"key_self_{i}", f"value_self_{i}"]])
    for i in range(num_layers):
        names.extend([prefix + s for s in [f"key_cross_{i}", f"value_cross_{i}"]])
    return names


def get_output_names(past_key_values: tuple[torch.Tensor]):
    """Process output names of model wrapper.

    As cross attention is unchanged during every iteration of beam search,
    we can only consider and calculate based on self attention past_key_values.

    Args:
        past_key_values: Only consider `self` past_key_values

    Returns:
        names (List[string]): input names
    """
    names = []
    num_layers = len(past_key_values)
    prefix = "present_"
    for i in range(num_layers):
        names.extend([prefix + s for s in [f"key_self_{i}", f"value_self_{i}"]])
    return names


def initialize_config(args):
    """Initialize BART config.

    Initilaize BartConfig from pretrained, and customize some of them with user parameters.

    Args:
        args: User inputs.

    Returns:
        config (BartConfig): Bart config.
        tokenizer (BartTokenizer): Bart tokenizer.

    """
    model_dir = args.model_dir
    config = BartConfig.from_pretrained(model_dir)
    tokenizer = BartTokenizer.from_pretrained(model_dir)
    if args.spm_path:
        tokenizer = BartTokenizer(args.spm_path, args.input_text, args.vocab_path, config=config)

    config.use_decoder = True
    assert tokenizer.eos_token_id == config.decoder_start_token_id

    config.do_blenderbot_90_layernorm = False
    config.extra_pos_embeddings = 2
    config.force_bos_token_to_be_generated = False
    config.static_position_embeddings = False

    return config, tokenizer


def initialize_model(config: BartConfig, tokenizer: BartTokenizer, args):
    """Initialize Bart model: BartForConditionalGeneration.

    model initialization and input data preprcessing.

    Args:
        config: Bart config.
        tokenizer: Bart tokenizer.

    Returns:
        model (BartForConditionalGeneration): Bart model.
        input_data (torch.LongTensor): encoded input data
    """

    model_dir = args.model_dir
    device = args.device
    input_text = args.input_text

    model = BartForConditionalGeneration.from_pretrained(model_dir, config=config).eval()

    model = model.to(device)

    lang = "__en__"
    features = [tokenizer.convert_tokens_to_ids(lang)]
    features.extend(
        tokenizer.encode_plus(input_text, add_special_tokens=False, max_length=510, truncation=True)["input_ids"]
    )
    features.append(tokenizer.eos_token_id)
    input_data = torch.LongTensor(features).unsqueeze(0).to(device)

    return model, input_data
