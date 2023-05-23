from .reranker_utils import RobertaRanker, LongformerRanker, BartRanker
from .generation_utils import BartForConditionalGeneration
from transformers import RobertaConfig, RobertaTokenizer, LongformerConfig, LongformerTokenizer, BartTokenizer, BartConfig
from transformers import BartConfig, BartTokenizer


def load_reranker_model(model_args, data_args):
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    model_args.reranker_model_type = model_args.reranker_model_type.lower()
    if  model_args.reranker_model_type not in ['roberta', 'longformer', 'bart']:
        raise ValueError('The model type should be either orberta or longformer')
    if model_args.reranker_model_type.lower() == 'roberta':
        reranker_config = RobertaConfig.from_pretrained(
            model_args.reranker_model_name_or_path,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        reranker_tokenizer = RobertaTokenizer.from_pretrained(
            model_args.reranker_model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        setattr(reranker_config, "loss_type", model_args.loss_type)
        setattr(reranker_config, "model_type", model_args.reranker_model_type)

        reranker_model = RobertaRanker.from_pretrained(
            model_args.reranker_model_name_or_path,
            from_tf=bool(".ckpt" in model_args.reranker_model_name_or_path),
            config=reranker_config,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    elif  model_args.reranker_model_type.lower() == 'longformer':
        reranker_config = LongformerConfig.from_pretrained(
            model_args.reranker_model_name_or_path,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        reranker_tokenizer = LongformerTokenizer.from_pretrained(
            model_args.reranker_model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        setattr(reranker_config, "loss_type", model_args.loss_type)
        setattr(reranker_config, "model_type", model_args.reranker_model_type)

        reranker_model = LongformerRanker.from_pretrained(
            model_args.reranker_model_name_or_path,
            from_tf=bool(".ckpt" in model_args.reranker_model_name_or_path),
            config=reranker_config,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        reranker_config = BartConfig.from_pretrained(
            model_args.reranker_model_name_or_path,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        reranker_tokenizer = BartTokenizer.from_pretrained(
            model_args.reranker_model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        setattr(reranker_config, "loss_type", model_args.loss_type)
        setattr(reranker_config, "model_type", model_args.reranker_model_type)

        reranker_model = BartRanker.from_pretrained(
            model_args.reranker_model_name_or_path,
            from_tf=bool(".ckpt" in model_args.reranker_model_name_or_path),
            config=reranker_config,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    if data_args.reranker_max_source_length + data_args.reranker_max_target_length + 4 > reranker_config.max_position_embeddings:
        # How to understand + 4: 
        # the total max input length is data_args.max_source_length + data_args.max_candidate_length + 2 (for bos and sep)
        # and for roberta positional ids starts for 2 (the first position is 2, padding is 1, 0 is unused)
        # therefore total number for new position embedding is data_args.max_source_length + data_args.max_candidate_length + 4
        reranker_model._resize_position_embedding(data_args.reranker_max_source_length + data_args.reranker_max_target_length + 4, extend_way = model_args.position_extend_way)
        reranker_config.max_position_embeddings = data_args.reranker_max_source_length + data_args.reranker_max_target_length + 4

    return reranker_config, reranker_tokenizer, reranker_model




def load_generator_model(model_args, data_args):
    generator_config = BartConfig.from_pretrained(
        model_args.generator_model_name_or_path,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    generator_tokenizer = BartTokenizer.from_pretrained(
        model_args.generator_model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    generator_model = BartForConditionalGeneration.from_pretrained(
        model_args.generator_model_name_or_path,
        from_tf=bool(".ckpt" in model_args.generator_model_name_or_path),
        config=generator_config,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    generator_model.resize_token_embeddings(len(generator_tokenizer))

    if generator_model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    return generator_config, generator_tokenizer, generator_model