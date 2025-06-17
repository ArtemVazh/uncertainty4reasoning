from typing import List, Optional
from omegaconf import OmegaConf

from lm_polygraph.stat_calculators import *
from lm_polygraph.utils.factory_stat_calculator import (
    StatCalculatorContainer,
)


def create_container(
        calculator_class: StatCalculator,
        builder="lm_polygraph.utils.builder_stat_calculator_simple",
        default_config=dict()
) -> StatCalculatorContainer:
    cfg = dict()
    cfg.update(default_config)
    cfg["obj"] = calculator_class.__name__

    return StatCalculatorContainer(
        name=calculator_class.__name__,
        obj=calculator_class,
        builder=builder,
        cfg=OmegaConf.create(cfg),
        dependencies=calculator_class.meta_info()[1],
        stats=calculator_class.meta_info()[0],
    )


def register_default_stat_calculators(
        model_type: str,
        language: str = "en",
        hf_cache: Optional[str] = None,
        blackbox_supports_logprobs: bool = False,
        output_attentions: bool = True,
        deberta_batch_size: int = 100,
        deberta_device: str | None = None,
) -> List[StatCalculatorContainer]:
    """
    Specifies the list of the default stat_calculators that could be used in the evaluation scripts and
    estimate_uncertainty() function with default configurations.
    """

    all_stat_calculators = []

    def _register(
            calculator_class: StatCalculator,
            builder="lm_polygraph.utils.builder_stat_calculator_simple",
            default_config=dict(),
    ):
        all_stat_calculators.append(create_container(calculator_class, builder, default_config))

    if language == "en":
        deberta_model_path = "microsoft/deberta-large-mnli"
    else:
        deberta_model_path = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

    nli_config = {
        "deberta_path": deberta_model_path,
        "hf_cache": hf_cache,
        "batch_size": deberta_batch_size,
        "device": deberta_device,
    }

    _register(InitialStateCalculator)
    _register(
        SemanticMatrixCalculator,
        "lm_polygraph.defaults.stat_calculator_builders.default_SemanticMatrixCalculator",
        {
            "nli_model": nli_config
        },
    )
    _register(SemanticClassesCalculator)

    if model_type == "Blackbox":
        _register(BlackboxGreedyTextsCalculator)
        _register(BlackboxSamplingGenerationCalculator)
        if blackbox_supports_logprobs:
            # For blackbox models that support logprobs (like OpenAI models)
            _register(EntropyCalculator)
            _register(
                GreedyAlternativesNLICalculator,
                "lm_polygraph.defaults.stat_calculator_builders.default_GreedyAlternativesNLICalculator",
                {
                    "nli_model": nli_config
                },
            )

    elif model_type == "Whitebox":
        _register(
            GreedyProbsCalculator,
            "lm_polygraph.defaults.stat_calculator_builders.default_GreedyProbsCalculator",
            {
                "output_attentions": True,
            },
        )
        _register(EntropyCalculator)
        _register(GreedyLMProbsCalculator)
        _register(PromptCalculator)
        _register(SamplingGenerationCalculator)
        _register(BartScoreCalculator)
        _register(ModelScoreCalculator)
        _register(EnsembleTokenLevelDataCalculator)
        _register(PromptCalculator)
        _register(SamplingPromptCalculator)
        _register(ClaimPromptCalculator)
        _register(
            CrossEncoderSimilarityMatrixCalculator,
            "lm_polygraph.defaults.stat_calculator_builders.default_CrossEncoderSimilarityMatrixCalculator",
            {
                "batch_size": 10,
                "cross_encoder_name": "cross-encoder/stsb-roberta-large",
            },
        )
        _register(
            GreedyAlternativesNLICalculator,
            "lm_polygraph.defaults.stat_calculator_builders.default_GreedyAlternativesNLICalculator",
            {
                "nli_model": nli_config
            },
        )
        _register(
            GreedyAlternativesFactPrefNLICalculator,
            "lm_polygraph.defaults.stat_calculator_builders.default_GreedyAlternativesFactPrefNLICalculator",
            {
                "nli_model": nli_config
            },
        )
        _register(
            ClaimsExtractor,
            "lm_polygraph.defaults.stat_calculator_builders.default_ClaimsExtractor",
            {"openai_model": "gpt-4o", "cache_path": "~/.cache", "language": language},
        )

    else:
        raise NotImplementedError(f"Unknown model type: {model_type}")

    return all_stat_calculators
