from baselines.prm import PRMStatCalculator
from baselines.reasoneval import ReasonEvalStatCalculator
from lm_polygraph import WhiteboxModel
from lm_polygraph.defaults.register_default_stat_calculators import register_default_stat_calculators, create_container
from lm_polygraph.utils.builder_enviroment_stat_calculator import BuilderEnvironmentStatCalculator
from lm_polygraph.utils.factory_stat_calculator import FactoryStatCalculator
from lm_polygraph.utils.manager import initialize_stat_calculators
from lm_polygraph.stat_calculators import StatCalculator
from lm_polygraph.estimators import Estimator
from bestofn.stat_calculators.sample_generation import SampleGenerationCalculator
from luh.calculator_apply_uq_head import CalculatorApplyUQHead
from synthetic_dataset_generation.utils.steps_extractor import StepsExtractor


def load_relevant_stat_calculators(
        estimators: list[Estimator],
        model: WhiteboxModel,
        uhead_hf_path: str,
        prompt_path: str,
        hf_cache: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
) -> list[StatCalculator]:
    factory = FactoryStatCalculator(BuilderEnvironmentStatCalculator(model=model))
    stat_containers = register_default_stat_calculators(model_type="Whitebox", hf_cache=hf_cache)
    stat_containers += [
        create_container(
            # should shadow GreedyProbsCalculator, so that later calculators
            # will calculate uncertainty wrt sample text, not greedy one
            SampleGenerationCalculator,
            builder="bestofn.stat_calculators.sample_generation",
            default_config={
                "predict_token_uncertainties": False,
                "uq_head_path": uhead_hf_path,
                "args_generate": {
                    "max_new_tokens": max_new_tokens,
                    "min_new_tokens": 2,
                    "temperature": temperature,
                    "length_penalty": 1.0,
                    "stop_strings": ["\n\n", "}\n"],
                },
            },
        ),
        # containers for UHead
        create_container(
            StepsExtractor,
            builder="synthetic_dataset_generation.utils.steps_extractor",
            default_config={},
        ),
        create_container(
            CalculatorApplyUQHead,
            builder="luh.builder_CalculatorApplyUQHead",
            default_config={},
        ),
        # baselines
        create_container(
            PRMStatCalculator,
            builder="baselines.prm",
            default_config={
                "prompt_path": prompt_path,
                "offload_to_cpu_between_calls": True,  # When I run ReasonEval with PRM at the same time, I get CUDA OOM
            },
        ),
        create_container(
            ReasonEvalStatCalculator,
            builder="baselines.reasoneval",
            default_config={
                "prompt_path": prompt_path,
                "offload_to_cpu_between_calls": True,
            },
        )
    ]
    return initialize_stat_calculators(factory, stat_containers, model, estimators)[0]
