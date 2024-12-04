from levanter.eval_harness import LmEvalHarnessConfig, TaskConfig
from test_utils import skip_if_module_missing


@skip_if_module_missing("lm_eval")
def test_task_config():
    task_spec = [
        TaskConfig(
            task="hellaswag",
            task_alias="hellaswag_10shot",
            num_fewshot=10,
        ),
        TaskConfig(
            task="hellaswag",
            task_alias="hellaswag_5shot",
            num_fewshot=5,
        ),
        "lambada_openai",
    ]

    config = LmEvalHarnessConfig(
        task_spec=task_spec,
    )

    q = config.to_task_dict()

    assert len(q) == 3
