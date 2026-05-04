import vector_ta


def test_module_identity_is_vector_ta():
    assert vector_ta.__name__ == "vector_ta"


def test_cuda_batch_plan_symbols_are_complete_when_cuda_enabled():
    expected = {
        "FramaCudaBatchPlan",
        "MabCudaBatchPlan",
        "MediumAdCudaBatchPlan",
        "RsmkCudaBatchPlan",
        "VpciCudaBatchPlan",
        "VpwmaCudaBatchPlan",
        "VwmaCudaBatchPlan",
        "VwmacdCudaBatchPlan",
        "frama_cuda_batch_plan_create",
        "mab_cuda_batch_plan_create",
        "medium_ad_cuda_batch_plan_create",
        "rsmk_cuda_batch_plan_create",
        "vpci_cuda_batch_plan_create",
        "vpwma_cuda_batch_plan_create",
        "vwma_cuda_batch_plan_create",
        "vwmacd_cuda_batch_plan_create",
    }
    present = {name for name in expected if hasattr(vector_ta, name)}
    assert present == set() or present == expected
