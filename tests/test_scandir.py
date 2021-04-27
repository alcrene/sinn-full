import sinnfull
sinnfull.setup('numpy')

def test_model_scandir():
    import sinnfull.models
    # Assert that collections are not empty
    assert sinnfull.models.models
    assert sinnfull.models.objectives
    assert sinnfull.models.priors
    assert sinnfull.models.paramsets

    # Assert that expected tags are there
    assert sinnfull.models.models.OU
    assert sinnfull.models.objectives.OU
    assert sinnfull.models.priors.OU
    assert sinnfull.models.paramsets.OU
    assert sinnfull.models.paramsets.WC

def test_optim_scandir():
    import sinnfull.optim
    # Assert that collections are not empty
    assert sinnfull.optim.paramsets

    # Assert that expected tags are there
    assert sinnfull.optim.paramsets.OU
