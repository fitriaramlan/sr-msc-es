import os
from pysr import PySRRegressor


def get_models(pysr_overrides: dict | None = None):
    # PySR defaults; override with env (e.g. PY_SR_NITERATIONS) or pysr_overrides
    params = dict(
        populations=1,
        population_size=200,
        model_selection="best",
        niterations=300,
        optimizer_iterations=0,
        optimizer_nrestarts=0,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["log", "sin", "cos", "exp", "square", "sqrt"],
        tournament_selection_n=10,
        maxsize=200,
        use_frequency=False,
        warmup_maxsize_by=0,
        deterministic=True,
        parallelism="serial",
        random_state=0,
        procs=0,
        batch_size=64,
        temp_equation_file=True,
        progress=False,
        verbosity=0,
    )

    env_overrides = {
        "PY_SR_NITERATIONS": ("niterations", int),
        "PY_SR_POPULATION_SIZE": ("population_size", int),
        "PY_SR_BATCH_SIZE": ("batch_size", int),
        "PY_SR_PARALLELISM": ("parallelism", str),
        "PY_SR_PROCS": ("procs", int),
        "PY_SR_MAXSIZE": ("maxsize", int),
    }

    for env_key, (param_key, caster) in env_overrides.items():
        val = os.environ.get(env_key)
        if val is not None:
            try:
                params[param_key] = caster(val)
            except ValueError:
                pass

    if pysr_overrides:
        params.update(pysr_overrides)

    pysr = PySRRegressor(**params)
    return {"PySR": pysr}
