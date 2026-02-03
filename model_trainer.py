import os
from pysr import PySRRegressor


def get_models(pysr_overrides: dict | None = None):
    """
    Configure and return symbolic regression models with protected operators.
    
    This version includes custom protected operators for more expressiveness:
    - protected_div: x / (y + epsilon) to avoid division by zero
    - protected_log: log(abs(x) + epsilon) to avoid log(0) and log(negative)
    - protected_sqrt: sqrt(abs(x)) to avoid sqrt(negative)
    
    NOTE: Check PySR documentation for the current syntax of defining custom operators.
    The syntax below may need adjustment based on your PySR version.
    
    Use environment variables (e.g. PY_SR_NITERATIONS) or `pysr_overrides`
    to tweak settings without modifying the source.
    """
    params = dict(
        populations=1,
        population_size=200,
        model_selection="best",
        niterations=300,
        optimizer_iterations=0,
        optimizer_nrestarts=0,
        
        # Binary operators: basic arithmetic + protected division
        # Note: You may need to define protected_div in Julia or Python depending on PySR version
        binary_operators=["+", "*", "-", "protected_div"],
        
        # Unary operators: safe functions + protected versions
        unary_operators=["sin", "cos", "exp", "square", "protected_log", "protected_sqrt"],
        
        # Define protected operators
        extra_sympy_mappings={
            "protected_div": lambda x, y: x / (y + 1e-10),  # Protected division
            "protected_log": lambda x: "log(abs(x) + 1e-10)",  # Protected log
            "protected_sqrt": lambda x: "sqrt(abs(x))",  # Protected sqrt
        },
        
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

    # Environment variable overrides
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
