from goedels_poetry.config.config import parsed_config

# Gather Kimina Lean Server configuration
KIMINA_LEAN_SERVER = {
    "url": parsed_config.get(section="KIMINA_LEAN_SERVER", option="url", fallback="http://0.0.0.0:8000"),
    "max_retries": parsed_config.getint(section="KIMINA_LEAN_SERVER", option="max_retries", fallback=5),
}
