import json
import os

from logger import logger

from transformers import AutoConfig as AC, AutoModelForCausalLM as AM
from .mpt import MPTConfig, MPTForCausalLM


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


class AutoConfig():
    def from_pretrained(path, *args, **kwargs):
        js = load_json(os.path.join(path, "config.json"))

        auto_map = js.get("auto_map")
        if auto_map:
            auto_config = auto_map.get("AutoConfig")

            if auto_config == "configuration_mpt.MPTConfig":
                logger.info("Patching MPT config load")
                return MPTConfig.from_pretrained(path, *args, **kwargs)

        return AC.from_pretrained(path, *args, **kwargs)


class AutoModelForCausalLM():
    def from_pretrained(path, *args, **kwargs):
        js = load_json(os.path.join(path, "config.json"))

        auto_map = js.get("auto_map")
        if auto_map:
            auto_config = auto_map.get("AutoModelForCausalLM")

            if auto_config == "modeling_mpt.MPTForCausalLM":
                logger.info("Patching MPT model load")
                return MPTForCausalLM.from_pretrained(path, *args, **kwargs)

        return AM.from_pretrained(path, *args, **kwargs)
