"""Next ASA baseline: direct legal portion selection with bounded JSON extraction."""
import importlib.util
from pathlib import Path


def _load_round09():
    path = Path(__file__).resolve().parent.parent / 'c_argon_portion_first_20261107/run.py'
    spec = importlib.util.spec_from_file_location('asa_portion_format_round09', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


round09 = _load_round09()


def make_asa(modules, chat, embed, catalog, search4):
    """Use round-09 pipeline and chat, even with the Argon transport's wrapper."""
    method = modules['pipeline']
    # Argon's provider wraps its extractor with a chat-local stage condition.
    # Parsing happens after chat returns, so take the frozen pure extractor.
    provider = round09.load_module('asa_portion_format_original_provider',
                                  round09.original.OLD / 'provider.py')
    extract = provider.extract_unique_json
    contract_error = method.current.ContractError
    base = round09.make_asa(method, chat, embed, catalog, search4)

    class CompatiblePortionASA(type(base)):
        def portion(self, image_path, identity, dataset, config, conditional_context=None, examples=None):
            if dataset == 'asa24':
                config = {**config, 'portion_mode': 'portion_first'}
            return super().portion(image_path, identity, dataset, config,
                                   conditional_context=conditional_context, examples=examples)

        def _stage(self, name, prompt, images, parser, stages):
            if name != 'portion_conditional_option_direct':
                return super()._stage(name, prompt, images, parser, stages)

            def strict(value):
                try:
                    return parser(value)
                except RecursionError:
                    raise contract_error('invalid_json') from None

            def parse_portion(raw):
                try:
                    return strict(raw)
                except contract_error as error:
                    if str(error) != 'invalid_json':
                        raise
                    try:
                        candidate = extract(raw)
                    except RecursionError:
                        raise error from None
                    if candidate == raw:
                        raise
                    return strict(candidate)

            return super()._stage(name, prompt, images, parse_portion, stages)

    return CompatiblePortionASA(chat, embed, catalog, base.retriever)
