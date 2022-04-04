import inspect
import logging
import typing
from collections import defaultdict
from typing import Callable, Dict, Type, TypeVar

T = TypeVar("T", bound="Registrable")

logger = logging.getLogger(__name__)


def get_signatures(klass):
    def _get_signatures(klass):
        return {k: v for k, v in inspect.signature(klass).parameters.items()}

    signatures = {}
    signatures_this_class = _get_signatures(klass)
    signatures.update(signatures_this_class)
    parent_class = klass.__bases__[0]
    while "kwargs" in signatures_this_class:
        signatures_this_class = _get_signatures(parent_class)
        signatures.update(signatures_this_class)
        parent_class = parent_class.__bases__[0]

    return signatures.items()


class FromConfig:
    @classmethod
    def from_config(cls: Type[T], config: Dict, **kwargs) -> T:
        return cls._from_config(config, **kwargs)

    @classmethod
    def _from_config(cls, config, **kwargs):
        for name, param in get_signatures(cls):
            child_class = param.annotation

            if name not in config:
                continue

            if isinstance(child_class, typing._GenericAlias):
                # special case for List[Registrable]
                contained_class = typing.get_args(child_class)[0]
                if inspect.isclass(contained_class) and issubclass(contained_class, FromConfig):
                    child_instance = [contained_class.from_config(c) for c in config[name]]
                    config[name] = child_instance
            # Recursively call the submodule's from_config
            elif issubclass(child_class, FromConfig):
                child_instance = child_class.from_config(config[name])
                config[name] = child_instance
        logger.info(f"Instantiating {cls}...")
        return cls(**config, **kwargs)


class Registrable(FromConfig):
    """
    A simplify version of my_ml's Registrable.
    (https://github.com/allenai/my_ml/blob/master/my_ml/common/registrable.py)
    """

    _registry: Dict[Type, Dict[str, Type]] = defaultdict(dict)

    @classmethod
    def register(cls: Type[T], name: str, exist_ok: bool = False):

        registry = Registrable._registry[cls]

        def add_subclass_to_registry(subclass: Type[T]):

            # Add to registry, raise an error if key has already been used.
            if name in registry:
                if exist_ok:
                    message = (
                        f"{name} has already been registered as {registry[name].__name__}, but "
                        f"exist_ok=True, so overwriting with {cls.__name__}"
                    )
                    logger.info(message)
                else:
                    message = (
                        f"Cannot register {name} as {cls.__name__}; "
                        f"name already in use for {registry[name].__name__}"
                    )
                    raise Exception(message)
            registry[name] = subclass
            return subclass

        return add_subclass_to_registry

    @classmethod
    def by_name(cls: Type[T], name: str) -> Callable[..., T]:
        """
        Returns a callable function that constructs an argument of the registered class.
        """
        logger.debug(f"instantiating registered subclass {name} of {cls}")
        if name not in Registrable._registry[cls]:
            raise Exception(f"``{name}`` not found for {cls}")
        return Registrable._registry[cls][name]

    @classmethod
    def from_config(cls: Type[T], config: Dict, **kwargs) -> T:
        if "type" in config:
            name = config.pop("type")
            logger.debug(f"instantiating registered subclass {name} of {cls} from config")
            logger.debug(config)
            this_class = Registrable._registry[cls][name]
        else:
            this_class = cls

        return this_class._from_config(config, **kwargs)
