from .classification import HEAD_REGISTRY
# Import submodules so that @register decorators execute
from . import classification  # noqa: F401
from . import token_classification  # noqa: F401
from . import qa  # noqa: F401
from . import similarity  # noqa: F401
from . import regression  # noqa: F401
