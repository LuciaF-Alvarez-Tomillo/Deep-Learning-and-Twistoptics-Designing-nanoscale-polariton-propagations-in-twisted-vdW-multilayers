from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_package_imports():
    import twistoptics  # noqa: F401
    import twistoptics.config  # noqa: F401
    import twistoptics.materials  # noqa: F401
    import twistoptics.models  # noqa: F401
    import twistoptics.physics  # noqa: F401
    import twistoptics.utils  # noqa: F401
