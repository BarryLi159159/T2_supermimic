"""SuperMimic tracker as a booster_deploy task (symlinked into booster_deploy/tasks/supermimic)."""

from .tracker import SuperMimicPolicy, SuperMimicPolicyCfg

__all__ = ["SuperMimicPolicy", "SuperMimicPolicyCfg"]
