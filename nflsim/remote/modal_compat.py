import warnings


def artifact_mount(volume):
    options = getattr(volume, "with_mount_options", None)
    if options is not None:
        return options(read_only=True)
    warnings.warn(
        "This Modal SDK lacks read-only mount options. Using a standard isolated "
        "volume mount; ridge workers only read immutable hash-addressed bundles. "
        "Upgrade Modal for mount-level read-only enforcement.",
        RuntimeWarning,
        stacklevel=2,
    )
    return volume