def _has_spconv():
    for module_name in ("spconv.pytorch", "spconv"):
        try:
            module = __import__(module_name, fromlist=["SparseConv3d"])
            getattr(module, "SparseConv3d")
            return True
        except Exception:
            continue
    return False


if _has_spconv():
    from .scn import SpMiddleResNetFHD
else:
    print("No spconv, sparse convolution disabled!")
