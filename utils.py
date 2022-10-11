
def to_device(obj, device):
    if isinstance(obj, list):
        return [to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(to_device(list(obj), device))
    elif isinstance(obj, dict):
        retval = dict()
        for key, value in obj.items():
            retval[to_device(key, device)] = to_device(value, device)
        return retval 
    elif hasattr(obj, "to"): 
        return obj.to(device)
    else: 
        return obj

