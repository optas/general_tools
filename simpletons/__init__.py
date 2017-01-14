import operator


def sort_dict_by_key(in_dict, reverse=False):
    return sorted(in_dict.items(), key=operator.itemgetter(1), reverse=reverse)
