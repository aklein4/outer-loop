
import datasets


def get_splits(url, subset, remove=[]):
    all_splits = list(datasets.get_dataset_split_names(url, subset))
    for s in remove:
        assert s in all_splits, f"Expected split {s} not found in dataset."
        all_splits.remove(s)
    return all_splits

def get_subsets(url, remove=[]):
    all_subsets = list(datasets.get_dataset_config_names(url))
    for s in remove:
        assert s in all_subsets, f"Expected subset {s} not found in dataset."
        all_subsets.remove(s)
    return all_subsets


def convert_role(role):
    if role in ["system", "user", "assistant"]:
        return role
    if role == "human":
        return "user"
    if role == "gpt":
        return "assistant"
    return role

def convert_role_conversation(conversation):
    for turn in conversation:

        if "role" not in turn:
            turn["role"] = turn.pop("from")
        if "content" not in turn:
            turn["content"] = turn.pop("value")

        turn["role"] = convert_role(turn["role"])

    return conversation


def remove_think(content):
    if "</think>" in content:
        return content.split("</think>")[-1].strip()
    return content


def key_filter(example, keys):
    for key in keys:
        if example.get(key, None) is None:
            return False
    return True


def single_turn(conversation):
    out = []
    for turn in conversation:
        out.append(turn)
        if turn["role"] == "assistant":
            break
    return out


def system_turn(content):
    return {"role": "system", "content": content}

def user_turn(content):
    return {"role": "user", "content": content}

def assistant_turn(content):
    return {"role": "assistant", "content": content}


def simple_format(*contents):
    if len(contents) == 2:
        return [
            user_turn(contents[0]), assistant_turn(contents[1])
        ]
    elif len(contents) == 3:
        return [
            system_turn(contents[0]), user_turn(contents[1]), assistant_turn(contents[2])
        ]
    else:
        raise ValueError("simple_format expects 2 or 3 arguments.")


def clean_conversation(conversation):
    for turn in conversation:
        if turn["content"] is not None:
            turn["content"] = turn["content"].strip()
    return conversation
