import torch
import clip

def text_prompt(classes, future=False, only_class_names=False):
    if future:
        text_aug = 'a video of a person {} in the future.'
    else:
        text_aug = 'a video of a person {}.'

    classes = torch.cat([clip.tokenize(c if only_class_names else text_aug.format(c)) for c in classes])

    return classes


def multi_text_prompt(classes):
    text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
                f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
                f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
                f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
                f"The man is {{}}", f"The woman is {{}}", f"a video of a person {{}}", f"{{}}"]
    
    text_dict = {}

    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for c in classes])

    return text_dict
