import re

def inject_extended_rules(rule_groups, expand_patterns):
    for rules in rule_groups:
        for rule in rules:
            for key, val in expand_patterns.items():
                rule[0] = rule[0].replace(key, val)
    return rule_groups

def apply_rule(text, rule):

    if re.match(rule[0], text.lower(), flags=re.I):
        # Make sure anything in "..." quotes is unaffected by rules
        text = text.replace('\\"', '"')
        quote_regex = r'"[^"]+"'
        quote_match = re.search(quote_regex, text, flags=re.I)
        if quote_match:
            quote = quote_match.group()
            text = re.sub(quote_regex, '"some random quoted stuff"', text, flags=re.I)
        text = re.sub(rule[0], rule[1], text, flags=re.I)
        text = text.replace('?', '.')
        if quote_match:
            text = text.replace('"some random quoted stuff"', quote)
        return text
    return text

def rule_transformations(text, groups):
    for rules in groups:
        for rule in rules:
            text = apply_rule(text, rule)
    return text

def post_rule_transformations(text):
    if text.strip().endswith("?"):
        text = text[:-1]+" something."
    return text

def question_to_implicit_hypothesis(text):
    transformed_text = rule_transformations(text, expanded_rule_groups)
    rule_applied = (text != transformed_text)
    transformed_text = post_rule_transformations(transformed_text)
    return transformed_text

expand_patterns = {'do': r'does|do|did',
                   'is': r'is|was|are|were',
                   'has': r'has|have|had',
                   'can': r'can|could|should|ought|may|might|would|will',
                   'enum': r'give|name|list|provide',
                   'break': r', |and |^'}

rule_groups = [
          [[r"(.*)what (do) (.+)", r"\1something \2 \3"],
           [r"(.*)what (is) (.+)", r"\1there \2 \3"],
           [r"(.*)what's (.+)", r"\1there is \2"],
           [r"(.*)what (has) (.+)", r"\1someone \2 \3"],
           [r"(.*)what (can) (.+)", r"\1someone \2 \3"],
           [r"(.*)what (.+) (do) (.+)", r"\1some \2 \3 \4"],
           [r"(.*)what (.+) (is) (.+)", r"\1some \2 \3 \4"],
           [r"(.*)what (.+) (has) (.+)", r"\1someone \2 \3 \4"],
           [r"(.*)what (.+) (can) (.+)", r"\1some \2 \3 \4"],
           [r"(.*)what(.+)", r"\1some\2"]],

          [[r"(.*)whose (.+)", r"\1someone's \2"],
           [r"(.*)who's (.+)", r"\1someone's \2"],
           [r"(.*)who (.+)", r"\1someone \2"],
           [r"(.*)who(.+)", r"\1some\2"]],

          [[r"(.*)why (is) (.+)", r"\1for some reason \3"],
           [r"(.*)why (do) (.+)", r"\1for some reason \3"],
           [r"(.*)why (has) (.+)", r"\1for some reason \3"],
           [r"(.*)why (can) (.+)", r"\1for some reason \3"],
           [r"(.*)why(.+)", r"\1for some reason\2"]],

          [[r"(.*)where(.+)", r"\1somewhere\2"],
           [r"(.*)where\?", r"\1somewhere."]],

          [[r"(.*)when(.+)", r"\1sometime\2"],
           [r"(.*)when\?", r"\1sometime."]],

          [[r"(.*)which (.+)", r"\1some \2"]],

          [[r"(.*)how many (.+)", r"\1some \2"],
           [r"(.*)how old (.+)", r"\1some age has \2"],
           [r"(.*)how much (.+)", r"\1some quantity \2"],
           [r"(.*)how often (.+)", r"\1sometimes \2"],
           [r"(.*)how (is) (.+)", r"\1somehow \2 \3"],
           [r"(.*)how's (.+)", r"\1somehow is \2"],
           [r"(.*)how (do) (.+)", r"\1somehow \2 \3"],
           [r"(.*)how (has) (.+)", r"\1somehow \2 \3"],
           [r"(.*)how (can) (.+)", r"\1somehow \2 \3"],
           [r"(.*)how (.+) (do) (.+)", r"\1somehow \2 \3 \4"],
           [r"(.*)how (.+) (is) (.+)", r"\1somehow \2 \3 \4"],
           [r"(.*)how (.+) (has) (.+)", r"\1somehow \2 \3 \4"],
           [r"(.*)how (.+) (can) (.+)", r"\1somehow \2 \3 \4"],
           [r"(.*)how (.+) (can) (.+)", r"\1somehow \2 \3 \4"],
           [r"(.*)how(.+)", r"\1somehow\2"]],

          [[r"^(is) (.+)", r"\2"], [r"(.+)(break)(is) (.+)", r"\1\2\4"],
           [r"^(has) (.+)", r"\2"], [r"(.+)(break)(has) (.+)", r"\1\2\4"],
           [r"^(can) (.+)", r"\2"], [r"(.+)(break)(can) (.+)", r"\1\2\4"],
           [r"^(do) (.+)", r"\2"], [r"(.+)(break)(do) (.+)", r"\1\2\4"]],

          [[r"(.*)true or false(.*)", r"\1 \2"]],

          [[r"(enum) (.+)", r"\2"]],

          [[r"explain (.+)", r"\1"],
           [r"describe (.+)", r"\1"]],

        ]

expanded_rule_groups = inject_extended_rules(rule_groups, expand_patterns)

if __name__ == "__main__":

    # TODO:
    # Make a usage example here.
    pass
