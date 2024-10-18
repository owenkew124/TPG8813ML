from typing import Any, List, Tuple, Set
from functools import lru_cache

# Implementations of each rule are pulled from https://gist.github.com/rarecoil/8b964b473eb8d47c70dea0a86b772f60#file-rulegen-py3

_i = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18,
    'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27,
    'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35,
}

# Rules are zero arg if they only use x
# Rules are one arg if they use x and y
# Rules are two arg if they use x, y, and z

# check number of args of function
def get_num_args(f):
    return f.__code__.co_argcount


hashcat_rule = {}
# Dummy rule
hashcat_rule[':'] = lambda x: x                                    # Do nothing

        # Case rules
hashcat_rule["l"] = lambda x: x.lower()                            # Lowercase all letters
hashcat_rule["u"] = lambda x: x.upper()                            # Capitalize all letters
hashcat_rule["c"] = lambda x: x.capitalize()                       # Capitalize the first letter
hashcat_rule["C"] = lambda x: x[0].lower() + x[1:].upper()         # Lowercase the first found character, uppercase the rest
hashcat_rule["t"] = lambda x: x.swapcase()                         # Toggle the case of all characters in word
hashcat_rule["E"] = lambda x: " ".join([i[0].upper()+i[1:] for i in x.split(" ")]) # Upper case the first letter and every letter after a space
hashcat_rule["T"] = lambda x,y: x[:y] + x[y].swapcase() + x[y+1:]  # Toggle the case of characters at position N

        # Rotation rules
hashcat_rule["r"] = lambda x: x[::-1]                              # Reverse the entire word
hashcat_rule["{"] = lambda x: x[1:]+x[0]                           # Rotate the word left
hashcat_rule["}"] = lambda x: x[-1]+x[:-1]                         # Rotate the word right

        # Duplication rules
hashcat_rule["d"] = lambda x: x+x                                  # Duplicate entire word
hashcat_rule["f"] = lambda x: x+x[::-1]                            # Duplicate word reversed
hashcat_rule["q"] = lambda x: "".join([i+i for i in x])            # Duplicate every character
hashcat_rule["p"] = lambda x,y: x*y                                # Duplicate entire word N times
hashcat_rule["z"] = lambda x,y: x[0]*y+x                           # Duplicate first character N times
hashcat_rule["Z"] = lambda x,y: x+x[-1]*y                          # Duplicate last character N times
hashcat_rule["y"] = lambda x,y: x[:y]+x                            # Duplicate first N characters
hashcat_rule["Y"] = lambda x,y: x+x[-y:]                           # Duplicate last N characters

        # Cutting rules
hashcat_rule["["] = lambda x: x[1:]                                # Delete first character
hashcat_rule["]"] = lambda x: x[:-1]                               # Delete last character
hashcat_rule["D"] = lambda x,y: x[:y]+x[y+1:]                      # Deletes character at position N
hashcat_rule["'"] = lambda x,y: x[:y]                              # Truncate word at position N
hashcat_rule["O"] = lambda x,y,z: x[:y]+x[y+z:]                    # Delete M characters, starting at position N
hashcat_rule["x"] = lambda x,y,z: x[y:y+z]                         # Extract M characters, starting at position N
hashcat_rule["@"] = lambda x,y: x.replace(y,'')                    # Purge all instances of X

        # Insertion rules
hashcat_rule["$"] = lambda x,y: x+y                                # Append character to end
hashcat_rule["^"] = lambda x,y: y+x                                # Prepend character to front
hashcat_rule["i"] = lambda x,y,z: x[:y]+z+x[y:]                    # Insert character X at position N

        # Replacement rules
hashcat_rule["o"] = lambda x,y,z: x[:y]+z+x[y+1:]                  # Overwrite character at position N with X
hashcat_rule["s"] = lambda x,y,z: x.replace(y,z)                   # Replace all instances of X with Y
hashcat_rule["L"] = lambda x,y: x[:y]+chr(ord(x[y])<<1)+x[y+1:]    # Bitwise shift left character @ N
hashcat_rule["R"] = lambda x,y: x[:y]+chr(ord(x[y])>>1)+x[y+1:]    # Bitwise shift right character @ N
hashcat_rule["+"] = lambda x,y: x[:y]+chr(ord(x[y])+1)+x[y+1:]     # Increment character @ N by 1 ascii value
hashcat_rule["-"] = lambda x,y: x[:y]+chr(ord(x[y])-1)+x[y+1:]     # Decrement character @ N by 1 ascii value
hashcat_rule["."] = lambda x,y: x[:y]+x[y+1]+x[y+1:]               # Replace character @ N with value at @ N plus 1
hashcat_rule[","] = lambda x,y: x[:y]+x[y-1]+x[y+1:]               # Replace character @ N with value at @ N minus 1

        # Swappping rules
hashcat_rule["k"] = lambda x: x[1]+x[0]+x[2:]                      # Swap first two characters
hashcat_rule["K"] = lambda x: x[:-2]+x[-1]+x[-2]                   # Swap last two characters

def star_func(x,y,z):
    return x[:y]+x[z]+x[y+1:z]+x[y]+x[z+1:] if z > y else x[:z]+x[y]+x[z+1:y]+x[z]+x[y+1:] # Swap character X with Y
hashcat_rule["*"] = star_func

num_args_dict = {k: get_num_args(v)-1 for k,v in hashcat_rule.items()}



# Each rule is a list of tuples, where the first element is the rule type and the second element is the arguments
RuleType = str
RuleArgs = List[str]
Rule = List[Tuple[RuleType, RuleArgs]]

def parse_rule(unparsed_rule: str) -> Rule:
    """Take a string representation of a rule and return a list of tuples 
    separating the rule type and arguments
    """
    rule = unparsed_rule
    try:
        result = []
        while rule != "":
            rule_type = rule[0]
            num_args = num_args_dict[rule_type]
            args : List[Any] = [*rule[1:1+num_args]] 
            if rule_type in ["T", "p", "Z", 'z', 'Y', 'y', 'D', "'", 'x', 'O' 
                            "o", "L", "R", "+", "-", ".", ",", "*", "i"]:
                args[0] = _i[args[0]]
            if rule_type in ['x', 'O', '*']:
                args[1] = _i[args[1]]

            rule = rule[1+num_args+1:] # additional +1 to skip space
            result.append((rule_type, args))
        return result
    except Exception as e:
        print(f"Error parsing rule '{unparsed_rule}': {e}")
        raise e
        return None
    
def apply_rule(password: str, rule: Rule) -> str:
    """Apply a rule to a password and return the transformed password"""
    result = password

    for rule_type, args in rule:
        result = hashcat_rule[rule_type](result, *args)
        
    return result

def load_rockyou() -> List[Rule]:
    return load_rules('rockyou-30000.rule')

def load_rockyou1000() -> List[Rule]:
    return load_rules('rockyou-top-1000.rule')

def load_best64() -> List[Rule]:
    return load_rules('best64.rule')
    
def load_rules(path: str) -> List[Rule]:
    """Load rules from a file and return a list of parsed rules"""
    with open(path) as f:
        results = []
        for x in f.readlines():
            if x.startswith('#'):
                continue
            if x == '\n':
                continue
            results.append(parse_rule(x.strip('\n')))
    return results    

def apply_rules(password: str, rules: List[Rule]) -> Set[str]:
    """Apply a list of rules to a password and return the results"""
    output = set()
    for r in rules:
        try:
            output.add(apply_rule(password, r))
        except Exception as e:
            # print(f"Error applying rule {r}: {e}")
            pass
    return output