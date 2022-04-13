vocab_path = './data/vocab.txt'

EN_WHITELIST = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'  # space is included in whitelist
SYMBOL_WHITELIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n '
VOCAB_WHITELIST = EN_WHITELIST + SYMBOL_WHITELIST
MAX_LEN = None

Start_Token = ["<s>"]
End_Token = ["</s>"]
