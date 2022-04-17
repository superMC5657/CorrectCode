vocab_path = './data/vocab.txt'

EN_WHITELIST = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'  # space is included in whitelist
SYMBOL_WHITELIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n '
MAX_LEN = None
SEQ_LEN = 128

Start_Token = "<s>"
End_Token = "</s>"
PAD_TOKEN = "<pad>"
VOCAB_WHITELIST = [PAD_TOKEN] + [Start_Token] + list(EN_WHITELIST) + list(SYMBOL_WHITELIST) + [End_Token]
