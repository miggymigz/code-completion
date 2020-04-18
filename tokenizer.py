from pygments.lexers import Python3Lexer


class Tokenizer:
    def __init__(self):
        self.lexer = Python3Lexer()

    def tokenize(self, src):
        return [token for _, token in self.lexer.get_tokens(src)]
